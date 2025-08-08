"""
GPU memory optimization and CUDA stream management for high-performance processing.
"""

import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Any, Tuple, Iterator
from dataclasses import dataclass
from contextlib import contextmanager
import logging
import time
import threading
from collections import defaultdict, deque
import gc

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    device_id: int
    total_memory: int
    allocated_memory: int
    cached_memory: int
    free_memory: int
    utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "total_memory_gb": self.total_memory / (1024**3),
            "allocated_memory_gb": self.allocated_memory / (1024**3),
            "cached_memory_gb": self.cached_memory / (1024**3),
            "free_memory_gb": self.free_memory / (1024**3),
            "utilization_percent": self.utilization * 100
        }


@dataclass
class StreamConfig:
    """Configuration for CUDA streams."""
    max_streams: int = 8
    priority_levels: int = 3
    enable_async_memory_ops: bool = True
    stream_synchronization_policy: str = "lazy"  # "lazy", "eager", "adaptive"


class GPUMemoryPool:
    """Memory pool for efficient GPU memory management."""
    
    def __init__(self, device_id: int, initial_size: int = 1024 * 1024 * 1024):  # 1GB
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.initial_size = initial_size
        
        # Memory blocks organized by size
        self.free_blocks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.allocated_blocks: Dict[int, torch.Tensor] = {}
        self.block_sizes: Dict[int, int] = {}
        
        # Statistics
        self.total_allocated = 0
        self.peak_allocated = 0
        self.allocation_count = 0
        self.deallocation_count = 0
        
        self._lock = threading.Lock()
        self._next_block_id = 0
        
        logger.info(f"Initialized GPU memory pool for device {device_id}")
    
    def allocate(self, size: int, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, int]:
        """Allocate memory block from pool."""
        with self._lock:
            # Round up to nearest power of 2 for efficient reuse
            actual_size = self._round_up_to_power_of_2(size)
            
            # Try to reuse existing block
            if actual_size in self.free_blocks and self.free_blocks[actual_size]:
                tensor = self.free_blocks[actual_size].pop()
                block_id = self._next_block_id
                self._next_block_id += 1
                
                self.allocated_blocks[block_id] = tensor
                self.block_sizes[block_id] = actual_size
                self.allocation_count += 1
                
                logger.debug(f"Reused memory block {block_id} of size {actual_size}")
                return tensor, block_id
            
            # Allocate new block
            try:
                tensor = torch.empty(actual_size // dtype.itemsize, dtype=dtype, device=self.device)
                block_id = self._next_block_id
                self._next_block_id += 1
                
                self.allocated_blocks[block_id] = tensor
                self.block_sizes[block_id] = actual_size
                
                self.total_allocated += actual_size
                self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                self.allocation_count += 1
                
                logger.debug(f"Allocated new memory block {block_id} of size {actual_size}")
                return tensor, block_id
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Try garbage collection and retry
                    self._garbage_collect()
                    tensor = torch.empty(actual_size // dtype.itemsize, dtype=dtype, device=self.device)
                    block_id = self._next_block_id
                    self._next_block_id += 1
                    
                    self.allocated_blocks[block_id] = tensor
                    self.block_sizes[block_id] = actual_size
                    
                    self.total_allocated += actual_size
                    self.peak_allocated = max(self.peak_allocated, self.total_allocated)
                    self.allocation_count += 1
                    
                    logger.warning(f"Allocated memory block {block_id} after garbage collection")
                    return tensor, block_id
                else:
                    raise
    
    def deallocate(self, block_id: int):
        """Return memory block to pool."""
        with self._lock:
            if block_id not in self.allocated_blocks:
                logger.warning(f"Attempting to deallocate unknown block {block_id}")
                return
            
            tensor = self.allocated_blocks.pop(block_id)
            size = self.block_sizes.pop(block_id)
            
            # Return to free pool
            self.free_blocks[size].append(tensor)
            self.deallocation_count += 1
            
            logger.debug(f"Deallocated memory block {block_id} of size {size}")
    
    def _round_up_to_power_of_2(self, size: int) -> int:
        """Round size up to nearest power of 2."""
        import math
        return 2 ** math.ceil(math.log2(size))
    
    def _garbage_collect(self):
        """Perform garbage collection to free memory."""
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("Performed garbage collection")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                "device_id": self.device_id,
                "total_allocated": self.total_allocated,
                "peak_allocated": self.peak_allocated,
                "allocation_count": self.allocation_count,
                "deallocation_count": self.deallocation_count,
                "active_blocks": len(self.allocated_blocks),
                "free_block_sizes": dict(self.free_blocks.keys())
            }


class CUDAStreamManager:
    """Manager for CUDA streams with priority scheduling."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.device_streams: Dict[int, List[cuda.Stream]] = {}
        self.stream_priorities: Dict[cuda.Stream, int] = {}
        self.stream_usage_count: Dict[cuda.Stream, int] = defaultdict(int)
        self.stream_queue: Dict[int, deque] = {}  # Device -> priority queue
        
        self._lock = threading.Lock()
        
        # Initialize streams for available devices
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                self._initialize_streams_for_device(device_id)
        
        logger.info(f"Initialized CUDA stream manager with {config.max_streams} streams per device")
    
    def _initialize_streams_for_device(self, device_id: int):
        """Initialize streams for a specific device."""
        streams = []
        
        with torch.cuda.device(device_id):
            for i in range(self.config.max_streams):
                # Calculate priority (higher number = higher priority)
                priority = i % self.config.priority_levels
                
                stream = cuda.Stream(priority=priority)
                streams.append(stream)
                self.stream_priorities[stream] = priority
        
        self.device_streams[device_id] = streams
        self.stream_queue[device_id] = deque(streams)
        
        logger.debug(f"Initialized {len(streams)} streams for device {device_id}")
    
    def get_stream(self, device_id: int, priority: int = 0) -> cuda.Stream:
        """Get a CUDA stream with specified priority."""
        if device_id not in self.device_streams:
            raise ValueError(f"Device {device_id} not available")
        
        with self._lock:
            # Try to get stream with requested priority
            available_streams = [
                s for s in self.device_streams[device_id]
                if self.stream_priorities[s] >= priority
            ]
            
            if not available_streams:
                # Fall back to any available stream
                available_streams = self.device_streams[device_id]
            
            # Get least used stream
            stream = min(available_streams, key=lambda s: self.stream_usage_count[s])
            self.stream_usage_count[stream] += 1
            
            logger.debug(f"Assigned stream with priority {self.stream_priorities[stream]} for device {device_id}")
            return stream
    
    def release_stream(self, stream: cuda.Stream):
        """Release a CUDA stream back to the pool."""
        with self._lock:
            if stream in self.stream_usage_count:
                self.stream_usage_count[stream] = max(0, self.stream_usage_count[stream] - 1)
    
    @contextmanager
    def stream_context(self, device_id: int, priority: int = 0):
        """Context manager for using a CUDA stream."""
        stream = self.get_stream(device_id, priority)
        try:
            with torch.cuda.stream(stream):
                yield stream
        finally:
            self.release_stream(stream)
    
    def synchronize_all_streams(self, device_id: Optional[int] = None):
        """Synchronize all streams on specified device(s)."""
        if device_id is not None:
            devices = [device_id]
        else:
            devices = list(self.device_streams.keys())
        
        for dev_id in devices:
            for stream in self.device_streams[dev_id]:
                stream.synchronize()
        
        logger.debug(f"Synchronized all streams for devices: {devices}")
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream usage statistics."""
        stats = {}
        
        for device_id, streams in self.device_streams.items():
            device_stats = {
                "total_streams": len(streams),
                "stream_usage": {},
                "priority_distribution": defaultdict(int)
            }
            
            for stream in streams:
                usage_count = self.stream_usage_count[stream]
                priority = self.stream_priorities[stream]
                
                device_stats["stream_usage"][str(stream)] = usage_count
                device_stats["priority_distribution"][priority] += 1
            
            stats[f"device_{device_id}"] = device_stats
        
        return stats


class GPUMemoryManager:
    """Comprehensive GPU memory management system."""
    
    def __init__(self, enable_memory_pool: bool = True, stream_config: Optional[StreamConfig] = None):
        self.enable_memory_pool = enable_memory_pool
        self.memory_pools: Dict[int, GPUMemoryPool] = {}
        
        # Initialize stream manager
        self.stream_manager = CUDAStreamManager(stream_config or StreamConfig())
        
        # Memory monitoring
        self.memory_stats_history: Dict[int, List[GPUMemoryStats]] = defaultdict(list)
        self.monitoring_active = False
        self._monitoring_thread = None
        
        if torch.cuda.is_available() and enable_memory_pool:
            for device_id in range(torch.cuda.device_count()):
                self.memory_pools[device_id] = GPUMemoryPool(device_id)
        
        logger.info(f"Initialized GPU memory manager for {len(self.memory_pools)} devices")
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                       device_id: int = 0) -> torch.Tensor:
        """Allocate tensor using memory pool if available."""
        if self.enable_memory_pool and device_id in self.memory_pools:
            size = torch.tensor(shape).prod().item() * dtype.itemsize
            tensor, block_id = self.memory_pools[device_id].allocate(size, dtype)
            
            # Reshape to desired shape
            return tensor[:torch.tensor(shape).prod().item()].view(shape)
        else:
            # Standard allocation
            return torch.empty(shape, dtype=dtype, device=f"cuda:{device_id}")
    
    def get_memory_stats(self, device_id: Optional[int] = None) -> Dict[int, GPUMemoryStats]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        devices = [device_id] if device_id is not None else list(range(torch.cuda.device_count()))
        stats = {}
        
        for dev_id in devices:
            with torch.cuda.device(dev_id):
                total_memory = torch.cuda.get_device_properties(dev_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(dev_id)
                cached_memory = torch.cuda.memory_reserved(dev_id)
                free_memory = total_memory - cached_memory
                utilization = allocated_memory / total_memory
                
                stats[dev_id] = GPUMemoryStats(
                    device_id=dev_id,
                    total_memory=total_memory,
                    allocated_memory=allocated_memory,
                    cached_memory=cached_memory,
                    free_memory=free_memory,
                    utilization=utilization
                )
        
        return stats
    
    def start_memory_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                stats = self.get_memory_stats()
                for device_id, stat in stats.items():
                    self.memory_stats_history[device_id].append(stat)
                    
                    # Keep only last 1000 entries
                    if len(self.memory_stats_history[device_id]) > 1000:
                        self.memory_stats_history[device_id].pop(0)
                
                time.sleep(interval)
        
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
        
        logger.info(f"Started memory monitoring with {interval}s interval")
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped memory monitoring")
    
    def optimize_memory_usage(self, device_id: Optional[int] = None):
        """Optimize memory usage through garbage collection and cache clearing."""
        devices = [device_id] if device_id is not None else list(range(torch.cuda.device_count()))
        
        for dev_id in devices:
            with torch.cuda.device(dev_id):
                # Clear unused cached memory
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
        
        logger.info(f"Optimized memory usage for devices: {devices}")
    
    def get_pool_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get memory pool statistics."""
        return {device_id: pool.get_stats() for device_id, pool in self.memory_pools.items()}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU management statistics."""
        memory_stats = {dev_id: stats.to_dict() for dev_id, stats in self.get_memory_stats().items()}
        
        return {
            "memory_stats": memory_stats,
            "pool_stats": self.get_pool_stats(),
            "stream_stats": self.stream_manager.get_stream_stats(),
            "monitoring_active": self.monitoring_active,
            "devices_managed": list(self.memory_pools.keys())
        }
    
    @contextmanager
    def managed_memory_context(self, device_id: int):
        """Context manager for automatic memory management."""
        initial_stats = self.get_memory_stats(device_id)
        
        try:
            yield
        finally:
            # Clean up after context
            self.optimize_memory_usage(device_id)
            
            final_stats = self.get_memory_stats(device_id)
            if device_id in initial_stats and device_id in final_stats:
                initial_mem = initial_stats[device_id].allocated_memory
                final_mem = final_stats[device_id].allocated_memory
                logger.debug(f"Memory change for device {device_id}: "
                           f"{(final_mem - initial_mem) / (1024**2):.2f} MB")
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.monitoring_active:
            self.stop_memory_monitoring()