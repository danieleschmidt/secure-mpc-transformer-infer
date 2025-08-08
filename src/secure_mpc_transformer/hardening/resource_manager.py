"""Production resource management, connection pooling, and memory optimization."""

import os
import gc
import time
import psutil
import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional, Union, Callable, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import weakref

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources being managed."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    FILE_HANDLES = "file_handles"


class ResourceStatus(Enum):
    """Resource allocation status."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    EXHAUSTED = "exhausted"
    CRITICAL = "critical"


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    
    # Memory limits (in MB)
    max_memory_mb: Optional[int] = None
    memory_warning_threshold: float = 0.8
    memory_critical_threshold: float = 0.9
    
    # CPU limits
    max_cpu_percent: Optional[float] = None
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 90.0
    
    # GPU limits (if available)
    max_gpu_memory_mb: Optional[int] = None
    gpu_warning_threshold: float = 0.8
    gpu_critical_threshold: float = 0.9
    
    # Connection limits
    max_connections: int = 1000
    max_connections_per_host: int = 100
    connection_timeout: float = 30.0
    connection_keepalive: float = 600.0
    
    # File handle limits
    max_file_handles: int = 1024
    
    # Request limits
    max_concurrent_requests: int = 100
    request_timeout: float = 300.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'max_memory_mb': self.max_memory_mb,
            'memory_warning_threshold': self.memory_warning_threshold,
            'memory_critical_threshold': self.memory_critical_threshold,
            'max_cpu_percent': self.max_cpu_percent,
            'cpu_warning_threshold': self.cpu_warning_threshold,
            'cpu_critical_threshold': self.cpu_critical_threshold,
            'max_gpu_memory_mb': self.max_gpu_memory_mb,
            'gpu_warning_threshold': self.gpu_warning_threshold,
            'gpu_critical_threshold': self.gpu_critical_threshold,
            'max_connections': self.max_connections,
            'max_connections_per_host': self.max_connections_per_host,
            'connection_timeout': self.connection_timeout,
            'connection_keepalive': self.connection_keepalive,
            'max_file_handles': self.max_file_handles,
            'max_concurrent_requests': self.max_concurrent_requests,
            'request_timeout': self.request_timeout
        }


@dataclass
class ResourceUsage:
    """Current resource usage."""
    
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    active_connections: int = 0
    open_file_handles: int = 0
    concurrent_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_memory_percent': self.gpu_memory_percent,
            'active_connections': self.active_connections,
            'open_file_handles': self.open_file_handles,
            'concurrent_requests': self.concurrent_requests
        }


class MemoryManager:
    """Memory management and optimization."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.allocated_objects = weakref.WeakSet()
        self.memory_pools = {}
        self.gc_stats = {'collections': 0, 'collected': 0}
        self._lock = threading.Lock()
        
        # Memory monitoring
        self.memory_history = deque(maxlen=1000)
        self.last_gc_time = time.time()
        self.gc_threshold = 60.0  # Force GC every minute
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            usage = {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'system_available_mb': system_memory.available / (1024 * 1024),
                'system_used_percent': system_memory.percent
            }
            
            # Add GPU memory if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                usage.update({
                    'gpu_allocated_mb': gpu_allocated,
                    'gpu_cached_mb': gpu_cached
                })
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def check_memory_limits(self) -> Dict[str, Any]:
        """Check if memory usage is within limits."""
        usage = self.get_memory_usage()
        warnings = []
        
        # Check process memory
        rss_mb = usage.get('rss_mb', 0)
        if self.limits.max_memory_mb:
            if rss_mb > self.limits.max_memory_mb:
                warnings.append(f"Process memory limit exceeded: {rss_mb:.1f}MB > {self.limits.max_memory_mb}MB")
        
        # Check system memory
        system_used = usage.get('system_used_percent', 0) / 100.0
        if system_used > self.limits.memory_critical_threshold:
            warnings.append(f"System memory critically low: {system_used:.1%}")
        elif system_used > self.limits.memory_warning_threshold:
            warnings.append(f"System memory warning: {system_used:.1%}")
        
        # Check GPU memory
        gpu_allocated = usage.get('gpu_allocated_mb', 0)
        if self.limits.max_gpu_memory_mb and gpu_allocated > self.limits.max_gpu_memory_mb:
            warnings.append(f"GPU memory limit exceeded: {gpu_allocated:.1f}MB > {self.limits.max_gpu_memory_mb}MB")
        
        return {
            'within_limits': len(warnings) == 0,
            'warnings': warnings,
            'usage': usage
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_usage = self.get_memory_usage()
        optimizations_applied = []
        
        # Force garbage collection
        collected_counts = gc.collect()
        if collected_counts:
            optimizations_applied.append(f"Garbage collection: {collected_counts} objects collected")
            self.gc_stats['collections'] += 1
            self.gc_stats['collected'] += collected_counts
        
        # Clear PyTorch cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimizations_applied.append("PyTorch GPU cache cleared")
        
        # Clear weak references
        self.allocated_objects.clear()
        optimizations_applied.append("Weak references cleared")
        
        end_usage = self.get_memory_usage()
        memory_freed = start_usage.get('rss_mb', 0) - end_usage.get('rss_mb', 0)
        
        self.last_gc_time = time.time()
        
        return {
            'memory_freed_mb': memory_freed,
            'optimizations_applied': optimizations_applied,
            'start_usage': start_usage,
            'end_usage': end_usage
        }
    
    def register_allocated_object(self, obj):
        """Register an allocated object for tracking."""
        with self._lock:
            self.allocated_objects.add(obj)
    
    def should_force_gc(self) -> bool:
        """Check if garbage collection should be forced."""
        return time.time() - self.last_gc_time > self.gc_threshold
    
    @contextmanager
    def memory_limit_context(self, limit_mb: int):
        """Context manager for temporary memory limits."""
        original_limit = self.limits.max_memory_mb
        self.limits.max_memory_mb = limit_mb
        
        try:
            yield
        finally:
            self.limits.max_memory_mb = original_limit


class ConnectionPool:
    """Generic connection pool implementation."""
    
    def __init__(self, name: str, create_connection: Callable,
                 max_connections: int = 10, max_idle_time: float = 300.0):
        self.name = name
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        # Connection tracking
        self.active_connections = set()
        self.idle_connections = deque()
        self.connection_stats = {
            'created': 0,
            'reused': 0,
            'closed': 0,
            'errors': 0
        }
        
        self._lock = threading.Lock()
        self._cleanup_task = None
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def get_connection(self):
        """Get a connection from the pool."""
        with self._lock:
            # Try to reuse an idle connection
            while self.idle_connections:
                connection, last_used = self.idle_connections.popleft()
                
                # Check if connection is still valid and not too old
                if time.time() - last_used < self.max_idle_time:
                    if self._is_connection_valid(connection):
                        self.active_connections.add(connection)
                        self.connection_stats['reused'] += 1
                        return connection
                
                # Connection is invalid or too old, close it
                self._close_connection(connection)
                self.connection_stats['closed'] += 1
            
            # Create new connection if under limit
            if len(self.active_connections) < self.max_connections:
                try:
                    connection = self.create_connection()
                    self.active_connections.add(connection)
                    self.connection_stats['created'] += 1
                    return connection
                except Exception as e:
                    logger.error(f"Failed to create connection for pool {self.name}: {e}")
                    self.connection_stats['errors'] += 1
                    raise
            
            raise RuntimeError(f"Connection pool {self.name} exhausted (max: {self.max_connections})")
    
    def return_connection(self, connection):
        """Return a connection to the pool."""
        with self._lock:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
                
                if self._is_connection_valid(connection):
                    self.idle_connections.append((connection, time.time()))
                else:
                    self._close_connection(connection)
                    self.connection_stats['closed'] += 1
    
    def close_connection(self, connection):
        """Explicitly close a connection."""
        with self._lock:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
            
            self._close_connection(connection)
            self.connection_stats['closed'] += 1
    
    def _is_connection_valid(self, connection) -> bool:
        """Check if connection is still valid."""
        # Override in subclasses for specific connection types
        return hasattr(connection, 'closed') and not getattr(connection, 'closed', True)
    
    def _close_connection(self, connection):
        """Close a connection."""
        try:
            if hasattr(connection, 'close'):
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection in pool {self.name}: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    self._cleanup_idle_connections()
                    await asyncio.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Connection pool cleanup error: {e}")
                    await asyncio.sleep(60)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._cleanup_task = asyncio.create_task(cleanup_loop())
        except RuntimeError:
            # Start manual cleanup
            threading.Timer(60, self._manual_cleanup).start()
    
    def _cleanup_idle_connections(self):
        """Clean up old idle connections."""
        current_time = time.time()
        
        with self._lock:
            connections_to_close = []
            valid_connections = deque()
            
            for connection, last_used in self.idle_connections:
                if current_time - last_used > self.max_idle_time:
                    connections_to_close.append(connection)
                else:
                    valid_connections.append((connection, last_used))
            
            self.idle_connections = valid_connections
            
            # Close old connections outside the lock
            for connection in connections_to_close:
                self._close_connection(connection)
                self.connection_stats['closed'] += 1
    
    def _manual_cleanup(self):
        """Manual cleanup for non-async environments."""
        try:
            self._cleanup_idle_connections()
        except Exception as e:
            logger.error(f"Manual connection pool cleanup error: {e}")
        finally:
            threading.Timer(60, self._manual_cleanup).start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                'name': self.name,
                'max_connections': self.max_connections,
                'active_connections': len(self.active_connections),
                'idle_connections': len(self.idle_connections),
                'stats': self.connection_stats.copy()
            }
    
    def shutdown(self):
        """Shutdown the connection pool."""
        with self._lock:
            # Close all active connections
            for connection in list(self.active_connections):
                self._close_connection(connection)
            self.active_connections.clear()
            
            # Close all idle connections
            while self.idle_connections:
                connection, _ = self.idle_connections.popleft()
                self._close_connection(connection)
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info(f"Connection pool {self.name} shutdown")


class ResourceManager:
    """Main resource management system."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        
        # Managers
        self.memory_manager = MemoryManager(self.limits)
        self.connection_pools: Dict[str, ConnectionPool] = {}
        
        # Resource monitoring
        self.usage_history = deque(maxlen=1000)
        self.resource_alerts = deque(maxlen=100)
        
        # Semaphores for request limiting
        self.request_semaphore = threading.Semaphore(self.limits.max_concurrent_requests)
        self.async_request_semaphore = asyncio.Semaphore(self.limits.max_concurrent_requests)
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'requests_rejected': 0,
            'memory_optimizations': 0,
            'connection_pool_hits': 0,
            'resource_alerts_triggered': 0
        }
        
        # Monitoring
        self._monitor_task = None
        self._start_monitoring()
        
        logger.info("Resource manager initialized")
    
    def create_connection_pool(self, name: str, create_func: Callable,
                             max_connections: int = None, max_idle_time: float = None) -> ConnectionPool:
        """Create a new connection pool."""
        max_connections = max_connections or self.limits.max_connections_per_host
        max_idle_time = max_idle_time or self.limits.connection_keepalive
        
        pool = ConnectionPool(name, create_func, max_connections, max_idle_time)
        self.connection_pools[name] = pool
        
        logger.info(f"Created connection pool '{name}' with max {max_connections} connections")
        return pool
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name."""
        return self.connection_pools.get(name)
    
    @contextmanager
    def resource_limit_context(self):
        """Context manager for resource-limited operations."""
        acquired = self.request_semaphore.acquire(blocking=False)
        if not acquired:
            self.stats['requests_rejected'] += 1
            raise RuntimeError("Resource limit reached - too many concurrent requests")
        
        start_time = time.time()
        try:
            # Check resource limits before proceeding
            memory_check = self.memory_manager.check_memory_limits()
            if not memory_check['within_limits']:
                self._trigger_resource_alert("memory_limit_exceeded", memory_check['warnings'])
                
                # Force memory optimization
                self.memory_manager.optimize_memory()
                self.stats['memory_optimizations'] += 1
            
            yield
            self.stats['requests_processed'] += 1
            
        finally:
            self.request_semaphore.release()
            
            # Record processing time
            processing_time = time.time() - start_time
            if processing_time > self.limits.request_timeout:
                self._trigger_resource_alert(
                    "request_timeout_exceeded",
                    [f"Request took {processing_time:.2f}s (limit: {self.limits.request_timeout}s)"]
                )
    
    @asynccontextmanager
    async def async_resource_limit_context(self):
        """Async context manager for resource-limited operations."""
        try:
            await asyncio.wait_for(
                self.async_request_semaphore.acquire(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            self.stats['requests_rejected'] += 1
            raise RuntimeError("Resource limit reached - too many concurrent requests")
        
        start_time = time.time()
        try:
            # Check resource limits
            memory_check = self.memory_manager.check_memory_limits()
            if not memory_check['within_limits']:
                self._trigger_resource_alert("memory_limit_exceeded", memory_check['warnings'])
                
                # Force memory optimization
                self.memory_manager.optimize_memory()
                self.stats['memory_optimizations'] += 1
            
            yield
            self.stats['requests_processed'] += 1
            
        finally:
            self.async_request_semaphore.release()
            
            # Record processing time
            processing_time = time.time() - start_time
            if processing_time > self.limits.request_timeout:
                self._trigger_resource_alert(
                    "request_timeout_exceeded",
                    [f"Request took {processing_time:.2f}s (limit: {self.limits.request_timeout}s)"]
                )
    
    def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            
            # GPU usage
            gpu_memory_mb = 0.0
            gpu_memory_percent = 0.0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_memory_percent = (gpu_memory_mb / gpu_total) * 100 if gpu_total > 0 else 0
            
            # Connection counts
            active_connections = sum(len(pool.active_connections) for pool in self.connection_pools.values())
            
            # File handles
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # Concurrent requests (approximate)
            concurrent_requests = self.limits.max_concurrent_requests - self.request_semaphore._value
            
            usage = ResourceUsage(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_memory_percent=gpu_memory_percent,
                active_connections=active_connections,
                open_file_handles=open_files,
                concurrent_requests=concurrent_requests
            )
            
            # Store in history
            self.usage_history.append(usage)
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return ResourceUsage(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0
            )
    
    def _trigger_resource_alert(self, alert_type: str, messages: List[str]):
        """Trigger a resource alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'messages': messages
        }
        
        self.resource_alerts.append(alert)
        self.stats['resource_alerts_triggered'] += 1
        
        logger.warning(f"Resource alert: {alert_type} - {'; '.join(messages)}")
    
    def _start_monitoring(self):
        """Start background resource monitoring."""
        async def monitor_loop():
            while True:
                try:
                    await self._monitor_resources()
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(30)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._monitor_task = asyncio.create_task(monitor_loop())
        except RuntimeError:
            # Start manual monitoring
            threading.Timer(30, self._manual_monitoring).start()
    
    async def _monitor_resources(self):
        """Monitor resource usage and trigger alerts."""
        usage = self.get_resource_usage()
        
        # Check CPU usage
        if (self.limits.max_cpu_percent and 
            usage.cpu_percent > self.limits.max_cpu_percent):
            self._trigger_resource_alert(
                "cpu_limit_exceeded",
                [f"CPU usage: {usage.cpu_percent:.1f}% > {self.limits.max_cpu_percent:.1f}%"]
            )
        
        # Check memory thresholds
        memory_ratio = usage.memory_percent / 100.0
        if memory_ratio > self.limits.memory_critical_threshold:
            self._trigger_resource_alert(
                "memory_critical",
                [f"Memory usage critical: {memory_ratio:.1%}"]
            )
        elif memory_ratio > self.limits.memory_warning_threshold:
            self._trigger_resource_alert(
                "memory_warning",
                [f"Memory usage high: {memory_ratio:.1%}"]
            )
        
        # Check file handles
        if usage.open_file_handles > self.limits.max_file_handles * 0.9:
            self._trigger_resource_alert(
                "file_handle_warning",
                [f"High file handle usage: {usage.open_file_handles}/{self.limits.max_file_handles}"]
            )
        
        # Force memory optimization if needed
        if self.memory_manager.should_force_gc():
            self.memory_manager.optimize_memory()
            self.stats['memory_optimizations'] += 1
    
    def _manual_monitoring(self):
        """Manual monitoring for non-async environments."""
        try:
            asyncio.run(self._monitor_resources())
        except Exception as e:
            logger.error(f"Manual resource monitoring error: {e}")
        finally:
            threading.Timer(30, self._manual_monitoring).start()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        current_usage = self.get_resource_usage()
        
        # Connection pool stats
        pool_stats = {name: pool.get_stats() for name, pool in self.connection_pools.items()}
        
        # Memory manager stats
        memory_stats = self.memory_manager.get_memory_usage()
        
        # Recent alerts
        recent_alerts = list(self.resource_alerts)[-10:]  # Last 10 alerts
        
        return {
            'limits': self.limits.to_dict(),
            'current_usage': current_usage.to_dict(),
            'memory_stats': memory_stats,
            'connection_pools': pool_stats,
            'recent_alerts': recent_alerts,
            'statistics': self.stats.copy(),
            'monitoring_active': self._monitor_task is not None and not self._monitor_task.done()
        }
    
    def shutdown(self):
        """Shutdown resource manager."""
        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
        
        # Shutdown connection pools
        for pool in self.connection_pools.values():
            pool.shutdown()
        
        # Final memory cleanup
        self.memory_manager.optimize_memory()
        
        logger.info("Resource manager shutdown completed")


# Global resource manager instance
resource_manager = ResourceManager()


def with_resource_limits(func):
    """Decorator for resource-limited function execution."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with resource_manager.async_resource_limit_context():
                return await func(*args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with resource_manager.resource_limit_context():
                return func(*args, **kwargs)
        return sync_wrapper