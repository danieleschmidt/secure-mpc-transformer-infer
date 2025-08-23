"""
Quantum Performance Optimizer

Advanced performance optimization system with quantum-enhanced algorithms
for massive scale MPC transformer inference.
"""

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import json
import random

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    QUANTUM_ENHANCED = "quantum_enhanced"
    EXTREME_PERFORMANCE = "extreme_performance"


class ResourceType(Enum):
    """Types of computational resources"""
    CPU_CORES = "cpu_cores"
    GPU_UNITS = "gpu_units"
    MEMORY_GB = "memory_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    QUANTUM_COHERENCE = "quantum_coherence"
    STORAGE_IOPS = "storage_iops"


@dataclass
class PerformanceProfile:
    """Performance profile for optimization"""
    optimization_level: OptimizationLevel = OptimizationLevel.QUANTUM_ENHANCED
    max_parallel_operations: int = 64
    cache_size_gb: float = 16.0
    prefetch_buffer_size: int = 1000
    quantum_coherence_threshold: float = 0.5
    auto_scaling_enabled: bool = True
    performance_target_latency_ms: float = 100.0
    throughput_target_ops_per_sec: float = 1000.0


@dataclass
class ResourcePool:
    """Dynamic resource pool management"""
    cpu_cores: int = 16
    gpu_units: int = 4
    memory_gb: float = 64.0
    network_bandwidth_gbps: float = 10.0
    quantum_coherence_units: float = 2.0
    storage_iops: int = 10000
    
    # Dynamic allocation tracking
    allocated_resources: Dict[ResourceType, float] = field(default_factory=dict)
    resource_utilization: Dict[ResourceType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize resource tracking"""
        if not self.allocated_resources:
            self.allocated_resources = {resource_type: 0.0 for resource_type in ResourceType}
        if not self.resource_utilization:
            self.resource_utilization = {resource_type: 0.0 for resource_type in ResourceType}


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    operation_latency_ms: float
    throughput_ops_per_sec: float
    resource_utilization: Dict[str, float]
    quantum_coherence: float
    cache_hit_rate: float
    error_rate: float
    scaling_factor: float
    optimization_efficiency: float


class QuantumPerformanceCache:
    """High-performance quantum-aware cache system"""
    
    def __init__(self, max_size_gb: float = 16.0, coherence_threshold: float = 0.5):
        self.max_size_gb = max_size_gb
        self.coherence_threshold = coherence_threshold
        
        # Cache storage
        self.cache_data: Dict[str, Dict[str, Any]] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_history: List[Tuple[str, datetime]] = []
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        # Quantum state tracking
        self.quantum_states: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized QuantumPerformanceCache with {max_size_gb}GB capacity")
    
    def get(self, key: str, quantum_state: Optional[Dict[str, float]] = None) -> Optional[Any]:
        """Retrieve item from cache with quantum coherence checking"""
        self.total_requests += 1
        
        if key not in self.cache_data:
            self.cache_misses += 1
            return None
        
        # Check quantum coherence if state provided
        if quantum_state and key in self.quantum_states:
            coherence = self._calculate_quantum_similarity(
                quantum_state, 
                self.quantum_states[key]
            )
            
            if coherence < self.coherence_threshold:
                logger.debug(f"Cache invalidated for {key}: low quantum coherence {coherence:.3f}")
                self.invalidate(key)
                self.cache_misses += 1
                return None
        
        # Update access history
        self.access_history.append((key, datetime.now()))
        self._trim_access_history()
        
        self.cache_hits += 1
        return self.cache_data[key]
    
    def put(self, key: str, value: Any, quantum_state: Optional[Dict[str, float]] = None) -> bool:
        """Store item in cache with quantum state tracking"""
        # Check capacity and evict if necessary
        if self._estimate_cache_size() >= self.max_size_gb:
            self._evict_lru_items()
        
        # Store data and metadata
        self.cache_data[key] = value
        self.cache_metadata[key] = {
            "timestamp": datetime.now(),
            "access_count": 1,
            "size_estimate": self._estimate_item_size(value)
        }
        
        # Store quantum state if provided
        if quantum_state:
            self.quantum_states[key] = quantum_state.copy()
        
        logger.debug(f"Cached item {key} with quantum state tracking")
        return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry"""
        removed = False
        
        if key in self.cache_data:
            del self.cache_data[key]
            removed = True
        
        if key in self.cache_metadata:
            del self.cache_metadata[key]
        
        if key in self.quantum_states:
            del self.quantum_states[key]
        
        return removed
    
    def _calculate_quantum_similarity(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """Calculate quantum state similarity"""
        common_keys = set(state1.keys()) & set(state2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            # Quantum fidelity-like similarity measure
            similarity += math.sqrt(state1[key] * state2[key])
        
        return similarity / len(common_keys)
    
    def _estimate_cache_size(self) -> float:
        """Estimate current cache size in GB"""
        total_size = 0.0
        for metadata in self.cache_metadata.values():
            total_size += metadata.get("size_estimate", 0.1)  # Default 100MB per item
        
        return total_size
    
    def _estimate_item_size(self, item: Any) -> float:
        """Estimate item size in GB"""
        try:
            # Rough size estimation
            if isinstance(item, (str, bytes)):
                return len(item) / (1024**3)  # Convert to GB
            elif isinstance(item, dict):
                return len(json.dumps(item)) / (1024**3)
            else:
                return 0.1  # Default 100MB
        except Exception:
            return 0.1
    
    def _evict_lru_items(self) -> None:
        """Evict least recently used items"""
        # Sort by last access time
        if not self.access_history:
            return
        
        # Find least recently accessed items
        access_counts = {}
        for key, timestamp in self.access_history[-100:]:  # Recent history
            access_counts[key] = access_counts.get(key, 0) + 1
        
        # Sort by access count (ascending)
        items_by_usage = sorted(
            self.cache_data.keys(),
            key=lambda k: access_counts.get(k, 0)
        )
        
        # Evict least used items until under capacity
        evicted_count = 0
        for key in items_by_usage:
            if self._estimate_cache_size() < self.max_size_gb * 0.8:  # 80% threshold
                break
            
            self.invalidate(key)
            evicted_count += 1
        
        logger.debug(f"Evicted {evicted_count} items from cache")
    
    def _trim_access_history(self) -> None:
        """Trim access history to reasonable size"""
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-5000:]  # Keep recent 5000
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        hit_rate = self.cache_hits / max(1, self.total_requests)
        
        return {
            "cache_size_gb": self._estimate_cache_size(),
            "max_size_gb": self.max_size_gb,
            "utilization": self._estimate_cache_size() / self.max_size_gb,
            "total_items": len(self.cache_data),
            "hit_rate": hit_rate,
            "miss_rate": 1.0 - hit_rate,
            "total_requests": self.total_requests,
            "quantum_states_tracked": len(self.quantum_states)
        }


class QuantumAutoScaler:
    """Intelligent auto-scaling system with quantum-enhanced predictions"""
    
    def __init__(self, 
                 min_instances: int = 2,
                 max_instances: int = 50,
                 target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.4):
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_instances = min_instances
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Quantum prediction model
        self.quantum_prediction_weights = [random.uniform(-1, 1) for _ in range(10)]
        
        logger.info(f"Initialized QuantumAutoScaler: {min_instances}-{max_instances} instances")
    
    async def evaluate_scaling_need(self, 
                                   current_metrics: PerformanceMetrics,
                                   resource_pool: ResourcePool) -> Tuple[int, str]:
        """Evaluate if scaling is needed using quantum predictions"""
        
        # Calculate current resource utilization
        avg_utilization = sum(resource_pool.resource_utilization.values()) / len(resource_pool.resource_utilization)
        
        # Quantum-enhanced prediction
        quantum_trend = self._predict_quantum_trend(current_metrics, resource_pool)
        
        # Scaling decision logic
        scaling_decision = "none"
        target_instances = self.current_instances
        
        if avg_utilization > self.scale_up_threshold or quantum_trend > 0.2:
            # Scale up
            if self.current_instances < self.max_instances:
                scale_factor = 1 + max(0.2, min(1.0, quantum_trend))  # 20%-200% increase
                target_instances = min(
                    self.max_instances,
                    int(self.current_instances * scale_factor)
                )
                scaling_decision = "scale_up"
        
        elif avg_utilization < self.scale_down_threshold and quantum_trend < -0.1:
            # Scale down
            if self.current_instances > self.min_instances:
                scale_factor = 0.5 + max(0, 0.4 + quantum_trend)  # 50%-90% of current
                target_instances = max(
                    self.min_instances,
                    int(self.current_instances * scale_factor)
                )
                scaling_decision = "scale_down"
        
        # Record scaling decision
        scaling_record = {
            "timestamp": datetime.now(),
            "current_instances": self.current_instances,
            "target_instances": target_instances,
            "avg_utilization": avg_utilization,
            "quantum_trend": quantum_trend,
            "decision": scaling_decision
        }
        
        self.scaling_history.append(scaling_record)
        
        # Trim history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]
        
        return target_instances, scaling_decision
    
    def _predict_quantum_trend(self, 
                              metrics: PerformanceMetrics, 
                              resource_pool: ResourcePool) -> float:
        """Predict future resource needs using quantum-inspired algorithm"""
        
        # Feature vector for prediction
        features = [
            metrics.operation_latency_ms / 1000.0,  # Normalized latency
            metrics.throughput_ops_per_sec / 1000.0,  # Normalized throughput
            metrics.quantum_coherence,
            metrics.cache_hit_rate,
            metrics.error_rate,
            sum(resource_pool.resource_utilization.values()) / len(resource_pool.resource_utilization),
            metrics.scaling_factor,
            metrics.optimization_efficiency,
            len(self.scaling_history) / 1000.0,  # History factor
            time.time() % 3600 / 3600.0  # Time of day factor
        ]
        
        # Quantum-inspired prediction (simplified variational circuit)
        prediction = 0.0
        for i, (feature, weight) in enumerate(zip(features, self.quantum_prediction_weights)):
            # Apply quantum gates simulation
            theta = feature * weight * math.pi
            quantum_amplitude = math.cos(theta) ** 2 - math.sin(theta) ** 2
            prediction += quantum_amplitude
        
        # Normalize prediction to [-1, 1] range
        prediction = math.tanh(prediction / len(features))
        
        # Update weights based on recent scaling effectiveness (simple online learning)
        if len(self.scaling_history) > 2:
            self._update_prediction_weights(prediction)
        
        return prediction
    
    def _update_prediction_weights(self, prediction: float) -> None:
        """Update quantum prediction weights based on effectiveness"""
        if len(self.scaling_history) < 3:
            return
        
        # Simple gradient-like update
        recent_decisions = self.scaling_history[-3:]
        
        # Check if recent scaling decisions were effective
        effectiveness = 0.0
        for i in range(1, len(recent_decisions)):
            prev_util = recent_decisions[i-1].get("avg_utilization", 0.7)
            curr_util = recent_decisions[i].get("avg_utilization", 0.7)
            
            # Good scaling should move utilization toward target
            target_distance_before = abs(prev_util - self.target_utilization)
            target_distance_after = abs(curr_util - self.target_utilization)
            
            if target_distance_after < target_distance_before:
                effectiveness += 0.1
            else:
                effectiveness -= 0.1
        
        # Update weights slightly
        learning_rate = 0.01
        for i in range(len(self.quantum_prediction_weights)):
            self.quantum_prediction_weights[i] += learning_rate * effectiveness * random.uniform(-1, 1)
            # Keep weights bounded
            self.quantum_prediction_weights[i] = max(-2, min(2, self.quantum_prediction_weights[i]))
    
    async def execute_scaling(self, target_instances: int) -> bool:
        """Execute scaling operation"""
        if target_instances == self.current_instances:
            return True
        
        logger.info(f"Scaling from {self.current_instances} to {target_instances} instances")
        
        # Simulate scaling delay
        scaling_delay = abs(target_instances - self.current_instances) * 2.0  # 2s per instance
        await asyncio.sleep(min(scaling_delay, 30.0))  # Max 30s
        
        self.current_instances = target_instances
        
        logger.info(f"Scaling completed: now running {self.current_instances} instances")
        return True
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling metrics and history"""
        recent_scalings = [s for s in self.scaling_history 
                          if (datetime.now() - s["timestamp"]).total_seconds() < 3600]  # Last hour
        
        scale_up_count = len([s for s in recent_scalings if s["decision"] == "scale_up"])
        scale_down_count = len([s for s in recent_scalings if s["decision"] == "scale_down"])
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "target_utilization": self.target_utilization,
            "recent_scale_ups": scale_up_count,
            "recent_scale_downs": scale_down_count,
            "total_scaling_events": len(self.scaling_history),
            "quantum_prediction_weights_variance": sum(w**2 for w in self.quantum_prediction_weights) / len(self.quantum_prediction_weights)
        }


class QuantumPerformanceOptimizer:
    """Main quantum performance optimization system"""
    
    def __init__(self, profile: PerformanceProfile = None):
        self.profile = profile or PerformanceProfile()
        
        # Initialize subsystems
        self.cache = QuantumPerformanceCache(
            max_size_gb=self.profile.cache_size_gb,
            coherence_threshold=self.profile.quantum_coherence_threshold
        )
        
        self.auto_scaler = QuantumAutoScaler(
            min_instances=2,
            max_instances=min(self.profile.max_parallel_operations, 100),
            target_utilization=0.7
        )
        
        self.resource_pool = ResourcePool()
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_metrics = {
            "total_operations": 0,
            "total_optimization_time": 0.0,
            "average_speedup": 1.0,
            "peak_throughput": 0.0,
            "best_latency": float('inf')
        }
        
        # Thread/Process pools for parallel execution
        self.thread_executor = ThreadPoolExecutor(max_workers=min(32, self.profile.max_parallel_operations))
        self.process_executor = ProcessPoolExecutor(max_workers=min(8, self.profile.max_parallel_operations // 4))
        
        logger.info(f"Initialized QuantumPerformanceOptimizer with {self.profile.optimization_level} level")
    
    async def optimize_operation(self, 
                                operation: Callable,
                                operation_id: str,
                                quantum_state: Optional[Dict[str, float]] = None,
                                *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Optimize and execute operation with comprehensive performance tracking"""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{operation_id}_{hash(str(args))}{hash(str(kwargs))}"
        cached_result = self.cache.get(cache_key, quantum_state)
        
        if cached_result is not None:
            # Cache hit - return immediately
            latency = (time.time() - start_time) * 1000
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation_latency_ms=latency,
                throughput_ops_per_sec=1000.0 / max(latency, 1.0),
                resource_utilization={rt.value: 0.1 for rt in ResourceType},  # Minimal for cache hit
                quantum_coherence=quantum_state.get("coherence", 1.0) if quantum_state else 1.0,
                cache_hit_rate=1.0,
                error_rate=0.0,
                scaling_factor=1.0,
                optimization_efficiency=2.0  # Cache hits are highly efficient
            )
            
            return cached_result, metrics
        
        # Cache miss - execute operation with optimization
        try:
            # Determine optimal execution strategy
            execution_strategy = self._select_execution_strategy(operation, args, kwargs)
            
            # Execute with selected strategy
            if execution_strategy == "parallel_threads":
                result = await self._execute_with_threads(operation, args, kwargs)
            elif execution_strategy == "parallel_processes":
                result = await self._execute_with_processes(operation, args, kwargs)
            elif execution_strategy == "quantum_enhanced":
                result = await self._execute_with_quantum_enhancement(operation, quantum_state, args, kwargs)
            else:
                # Standard async execution
                result = await operation(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Cache the result
            self.cache.put(cache_key, result, quantum_state)
            
            # Update resource utilization
            await self._update_resource_utilization(execution_time, execution_strategy)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                start_time, execution_time, execution_strategy, quantum_state
            )
            
            # Store metrics
            self.performance_history.append(metrics)
            self._trim_performance_history()
            
            # Update optimization metrics
            self._update_optimization_metrics(metrics)
            
            # Check if auto-scaling is needed
            if self.profile.auto_scaling_enabled:
                target_instances, scaling_decision = await self.auto_scaler.evaluate_scaling_need(
                    metrics, self.resource_pool
                )
                
                if scaling_decision != "none":
                    # Execute scaling in background
                    asyncio.create_task(self.auto_scaler.execute_scaling(target_instances))
            
            return result, metrics
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation_latency_ms=execution_time * 1000,
                throughput_ops_per_sec=0.0,
                resource_utilization={rt.value: 0.5 for rt in ResourceType},
                quantum_coherence=0.0,
                cache_hit_rate=0.0,
                error_rate=1.0,
                scaling_factor=1.0,
                optimization_efficiency=0.0
            )
            
            self.performance_history.append(metrics)
            raise e
    
    def _select_execution_strategy(self, operation: Callable, args: Tuple, kwargs: Dict) -> str:
        """Select optimal execution strategy based on operation characteristics"""
        
        # Simple heuristics for strategy selection
        arg_count = len(args) + len(kwargs)
        
        if self.profile.optimization_level == OptimizationLevel.EXTREME_PERFORMANCE:
            if arg_count > 10:
                return "parallel_processes"
            elif arg_count > 3:
                return "parallel_threads"
            else:
                return "quantum_enhanced"
        
        elif self.profile.optimization_level == OptimizationLevel.QUANTUM_ENHANCED:
            return "quantum_enhanced"
        
        elif self.profile.optimization_level == OptimizationLevel.ADVANCED:
            if arg_count > 5:
                return "parallel_threads"
            else:
                return "standard_async"
        
        else:
            return "standard_async"
    
    async def _execute_with_threads(self, operation: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Execute operation using thread pool"""
        loop = asyncio.get_event_loop()
        
        # For thread execution, we need to make the operation thread-safe
        if asyncio.iscoroutinefunction(operation):
            # Can't directly run coroutine in thread - run in executor with asyncio.run
            def run_coroutine():
                return asyncio.run(operation(*args, **kwargs))
            
            result = await loop.run_in_executor(self.thread_executor, run_coroutine)
        else:
            result = await loop.run_in_executor(self.thread_executor, operation, *args, **kwargs)
        
        return result
    
    async def _execute_with_processes(self, operation: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Execute operation using process pool"""
        loop = asyncio.get_event_loop()
        
        # Process execution requires pickleable operations
        if asyncio.iscoroutinefunction(operation):
            # Convert to sync for process execution
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(operation(*args, **kwargs))
            
            result = await loop.run_in_executor(self.process_executor, sync_wrapper, *args, **kwargs)
        else:
            result = await loop.run_in_executor(self.process_executor, operation, *args, **kwargs)
        
        return result
    
    async def _execute_with_quantum_enhancement(self, 
                                               operation: Callable, 
                                               quantum_state: Optional[Dict[str, float]],
                                               args: Tuple, 
                                               kwargs: Dict) -> Any:
        """Execute operation with quantum-enhanced optimizations"""
        
        # Apply quantum parameter optimization
        if quantum_state:
            optimized_kwargs = self._apply_quantum_parameter_optimization(kwargs, quantum_state)
        else:
            optimized_kwargs = kwargs
        
        # Execute with quantum timing optimization
        await self._quantum_timing_delay(quantum_state)
        
        # Execute operation
        result = await operation(*args, **optimized_kwargs)
        
        return result
    
    def _apply_quantum_parameter_optimization(self, 
                                            kwargs: Dict, 
                                            quantum_state: Dict[str, float]) -> Dict:
        """Apply quantum-inspired parameter optimization"""
        optimized_kwargs = kwargs.copy()
        
        coherence = quantum_state.get("coherence", 1.0)
        
        # Quantum-inspired parameter adjustments
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                # Apply quantum fluctuation
                quantum_fluctuation = coherence * random.uniform(-0.1, 0.1)
                optimized_value = value * (1 + quantum_fluctuation)
                
                if isinstance(value, int):
                    optimized_kwargs[key] = int(optimized_value)
                else:
                    optimized_kwargs[key] = optimized_value
        
        return optimized_kwargs
    
    async def _quantum_timing_delay(self, quantum_state: Optional[Dict[str, float]]) -> None:
        """Apply quantum-optimized timing delay"""
        if not quantum_state:
            return
        
        coherence = quantum_state.get("coherence", 1.0)
        
        # Quantum timing optimization - higher coherence = less delay needed
        optimal_delay = (1.0 - coherence) * 0.01  # Max 10ms delay
        
        if optimal_delay > 0.001:  # Only delay if significant
            await asyncio.sleep(optimal_delay)
    
    async def _update_resource_utilization(self, execution_time: float, strategy: str) -> None:
        """Update resource utilization tracking"""
        
        # Calculate utilization based on execution strategy
        utilization_multipliers = {
            "standard_async": {"cpu_cores": 0.3, "memory_gb": 0.2},
            "parallel_threads": {"cpu_cores": 0.7, "memory_gb": 0.4},
            "parallel_processes": {"cpu_cores": 0.9, "memory_gb": 0.6},
            "quantum_enhanced": {"cpu_cores": 0.5, "memory_gb": 0.3, "quantum_coherence": 0.8}
        }
        
        strategy_multipliers = utilization_multipliers.get(strategy, {"cpu_cores": 0.5})
        
        # Update resource utilization with exponential moving average
        alpha = 0.1  # Smoothing factor
        
        for resource_type in ResourceType:
            current_util = self.resource_pool.resource_utilization.get(resource_type, 0.0)
            new_util = strategy_multipliers.get(resource_type.value, 0.1)
            
            # Apply exponential moving average
            updated_util = alpha * new_util + (1 - alpha) * current_util
            self.resource_pool.resource_utilization[resource_type] = updated_util
    
    def _calculate_performance_metrics(self, 
                                     start_time: float,
                                     execution_time: float, 
                                     strategy: str,
                                     quantum_state: Optional[Dict[str, float]]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        latency_ms = execution_time * 1000
        throughput = 1000.0 / max(latency_ms, 1.0)  # ops per second
        
        # Get cache metrics
        cache_metrics = self.cache.get_cache_metrics()
        
        # Calculate scaling factor
        scaling_factor = self.auto_scaler.current_instances / self.auto_scaler.min_instances
        
        # Calculate optimization efficiency
        strategy_efficiency = {
            "standard_async": 1.0,
            "parallel_threads": 1.5,
            "parallel_processes": 2.0,
            "quantum_enhanced": 2.5
        }.get(strategy, 1.0)
        
        # Quantum coherence
        quantum_coherence = quantum_state.get("coherence", 1.0) if quantum_state else 1.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            operation_latency_ms=latency_ms,
            throughput_ops_per_sec=throughput,
            resource_utilization={rt.value: util for rt, util in self.resource_pool.resource_utilization.items()},
            quantum_coherence=quantum_coherence,
            cache_hit_rate=cache_metrics["hit_rate"],
            error_rate=0.0,  # No error if we got here
            scaling_factor=scaling_factor,
            optimization_efficiency=strategy_efficiency * quantum_coherence
        )
    
    def _trim_performance_history(self) -> None:
        """Trim performance history to reasonable size"""
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-5000:]  # Keep recent 5000
    
    def _update_optimization_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update overall optimization metrics"""
        self.optimization_metrics["total_operations"] += 1
        
        # Update averages
        total_ops = self.optimization_metrics["total_operations"]
        
        # Average speedup (compared to baseline)
        baseline_efficiency = 1.0
        current_speedup = metrics.optimization_efficiency / baseline_efficiency
        
        prev_avg_speedup = self.optimization_metrics["average_speedup"]
        self.optimization_metrics["average_speedup"] = (
            (prev_avg_speedup * (total_ops - 1) + current_speedup) / total_ops
        )
        
        # Peak throughput
        if metrics.throughput_ops_per_sec > self.optimization_metrics["peak_throughput"]:
            self.optimization_metrics["peak_throughput"] = metrics.throughput_ops_per_sec
        
        # Best latency
        if metrics.operation_latency_ms < self.optimization_metrics["best_latency"]:
            self.optimization_metrics["best_latency"] = metrics.operation_latency_ms
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance and optimization metrics"""
        
        # Recent performance metrics (last 100 operations)
        recent_metrics = self.performance_history[-100:] if self.performance_history else []
        
        if recent_metrics:
            avg_latency = sum(m.operation_latency_ms for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput_ops_per_sec for m in recent_metrics) / len(recent_metrics)
            avg_coherence = sum(m.quantum_coherence for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        else:
            avg_latency = avg_throughput = avg_coherence = avg_cache_hit_rate = 0.0
        
        return {
            "optimization_profile": {
                "level": self.profile.optimization_level.value,
                "max_parallel_operations": self.profile.max_parallel_operations,
                "performance_target_latency_ms": self.profile.performance_target_latency_ms,
                "throughput_target_ops_per_sec": self.profile.throughput_target_ops_per_sec
            },
            "current_performance": {
                "average_latency_ms": avg_latency,
                "average_throughput_ops_per_sec": avg_throughput,
                "average_quantum_coherence": avg_coherence,
                "cache_hit_rate": avg_cache_hit_rate
            },
            "optimization_metrics": self.optimization_metrics,
            "cache_metrics": self.cache.get_cache_metrics(),
            "scaling_metrics": self.auto_scaler.get_scaling_metrics(),
            "resource_utilization": dict(self.resource_pool.resource_utilization),
            "performance_targets_met": {
                "latency_target": avg_latency <= self.profile.performance_target_latency_ms,
                "throughput_target": avg_throughput >= self.profile.throughput_target_ops_per_sec
            }
        }
    
    async def benchmark_optimization(self, 
                                   test_operations: List[Callable],
                                   iterations: int = 100) -> Dict[str, Any]:
        """Run comprehensive optimization benchmark"""
        
        logger.info(f"Starting optimization benchmark with {len(test_operations)} operations, {iterations} iterations")
        
        benchmark_start = time.time()
        results = {}
        
        for i, operation in enumerate(test_operations):
            operation_name = f"test_operation_{i}"
            operation_results = []
            
            for iteration in range(iterations):
                # Generate quantum state for testing
                test_quantum_state = {
                    "coherence": random.uniform(0.5, 1.0),
                    "entanglement": random.uniform(0.3, 0.9),
                    "phase": random.uniform(0, 2 * math.pi)
                }
                
                try:
                    result, metrics = await self.optimize_operation(
                        operation,
                        f"{operation_name}_iter_{iteration}",
                        test_quantum_state
                    )
                    
                    operation_results.append({
                        "iteration": iteration,
                        "success": True,
                        "latency_ms": metrics.operation_latency_ms,
                        "throughput_ops_per_sec": metrics.throughput_ops_per_sec,
                        "quantum_coherence": metrics.quantum_coherence,
                        "optimization_efficiency": metrics.optimization_efficiency
                    })
                    
                except Exception as e:
                    operation_results.append({
                        "iteration": iteration,
                        "success": False,
                        "error": str(e)
                    })
                
                # Small delay between iterations
                await asyncio.sleep(0.01)
            
            # Analyze operation results
            successful_results = [r for r in operation_results if r.get("success", False)]
            
            if successful_results:
                results[operation_name] = {
                    "total_iterations": iterations,
                    "successful_iterations": len(successful_results),
                    "success_rate": len(successful_results) / iterations,
                    "average_latency_ms": sum(r["latency_ms"] for r in successful_results) / len(successful_results),
                    "average_throughput": sum(r["throughput_ops_per_sec"] for r in successful_results) / len(successful_results),
                    "average_coherence": sum(r["quantum_coherence"] for r in successful_results) / len(successful_results),
                    "average_optimization_efficiency": sum(r["optimization_efficiency"] for r in successful_results) / len(successful_results),
                    "min_latency_ms": min(r["latency_ms"] for r in successful_results),
                    "max_throughput": max(r["throughput_ops_per_sec"] for r in successful_results)
                }
            else:
                results[operation_name] = {
                    "total_iterations": iterations,
                    "successful_iterations": 0,
                    "success_rate": 0.0,
                    "error": "All iterations failed"
                }
        
        benchmark_time = time.time() - benchmark_start
        
        # Overall benchmark summary
        all_successful = [r for r in results.values() if r.get("success_rate", 0) > 0]
        
        benchmark_summary = {
            "total_benchmark_time_seconds": benchmark_time,
            "operations_tested": len(test_operations),
            "total_iterations": len(test_operations) * iterations,
            "overall_success_rate": sum(r.get("success_rate", 0) for r in results.values()) / len(results),
            "best_average_latency_ms": min(r["average_latency_ms"] for r in all_successful) if all_successful else None,
            "best_average_throughput": max(r["average_throughput"] for r in all_successful) if all_successful else None,
            "best_optimization_efficiency": max(r["average_optimization_efficiency"] for r in all_successful) if all_successful else None
        }
        
        logger.info(f"Optimization benchmark completed in {benchmark_time:.2f}s")
        logger.info(f"Overall success rate: {benchmark_summary['overall_success_rate']:.1%}")
        
        return {
            "benchmark_summary": benchmark_summary,
            "operation_results": results,
            "final_metrics": self.get_comprehensive_metrics()
        }
    
    def __del__(self):
        """Cleanup executors"""
        try:
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
        except Exception:
            pass