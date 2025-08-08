"""
Dynamic worker pools with auto-scaling for high-throughput processing.

This module provides worker pool implementations that can dynamically
scale based on workload and resource availability.
"""

import time
import threading
import multiprocessing as mp
import queue
import concurrent.futures
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import psutil
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    """Types of worker pools."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"


class ScalingStrategy(Enum):
    """Scaling strategies for worker pools."""
    FIXED = "fixed"           # Fixed number of workers
    ADAPTIVE = "adaptive"     # Scale based on queue size
    PREDICTIVE = "predictive" # Scale based on predicted load
    REACTIVE = "reactive"     # Scale based on current metrics


@dataclass
class WorkerConfig:
    """Configuration for worker pools."""
    min_workers: int = 2
    max_workers: int = mp.cpu_count() * 2
    initial_workers: int = 4
    worker_type: WorkerType = WorkerType.THREAD
    scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    
    # Scaling parameters
    scale_up_threshold: float = 0.8    # Scale up when queue > 80% capacity
    scale_down_threshold: float = 0.2  # Scale down when queue < 20% capacity
    scaling_cooldown_seconds: int = 30  # Minimum time between scaling actions
    queue_size_factor: float = 2.0     # Queue size = workers * factor
    
    # Performance parameters
    max_queue_size: Optional[int] = None
    worker_timeout_seconds: int = 300  # 5 minutes
    task_timeout_seconds: int = 60     # 1 minute per task
    enable_task_cancellation: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval_seconds: int = 10
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60


@dataclass
class WorkerMetrics:
    """Metrics for worker pool performance."""
    active_workers: int = 0
    total_workers: int = 0
    queue_size: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    worker_utilization: float = 0.0
    throughput: float = 0.0  # tasks per second
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'active_workers': self.active_workers,
            'total_workers': self.total_workers,
            'queue_size': self.queue_size,
            'pending_tasks': self.pending_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'average_task_time': self.average_task_time,
            'worker_utilization': self.worker_utilization,
            'throughput': self.throughput
        }


class BaseWorkerPool(ABC):
    """Abstract base class for worker pools."""
    
    @abstractmethod
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the worker pool."""
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> WorkerMetrics:
        """Get current worker pool metrics."""
        pass
    
    @abstractmethod
    def scale_workers(self, target_count: int):
        """Scale workers to target count."""
        pass


class WorkerPool(BaseWorkerPool):
    """High-performance worker pool with dynamic scaling."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self._executor: Optional[concurrent.futures.Executor] = None
        self._task_queue: queue.Queue = queue.Queue(
            maxsize=config.max_queue_size or (config.max_workers * int(config.queue_size_factor))
        )
        
        # Metrics and monitoring
        self._metrics = WorkerMetrics()
        self._task_times: deque = deque(maxlen=1000)  # Keep last 1000 task times
        self._last_scaling_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring threads
        self._metrics_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # Initialize executor
        self._initialize_executor()
        
        # Start monitoring if enabled
        if config.enable_metrics:
            self._start_monitoring()
        
        logger.info(f"Worker pool initialized: {config.worker_type.value}, "
                   f"{config.initial_workers} workers")
    
    def _initialize_executor(self):
        """Initialize the appropriate executor based on worker type."""
        if self.config.worker_type == WorkerType.THREAD:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.initial_workers,
                thread_name_prefix="mpc_worker"
            )
        elif self.config.worker_type == WorkerType.PROCESS:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.initial_workers,
                mp_context=mp.get_context('spawn')
            )
        else:
            raise ValueError(f"Unsupported worker type: {self.config.worker_type}")
        
        self._metrics.total_workers = self.config.initial_workers
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the worker pool."""
        if not self._executor:
            raise RuntimeError("Worker pool is not initialized")
        
        # Record task submission
        start_time = time.perf_counter()
        
        # Submit task with timing wrapper
        def timed_task():
            try:
                result = fn(*args, **kwargs)
                task_time = time.perf_counter() - start_time
                self._record_task_completion(task_time, success=True)
                return result
            except Exception as e:
                task_time = time.perf_counter() - start_time
                self._record_task_completion(task_time, success=False)
                raise
        
        try:
            future = self._executor.submit(timed_task)
            self._metrics.pending_tasks += 1
            
            # Check if scaling is needed
            self._check_scaling_conditions()
            
            return future
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def _record_task_completion(self, task_time: float, success: bool):
        """Record task completion metrics."""
        with self._lock:
            self._task_times.append(task_time)
            self._metrics.pending_tasks = max(0, self._metrics.pending_tasks - 1)
            
            if success:
                self._metrics.completed_tasks += 1
            else:
                self._metrics.failed_tasks += 1
            
            # Update average task time
            if self._task_times:
                self._metrics.average_task_time = sum(self._task_times) / len(self._task_times)
    
    def _check_scaling_conditions(self):
        """Check if worker pool scaling is needed."""
        if self.config.scaling_strategy == ScalingStrategy.FIXED:
            return
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scaling_time < self.config.scaling_cooldown_seconds:
            return
        
        with self._lock:
            queue_utilization = self._metrics.pending_tasks / max(self._task_queue.maxsize, 1)
            current_workers = self._metrics.total_workers
            
            target_workers = current_workers
            
            if self.config.scaling_strategy == ScalingStrategy.ADAPTIVE:
                # Scale based on queue utilization
                if queue_utilization > self.config.scale_up_threshold:
                    target_workers = min(current_workers + 1, self.config.max_workers)
                elif queue_utilization < self.config.scale_down_threshold:
                    target_workers = max(current_workers - 1, self.config.min_workers)
            
            elif self.config.scaling_strategy == ScalingStrategy.REACTIVE:
                # Scale based on current system load
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent < 50 and queue_utilization > 0.5:
                    target_workers = min(current_workers + 2, self.config.max_workers)
                elif cpu_percent > 90:
                    target_workers = max(current_workers - 1, self.config.min_workers)
            
            # Apply scaling if needed
            if target_workers != current_workers:
                self.scale_workers(target_workers)
                self._last_scaling_time = current_time
    
    def scale_workers(self, target_count: int):
        """Scale workers to target count."""
        target_count = max(self.config.min_workers, min(target_count, self.config.max_workers))
        
        current_count = self._metrics.total_workers
        
        if target_count == current_count:
            return
        
        logger.info(f"Scaling workers from {current_count} to {target_count}")
        
        # For thread/process pools, we need to recreate the executor
        # This is a limitation of the concurrent.futures implementation
        if self._executor:
            # Gracefully shutdown current executor
            self._executor.shutdown(wait=False)
        
        # Create new executor with target worker count
        if self.config.worker_type == WorkerType.THREAD:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=target_count,
                thread_name_prefix="mpc_worker"
            )
        elif self.config.worker_type == WorkerType.PROCESS:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=target_count,
                mp_context=mp.get_context('spawn')
            )
        
        self._metrics.total_workers = target_count
        
        logger.info(f"Worker pool scaled to {target_count} workers")
    
    def _start_monitoring(self):
        """Start monitoring threads."""
        self._monitoring_active = True
        
        if self.config.enable_metrics:
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop,
                name="worker_pool_metrics",
                daemon=True
            )
            self._metrics_thread.start()
        
        if self.config.enable_health_checks:
            self._health_thread = threading.Thread(
                target=self._health_check_loop,
                name="worker_pool_health",
                daemon=True
            )
            self._health_thread.start()
        
        logger.info("Worker pool monitoring started")
    
    def _metrics_loop(self):
        """Metrics collection loop."""
        while self._monitoring_active:
            try:
                self._update_metrics()
                time.sleep(self.config.metrics_interval_seconds)
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                time.sleep(self.config.metrics_interval_seconds)
    
    def _health_check_loop(self):
        """Health check loop."""
        while self._monitoring_active:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(self.config.health_check_interval_seconds)
    
    def _update_metrics(self):
        """Update worker pool metrics."""
        with self._lock:
            # Calculate throughput (tasks per second)
            if self._task_times:
                # Calculate based on recent task completions
                recent_completions = len([t for t in self._task_times if t < 1.0])  # Tasks < 1s
                self._metrics.throughput = recent_completions / max(self.config.metrics_interval_seconds, 1)
            
            # Calculate worker utilization
            if self._metrics.total_workers > 0:
                active_tasks = self._metrics.pending_tasks
                self._metrics.worker_utilization = min(
                    active_tasks / self._metrics.total_workers, 1.0
                )
            
            # Update queue size
            self._metrics.queue_size = self._task_queue.qsize() if hasattr(self._task_queue, 'qsize') else 0
    
    def _perform_health_check(self):
        """Perform health check on worker pool."""
        if not self._executor:
            logger.warning("Executor is not available for health check")
            return
        
        # Submit a simple health check task
        def health_task():
            return "healthy"
        
        try:
            future = self._executor.submit(health_task)
            result = future.result(timeout=5.0)  # 5 second timeout
            
            if result == "healthy":
                logger.debug("Worker pool health check passed")
            else:
                logger.warning("Worker pool health check returned unexpected result")
                
        except concurrent.futures.TimeoutError:
            logger.error("Worker pool health check timed out - workers may be unresponsive")
        except Exception as e:
            logger.error(f"Worker pool health check failed: {e}")
    
    def get_metrics(self) -> WorkerMetrics:
        """Get current worker pool metrics."""
        with self._lock:
            return WorkerMetrics(
                active_workers=self._metrics.active_workers,
                total_workers=self._metrics.total_workers,
                queue_size=self._metrics.queue_size,
                pending_tasks=self._metrics.pending_tasks,
                completed_tasks=self._metrics.completed_tasks,
                failed_tasks=self._metrics.failed_tasks,
                average_task_time=self._metrics.average_task_time,
                worker_utilization=self._metrics.worker_utilization,
                throughput=self._metrics.throughput
            )
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        logger.info("Shutting down worker pool")
        
        # Stop monitoring
        self._monitoring_active = False
        
        if self._metrics_thread:
            self._metrics_thread.join(timeout=5.0)
        
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=wait)
        
        logger.info("Worker pool shutdown completed")


class DynamicWorkerPool(BaseWorkerPool):
    """Advanced worker pool with predictive scaling and load balancing."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self._worker_pools: Dict[str, WorkerPool] = {}
        self._load_balancer = LoadBalancer()
        self._predictor = LoadPredictor()
        
        # Create initial worker pools
        self._initialize_worker_pools()
        
        logger.info("Dynamic worker pool system initialized")
    
    def _initialize_worker_pools(self):
        """Initialize multiple worker pools for different workload types."""
        # Create separate pools for different task types
        pool_configs = {
            'cpu_intensive': WorkerConfig(
                min_workers=2,
                max_workers=mp.cpu_count(),
                worker_type=WorkerType.PROCESS,
                scaling_strategy=ScalingStrategy.REACTIVE
            ),
            'io_intensive': WorkerConfig(
                min_workers=4,
                max_workers=mp.cpu_count() * 4,
                worker_type=WorkerType.THREAD,
                scaling_strategy=ScalingStrategy.ADAPTIVE
            ),
            'general': self.config
        }
        
        for pool_name, config in pool_configs.items():
            self._worker_pools[pool_name] = WorkerPool(config)
    
    def submit(self, fn: Callable, *args, task_type: str = "general", **kwargs) -> concurrent.futures.Future:
        """Submit task to appropriate worker pool."""
        # Select appropriate pool
        pool = self._worker_pools.get(task_type, self._worker_pools['general'])
        
        # Record submission for load prediction
        self._predictor.record_task_submission(task_type)
        
        return pool.submit(fn, *args, **kwargs)
    
    def get_metrics(self) -> Dict[str, WorkerMetrics]:
        """Get metrics for all worker pools."""
        return {
            pool_name: pool.get_metrics()
            for pool_name, pool in self._worker_pools.items()
        }
    
    def scale_workers(self, target_count: int):
        """Scale all worker pools."""
        for pool in self._worker_pools.values():
            pool.scale_workers(target_count)
    
    def shutdown(self, wait: bool = True):
        """Shutdown all worker pools."""
        for pool in self._worker_pools.values():
            pool.shutdown(wait=wait)


class LoadBalancer:
    """Load balancer for distributing tasks across worker pools."""
    
    def __init__(self):
        self._pool_loads: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def select_pool(self, available_pools: List[str]) -> str:
        """Select the best pool based on current load."""
        if not available_pools:
            raise ValueError("No available pools")
        
        if len(available_pools) == 1:
            return available_pools[0]
        
        with self._lock:
            # Select pool with lowest load
            best_pool = min(available_pools, key=lambda p: self._pool_loads.get(p, 0.0))
            return best_pool
    
    def update_load(self, pool_name: str, load: float):
        """Update load information for a pool."""
        with self._lock:
            self._pool_loads[pool_name] = load


class LoadPredictor:
    """Predict future load patterns for proactive scaling."""
    
    def __init__(self):
        self._task_history: deque = deque(maxlen=10000)
        self._predictions: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def record_task_submission(self, task_type: str):
        """Record a task submission for pattern learning."""
        with self._lock:
            timestamp = time.time()
            self._task_history.append((timestamp, task_type))
    
    def predict_load(self, task_type: str, horizon_seconds: int = 300) -> float:
        """Predict load for the next time horizon."""
        with self._lock:
            current_time = time.time()
            
            # Count recent tasks of this type
            recent_tasks = [
                t for timestamp, t in self._task_history
                if current_time - timestamp < horizon_seconds and t == task_type
            ]
            
            # Simple prediction: assume similar rate continues
            if recent_tasks:
                return len(recent_tasks) / horizon_seconds
            
            return 0.0