"""Advanced Performance Optimization System - Generation 3 Scaling Enhancement."""

import asyncio
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """Performance optimization targets."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    BALANCED = "balanced"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


@dataclass
class PerformanceMetric:
    """Performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    context: dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class OptimizationProfile:
    """Performance optimization profile."""
    name: str
    target: OptimizationTarget
    max_memory_mb: int
    max_cpu_percent: float
    target_latency_ms: float
    min_throughput_rps: float
    cache_strategy: CacheStrategy
    concurrency_level: int
    prefetch_enabled: bool = True
    batch_processing: bool = True


class AdaptiveLoadBalancer:
    """Adaptive load balancer for request distribution."""

    def __init__(self, initial_workers: int = 4, max_workers: int = 16):
        self.workers: list[dict[str, Any]] = []
        self.max_workers = max_workers
        self.current_request_id = 0
        self.worker_stats: dict[int, dict[str, Any]] = {}
        self.load_threshold = 0.8
        self.scale_up_cooldown = 30.0  # seconds
        self.scale_down_cooldown = 60.0  # seconds
        self.last_scale_time = 0.0

        # Initialize workers
        for i in range(initial_workers):
            self._add_worker(i)

        logger.info(f"Adaptive load balancer initialized with {initial_workers} workers")

    def _add_worker(self, worker_id: int):
        """Add new worker."""
        worker = {
            "id": worker_id,
            "active_requests": 0,
            "total_requests": 0,
            "total_latency": 0.0,
            "last_request_time": 0.0,
            "created_at": time.time()
        }
        self.workers.append(worker)
        self.worker_stats[worker_id] = {
            "avg_latency": 0.0,
            "requests_per_second": 0.0,
            "utilization": 0.0
        }
        logger.info(f"Added worker {worker_id}")

    def _remove_worker(self, worker_id: int):
        """Remove worker."""
        self.workers = [w for w in self.workers if w["id"] != worker_id]
        if worker_id in self.worker_stats:
            del self.worker_stats[worker_id]
        logger.info(f"Removed worker {worker_id}")

    def select_worker(self) -> int:
        """Select optimal worker for next request."""
        if not self.workers:
            return 0

        # Select worker with lowest load
        best_worker = min(self.workers, key=lambda w: w["active_requests"])

        # Update request assignment
        best_worker["active_requests"] += 1
        best_worker["total_requests"] += 1
        best_worker["last_request_time"] = time.time()

        return best_worker["id"]

    def complete_request(self, worker_id: int, latency: float):
        """Mark request as completed."""
        worker = next((w for w in self.workers if w["id"] == worker_id), None)
        if worker:
            worker["active_requests"] = max(0, worker["active_requests"] - 1)
            worker["total_latency"] += latency

            # Update stats
            if worker["total_requests"] > 0:
                self.worker_stats[worker_id]["avg_latency"] = (
                    worker["total_latency"] / worker["total_requests"]
                )

        # Check if scaling is needed
        self._evaluate_scaling()

    def _evaluate_scaling(self):
        """Evaluate if workers need to be scaled up or down."""
        current_time = time.time()

        # Cooldown check
        if current_time - self.last_scale_time < self.scale_up_cooldown:
            return

        # Calculate overall load
        total_active = sum(w["active_requests"] for w in self.workers)
        avg_load = total_active / len(self.workers) if self.workers else 0

        # Scale up if load is high
        if (avg_load > self.load_threshold and
            len(self.workers) < self.max_workers):
            new_worker_id = max([w["id"] for w in self.workers], default=-1) + 1
            self._add_worker(new_worker_id)
            self.last_scale_time = current_time

        # Scale down if load is low (and we have more than 2 workers)
        elif (avg_load < 0.3 and
              len(self.workers) > 2 and
              current_time - self.last_scale_time > self.scale_down_cooldown):
            # Remove least utilized worker
            idle_workers = [w for w in self.workers if w["active_requests"] == 0]
            if idle_workers:
                worker_to_remove = max(idle_workers, key=lambda w: current_time - w["last_request_time"])
                self._remove_worker(worker_to_remove["id"])
                self.last_scale_time = current_time

    def get_stats(self) -> dict[str, Any]:
        """Get load balancer statistics."""
        total_active = sum(w["active_requests"] for w in self.workers)
        total_requests = sum(w["total_requests"] for w in self.workers)

        return {
            "worker_count": len(self.workers),
            "total_active_requests": total_active,
            "total_requests": total_requests,
            "avg_load": total_active / len(self.workers) if self.workers else 0,
            "worker_stats": self.worker_stats.copy(),
            "last_scale_time": self.last_scale_time
        }


class IntelligentCache:
    """Intelligent multi-layer cache with predictive prefetching."""

    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy

        # Cache layers
        self.l1_cache: dict[str, Any] = {}  # Hot cache
        self.l2_cache: dict[str, Any] = {}  # Warm cache
        self.l3_cache: dict[str, Any] = {}  # Cold cache

        # Cache metadata
        self.access_counts: dict[str, int] = {}
        self.access_times: dict[str, list[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Predictive components
        self.access_patterns: dict[str, list[str]] = {}  # key -> frequently accessed after
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_task: asyncio.Task | None = None

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"Intelligent cache initialized with strategy: {strategy.value}")

    async def start_prefetching(self):
        """Start background prefetching task."""
        if not self.prefetch_task:
            self.prefetch_task = asyncio.create_task(self._prefetch_worker())

    def stop_prefetching(self):
        """Stop background prefetching."""
        if self.prefetch_task:
            self.prefetch_task.cancel()
            self.prefetch_task = None

    async def get(self, key: str, compute_func: Callable | None = None) -> Any | None:
        """Get item from cache with optional computation."""
        with self.lock:
            # Check L1 cache first (hottest)
            if key in self.l1_cache:
                self._record_access(key)
                self.cache_hits += 1
                return self.l1_cache[key]

            # Check L2 cache
            if key in self.l2_cache:
                value = self.l2_cache[key]
                # Promote to L1
                self._promote_to_l1(key, value)
                self._record_access(key)
                self.cache_hits += 1
                return value

            # Check L3 cache
            if key in self.l3_cache:
                value = self.l3_cache[key]
                # Promote to L2
                self._promote_to_l2(key, value)
                self._record_access(key)
                self.cache_hits += 1
                return value

        # Cache miss - compute if function provided
        self.cache_misses += 1

        if compute_func:
            if asyncio.iscoroutinefunction(compute_func):
                value = await compute_func(key)
            else:
                value = compute_func(key)

            await self.put(key, value)
            return value

        return None

    async def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            # Always start in L1 cache
            self.l1_cache[key] = value
            self._record_access(key)

            # Trigger cache rebalancing if needed
            await self._rebalance_cache()

            # Update access patterns for prefetching
            self._update_access_patterns(key)

            # Schedule prefetching for related items
            await self._schedule_prefetch(key)

    def _record_access(self, key: str):
        """Record cache access for analytics."""
        current_time = time.time()

        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

        # Update access times
        if key not in self.access_times:
            self.access_times[key] = []
        self.access_times[key].append(current_time)

        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self.access_times[key] = [t for t in self.access_times[key] if t > cutoff_time]

    def _promote_to_l1(self, key: str, value: Any):
        """Promote item to L1 cache."""
        # Remove from L2
        if key in self.l2_cache:
            del self.l2_cache[key]

        # Add to L1
        self.l1_cache[key] = value

    def _promote_to_l2(self, key: str, value: Any):
        """Promote item to L2 cache."""
        # Remove from L3
        if key in self.l3_cache:
            del self.l3_cache[key]

        # Add to L2
        self.l2_cache[key] = value

    async def _rebalance_cache(self):
        """Rebalance cache layers based on strategy."""
        total_items = len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)

        if total_items <= self.max_size:
            return

        # Determine eviction based on strategy
        if self.strategy == CacheStrategy.LRU:
            await self._evict_lru()
        elif self.strategy == CacheStrategy.LFU:
            await self._evict_lfu()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            await self._evict_adaptive()

    async def _evict_lru(self):
        """Evict least recently used items."""
        # Find LRU items across all layers
        all_items = []

        for key in self.l1_cache:
            last_access = max(self.access_times.get(key, [0]))
            all_items.append((key, last_access, "l1"))

        for key in self.l2_cache:
            last_access = max(self.access_times.get(key, [0]))
            all_items.append((key, last_access, "l2"))

        for key in self.l3_cache:
            last_access = max(self.access_times.get(key, [0]))
            all_items.append((key, last_access, "l3"))

        # Sort by access time and evict oldest
        all_items.sort(key=lambda x: x[1])

        items_to_evict = len(all_items) - self.max_size + 1
        for i in range(items_to_evict):
            key, _, layer = all_items[i]
            await self._evict_key(key, layer)

    async def _evict_lfu(self):
        """Evict least frequently used items."""
        # Find LFU items across all layers
        all_items = []

        for key in self.l1_cache:
            frequency = self.access_counts.get(key, 0)
            all_items.append((key, frequency, "l1"))

        for key in self.l2_cache:
            frequency = self.access_counts.get(key, 0)
            all_items.append((key, frequency, "l2"))

        for key in self.l3_cache:
            frequency = self.access_counts.get(key, 0)
            all_items.append((key, frequency, "l3"))

        # Sort by frequency and evict least frequent
        all_items.sort(key=lambda x: x[1])

        items_to_evict = len(all_items) - self.max_size + 1
        for i in range(items_to_evict):
            key, _, layer = all_items[i]
            await self._evict_key(key, layer)

    async def _evict_adaptive(self):
        """Adaptive eviction based on access patterns."""
        # Combine recency and frequency for scoring
        all_items = []
        current_time = time.time()

        for key in self.l1_cache:
            last_access = max(self.access_times.get(key, [0]))
            frequency = self.access_counts.get(key, 0)
            recency_score = 1.0 / max(current_time - last_access, 1)
            combined_score = frequency * recency_score
            all_items.append((key, combined_score, "l1"))

        for key in self.l2_cache:
            last_access = max(self.access_times.get(key, [0]))
            frequency = self.access_counts.get(key, 0)
            recency_score = 1.0 / max(current_time - last_access, 1)
            combined_score = frequency * recency_score * 0.8  # L2 penalty
            all_items.append((key, combined_score, "l2"))

        for key in self.l3_cache:
            last_access = max(self.access_times.get(key, [0]))
            frequency = self.access_counts.get(key, 0)
            recency_score = 1.0 / max(current_time - last_access, 1)
            combined_score = frequency * recency_score * 0.6  # L3 penalty
            all_items.append((key, combined_score, "l3"))

        # Sort by combined score and evict lowest
        all_items.sort(key=lambda x: x[1])

        items_to_evict = len(all_items) - self.max_size + 1
        for i in range(items_to_evict):
            key, _, layer = all_items[i]
            await self._evict_key(key, layer)

    async def _evict_key(self, key: str, layer: str):
        """Evict specific key from specific layer."""
        if layer == "l1" and key in self.l1_cache:
            del self.l1_cache[key]
        elif layer == "l2" and key in self.l2_cache:
            del self.l2_cache[key]
        elif layer == "l3" and key in self.l3_cache:
            del self.l3_cache[key]

        # Clean up metadata
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_patterns:
            del self.access_patterns[key]

    def _update_access_patterns(self, key: str):
        """Update access patterns for predictive prefetching."""
        # Simple pattern tracking - could be more sophisticated
        if hasattr(self, '_last_accessed_key'):
            last_key = self._last_accessed_key
            if last_key not in self.access_patterns:
                self.access_patterns[last_key] = []

            if key not in self.access_patterns[last_key]:
                self.access_patterns[last_key].append(key)

                # Keep only recent patterns
                if len(self.access_patterns[last_key]) > 10:
                    self.access_patterns[last_key] = self.access_patterns[last_key][-10:]

        self._last_accessed_key = key

    async def _schedule_prefetch(self, key: str):
        """Schedule prefetching for related items."""
        if key in self.access_patterns:
            for related_key in self.access_patterns[key][:3]:  # Top 3 related
                if not self._is_cached(related_key):
                    try:
                        await self.prefetch_queue.put(related_key)
                    except asyncio.QueueFull:
                        pass  # Queue full, skip

    def _is_cached(self, key: str) -> bool:
        """Check if key is in any cache layer."""
        return key in self.l1_cache or key in self.l2_cache or key in self.l3_cache

    async def _prefetch_worker(self):
        """Background worker for prefetching."""
        while True:
            try:
                key = await asyncio.wait_for(self.prefetch_queue.get(), timeout=1.0)

                # Prefetch logic would go here
                # For now, just mark the task as done
                self.prefetch_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)

        return {
            "hit_rate": hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_items": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache),
            "l1_items": len(self.l1_cache),
            "l2_items": len(self.l2_cache),
            "l3_items": len(self.l3_cache),
            "strategy": self.strategy.value,
            "access_patterns": len(self.access_patterns)
        }


class ConcurrentExecutionManager:
    """Advanced concurrent execution management."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(mp.cpu_count() or 1, 4))

        # Execution queues
        self.high_priority_queue: asyncio.Queue = asyncio.Queue()
        self.normal_priority_queue: asyncio.Queue = asyncio.Queue()
        self.low_priority_queue: asyncio.Queue = asyncio.Queue()

        # Worker tasks
        self.workers: list[asyncio.Task] = []
        self.running = False

        # Performance tracking
        self.execution_times: list[float] = []
        self.queue_times: list[float] = []

        logger.info(f"Concurrent execution manager initialized with {self.max_workers} workers")

    async def start(self):
        """Start execution workers."""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(min(self.max_workers, 8)):  # Async workers
            worker_task = asyncio.create_task(self._execution_worker(f"worker_{i}"))
            self.workers.append(worker_task)

        logger.info(f"Started {len(self.workers)} execution workers")

    async def stop(self):
        """Stop execution workers."""
        if not self.running:
            return

        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        logger.info("Execution workers stopped")

    async def execute_async(self, func: Callable, *args, priority: str = "normal", **kwargs) -> Any:
        """Execute function asynchronously with priority."""
        task_info = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "queued_time": time.time(),
            "future": asyncio.Future()
        }

        # Add to appropriate queue based on priority
        if priority == "high":
            await self.high_priority_queue.put(task_info)
        elif priority == "low":
            await self.low_priority_queue.put(task_info)
        else:
            await self.normal_priority_queue.put(task_info)

        # Wait for result
        return await task_info["future"]

    async def execute_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Execute CPU-bound function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)

    async def execute_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Execute CPU-intensive function in process pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)

    async def _execution_worker(self, worker_name: str):
        """Worker for processing execution queue."""
        while self.running:
            try:
                task_info = await self._get_next_task()

                if not task_info:
                    await asyncio.sleep(0.1)
                    continue

                # Record queue time
                queue_time = time.time() - task_info["queued_time"]
                self.queue_times.append(queue_time)

                # Execute task
                start_time = time.time()

                try:
                    func = task_info["func"]
                    args = task_info["args"]
                    kwargs = task_info["kwargs"]

                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    task_info["future"].set_result(result)

                except Exception as e:
                    task_info["future"].set_exception(e)

                # Record execution time
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)

                # Trim metrics lists
                if len(self.execution_times) > 1000:
                    self.execution_times = self.execution_times[-1000:]
                if len(self.queue_times) > 1000:
                    self.queue_times = self.queue_times[-1000:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Execution worker {worker_name} error: {e}")

    async def _get_next_task(self) -> dict[str, Any] | None:
        """Get next task from priority queues."""
        # Check high priority first
        try:
            return self.high_priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Check normal priority
        try:
            return self.normal_priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Check low priority
        try:
            return self.low_priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        avg_execution_time = sum(self.execution_times) / max(len(self.execution_times), 1)
        avg_queue_time = sum(self.queue_times) / max(len(self.queue_times), 1)

        return {
            "active_workers": len(self.workers),
            "max_workers": self.max_workers,
            "avg_execution_time_ms": avg_execution_time * 1000,
            "avg_queue_time_ms": avg_queue_time * 1000,
            "high_priority_queue_size": self.high_priority_queue.qsize(),
            "normal_priority_queue_size": self.normal_priority_queue.qsize(),
            "low_priority_queue_size": self.low_priority_queue.qsize(),
            "total_executions": len(self.execution_times),
            "running": self.running
        }


class AdvancedPerformanceOptimizer:
    """Advanced performance optimization system."""

    def __init__(self, profile: OptimizationProfile = None):
        self.profile = profile or OptimizationProfile(
            name="default",
            target=OptimizationTarget.BALANCED,
            max_memory_mb=4000,
            max_cpu_percent=80.0,
            target_latency_ms=100.0,
            min_throughput_rps=10.0,
            cache_strategy=CacheStrategy.ADAPTIVE,
            concurrency_level=8
        )

        # Core components
        self.load_balancer = AdaptiveLoadBalancer(
            initial_workers=self.profile.concurrency_level,
            max_workers=self.profile.concurrency_level * 2
        )

        self.cache = IntelligentCache(
            max_size=1000,
            strategy=self.profile.cache_strategy
        )

        self.executor = ConcurrentExecutionManager(
            max_workers=self.profile.concurrency_level
        )

        # Performance tracking
        self.metrics: list[PerformanceMetric] = []
        self.optimization_history: list[dict[str, Any]] = []

        # Auto-optimization
        self.auto_optimize_enabled = True
        self.last_optimization = 0.0
        self.optimization_interval = 300.0  # 5 minutes

        logger.info(f"Advanced performance optimizer initialized with profile: {self.profile.name}")

    async def start(self):
        """Start optimization system."""
        await self.cache.start_prefetching()
        await self.executor.start()

        # Start auto-optimization task
        if self.auto_optimize_enabled:
            asyncio.create_task(self._auto_optimization_loop())

        logger.info("Performance optimizer started")

    async def stop(self):
        """Stop optimization system."""
        self.cache.stop_prefetching()
        await self.executor.stop()

        logger.info("Performance optimizer stopped")

    async def execute_optimized(self, operation_name: str, func: Callable,
                              *args, **kwargs) -> Any:
        """Execute function with comprehensive optimization."""
        start_time = time.time()

        # Select worker
        worker_id = self.load_balancer.select_worker()

        # Check cache first
        cache_key = self._generate_cache_key(operation_name, args, kwargs)
        cached_result = await self.cache.get(cache_key)

        if cached_result is not None:
            # Cache hit - record metrics and return
            latency = (time.time() - start_time) * 1000
            self.load_balancer.complete_request(worker_id, latency)

            self._record_metric("cache_hit_latency", latency, "ms", {
                "operation": operation_name,
                "worker_id": worker_id
            })

            return cached_result

        # Cache miss - execute with optimization
        try:
            # Determine execution strategy based on function characteristics
            if self._is_cpu_intensive(func):
                result = await self.executor.execute_in_thread(func, *args, **kwargs)
            elif self._is_compute_intensive(func):
                result = await self.executor.execute_in_process(func, *args, **kwargs)
            else:
                result = await self.executor.execute_async(func, *args, **kwargs)

            # Cache result if cacheable
            if self._is_cacheable(result):
                await self.cache.put(cache_key, result)

            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.load_balancer.complete_request(worker_id, latency)

            self._record_metric("execution_latency", latency, "ms", {
                "operation": operation_name,
                "worker_id": worker_id,
                "cache_miss": True
            })

            return result

        except Exception as e:
            # Record error metrics
            latency = (time.time() - start_time) * 1000
            self.load_balancer.complete_request(worker_id, latency)

            self._record_metric("execution_error", 1, "count", {
                "operation": operation_name,
                "error_type": type(e).__name__
            })

            raise

    def _generate_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for operation."""
        # Simple hash-based key generation
        import hashlib

        key_data = f"{operation_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Determine if function is CPU intensive."""
        # Simple heuristic - could be more sophisticated
        func_name = getattr(func, '__name__', str(func))
        cpu_intensive_keywords = ['compute', 'calculate', 'process', 'transform', 'encode']

        return any(keyword in func_name.lower() for keyword in cpu_intensive_keywords)

    def _is_compute_intensive(self, func: Callable) -> bool:
        """Determine if function is compute intensive (process pool candidate)."""
        func_name = getattr(func, '__name__', str(func))
        compute_keywords = ['matrix', 'numpy', 'scientific', 'math', 'algorithm']

        return any(keyword in func_name.lower() for keyword in compute_keywords)

    def _is_cacheable(self, result: Any) -> bool:
        """Determine if result should be cached."""
        # Don't cache very large objects or streams
        try:
            import sys
            return sys.getsizeof(result) < 1024 * 1024  # 1MB limit
        except (TypeError, OSError):
            return False

    def _record_metric(self, name: str, value: float, unit: str, context: dict[str, Any]):
        """Record performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            context=context
        )

        self.metrics.append(metric)

        # Trim metrics to prevent memory growth
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]

    async def _auto_optimization_loop(self):
        """Background auto-optimization loop."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)

                if time.time() - self.last_optimization >= self.optimization_interval:
                    await self._perform_auto_optimization()
                    self.last_optimization = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-optimization error: {e}")

    async def _perform_auto_optimization(self):
        """Perform automatic optimization based on metrics."""
        logger.info("Performing auto-optimization...")

        # Analyze recent metrics
        recent_metrics = [m for m in self.metrics if time.time() - m.timestamp < 600]  # 10 minutes

        if not recent_metrics:
            return

        # Calculate key performance indicators
        latency_metrics = [m for m in recent_metrics if m.name == "execution_latency"]
        avg_latency = sum(m.value for m in latency_metrics) / max(len(latency_metrics), 1)

        cache_stats = self.cache.get_stats()
        load_balancer_stats = self.load_balancer.get_stats()
        executor_stats = self.executor.get_stats()

        # Optimization decisions
        optimizations = []

        # Latency optimization
        if avg_latency > self.profile.target_latency_ms:
            if cache_stats["hit_rate"] < 0.7:
                optimizations.append("increase_cache_size")

            if load_balancer_stats["avg_load"] > 0.8:
                optimizations.append("scale_up_workers")

        # Cache optimization
        if cache_stats["hit_rate"] < 0.5:
            optimizations.append("improve_caching_strategy")

        # Apply optimizations
        for optimization in optimizations:
            await self._apply_optimization(optimization)

        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "avg_latency": avg_latency,
            "cache_hit_rate": cache_stats["hit_rate"],
            "worker_count": load_balancer_stats["worker_count"],
            "optimizations": optimizations
        })

    async def _apply_optimization(self, optimization: str):
        """Apply specific optimization."""
        logger.info(f"Applying optimization: {optimization}")

        if optimization == "increase_cache_size":
            # Increase cache size by 20%
            new_size = int(self.cache.max_size * 1.2)
            self.cache.max_size = min(new_size, 5000)

        elif optimization == "scale_up_workers":
            # Add worker to load balancer
            current_workers = len(self.load_balancer.workers)
            if current_workers < self.load_balancer.max_workers:
                new_worker_id = max([w["id"] for w in self.load_balancer.workers], default=-1) + 1
                self.load_balancer._add_worker(new_worker_id)

        elif optimization == "improve_caching_strategy":
            # Switch to more aggressive caching
            if self.cache.strategy == CacheStrategy.LRU:
                self.cache.strategy = CacheStrategy.ADAPTIVE

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        recent_metrics = [m for m in self.metrics if time.time() - m.timestamp < 600]

        # Calculate averages
        latency_metrics = [m for m in recent_metrics if m.name == "execution_latency"]
        avg_latency = sum(m.value for m in latency_metrics) / max(len(latency_metrics), 1)

        throughput_metrics = len(latency_metrics) / 600 if latency_metrics else 0  # per second

        return {
            "profile": asdict(self.profile),
            "performance": {
                "avg_latency_ms": avg_latency,
                "throughput_rps": throughput_metrics,
                "total_operations": len(recent_metrics)
            },
            "cache_stats": self.cache.get_stats(),
            "load_balancer_stats": self.load_balancer.get_stats(),
            "executor_stats": self.executor.get_stats(),
            "optimization_count": len(self.optimization_history),
            "last_optimization": self.last_optimization,
            "timestamp": time.time()
        }

    def set_optimization_profile(self, profile: OptimizationProfile):
        """Update optimization profile."""
        self.profile = profile
        logger.info(f"Updated optimization profile: {profile.name}")

    def enable_auto_optimization(self, enabled: bool = True):
        """Enable or disable auto-optimization."""
        self.auto_optimize_enabled = enabled
        logger.info(f"Auto-optimization {'enabled' if enabled else 'disabled'}")


# Default optimization profiles
DEFAULT_PROFILES = {
    "low_latency": OptimizationProfile(
        name="low_latency",
        target=OptimizationTarget.LATENCY,
        max_memory_mb=8000,
        max_cpu_percent=90.0,
        target_latency_ms=50.0,
        min_throughput_rps=5.0,
        cache_strategy=CacheStrategy.ADAPTIVE,
        concurrency_level=16,
        prefetch_enabled=True,
        batch_processing=False
    ),

    "high_throughput": OptimizationProfile(
        name="high_throughput",
        target=OptimizationTarget.THROUGHPUT,
        max_memory_mb=16000,
        max_cpu_percent=95.0,
        target_latency_ms=200.0,
        min_throughput_rps=50.0,
        cache_strategy=CacheStrategy.LFU,
        concurrency_level=32,
        prefetch_enabled=True,
        batch_processing=True
    ),

    "memory_efficient": OptimizationProfile(
        name="memory_efficient",
        target=OptimizationTarget.MEMORY,
        max_memory_mb=2000,
        max_cpu_percent=70.0,
        target_latency_ms=300.0,
        min_throughput_rps=2.0,
        cache_strategy=CacheStrategy.LRU,
        concurrency_level=4,
        prefetch_enabled=False,
        batch_processing=True
    )
}
