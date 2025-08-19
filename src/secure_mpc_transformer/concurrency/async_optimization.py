"""
Async/await optimization for high-performance concurrent operations.

This module provides optimized async patterns and tools for the
secure MPC transformer system.
"""

import asyncio
import contextvars
import inspect
import logging
import time
from collections import deque
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AsyncExecutionMode(Enum):
    """Execution modes for async operations."""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"


@dataclass
class AsyncConfig:
    """Configuration for async optimization."""
    max_concurrent_tasks: int = 100
    task_timeout_seconds: float = 300.0  # 5 minutes
    enable_task_cancellation: bool = True
    enable_result_caching: bool = True
    cache_size: int = 1000
    enable_metrics: bool = True

    # Concurrency control
    concurrency_limit: int = 50
    rate_limit_per_second: float | None = None
    burst_limit: int = 10

    # Optimization settings
    enable_batching: bool = True
    batch_size: int = 32
    batch_timeout_ms: float = 100.0
    enable_prefetching: bool = True
    prefetch_factor: int = 2


@dataclass
class AsyncMetrics:
    """Metrics for async operations."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_execution_time: float = 0.0
    concurrent_tasks: int = 0
    queue_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def success_rate(self) -> float:
        """Calculate task success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests


class TaskContext:
    """Context for async task execution."""

    def __init__(self, task_id: str, priority: float = 1.0, metadata: dict | None = None):
        self.task_id = task_id
        self.priority = priority
        self.metadata = metadata or {}
        self.start_time = time.perf_counter()
        self.execution_time: float | None = None
        self.result: Any | None = None
        self.exception: Exception | None = None
        self.cancelled = False


class AsyncCache:
    """Async-aware result cache."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (result, timestamp)
        self._access_order: deque = deque()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get cached result."""
        async with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                # Move to end for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return result
            return None

    async def put(self, key: str, value: Any):
        """Cache result."""
        async with self._lock:
            # Remove oldest items if cache is full
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.popleft()
                self._cache.pop(oldest_key, None)

            self._cache[key] = (value, time.time())
            if key not in self._access_order:
                self._access_order.append(key)

    async def clear(self):
        """Clear cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()


class RateLimiter:
    """Async rate limiter with token bucket algorithm."""

    def __init__(self, rate_per_second: float, burst_limit: int = 10):
        self.rate = rate_per_second
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_limit,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available."""
        while not await self.acquire(tokens):
            # Calculate wait time
            wait_time = tokens / self.rate
            await asyncio.sleep(min(wait_time, 0.1))  # Max 100ms sleep


class AsyncBatcher:
    """Batches async operations for efficiency."""

    def __init__(self, batch_size: int = 32, timeout_ms: float = 100.0):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_ms / 1000.0
        self._pending_items: list[tuple[Any, asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: asyncio.Task | None = None

    async def submit(self, item: Any, processor: Callable) -> Any:
        """Submit item for batch processing."""
        future = asyncio.Future()

        async with self._batch_lock:
            self._pending_items.append((item, future))

            # Start batch processing if needed
            if len(self._pending_items) >= self.batch_size:
                if self._batch_task is None or self._batch_task.done():
                    self._batch_task = asyncio.create_task(
                        self._process_batch(processor)
                    )
            elif self._batch_task is None:
                # Start timeout-based batch processing
                self._batch_task = asyncio.create_task(
                    self._process_batch_with_timeout(processor)
                )

        return await future

    async def _process_batch(self, processor: Callable):
        """Process current batch."""
        async with self._batch_lock:
            if not self._pending_items:
                return

            batch_items = self._pending_items.copy()
            self._pending_items.clear()

        # Process batch
        items = [item for item, _ in batch_items]
        futures = [future for _, future in batch_items]

        try:
            results = await processor(items)

            # Set results
            for future, result in zip(futures, results, strict=False):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def _process_batch_with_timeout(self, processor: Callable):
        """Process batch with timeout."""
        await asyncio.sleep(self.timeout_seconds)
        await self._process_batch(processor)


class AsyncOptimizer:
    """Main async optimization coordinator."""

    def __init__(self, config: AsyncConfig):
        self.config = config

        # Components
        self.cache = AsyncCache(config.cache_size) if config.enable_result_caching else None
        self.rate_limiter = RateLimiter(
            config.rate_limit_per_second,
            config.burst_limit
        ) if config.rate_limit_per_second else None
        self.batcher = AsyncBatcher(
            config.batch_size,
            config.batch_timeout_ms
        ) if config.enable_batching else None

        # Task management
        self._active_tasks: set[asyncio.Task] = set()
        self._task_contexts: dict[str, TaskContext] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_tasks)

        # Metrics
        self._metrics = AsyncMetrics()
        self._execution_times: deque = deque(maxlen=1000)

        logger.info("Async optimizer initialized")

    async def execute_optimized(self,
                              coro_or_func: Coroutine | Callable,
                              *args,
                              task_id: str | None = None,
                              priority: float = 1.0,
                              cache_key: str | None = None,
                              use_batching: bool = False,
                              **kwargs) -> Any:
        """Execute coroutine or function with optimizations."""

        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"

        # Check cache first
        if cache_key and self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                self._metrics.cache_hits += 1
                return cached_result
            self._metrics.cache_misses += 1

        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.wait_for_tokens()

        # Create task context
        context = TaskContext(task_id, priority)
        self._task_contexts[task_id] = context
        self._metrics.total_tasks += 1

        try:
            # Execute with concurrency control
            async with self._semaphore:
                self._metrics.concurrent_tasks += 1

                try:
                    # Determine execution method
                    if use_batching and self.batcher:
                        result = await self._execute_batched(coro_or_func, args, kwargs)
                    else:
                        result = await self._execute_single(coro_or_func, args, kwargs)

                    # Cache result if specified
                    if cache_key and self.cache:
                        await self.cache.put(cache_key, result)

                    # Update metrics
                    execution_time = time.perf_counter() - context.start_time
                    self._execution_times.append(execution_time)
                    self._metrics.completed_tasks += 1

                    if self._execution_times:
                        self._metrics.average_execution_time = (
                            sum(self._execution_times) / len(self._execution_times)
                        )

                    context.result = result
                    context.execution_time = execution_time

                    return result

                finally:
                    self._metrics.concurrent_tasks = max(0, self._metrics.concurrent_tasks - 1)

        except asyncio.CancelledError:
            self._metrics.cancelled_tasks += 1
            context.cancelled = True
            raise
        except Exception as e:
            self._metrics.failed_tasks += 1
            context.exception = e
            raise
        finally:
            # Cleanup
            self._task_contexts.pop(task_id, None)

    async def _execute_single(self, coro_or_func: Coroutine | Callable, args: tuple, kwargs: dict) -> Any:
        """Execute single coroutine or function."""
        if inspect.iscoroutine(coro_or_func):
            return await coro_or_func
        elif inspect.iscoroutinefunction(coro_or_func):
            return await coro_or_func(*args, **kwargs)
        elif callable(coro_or_func):
            # Run in executor for blocking functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: coro_or_func(*args, **kwargs))
        else:
            raise TypeError("Expected coroutine, coroutine function, or callable")

    async def _execute_batched(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute using batching."""
        if not self.batcher:
            return await self._execute_single(func, args, kwargs)

        # Create batch item
        batch_item = (args, kwargs)

        # Define batch processor
        async def batch_processor(items):
            results = []
            for item_args, item_kwargs in items:
                result = await self._execute_single(func, item_args, item_kwargs)
                results.append(result)
            return results

        return await self.batcher.submit(batch_item, batch_processor)

    async def execute_concurrent(self,
                               tasks: list[Coroutine | Callable],
                               return_exceptions: bool = False) -> list[Any]:
        """Execute multiple tasks concurrently."""
        if not tasks:
            return []

        # Create optimized tasks
        optimized_tasks = []
        for i, task in enumerate(tasks):
            task_id = f"concurrent_{i}"

            if inspect.iscoroutine(task) or callable(task):
                optimized_task = self.execute_optimized(task, task_id=task_id)
            else:
                optimized_task = task

            optimized_tasks.append(optimized_task)

        # Execute concurrently
        return await asyncio.gather(*optimized_tasks, return_exceptions=return_exceptions)

    async def execute_pipeline(self,
                             stages: list[Callable],
                             initial_data: Any) -> Any:
        """Execute pipeline of async operations."""
        data = initial_data

        for i, stage in enumerate(stages):
            stage_id = f"pipeline_stage_{i}"
            data = await self.execute_optimized(
                stage,
                data,
                task_id=stage_id
            )

        return data

    @asynccontextmanager
    async def task_group(self, max_concurrent: int | None = None):
        """Context manager for grouped task execution."""
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)
        else:
            semaphore = self._semaphore

        tasks = []

        class TaskGroup:
            async def create_task(self, coro_or_func: Coroutine | Callable, *args, **kwargs):
                async with semaphore:
                    task = asyncio.create_task(
                        self.parent.execute_optimized(coro_or_func, *args, **kwargs)
                    )
                    tasks.append(task)
                    return task

            def __init__(self, parent):
                self.parent = parent

        group = TaskGroup(self)

        try:
            yield group
        finally:
            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def prefetch(self,
                      generators: list[Callable],
                      buffer_size: int | None = None) -> AsyncIterator[Any]:
        """Prefetch data from async generators."""
        buffer_size = buffer_size or (self.config.prefetch_factor * self.config.batch_size)

        async def prefetch_generator():
            buffer = asyncio.Queue(maxsize=buffer_size)
            generators_tasks = []

            # Start all generators
            for gen in generators:
                task = asyncio.create_task(self._fill_buffer_from_generator(gen, buffer))
                generators_tasks.append(task)

            try:
                while True:
                    # Get next item from buffer
                    try:
                        item = await asyncio.wait_for(buffer.get(), timeout=1.0)
                        if item is StopAsyncIteration:
                            break
                        yield item
                    except asyncio.TimeoutError:
                        # Check if all generators are done
                        if all(task.done() for task in generators_tasks):
                            break
            finally:
                # Cancel remaining tasks
                for task in generators_tasks:
                    if not task.done():
                        task.cancel()

        return prefetch_generator()

    async def _fill_buffer_from_generator(self, generator: Callable, buffer: asyncio.Queue):
        """Fill buffer from async generator."""
        try:
            async for item in generator():
                await buffer.put(item)
        except Exception as e:
            logger.error(f"Generator error: {e}")
        finally:
            await buffer.put(StopAsyncIteration)

    def get_metrics(self) -> AsyncMetrics:
        """Get current async metrics."""
        self._metrics.queue_size = len(self._task_contexts)
        return self._metrics

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on async optimizer."""
        start_time = time.perf_counter()

        # Test basic async operation
        async def test_task():
            await asyncio.sleep(0.001)  # 1ms
            return "healthy"

        try:
            result = await self.execute_optimized(test_task, task_id="health_check")
            execution_time = time.perf_counter() - start_time

            return {
                "status": "healthy",
                "test_result": result,
                "execution_time_ms": execution_time * 1000,
                "metrics": self.get_metrics().to_dict() if hasattr(self.get_metrics(), 'to_dict') else str(self.get_metrics()),
                "active_tasks": len(self._task_contexts)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000
            }

    async def shutdown(self):
        """Shutdown async optimizer."""
        logger.info("Shutting down async optimizer")

        # Cancel all active tasks
        for task_id, context in self._task_contexts.items():
            context.cancelled = True

        # Clear cache
        if self.cache:
            await self.cache.clear()

        logger.info("Async optimizer shutdown completed")


class AsyncContext:
    """Context manager for async optimization within a specific scope."""

    def __init__(self, config: AsyncConfig | None = None):
        self.config = config or AsyncConfig()
        self._optimizer: AsyncOptimizer | None = None
        self._context_var = contextvars.ContextVar('async_context', default=None)

    async def __aenter__(self):
        self._optimizer = AsyncOptimizer(self.config)
        self._context_var.set(self._optimizer)
        return self._optimizer

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._optimizer:
            await self._optimizer.shutdown()
        self._context_var.set(None)

    def get_current_optimizer(self) -> AsyncOptimizer | None:
        """Get the current async optimizer from context."""
        return self._context_var.get()


# Global async context
_global_async_context = AsyncContext()


async def optimized_async(coro_or_func: Coroutine | Callable, *args, **kwargs) -> Any:
    """Convenience function for optimized async execution."""
    context = _global_async_context.get_current_optimizer()
    if context:
        return await context.execute_optimized(coro_or_func, *args, **kwargs)
    else:
        # Fallback to direct execution
        if inspect.iscoroutine(coro_or_func):
            return await coro_or_func
        elif inspect.iscoroutinefunction(coro_or_func):
            return await coro_or_func(*args, **kwargs)
        elif callable(coro_or_func):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: coro_or_func(*args, **kwargs))


def async_cached(cache_key_func: Callable | None = None):
    """Decorator for async result caching."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args))}:{hash(str(kwargs))}"

            return await optimized_async(func, *args, cache_key=cache_key, **kwargs)

        return wrapper
    return decorator
