"""
High-performance concurrent processing module for secure MPC transformer system.

This module provides:
- Dynamic worker pools with auto-scaling
- Async/await optimization
- Lock-free data structures
- Thread-local storage optimization
- Coroutine-based processing
- Producer-consumer patterns with backpressure
"""

from .worker_pools import WorkerPool, DynamicWorkerPool, WorkerConfig
from .async_optimization import AsyncOptimizer, AsyncContext
from .lockfree_structures import LockFreeQueue, LockFreeStack, AtomicCounter
from .thread_local_cache import ThreadLocalCache, ThreadLocalCacheManager
from .coroutine_processor import CoroutineProcessor, TaskScheduler
from .producer_consumer import ProducerConsumer, BackpressureManager

__all__ = [
    'WorkerPool',
    'DynamicWorkerPool', 
    'WorkerConfig',
    'AsyncOptimizer',
    'AsyncContext',
    'LockFreeQueue',
    'LockFreeStack',
    'AtomicCounter',
    'ThreadLocalCache',
    'ThreadLocalCacheManager',
    'CoroutineProcessor',
    'TaskScheduler',
    'ProducerConsumer',
    'BackpressureManager'
]