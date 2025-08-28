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

from .async_optimization import AsyncContext, AsyncOptimizer
# from .coroutine_processor import CoroutineProcessor, TaskScheduler  # Module not yet implemented
# from .lockfree_structures import AtomicCounter, LockFreeQueue, LockFreeStack  # Module not yet implemented
# from .producer_consumer import BackpressureManager, ProducerConsumer  # Module not yet implemented  
# from .thread_local_cache import ThreadLocalCache, ThreadLocalCacheManager  # Module not yet implemented
from .worker_pools import DynamicWorkerPool, WorkerConfig, WorkerPool

__all__ = [
    'WorkerPool',
    'DynamicWorkerPool', 
    'WorkerConfig',
    'AsyncOptimizer',
    'AsyncContext',
    # 'LockFreeQueue',
    # 'LockFreeStack', 
    # 'AtomicCounter',
    # 'ThreadLocalCache',
    # 'ThreadLocalCacheManager',
    # 'CoroutineProcessor',
    # 'TaskScheduler',
    # 'ProducerConsumer',
    # 'BackpressureManager'
]
