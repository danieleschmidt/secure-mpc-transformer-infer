"""
Main cache management system coordinating multi-level caching.

This module orchestrates L1, L2, and distributed caching layers
with intelligent routing and optimization.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .coherence_manager import CacheCoherenceManager
from .distributed_cache import DistributedCache, RedisConfig
from .eviction_policies import LRUPolicy
from .l1_cache import L1TensorCache
from .l2_cache import CompressionConfig, L2ComponentCache

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1 = "l1"  # In-memory tensor cache
    L2 = "l2"  # Compressed component cache
    DISTRIBUTED = "distributed"  # Redis/Memcached cache


@dataclass
class CacheConfig:
    """Configuration for the multi-level cache system."""

    # L1 Cache settings
    l1_max_memory_mb: float = 4096.0  # 4GB default
    l1_max_entries: int = 10000
    l1_enable: bool = True

    # L2 Cache settings
    l2_max_memory_mb: float = 16384.0  # 16GB default
    l2_max_entries: int = 1000
    l2_enable: bool = True
    l2_compression_enabled: bool = True

    # Distributed cache settings
    distributed_enable: bool = False
    redis_config: RedisConfig | None = None

    # General settings
    default_ttl_seconds: int = 3600  # 1 hour
    enable_cache_warming: bool = True
    enable_coherence: bool = True
    cache_hit_promotion: bool = True  # Promote L2/distributed hits to L1
    cache_statistics: bool = True

    # Performance tuning
    async_operations: bool = True
    background_cleanup: bool = True
    cleanup_interval_seconds: int = 300  # 5 minutes


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    distributed_hits: int = 0
    total_misses: int = 0

    # Timing statistics
    avg_l1_latency_ms: float = 0.0
    avg_l2_latency_ms: float = 0.0
    avg_distributed_latency_ms: float = 0.0

    # Memory statistics
    l1_memory_usage_mb: float = 0.0
    l2_memory_usage_mb: float = 0.0

    # Efficiency metrics
    hit_rate: float = 0.0
    miss_rate: float = 0.0

    def update_hit_rate(self):
        """Update hit and miss rates."""
        total_hits = self.l1_hits + self.l2_hits + self.distributed_hits
        if self.total_requests > 0:
            self.hit_rate = total_hits / self.total_requests
            self.miss_rate = self.total_misses / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'total_requests': self.total_requests,
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'distributed_hits': self.distributed_hits,
            'total_misses': self.total_misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'avg_l1_latency_ms': self.avg_l1_latency_ms,
            'avg_l2_latency_ms': self.avg_l2_latency_ms,
            'avg_distributed_latency_ms': self.avg_distributed_latency_ms,
            'l1_memory_usage_mb': self.l1_memory_usage_mb,
            'l2_memory_usage_mb': self.l2_memory_usage_mb
        }


class CacheManager:
    """Main cache manager coordinating multi-level caching."""

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()

        # Cache layers
        self.l1_cache: L1TensorCache | None = None
        self.l2_cache: L2ComponentCache | None = None
        self.distributed_cache: DistributedCache | None = None

        # Cache coherence manager
        self.coherence_manager: CacheCoherenceManager | None = None

        # Statistics and monitoring
        self.stats = CacheStats()
        self.latency_history: dict[CacheLevel, list[float]] = defaultdict(list)

        # Thread safety
        self._lock = threading.RLock()

        # Background cleanup
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_running = False

        # Initialize cache layers
        self._initialize_caches()

        logger.info("Cache manager initialized with multi-level caching")

    def _initialize_caches(self):
        """Initialize all cache layers."""

        # Initialize L1 cache
        if self.config.l1_enable:
            eviction_policy = LRUPolicy(max_entries=self.config.l1_max_entries)
            self.l1_cache = L1TensorCache(
                max_memory_mb=self.config.l1_max_memory_mb,
                eviction_policy=eviction_policy,
                default_ttl=self.config.default_ttl_seconds
            )
            logger.info(f"L1 cache initialized: {self.config.l1_max_memory_mb}MB")

        # Initialize L2 cache
        if self.config.l2_enable:
            compression_config = CompressionConfig(
                enabled=self.config.l2_compression_enabled
            )
            eviction_policy = LRUPolicy(max_entries=self.config.l2_max_entries)
            self.l2_cache = L2ComponentCache(
                max_memory_mb=self.config.l2_max_memory_mb,
                compression_config=compression_config,
                eviction_policy=eviction_policy
            )
            logger.info(f"L2 cache initialized: {self.config.l2_max_memory_mb}MB")

        # Initialize distributed cache
        if self.config.distributed_enable and self.config.redis_config:
            self.distributed_cache = DistributedCache(self.config.redis_config)
            logger.info("Distributed cache initialized")

        # Initialize coherence manager
        if self.config.enable_coherence:
            from .coherence_manager import CoherenceConfig
            coherence_config = CoherenceConfig()
            self.coherence_manager = CacheCoherenceManager(
                config=coherence_config,
                l1_cache=self.l1_cache,
                l2_cache=self.l2_cache,
                distributed_cache=self.distributed_cache
            )
            logger.info("Cache coherence manager initialized")

        # Start background cleanup
        if self.config.background_cleanup:
            self._start_background_cleanup()

    def get(self, key: str, namespace: str = "default") -> Any | None:
        """Get value from cache, checking all levels in order."""
        self.stats.total_requests += 1
        full_key = f"{namespace}:{key}"

        with self._lock:
            # Try L1 cache first
            if self.l1_cache:
                start_time = time.perf_counter()
                value = self.l1_cache.get(full_key)
                l1_latency = (time.perf_counter() - start_time) * 1000

                self._update_latency_stats(CacheLevel.L1, l1_latency)

                if value is not None:
                    self.stats.l1_hits += 1
                    logger.debug(f"L1 cache hit: {full_key}")
                    return value

            # Try L2 cache
            if self.l2_cache:
                start_time = time.perf_counter()
                value = self.l2_cache.get(full_key)
                l2_latency = (time.perf_counter() - start_time) * 1000

                self._update_latency_stats(CacheLevel.L2, l2_latency)

                if value is not None:
                    self.stats.l2_hits += 1
                    logger.debug(f"L2 cache hit: {full_key}")

                    # Promote to L1 if enabled
                    if self.config.cache_hit_promotion and self.l1_cache:
                        self.l1_cache.put(full_key, value)

                    return value

            # Try distributed cache
            if self.distributed_cache:
                start_time = time.perf_counter()
                try:
                    value = self.distributed_cache.get(full_key)
                    distributed_latency = (time.perf_counter() - start_time) * 1000

                    self._update_latency_stats(CacheLevel.DISTRIBUTED, distributed_latency)

                    if value is not None:
                        self.stats.distributed_hits += 1
                        logger.debug(f"Distributed cache hit: {full_key}")

                        # Promote to higher levels if enabled
                        if self.config.cache_hit_promotion:
                            if self.l2_cache:
                                self.l2_cache.put(full_key, value)
                            if self.l1_cache:
                                self.l1_cache.put(full_key, value)

                        return value

                except Exception as e:
                    logger.error(f"Distributed cache error: {e}")

            # Cache miss
            self.stats.total_misses += 1
            logger.debug(f"Cache miss: {full_key}")
            return None

    def put(self, key: str, value: Any, namespace: str = "default",
            ttl: int | None = None, cache_levels: list[CacheLevel] | None = None):
        """Put value into cache at specified levels."""
        full_key = f"{namespace}:{key}"
        ttl = ttl or self.config.default_ttl_seconds

        # Default to all available levels
        if cache_levels is None:
            cache_levels = []
            if self.l1_cache:
                cache_levels.append(CacheLevel.L1)
            if self.l2_cache:
                cache_levels.append(CacheLevel.L2)
            if self.distributed_cache:
                cache_levels.append(CacheLevel.DISTRIBUTED)

        with self._lock:
            # Store in requested cache levels
            for level in cache_levels:
                try:
                    if level == CacheLevel.L1 and self.l1_cache:
                        self.l1_cache.put(full_key, value, ttl=ttl)
                        logger.debug(f"Stored in L1 cache: {full_key}")

                    elif level == CacheLevel.L2 and self.l2_cache:
                        self.l2_cache.put(full_key, value, ttl=ttl)
                        logger.debug(f"Stored in L2 cache: {full_key}")

                    elif level == CacheLevel.DISTRIBUTED and self.distributed_cache:
                        self.distributed_cache.put(full_key, value, ttl=ttl)
                        logger.debug(f"Stored in distributed cache: {full_key}")

                except Exception as e:
                    logger.error(f"Failed to store in {level.value} cache: {e}")

            # Update cache coherence
            if self.coherence_manager:
                try:
                    self.coherence_manager.on_cache_write(full_key, value)
                except Exception as e:
                    logger.error(f"Cache coherence update failed: {e}")

    def invalidate(self, key: str, namespace: str = "default"):
        """Invalidate key from all cache levels."""
        full_key = f"{namespace}:{key}"

        with self._lock:
            if self.l1_cache:
                self.l1_cache.remove(full_key)

            if self.l2_cache:
                self.l2_cache.remove(full_key)

            if self.distributed_cache:
                try:
                    self.distributed_cache.remove(full_key)
                except Exception as e:
                    logger.error(f"Distributed cache invalidation error: {e}")

            # Update cache coherence
            if self.coherence_manager:
                try:
                    self.coherence_manager.on_cache_invalidate(full_key)
                except Exception as e:
                    logger.error(f"Cache coherence invalidation failed: {e}")

        logger.debug(f"Invalidated from all levels: {full_key}")

    def warm_cache(self, key_value_pairs: list[tuple[str, Any]],
                   namespace: str = "default", priority_levels: list[CacheLevel] | None = None):
        """Warm cache with preloaded data."""
        if not key_value_pairs:
            return

        priority_levels = priority_levels or [CacheLevel.L1, CacheLevel.L2]

        logger.info(f"Warming cache with {len(key_value_pairs)} entries")

        for key, value in key_value_pairs:
            self.put(key, value, namespace=namespace, cache_levels=priority_levels)

        logger.info("Cache warming completed")

    async def warm_cache_async(self, key_value_pairs: list[tuple[str, Any]],
                              namespace: str = "default"):
        """Asynchronously warm cache with preloaded data."""
        if not key_value_pairs:
            return

        logger.info(f"Starting async cache warming with {len(key_value_pairs)} entries")

        # Process in batches to avoid blocking
        batch_size = 100
        for i in range(0, len(key_value_pairs), batch_size):
            batch = key_value_pairs[i:i + batch_size]

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._warm_batch, batch, namespace)

            # Allow other coroutines to run
            await asyncio.sleep(0.01)

        logger.info("Async cache warming completed")

    def _warm_batch(self, batch: list[tuple[str, Any]], namespace: str):
        """Warm a batch of cache entries."""
        for key, value in batch:
            self.put(key, value, namespace=namespace)

    def clear_cache(self, namespace: str | None = None, cache_levels: list[CacheLevel] | None = None):
        """Clear cache entries from specified levels."""
        cache_levels = cache_levels or [CacheLevel.L1, CacheLevel.L2, CacheLevel.DISTRIBUTED]

        with self._lock:
            for level in cache_levels:
                try:
                    if level == CacheLevel.L1 and self.l1_cache:
                        if namespace:
                            self.l1_cache.clear_namespace(namespace)
                        else:
                            self.l1_cache.clear()
                        logger.info(f"Cleared L1 cache{f' for namespace {namespace}' if namespace else ''}")

                    elif level == CacheLevel.L2 and self.l2_cache:
                        if namespace:
                            self.l2_cache.clear_namespace(namespace)
                        else:
                            self.l2_cache.clear()
                        logger.info(f"Cleared L2 cache{f' for namespace {namespace}' if namespace else ''}")

                    elif level == CacheLevel.DISTRIBUTED and self.distributed_cache:
                        if namespace:
                            self.distributed_cache.clear_namespace(namespace)
                        else:
                            self.distributed_cache.clear()
                        logger.info(f"Cleared distributed cache{f' for namespace {namespace}' if namespace else ''}")

                except Exception as e:
                    logger.error(f"Failed to clear {level.value} cache: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            # Update hit rate
            self.stats.update_hit_rate()

            # Get per-level statistics
            level_stats = {}

            if self.l1_cache:
                l1_stats = self.l1_cache.get_stats()
                level_stats['l1'] = l1_stats
                self.stats.l1_memory_usage_mb = l1_stats.get('memory_usage_mb', 0)

            if self.l2_cache:
                l2_stats = self.l2_cache.get_stats()
                level_stats['l2'] = l2_stats
                self.stats.l2_memory_usage_mb = l2_stats.get('memory_usage_mb', 0)

            if self.distributed_cache:
                try:
                    dist_stats = self.distributed_cache.get_stats()
                    level_stats['distributed'] = dist_stats
                except Exception as e:
                    logger.error(f"Failed to get distributed cache stats: {e}")
                    level_stats['distributed'] = {'error': str(e)}

            return {
                'overall': self.stats.to_dict(),
                'levels': level_stats,
                'latency_percentiles': self._calculate_latency_percentiles(),
                'coherence_stats': self.coherence_manager.get_stats() if self.coherence_manager else None
            }

    def _update_latency_stats(self, level: CacheLevel, latency_ms: float):
        """Update latency statistics for a cache level."""
        self.latency_history[level].append(latency_ms)

        # Keep only recent history
        if len(self.latency_history[level]) > 1000:
            self.latency_history[level].pop(0)

        # Update average latency
        avg_latency = sum(self.latency_history[level]) / len(self.latency_history[level])

        if level == CacheLevel.L1:
            self.stats.avg_l1_latency_ms = avg_latency
        elif level == CacheLevel.L2:
            self.stats.avg_l2_latency_ms = avg_latency
        elif level == CacheLevel.DISTRIBUTED:
            self.stats.avg_distributed_latency_ms = avg_latency

    def _calculate_latency_percentiles(self) -> dict[str, dict[str, float]]:
        """Calculate latency percentiles for each cache level."""
        percentiles = {}

        for level, latencies in self.latency_history.items():
            if latencies:
                import numpy as np
                percentiles[level.value] = {
                    'p50': float(np.percentile(latencies, 50)),
                    'p90': float(np.percentile(latencies, 90)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                    'min': float(min(latencies)),
                    'max': float(max(latencies))
                }

        return percentiles

    def _start_background_cleanup(self):
        """Start background cleanup thread."""
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._background_cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started background cache cleanup")

    def _background_cleanup_loop(self):
        """Background cleanup loop."""
        while self._cleanup_running:
            try:
                # Cleanup expired entries
                if self.l1_cache:
                    self.l1_cache.cleanup_expired()

                if self.l2_cache:
                    self.l2_cache.cleanup_expired()

                # Run cache coherence maintenance
                if self.coherence_manager:
                    self.coherence_manager.perform_maintenance()

                time.sleep(self.config.cleanup_interval_seconds)

            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                time.sleep(self.config.cleanup_interval_seconds)

    @contextmanager
    def cache_context(self, namespace: str = "temp"):
        """Context manager for temporary caching."""
        try:
            yield self
        finally:
            # Clean up temporary cache entries
            self.clear_cache(namespace=namespace)

    def optimize_cache_configuration(self) -> CacheConfig:
        """Automatically optimize cache configuration based on usage patterns."""
        logger.info("Optimizing cache configuration")

        stats = self.get_cache_stats()
        overall_stats = stats['overall']

        # Create optimized configuration
        optimized_config = CacheConfig()

        # Adjust L1 cache size based on hit rate
        if overall_stats['hit_rate'] < 0.5:
            # Low hit rate - increase L1 cache size
            optimized_config.l1_max_memory_mb = self.config.l1_max_memory_mb * 1.5
            optimized_config.l1_max_entries = self.config.l1_max_entries * 2
        elif overall_stats['hit_rate'] > 0.9:
            # Very high hit rate - can reduce L1 cache size
            optimized_config.l1_max_memory_mb = self.config.l1_max_memory_mb * 0.8

        # Adjust TTL based on access patterns
        if overall_stats['miss_rate'] > 0.5:
            # High miss rate - increase TTL
            optimized_config.default_ttl_seconds = self.config.default_ttl_seconds * 2

        # Enable distributed cache if L2 hit rate is low
        if stats['levels'].get('l2', {}).get('hit_rate', 0) < 0.3:
            optimized_config.distributed_enable = True

        logger.info("Cache configuration optimization completed")
        return optimized_config

    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        logger.info("Shutting down cache manager")

        # Stop background cleanup
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        # Shutdown cache layers
        if self.distributed_cache:
            try:
                self.distributed_cache.close()
            except Exception as e:
                logger.error(f"Error closing distributed cache: {e}")

        if self.coherence_manager:
            try:
                self.coherence_manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down coherence manager: {e}")

        logger.info("Cache manager shutdown completed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass
