"""
Cache eviction policies for intelligent cache management.

This module provides various eviction strategies including LRU, LFU, TTL-based,
and adaptive policies.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def on_access(self, key: str, value: Any):
        """Called when a cache entry is accessed."""
        pass

    @abstractmethod
    def on_insert(self, key: str, value: Any):
        """Called when a new entry is inserted."""
        pass

    @abstractmethod
    def on_evict(self, key: str, value: Any):
        """Called when an entry is evicted."""
        pass

    @abstractmethod
    def get_eviction_candidate(self, keys: list[str]) -> str | None:
        """Get the next key to evict."""
        pass

    @abstractmethod
    def clear(self):
        """Clear policy state."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get policy statistics."""
        pass


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._access_order: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'accesses': 0,
            'evictions': 0,
            'insertions': 0
        }

    def on_access(self, key: str, value: Any):
        """Update access order for LRU."""
        with self._lock:
            self._access_order[key] = time.time()
            self._access_order.move_to_end(key)
            self._stats['accesses'] += 1

    def on_insert(self, key: str, value: Any):
        """Track new insertion."""
        with self._lock:
            self._access_order[key] = time.time()
            self._stats['insertions'] += 1

    def on_evict(self, key: str, value: Any):
        """Clean up evicted entry."""
        with self._lock:
            self._access_order.pop(key, None)
            self._stats['evictions'] += 1

    def get_eviction_candidate(self, keys: list[str]) -> str | None:
        """Get least recently used key."""
        with self._lock:
            if not self._access_order:
                return keys[0] if keys else None

            # Return the first (least recently used) key
            return next(iter(self._access_order))

    def clear(self):
        """Clear LRU tracking."""
        with self._lock:
            self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get LRU policy statistics."""
        return {
            'policy': 'LRU',
            **self._stats,
            'tracked_keys': len(self._access_order)
        }


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._access_counts: dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'accesses': 0,
            'evictions': 0,
            'insertions': 0
        }

    def on_access(self, key: str, value: Any):
        """Increment access count."""
        with self._lock:
            self._access_counts[key] += 1
            self._stats['accesses'] += 1

    def on_insert(self, key: str, value: Any):
        """Initialize access count for new entry."""
        with self._lock:
            self._access_counts[key] = 1
            self._stats['insertions'] += 1

    def on_evict(self, key: str, value: Any):
        """Clean up evicted entry."""
        with self._lock:
            self._access_counts.pop(key, None)
            self._stats['evictions'] += 1

    def get_eviction_candidate(self, keys: list[str]) -> str | None:
        """Get least frequently used key."""
        with self._lock:
            if not self._access_counts:
                return keys[0] if keys else None

            # Find key with minimum access count
            min_count = float('inf')
            candidate = None

            for key in keys:
                count = self._access_counts.get(key, 0)
                if count < min_count:
                    min_count = count
                    candidate = key

            return candidate

    def clear(self):
        """Clear LFU tracking."""
        with self._lock:
            self._access_counts.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get LFU policy statistics."""
        with self._lock:
            avg_access_count = (
                sum(self._access_counts.values()) / len(self._access_counts)
                if self._access_counts else 0
            )

            return {
                'policy': 'LFU',
                **self._stats,
                'tracked_keys': len(self._access_counts),
                'avg_access_count': avg_access_count,
                'max_access_count': max(self._access_counts.values()) if self._access_counts else 0
            }


class TTLPolicy(EvictionPolicy):
    """Time-To-Live based eviction policy."""

    def __init__(self, default_ttl: int):
        self.default_ttl = default_ttl
        self._expiration_times: dict[str, float] = {}
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'accesses': 0,
            'evictions': 0,
            'insertions': 0,
            'expired_evictions': 0
        }

    def on_access(self, key: str, value: Any):
        """Update access time but don't change expiration."""
        with self._lock:
            self._stats['accesses'] += 1

    def on_insert(self, key: str, value: Any):
        """Set expiration time for new entry."""
        with self._lock:
            self._expiration_times[key] = time.time() + self.default_ttl
            self._stats['insertions'] += 1

    def on_evict(self, key: str, value: Any):
        """Clean up evicted entry."""
        with self._lock:
            self._expiration_times.pop(key, None)
            self._stats['evictions'] += 1

    def get_eviction_candidate(self, keys: list[str]) -> str | None:
        """Get earliest expiring key or expired key."""
        with self._lock:
            current_time = time.time()
            earliest_expiry = float('inf')
            candidate = None

            for key in keys:
                expiry_time = self._expiration_times.get(key, current_time)

                # If already expired, prioritize for eviction
                if expiry_time <= current_time:
                    self._stats['expired_evictions'] += 1
                    return key

                # Track earliest expiring key
                if expiry_time < earliest_expiry:
                    earliest_expiry = expiry_time
                    candidate = key

            return candidate

    def get_expired_keys(self, keys: list[str]) -> list[str]:
        """Get all expired keys."""
        with self._lock:
            current_time = time.time()
            expired = []

            for key in keys:
                expiry_time = self._expiration_times.get(key, current_time)
                if expiry_time <= current_time:
                    expired.append(key)

            return expired

    def clear(self):
        """Clear TTL tracking."""
        with self._lock:
            self._expiration_times.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get TTL policy statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for expiry in self._expiration_times.values()
                if expiry <= current_time
            )

            return {
                'policy': 'TTL',
                **self._stats,
                'tracked_keys': len(self._expiration_times),
                'expired_keys': expired_count,
                'default_ttl': self.default_ttl
            }


class AdaptivePolicy(EvictionPolicy):
    """Adaptive eviction policy that combines multiple strategies."""

    def __init__(self, max_entries: int, default_ttl: int = 3600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl

        # Component policies
        self.lru_policy = LRUPolicy(max_entries)
        self.lfu_policy = LFUPolicy(max_entries)
        self.ttl_policy = TTLPolicy(default_ttl)

        # Adaptation parameters
        self._current_strategy = 'lru'  # Start with LRU
        self._strategy_performance: dict[str, list[float]] = {
            'lru': [],
            'lfu': [],
            'ttl': []
        }

        # Monitoring
        self._adaptation_interval = 100  # Adapt every 100 operations
        self._operation_count = 0
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'adaptations': 0,
            'strategy_switches': defaultdict(int),
            'current_strategy': self._current_strategy
        }

    def on_access(self, key: str, value: Any):
        """Forward to all component policies."""
        with self._lock:
            self.lru_policy.on_access(key, value)
            self.lfu_policy.on_access(key, value)
            self.ttl_policy.on_access(key, value)

            self._operation_count += 1
            self._maybe_adapt()

    def on_insert(self, key: str, value: Any):
        """Forward to all component policies."""
        with self._lock:
            self.lru_policy.on_insert(key, value)
            self.lfu_policy.on_insert(key, value)
            self.ttl_policy.on_insert(key, value)

    def on_evict(self, key: str, value: Any):
        """Forward to all component policies."""
        with self._lock:
            self.lru_policy.on_evict(key, value)
            self.lfu_policy.on_evict(key, value)
            self.ttl_policy.on_evict(key, value)

    def get_eviction_candidate(self, keys: list[str]) -> str | None:
        """Get eviction candidate based on current strategy."""
        with self._lock:
            # Check for expired keys first
            expired_keys = self.ttl_policy.get_expired_keys(keys)
            if expired_keys:
                return expired_keys[0]

            # Use current strategy
            if self._current_strategy == 'lru':
                return self.lru_policy.get_eviction_candidate(keys)
            elif self._current_strategy == 'lfu':
                return self.lfu_policy.get_eviction_candidate(keys)
            elif self._current_strategy == 'ttl':
                return self.ttl_policy.get_eviction_candidate(keys)

            # Fallback to LRU
            return self.lru_policy.get_eviction_candidate(keys)

    def _maybe_adapt(self):
        """Adapt strategy if needed."""
        if self._operation_count % self._adaptation_interval == 0:
            self._adapt_strategy()

    def _adapt_strategy(self):
        """Adapt the eviction strategy based on performance."""
        # Simple adaptation: switch strategies periodically to measure performance
        strategies = ['lru', 'lfu', 'ttl']

        # Get performance scores (simplified)
        current_performance = self._calculate_performance_score()
        self._strategy_performance[self._current_strategy].append(current_performance)

        # Keep limited history
        for strategy in strategies:
            if len(self._strategy_performance[strategy]) > 10:
                self._strategy_performance[strategy].pop(0)

        # Find best performing strategy
        best_strategy = self._current_strategy
        best_performance = 0.0

        for strategy in strategies:
            if self._strategy_performance[strategy]:
                avg_performance = sum(self._strategy_performance[strategy]) / len(self._strategy_performance[strategy])
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_strategy = strategy

        # Switch strategy if needed
        if best_strategy != self._current_strategy:
            old_strategy = self._current_strategy
            self._current_strategy = best_strategy
            self._stats['strategy_switches'][f"{old_strategy}_to_{best_strategy}"] += 1
            self._stats['adaptations'] += 1
            self._stats['current_strategy'] = self._current_strategy

            logger.info(f"Adaptive cache policy switched from {old_strategy} to {best_strategy}")

    def _calculate_performance_score(self) -> float:
        """Calculate current performance score."""
        # Simple performance metric based on component policy stats
        lru_stats = self.lru_policy.get_stats()
        lfu_stats = self.lfu_policy.get_stats()

        # Calculate hit rates (simplified)
        lru_accesses = lru_stats.get('accesses', 1)
        lru_evictions = lru_stats.get('evictions', 0)
        lru_score = (lru_accesses - lru_evictions) / lru_accesses if lru_accesses > 0 else 0

        lfu_accesses = lfu_stats.get('accesses', 1)
        lfu_evictions = lfu_stats.get('evictions', 0)
        lfu_score = (lfu_accesses - lfu_evictions) / lfu_accesses if lfu_accesses > 0 else 0

        # Return score based on current strategy
        if self._current_strategy == 'lru':
            return lru_score
        elif self._current_strategy == 'lfu':
            return lfu_score
        else:
            return (lru_score + lfu_score) / 2

    def clear(self):
        """Clear all component policies."""
        with self._lock:
            self.lru_policy.clear()
            self.lfu_policy.clear()
            self.ttl_policy.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive adaptive policy statistics."""
        return {
            'policy': 'Adaptive',
            'current_strategy': self._current_strategy,
            'operation_count': self._operation_count,
            **self._stats,
            'component_stats': {
                'lru': self.lru_policy.get_stats(),
                'lfu': self.lfu_policy.get_stats(),
                'ttl': self.ttl_policy.get_stats()
            },
            'strategy_performance': {
                strategy: {
                    'avg_score': sum(scores) / len(scores) if scores else 0,
                    'sample_count': len(scores)
                }
                for strategy, scores in self._strategy_performance.items()
            }
        }


class SizeBasedPolicy(EvictionPolicy):
    """Eviction policy based on entry size."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._entry_sizes: dict[str, int] = {}
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'accesses': 0,
            'evictions': 0,
            'insertions': 0,
            'total_size_evicted': 0
        }

    def on_access(self, key: str, value: Any):
        """Track access but don't change size."""
        with self._lock:
            self._stats['accesses'] += 1

    def on_insert(self, key: str, value: Any):
        """Track size of new entry."""
        with self._lock:
            import sys
            if hasattr(value, 'numel') and hasattr(value, 'element_size'):
                # Tensor size
                size = value.numel() * value.element_size()
            else:
                # General object size
                size = sys.getsizeof(value)

            self._entry_sizes[key] = size
            self._stats['insertions'] += 1

    def on_evict(self, key: str, value: Any):
        """Clean up evicted entry."""
        with self._lock:
            size = self._entry_sizes.pop(key, 0)
            self._stats['evictions'] += 1
            self._stats['total_size_evicted'] += size

    def get_eviction_candidate(self, keys: list[str]) -> str | None:
        """Get largest entry for eviction."""
        with self._lock:
            if not self._entry_sizes:
                return keys[0] if keys else None

            # Find largest entry
            max_size = 0
            candidate = None

            for key in keys:
                size = self._entry_sizes.get(key, 0)
                if size > max_size:
                    max_size = size
                    candidate = key

            return candidate

    def clear(self):
        """Clear size tracking."""
        with self._lock:
            self._entry_sizes.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get size-based policy statistics."""
        with self._lock:
            total_size = sum(self._entry_sizes.values())
            avg_size = total_size / len(self._entry_sizes) if self._entry_sizes else 0

            return {
                'policy': 'SizeBased',
                **self._stats,
                'tracked_keys': len(self._entry_sizes),
                'total_size': total_size,
                'avg_entry_size': avg_size,
                'max_entry_size': max(self._entry_sizes.values()) if self._entry_sizes else 0
            }
