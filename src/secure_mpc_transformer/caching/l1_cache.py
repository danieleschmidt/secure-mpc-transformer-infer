"""
L1 in-memory tensor cache for high-speed access.

This module provides fast in-memory caching for frequently accessed
tensors and small objects.
"""

import torch
import time
import threading
import sys
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
import logging
import gc
import weakref
from collections import OrderedDict
import pickle

from .eviction_policies import EvictionPolicy, LRUPolicy

logger = logging.getLogger(__name__)


@dataclass
class TensorCacheEntry:
    """Entry in the tensor cache."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of the cached value."""
        try:
            if isinstance(self.value, torch.Tensor):
                return self.value.numel() * self.value.element_size()
            elif hasattr(self.value, '__sizeof__'):
                return sys.getsizeof(self.value)
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(self.value))
        except:
            return 1024  # Fallback size estimate
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def touch(self):
        """Update access timestamp and increment access count."""
        self.accessed_at = time.time()
        self.access_count += 1


class L1TensorCache:
    """High-performance L1 in-memory tensor cache."""
    
    def __init__(self, max_memory_mb: float = 1024.0, 
                 max_entries: int = 10000,
                 eviction_policy: Optional[EvictionPolicy] = None,
                 default_ttl: Optional[int] = None,
                 enable_compression: bool = False):
        
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        
        # Storage
        self._cache: OrderedDict[str, TensorCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Eviction policy
        self.eviction_policy = eviction_policy or LRUPolicy(max_entries=max_entries)
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_bytes': 0,
            'total_entries': 0
        }
        
        # Memory tracking
        self._current_memory_usage = 0
        
        logger.info(f"L1 cache initialized: {max_memory_mb}MB, {max_entries} entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                return None
            
            # Update access information
            entry.touch()
            
            # Move to end for LRU ordering
            self._cache.move_to_end(key)
            
            # Update eviction policy
            self.eviction_policy.on_access(key, entry.value)
            
            self._stats['hits'] += 1
            logger.debug(f"L1 cache hit: {key}")
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate value size
            if isinstance(value, torch.Tensor):
                value_size = value.numel() * value.element_size()
            else:
                value_size = sys.getsizeof(value)
            
            # Check if value is too large
            if value_size > self.max_memory_bytes:
                logger.warning(f"Value too large for L1 cache: {value_size} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure space is available
            self._ensure_space(value_size)
            
            # Create cache entry
            entry = TensorCacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=value_size
            )
            
            # Store in cache
            self._cache[key] = entry
            self._current_memory_usage += value_size
            
            # Update eviction policy
            self.eviction_policy.on_insert(key, value)
            
            logger.debug(f"L1 cache put: {key}, size: {value_size} bytes")
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def _remove_entry(self, key: str):
        """Internal method to remove cache entry."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory_usage -= entry.size_bytes
            del self._cache[key]
            
            # Update eviction policy
            self.eviction_policy.on_evict(key, entry.value)
            
            logger.debug(f"Removed L1 cache entry: {key}")
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient space is available in cache."""
        # Check entry count limit
        while len(self._cache) >= self.max_entries:
            self._evict_one_entry()
        
        # Check memory limit
        while (self._current_memory_usage + required_bytes) > self.max_memory_bytes:
            if not self._evict_one_entry():
                break  # No more entries to evict
    
    def _evict_one_entry(self) -> bool:
        """Evict one entry from cache."""
        if not self._cache:
            return False
        
        # Get eviction candidate from policy
        evict_key = self.eviction_policy.get_eviction_candidate(list(self._cache.keys()))
        
        if evict_key and evict_key in self._cache:
            self._remove_entry(evict_key)
            self._stats['evictions'] += 1
            return True
        
        # Fallback to LRU eviction
        if self._cache:
            evict_key = next(iter(self._cache))
            self._remove_entry(evict_key)
            self._stats['evictions'] += 1
            return True
        
        return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            self.remove(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired L1 cache entries")
        
        return len(expired_keys)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory_usage = 0
            self.eviction_policy.clear()
        
        logger.info("L1 cache cleared")
    
    def clear_namespace(self, namespace: str):
        """Clear entries for a specific namespace."""
        namespace_prefix = f"{namespace}:"
        keys_to_remove = []
        
        with self._lock:
            for key in self._cache:
                if key.startswith(namespace_prefix):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.remove(key)
        
        logger.info(f"Cleared {len(keys_to_remove)} entries from namespace: {namespace}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'type': 'L1TensorCache',
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': hit_rate,
                'total_entries': len(self._cache),
                'max_entries': self.max_entries,
                'memory_usage_bytes': self._current_memory_usage,
                'memory_usage_mb': self._current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self._current_memory_usage / self.max_memory_bytes,
                'average_entry_size': self._current_memory_usage / len(self._cache) if self._cache else 0,
                'eviction_policy': type(self.eviction_policy).__name__
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        with self._lock:
            entry_sizes = {}
            for key, entry in self._cache.items():
                entry_sizes[key] = entry.size_bytes
            
            # Sort by size
            sorted_entries = sorted(entry_sizes.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'total_bytes': self._current_memory_usage,
                'total_mb': self._current_memory_usage / (1024 * 1024),
                'entry_count': len(self._cache),
                'largest_entries': sorted_entries[:10],  # Top 10 largest entries
                'utilization': self._current_memory_usage / self.max_memory_bytes
            }
    
    def get_hot_keys(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequently accessed keys."""
        with self._lock:
            key_access_counts = [(key, entry.access_count) for key, entry in self._cache.items()]
            key_access_counts.sort(key=lambda x: x[1], reverse=True)
            
            return key_access_counts[:top_n]
    
    def optimize_memory(self):
        """Optimize memory usage by running garbage collection and cleanup."""
        logger.info("Optimizing L1 cache memory usage")
        
        # Clean up expired entries
        expired_count = self.cleanup_expired()
        
        # Force garbage collection
        gc.collect()
        
        # If still over memory limit, evict some entries
        with self._lock:
            while self._current_memory_usage > (self.max_memory_bytes * 0.9):
                if not self._evict_one_entry():
                    break
        
        logger.info(f"L1 cache optimization complete. Removed {expired_count} expired entries.")
    
    def prefetch(self, keys: List[str], values: List[Any]):
        """Prefetch multiple key-value pairs into cache."""
        if len(keys) != len(values):
            raise ValueError("Keys and values lists must have same length")
        
        successful_prefetch = 0
        
        for key, value in zip(keys, values):
            if self.put(key, value):
                successful_prefetch += 1
        
        logger.info(f"Prefetched {successful_prefetch}/{len(keys)} entries into L1 cache")
    
    def get_cache_efficiency(self) -> Dict[str, float]:
        """Calculate cache efficiency metrics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            
            if total_requests == 0:
                return {
                    'hit_rate': 0.0,
                    'miss_rate': 0.0,
                    'eviction_rate': 0.0,
                    'memory_efficiency': 0.0
                }
            
            hit_rate = self._stats['hits'] / total_requests
            miss_rate = self._stats['misses'] / total_requests
            eviction_rate = self._stats['evictions'] / total_requests
            memory_efficiency = len(self._cache) / self.max_entries
            
            return {
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'eviction_rate': eviction_rate,
                'memory_efficiency': memory_efficiency,
                'total_requests': total_requests
            }
    
    def __len__(self) -> int:
        """Get number of entries in cache."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache and not self._cache[key].is_expired()