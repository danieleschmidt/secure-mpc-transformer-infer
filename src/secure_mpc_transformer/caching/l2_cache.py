"""
L2 compressed component cache for larger objects with compression.

This module provides compressed storage for larger model components
and intermediate computation results.
"""

import torch
import time
import threading
import pickle
import gzip
import lz4.frame
import zstd
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import os
import tempfile
import hashlib
from pathlib import Path

from .eviction_policies import EvictionPolicy, LRUPolicy

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class CompressionConfig:
    """Configuration for compression settings."""
    enabled: bool = True
    algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4
    compression_level: int = 1  # Fast compression by default
    min_size_for_compression: int = 1024  # Only compress objects > 1KB
    enable_dictionary_compression: bool = False  # For ZSTD


@dataclass
class L2CacheEntry:
    """Entry in the L2 cache."""
    key: str
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_algorithm: CompressionAlgorithm
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def touch(self):
        """Update access timestamp and increment access count."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 1.0


class CompressionEngine:
    """Engine for handling different compression algorithms."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self._compression_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'bytes_compressed': 0,
            'bytes_decompressed': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        }
    
    def compress(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm, Dict[str, Any]]:
        """Compress data using configured algorithm."""
        if not self.config.enabled or len(data) < self.config.min_size_for_compression:
            return data, CompressionAlgorithm.NONE, {}
        
        start_time = time.perf_counter()
        compressed_data = data
        algorithm = CompressionAlgorithm.NONE
        metadata = {}
        
        try:
            if self.config.algorithm == CompressionAlgorithm.GZIP:
                compressed_data = gzip.compress(data, compresslevel=self.config.compression_level)
                algorithm = CompressionAlgorithm.GZIP
            
            elif self.config.algorithm == CompressionAlgorithm.LZ4:
                compressed_data = lz4.frame.compress(
                    data, 
                    compression_level=self.config.compression_level,
                    auto_flush=True
                )
                algorithm = CompressionAlgorithm.LZ4
            
            elif self.config.algorithm == CompressionAlgorithm.ZSTD:
                compressed_data = zstd.compress(
                    data, 
                    level=self.config.compression_level
                )
                algorithm = CompressionAlgorithm.ZSTD
            
            compression_time = time.perf_counter() - start_time
            
            # Update statistics
            self._compression_stats['total_compressions'] += 1
            self._compression_stats['bytes_compressed'] += len(data)
            self._compression_stats['compression_time'] += compression_time
            
            metadata = {
                'compression_time_ms': compression_time * 1000,
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': len(data) / len(compressed_data) if len(compressed_data) > 0 else 1.0
            }
            
            logger.debug(f"Compressed {len(data)} bytes to {len(compressed_data)} bytes "
                        f"using {algorithm.value} (ratio: {metadata['compression_ratio']:.2f})")
        
        except Exception as e:
            logger.error(f"Compression failed with {self.config.algorithm.value}: {e}")
            compressed_data = data
            algorithm = CompressionAlgorithm.NONE
        
        return compressed_data, algorithm, metadata
    
    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data using specified algorithm."""
        if algorithm == CompressionAlgorithm.NONE:
            return data
        
        start_time = time.perf_counter()
        decompressed_data = data
        
        try:
            if algorithm == CompressionAlgorithm.GZIP:
                decompressed_data = gzip.decompress(data)
            
            elif algorithm == CompressionAlgorithm.LZ4:
                decompressed_data = lz4.frame.decompress(data)
            
            elif algorithm == CompressionAlgorithm.ZSTD:
                decompressed_data = zstd.decompress(data)
            
            decompression_time = time.perf_counter() - start_time
            
            # Update statistics
            self._compression_stats['total_decompressions'] += 1
            self._compression_stats['bytes_decompressed'] += len(decompressed_data)
            self._compression_stats['decompression_time'] += decompression_time
            
            logger.debug(f"Decompressed {len(data)} bytes to {len(decompressed_data)} bytes "
                        f"using {algorithm.value}")
        
        except Exception as e:
            logger.error(f"Decompression failed with {algorithm.value}: {e}")
            decompressed_data = data
        
        return decompressed_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        stats = self._compression_stats.copy()
        
        if stats['total_compressions'] > 0:
            stats['avg_compression_time_ms'] = (stats['compression_time'] / stats['total_compressions']) * 1000
            stats['avg_compression_ratio'] = (stats['bytes_compressed'] / 
                                            sum(len(self.compress(b'x' * size)[0]) 
                                                for size in [100, 1000, 10000]) if stats['bytes_compressed'] > 0 else 1.0)
        else:
            stats['avg_compression_time_ms'] = 0.0
            stats['avg_compression_ratio'] = 1.0
        
        if stats['total_decompressions'] > 0:
            stats['avg_decompression_time_ms'] = (stats['decompression_time'] / stats['total_decompressions']) * 1000
        else:
            stats['avg_decompression_time_ms'] = 0.0
        
        return stats


class L2ComponentCache:
    """L2 cache with compression support for larger components."""
    
    def __init__(self, max_memory_mb: float = 4096.0,
                 max_entries: int = 1000,
                 compression_config: Optional[CompressionConfig] = None,
                 eviction_policy: Optional[EvictionPolicy] = None,
                 use_disk_spillover: bool = True,
                 spillover_directory: Optional[str] = None):
        
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.compression_config = compression_config or CompressionConfig()
        self.use_disk_spillover = use_disk_spillover
        
        # Storage
        self._memory_cache: Dict[str, L2CacheEntry] = {}
        self._disk_cache: Dict[str, str] = {}  # key -> file_path mapping
        self._lock = threading.RLock()
        
        # Compression engine
        self.compression_engine = CompressionEngine(self.compression_config)
        
        # Eviction policy
        self.eviction_policy = eviction_policy or LRUPolicy(max_entries=max_entries)
        
        # Disk spillover setup
        if use_disk_spillover:
            if spillover_directory:
                self.spillover_dir = Path(spillover_directory)
            else:
                self.spillover_dir = Path(tempfile.gettempdir()) / "l2_cache_spillover"
            
            self.spillover_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"L2 cache spillover directory: {self.spillover_dir}")
        
        # Statistics
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'memory_evictions': 0,
            'disk_spillovers': 0,
            'total_entries': 0,
            'memory_usage_bytes': 0
        }
        
        # Current memory usage
        self._current_memory_usage = 0
        
        logger.info(f"L2 cache initialized: {max_memory_mb}MB memory, "
                   f"compression: {compression_config.algorithm.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache."""
        with self._lock:
            # Try memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_memory_entry(key)
                    self._stats['misses'] += 1
                    return None
                
                # Decompress and deserialize
                try:
                    decompressed_data = self.compression_engine.decompress(
                        entry.compressed_data, entry.compression_algorithm
                    )
                    value = pickle.loads(decompressed_data)
                    
                    entry.touch()
                    self.eviction_policy.on_access(key, value)
                    
                    self._stats['memory_hits'] += 1
                    logger.debug(f"L2 memory cache hit: {key}")
                    
                    return value
                
                except Exception as e:
                    logger.error(f"L2 cache decompression/deserialization failed: {e}")
                    self._remove_memory_entry(key)
                    self._stats['misses'] += 1
                    return None
            
            # Try disk cache if enabled
            if self.use_disk_spillover and key in self._disk_cache:
                file_path = self._disk_cache[key]
                
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            # Read entry metadata and data
                            entry_data = pickle.load(f)
                        
                        entry = L2CacheEntry(**entry_data)
                        
                        # Check expiration
                        if entry.is_expired():
                            self._remove_disk_entry(key)
                            self._stats['misses'] += 1
                            return None
                        
                        # Decompress and deserialize
                        decompressed_data = self.compression_engine.decompress(
                            entry.compressed_data, entry.compression_algorithm
                        )
                        value = pickle.loads(decompressed_data)
                        
                        # Promote back to memory cache if there's space
                        if self._current_memory_usage + entry.compressed_size <= self.max_memory_bytes:
                            self._memory_cache[key] = entry
                            self._current_memory_usage += entry.compressed_size
                            self._remove_disk_entry(key)
                        
                        entry.touch()
                        self.eviction_policy.on_access(key, value)
                        
                        self._stats['disk_hits'] += 1
                        logger.debug(f"L2 disk cache hit: {key}")
                        
                        return value
                    else:
                        # File doesn't exist, clean up reference
                        self._remove_disk_entry(key)
                
                except Exception as e:
                    logger.error(f"L2 disk cache read failed: {e}")
                    self._remove_disk_entry(key)
            
            self._stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in L2 cache with compression."""
        try:
            # Serialize value
            serialized_data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress data
            compressed_data, algorithm, compression_metadata = self.compression_engine.compress(serialized_data)
            
            # Create cache entry
            entry = L2CacheEntry(
                key=key,
                compressed_data=compressed_data,
                original_size=len(serialized_data),
                compressed_size=len(compressed_data),
                compression_algorithm=algorithm,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl,
                metadata=compression_metadata
            )
            
            with self._lock:
                # Remove existing entry if present
                if key in self._memory_cache:
                    self._remove_memory_entry(key)
                elif key in self._disk_cache:
                    self._remove_disk_entry(key)
                
                # Try to store in memory
                if self._try_store_in_memory(key, entry):
                    self._stats['total_entries'] += 1
                    logger.debug(f"L2 cache stored in memory: {key}")
                    return True
                
                # Fallback to disk if enabled
                elif self.use_disk_spillover and self._store_on_disk(key, entry):
                    self._stats['total_entries'] += 1
                    self._stats['disk_spillovers'] += 1
                    logger.debug(f"L2 cache stored on disk: {key}")
                    return True
                
                else:
                    logger.warning(f"L2 cache storage failed: {key}")
                    return False
        
        except Exception as e:
            logger.error(f"L2 cache put failed for {key}: {e}")
            return False
    
    def _try_store_in_memory(self, key: str, entry: L2CacheEntry) -> bool:
        """Try to store entry in memory cache."""
        # Check if there's enough space
        required_space = entry.compressed_size
        
        # Make space if needed
        self._ensure_memory_space(required_space)
        
        # Check if we can fit after eviction
        if (self._current_memory_usage + required_space) <= self.max_memory_bytes:
            self._memory_cache[key] = entry
            self._current_memory_usage += required_space
            self.eviction_policy.on_insert(key, entry)
            return True
        
        return False
    
    def _ensure_memory_space(self, required_bytes: int):
        """Ensure sufficient memory space by evicting entries."""
        # Check entry count limit
        while len(self._memory_cache) >= self.max_entries:
            if not self._evict_one_memory_entry():
                break
        
        # Check memory limit
        while (self._current_memory_usage + required_bytes) > self.max_memory_bytes:
            if not self._evict_one_memory_entry():
                break
    
    def _evict_one_memory_entry(self) -> bool:
        """Evict one entry from memory cache."""
        if not self._memory_cache:
            return False
        
        # Get eviction candidate
        evict_key = self.eviction_policy.get_eviction_candidate(list(self._memory_cache.keys()))
        
        if evict_key and evict_key in self._memory_cache:
            entry = self._memory_cache[evict_key]
            
            # Try to spill to disk if enabled
            if self.use_disk_spillover:
                if self._store_on_disk(evict_key, entry):
                    logger.debug(f"Evicted L2 entry to disk: {evict_key}")
                else:
                    logger.debug(f"Failed to spill L2 entry to disk: {evict_key}")
            
            self._remove_memory_entry(evict_key)
            self._stats['memory_evictions'] += 1
            return True
        
        return False
    
    def _store_on_disk(self, key: str, entry: L2CacheEntry) -> bool:
        """Store entry on disk."""
        try:
            # Generate filename
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            file_path = self.spillover_dir / f"{key_hash}.l2cache"
            
            # Store entry data
            entry_data = {
                'key': entry.key,
                'compressed_data': entry.compressed_data,
                'original_size': entry.original_size,
                'compressed_size': entry.compressed_size,
                'compression_algorithm': entry.compression_algorithm,
                'created_at': entry.created_at,
                'accessed_at': entry.accessed_at,
                'access_count': entry.access_count,
                'ttl': entry.ttl,
                'metadata': entry.metadata
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(entry_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._disk_cache[key] = str(file_path)
            return True
        
        except Exception as e:
            logger.error(f"Failed to store L2 entry on disk: {e}")
            return False
    
    def _remove_memory_entry(self, key: str):
        """Remove entry from memory cache."""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            self._current_memory_usage -= entry.compressed_size
            del self._memory_cache[key]
            self.eviction_policy.on_evict(key, entry)
    
    def _remove_disk_entry(self, key: str):
        """Remove entry from disk cache."""
        if key in self._disk_cache:
            file_path = self._disk_cache[key]
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to remove disk cache file: {e}")
            
            del self._disk_cache[key]
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            removed = False
            
            if key in self._memory_cache:
                self._remove_memory_entry(key)
                removed = True
            
            if key in self._disk_cache:
                self._remove_disk_entry(key)
                removed = True
            
            if removed:
                self._stats['total_entries'] -= 1
            
            return removed
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()
            self._current_memory_usage = 0
            
            # Clear disk cache
            for file_path in self._disk_cache.values():
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to remove disk cache file: {e}")
            
            self._disk_cache.clear()
            
            # Clear eviction policy
            self.eviction_policy.clear()
            
            self._stats['total_entries'] = 0
        
        logger.info("L2 cache cleared")
    
    def clear_namespace(self, namespace: str):
        """Clear entries for a specific namespace."""
        namespace_prefix = f"{namespace}:"
        keys_to_remove = []
        
        with self._lock:
            # Find keys to remove
            for key in self._memory_cache:
                if key.startswith(namespace_prefix):
                    keys_to_remove.append(key)
            
            for key in self._disk_cache:
                if key.startswith(namespace_prefix):
                    keys_to_remove.append(key)
        
        # Remove found keys
        for key in keys_to_remove:
            self.remove(key)
        
        logger.info(f"Cleared {len(keys_to_remove)} L2 cache entries from namespace: {namespace}")
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        expired_keys = []
        
        with self._lock:
            # Check memory cache
            for key, entry in self._memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # Check disk cache (sample check to avoid full scan)
            disk_keys_sample = list(self._disk_cache.keys())[:100]  # Check first 100
            for key in disk_keys_sample:
                file_path = self._disk_cache[key]
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            entry_data = pickle.load(f)
                        entry = L2CacheEntry(**entry_data)
                        if entry.is_expired():
                            expired_keys.append(key)
                except:
                    # If we can't read it, consider it for removal
                    expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            self.remove(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired L2 cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L2 cache statistics."""
        with self._lock:
            total_requests = self._stats['memory_hits'] + self._stats['disk_hits'] + self._stats['misses']
            hit_rate = (self._stats['memory_hits'] + self._stats['disk_hits']) / total_requests if total_requests > 0 else 0.0
            
            # Calculate compression statistics
            total_original_size = sum(entry.original_size for entry in self._memory_cache.values())
            total_compressed_size = sum(entry.compressed_size for entry in self._memory_cache.values())
            avg_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
            
            stats = {
                'type': 'L2ComponentCache',
                'memory_hits': self._stats['memory_hits'],
                'disk_hits': self._stats['disk_hits'],
                'total_hits': self._stats['memory_hits'] + self._stats['disk_hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'memory_entries': len(self._memory_cache),
                'disk_entries': len(self._disk_cache),
                'total_entries': len(self._memory_cache) + len(self._disk_cache),
                'memory_usage_bytes': self._current_memory_usage,
                'memory_usage_mb': self._current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self._current_memory_usage / self.max_memory_bytes,
                'disk_spillovers': self._stats['disk_spillovers'],
                'memory_evictions': self._stats['memory_evictions'],
                'compression_enabled': self.compression_config.enabled,
                'compression_algorithm': self.compression_config.algorithm.value,
                'avg_compression_ratio': avg_compression_ratio,
                'eviction_policy': type(self.eviction_policy).__name__
            }
            
            # Add compression engine stats
            stats['compression_stats'] = self.compression_engine.get_stats()
            
            return stats
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics."""
        if not self.use_disk_spillover:
            return {'disk_spillover_enabled': False}
        
        total_disk_size = 0
        file_count = 0
        
        try:
            for file_path in self._disk_cache.values():
                if os.path.exists(file_path):
                    total_disk_size += os.path.getsize(file_path)
                    file_count += 1
        except Exception as e:
            logger.error(f"Error calculating disk usage: {e}")
        
        return {
            'disk_spillover_enabled': True,
            'spillover_directory': str(self.spillover_dir),
            'disk_entries': len(self._disk_cache),
            'disk_files_found': file_count,
            'total_disk_size_bytes': total_disk_size,
            'total_disk_size_mb': total_disk_size / (1024 * 1024),
            'avg_file_size_bytes': total_disk_size / file_count if file_count > 0 else 0
        }