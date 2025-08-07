"""
Quantum Computation Caching and Optimization

Advanced caching system for quantum states, optimization results,
and computational patterns to accelerate repeated operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import hashlib
import pickle
import time
import threading
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import OrderedDict, defaultdict
import weakref
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheLevel(Enum):
    """Cache storage levels"""
    MEMORY = "memory"    # In-memory cache
    DISK = "disk"       # Persistent disk cache
    DISTRIBUTED = "distributed"  # Distributed cache


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    ttl: Optional[float] = None  # seconds
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl
    
    def touch(self):
        """Update access information"""
        self.access_count += 1
        self.last_access = datetime.now()


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class QuantumStateCache:
    """
    High-performance cache for quantum states with specialized
    quantum computation optimizations.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 512,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 enable_compression: bool = True):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.enable_compression = enable_compression
        
        # Storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_frequencies: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Quantum-specific optimizations
        self.quantum_similarity_threshold = 0.95  # For finding similar states
        self.state_fingerprints: Dict[str, str] = {}  # Quick similarity checks
        
        logger.info(f"Initialized QuantumStateCache with {max_size} entries, {max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve quantum state from cache"""
        with self._lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access info
            entry.touch()
            self.access_frequencies[key] += 1
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            self.stats.hits += 1
            
            # Decompress if needed
            state = entry.value
            if self.enable_compression and isinstance(state, bytes):
                state = self._decompress_state(state)
            
            logger.debug(f"Cache hit for quantum state: {key}")
            return state
    
    def put(self, key: str, state: np.ndarray, ttl: Optional[float] = None) -> bool:
        """Store quantum state in cache"""
        with self._lock:
            # Validate quantum state
            if not self._validate_quantum_state(state):
                logger.warning(f"Invalid quantum state for key: {key}")
                return False
            
            # Calculate size
            state_data = state
            if self.enable_compression:
                state_data = self._compress_state(state)
            
            size_bytes = self._estimate_size(state_data)
            
            # Check if we need to make space
            while (len(self.cache) >= self.max_size or 
                   self.stats.size_bytes + size_bytes > self.max_memory_bytes):
                if not self._evict_entry():
                    logger.warning("Unable to evict entries for new quantum state")
                    return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=state_data,
                timestamp=datetime.now(),
                ttl=ttl,
                size_bytes=size_bytes,
                metadata={
                    "quantum_norm": np.linalg.norm(state),
                    "quantum_entropy": self._calculate_entropy(state),
                    "state_shape": state.shape
                }
            )
            
            # Store quantum fingerprint for similarity search
            self.state_fingerprints[key] = self._calculate_state_fingerprint(state)
            
            # Add to cache
            self.cache[key] = entry
            self.stats.size_bytes += size_bytes
            self.stats.entry_count += 1
            
            logger.debug(f"Cached quantum state: {key} ({size_bytes} bytes)")
            return True
    
    def find_similar_state(self, 
                          target_state: np.ndarray, 
                          threshold: Optional[float] = None) -> Optional[Tuple[str, np.ndarray]]:
        """
        Find similar quantum state in cache using fidelity measure.
        
        Args:
            target_state: Quantum state to find similarity for
            threshold: Similarity threshold (default uses instance threshold)
            
        Returns:
            Tuple of (key, state) if similar state found, None otherwise
        """
        threshold = threshold or self.quantum_similarity_threshold
        target_fingerprint = self._calculate_state_fingerprint(target_state)
        
        with self._lock:
            best_similarity = 0.0
            best_match = None
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    continue
                
                # Quick fingerprint check
                cached_fingerprint = self.state_fingerprints.get(key, "")
                if self._fingerprint_similarity(target_fingerprint, cached_fingerprint) < 0.8:
                    continue  # Skip expensive fidelity calculation
                
                # Calculate quantum fidelity
                cached_state = entry.value
                if self.enable_compression and isinstance(cached_state, bytes):
                    cached_state = self._decompress_state(cached_state)
                
                fidelity = self._quantum_fidelity(target_state, cached_state)
                
                if fidelity > threshold and fidelity > best_similarity:
                    best_similarity = fidelity
                    best_match = (key, cached_state)
            
            if best_match:
                logger.debug(f"Found similar quantum state with fidelity {best_similarity:.3f}")
                # Update access for similar state
                self.cache[best_match[0]].touch()
                self.access_frequencies[best_match[0]] += 1
            
            return best_match
    
    def _validate_quantum_state(self, state: np.ndarray) -> bool:
        """Validate quantum state properties"""
        if not np.iscomplexobj(state):
            return False
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return False
        
        # Check normalization (with tolerance)
        norm = np.linalg.norm(state)
        if abs(norm - 1.0) > 1e-6:
            return False
        
        return True
    
    def _calculate_state_fingerprint(self, state: np.ndarray) -> str:
        """Calculate fast fingerprint for quantum state"""
        # Use amplitude magnitudes and phases for fingerprint
        amplitudes = np.abs(state)
        phases = np.angle(state)
        
        # Create histogram-based fingerprint
        amp_hist, _ = np.histogram(amplitudes, bins=10)
        phase_hist, _ = np.histogram(phases, bins=10)
        
        fingerprint_data = np.concatenate([amp_hist, phase_hist])
        return hashlib.md5(fingerprint_data.tobytes()).hexdigest()[:16]
    
    def _fingerprint_similarity(self, fp1: str, fp2: str) -> float:
        """Calculate similarity between fingerprints"""
        if len(fp1) != len(fp2):
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(fp1, fp2))
        return matches / len(fp1)
    
    def _quantum_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate quantum fidelity between two states"""
        return abs(np.vdot(state1, state2)) ** 2
    
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calculate von Neumann entropy of quantum state"""
        amplitudes = np.abs(state) ** 2
        return -np.sum(amplitudes * np.log2(amplitudes + 1e-12))
    
    def _compress_state(self, state: np.ndarray) -> bytes:
        """Compress quantum state for storage"""
        # Use custom compression for complex arrays
        real_part = state.real.astype(np.float32)
        imag_part = state.imag.astype(np.float32)
        
        data = {
            'real': real_part,
            'imag': imag_part,
            'shape': state.shape
        }
        
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _decompress_state(self, compressed_data: bytes) -> np.ndarray:
        """Decompress quantum state from storage"""
        data = pickle.loads(compressed_data)
        
        real_part = data['real'].astype(np.complex128)
        imag_part = data['imag'].astype(np.complex128)
        
        state = real_part + 1j * imag_part
        return state.reshape(data['shape'])
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, bytes):
            return len(data)
        else:
            return len(pickle.dumps(data))
    
    def _evict_entry(self) -> bool:
        """Evict entry based on cache policy"""
        if not self.cache:
            return False
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            key_to_evict = next(iter(self.cache))  # First item (oldest)
        
        elif self.policy == CachePolicy.LFU:
            min_freq = min(self.access_frequencies.get(k, 0) for k in self.cache.keys())
            for key in self.cache.keys():
                if self.access_frequencies.get(key, 0) == min_freq:
                    key_to_evict = key
                    break
        
        elif self.policy == CachePolicy.TTL:
            # Evict expired entries first
            for key, entry in self.cache.items():
                if entry.is_expired():
                    key_to_evict = key
                    break
            
            # If no expired entries, fall back to LRU
            if key_to_evict is None:
                key_to_evict = next(iter(self.cache))
        
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive policy based on access patterns and quantum metrics
            key_to_evict = self._adaptive_eviction()
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            return True
        
        return False
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on quantum state properties and access patterns"""
        if not self.cache:
            return None
        
        # Score each entry for eviction (higher score = more likely to evict)
        eviction_scores = {}
        
        for key, entry in self.cache.items():
            score = 0.0
            
            # Time since last access
            if entry.last_access:
                hours_since_access = (datetime.now() - entry.last_access).total_seconds() / 3600
                score += hours_since_access * 10
            
            # Frequency penalty (lower frequency = higher eviction score)
            freq = self.access_frequencies.get(key, 1)
            score += 100 / freq
            
            # Quantum entropy (higher entropy states might be less useful)
            entropy = entry.metadata.get("quantum_entropy", 0)
            score += entropy * 5
            
            # Size penalty (larger entries get higher eviction score)
            size_mb = entry.size_bytes / (1024 * 1024)
            score += size_mb * 2
            
            eviction_scores[key] = score
        
        # Return key with highest eviction score
        return max(eviction_scores.items(), key=lambda x: x[1])[0]
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            self.stats.evictions += 1
            
            del self.cache[key]
            if key in self.access_frequencies:
                del self.access_frequencies[key]
            if key in self.state_fingerprints:
                del self.state_fingerprints[key]
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_frequencies.clear()
            self.state_fingerprints.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats


class OptimizationResultCache:
    """
    Cache for optimization results with pattern recognition
    to avoid redundant quantum optimization runs.
    """
    
    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.pattern_index: Dict[str, List[str]] = defaultdict(list)  # Pattern -> cache keys
        self._lock = threading.RLock()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "pattern_matches": 0
        }
    
    def get_optimization_result(self, 
                              task_pattern: str, 
                              constraints: Dict[str, Any],
                              resources: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Get cached optimization result for similar problem configuration.
        
        Args:
            task_pattern: Pattern describing task structure
            constraints: Optimization constraints
            resources: Available resources
            
        Returns:
            Cached optimization result if found, None otherwise
        """
        cache_key = self._generate_cache_key(task_pattern, constraints, resources)
        
        with self._lock:
            # Direct cache hit
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not self._is_expired(entry):
                    self.stats["hits"] += 1
                    entry["access_count"] += 1
                    entry["last_access"] = datetime.now()
                    return entry["result"]
                else:
                    del self.cache[cache_key]
            
            # Pattern-based search
            pattern_key = self._extract_pattern_key(task_pattern, constraints)
            similar_keys = self.pattern_index.get(pattern_key, [])
            
            for similar_key in similar_keys:
                if similar_key in self.cache:
                    entry = self.cache[similar_key]
                    if (not self._is_expired(entry) and 
                        self._is_configuration_similar(cache_key, similar_key)):
                        self.stats["pattern_matches"] += 1
                        entry["access_count"] += 1
                        return self._adapt_result_to_configuration(entry["result"], resources)
            
            self.stats["misses"] += 1
            return None
    
    def store_optimization_result(self, 
                                task_pattern: str,
                                constraints: Dict[str, Any],
                                resources: Dict[str, float],
                                result: Dict[str, Any],
                                ttl_hours: float = 24.0):
        """Store optimization result in cache"""
        cache_key = self._generate_cache_key(task_pattern, constraints, resources)
        pattern_key = self._extract_pattern_key(task_pattern, constraints)
        
        with self._lock:
            # Make space if needed
            while len(self.cache) >= self.max_entries:
                self._evict_oldest()
            
            # Store result
            entry = {
                "result": result,
                "timestamp": datetime.now(),
                "ttl_hours": ttl_hours,
                "access_count": 1,
                "last_access": datetime.now(),
                "pattern_key": pattern_key
            }
            
            self.cache[cache_key] = entry
            self.pattern_index[pattern_key].append(cache_key)
    
    def _generate_cache_key(self, task_pattern: str, constraints: Dict[str, Any], resources: Dict[str, float]) -> str:
        """Generate unique cache key for optimization configuration"""
        key_data = {
            "pattern": task_pattern,
            "constraints": sorted(constraints.items()),
            "resources": sorted(resources.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _extract_pattern_key(self, task_pattern: str, constraints: Dict[str, Any]) -> str:
        """Extract pattern key for similarity matching"""
        # Simplified pattern extraction - could be more sophisticated
        pattern_elements = [
            task_pattern[:10],  # First part of pattern
            str(len(constraints)),
            str(constraints.get("max_execution_time", 0))[:5]
        ]
        return "_".join(pattern_elements)
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired"""
        age_hours = (datetime.now() - entry["timestamp"]).total_seconds() / 3600
        return age_hours > entry["ttl_hours"]
    
    def _is_configuration_similar(self, key1: str, key2: str) -> bool:
        """Check if two configurations are similar enough for cache reuse"""
        # Simplified similarity check - could use more sophisticated comparison
        return key1[:8] == key2[:8]  # Compare first 8 chars of hash
    
    def _adapt_result_to_configuration(self, cached_result: Dict[str, Any], resources: Dict[str, float]) -> Dict[str, Any]:
        """Adapt cached result to current resource configuration"""
        # Simple adaptation - in practice would do more sophisticated scaling
        adapted_result = cached_result.copy()
        adapted_result["adapted"] = True
        adapted_result["original_resources"] = cached_result.get("resources", {})
        adapted_result["current_resources"] = resources
        
        return adapted_result
    
    def _evict_oldest(self):
        """Evict oldest entry from cache"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
        entry = self.cache[oldest_key]
        
        # Remove from pattern index
        pattern_key = entry["pattern_key"]
        if oldest_key in self.pattern_index[pattern_key]:
            self.pattern_index[pattern_key].remove(oldest_key)
        
        del self.cache[oldest_key]


class PersistentQuantumCache:
    """
    Persistent cache for quantum computations that survives process restarts.
    Uses efficient serialization and compression for quantum states.
    """
    
    def __init__(self, cache_dir: str = "quantum_cache", max_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.index_file = self.cache_dir / "cache_index.json"
        
        # Load existing index
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from persistent cache"""
        with self._lock:
            if key not in self.index:
                return None
            
            entry = self.index[key]
            
            # Check if file exists
            file_path = self.cache_dir / entry["filename"]
            if not file_path.exists():
                del self.index[key]
                return None
            
            # Load and return data
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access info
                entry["last_access"] = datetime.now().isoformat()
                entry["access_count"] = entry.get("access_count", 0) + 1
                
                return data
            
            except Exception as e:
                logger.error(f"Failed to load cached item {key}: {e}")
                return None
    
    def put(self, key: str, data: Any, ttl_hours: Optional[float] = None) -> bool:
        """Store item in persistent cache"""
        with self._lock:
            try:
                # Generate filename
                filename = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
                file_path = self.cache_dir / filename
                
                # Serialize data
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update index
                file_size = file_path.stat().st_size
                self.index[key] = {
                    "filename": filename,
                    "size": file_size,
                    "created": datetime.now().isoformat(),
                    "last_access": datetime.now().isoformat(),
                    "access_count": 0,
                    "ttl_hours": ttl_hours
                }
                
                # Check size limits and evict if necessary
                self._enforce_size_limit()
                
                # Save index
                self._save_index()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache item {key}: {e}")
                return False
    
    def _load_index(self):
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
                self.index = {}
    
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _enforce_size_limit(self):
        """Enforce cache size limit by evicting old entries"""
        total_size = sum(entry["size"] for entry in self.index.values())
        
        while total_size > self.max_size_bytes and self.index:
            # Find oldest accessed file
            oldest_key = min(self.index.keys(), 
                           key=lambda k: self.index[k]["last_access"])
            
            entry = self.index[oldest_key]
            file_path = self.cache_dir / entry["filename"]
            
            # Remove file and index entry
            if file_path.exists():
                file_path.unlink()
                total_size -= entry["size"]
            
            del self.index[oldest_key]


def create_quantum_cache_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create complete quantum caching system with all cache types.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing all cache instances
    """
    config = config or {}
    
    # Quantum state cache
    state_cache = QuantumStateCache(
        max_size=config.get("quantum_cache_size", 1000),
        max_memory_mb=config.get("quantum_cache_memory_mb", 512),
        policy=CachePolicy(config.get("quantum_cache_policy", "adaptive")),
        enable_compression=config.get("enable_compression", True)
    )
    
    # Optimization result cache
    optimization_cache = OptimizationResultCache(
        max_entries=config.get("optimization_cache_size", 500)
    )
    
    # Persistent cache
    persistent_cache = PersistentQuantumCache(
        cache_dir=config.get("persistent_cache_dir", "quantum_cache"),
        max_size_gb=config.get("persistent_cache_size_gb", 1.0)
    )
    
    return {
        "quantum_state_cache": state_cache,
        "optimization_cache": optimization_cache,
        "persistent_cache": persistent_cache
    }