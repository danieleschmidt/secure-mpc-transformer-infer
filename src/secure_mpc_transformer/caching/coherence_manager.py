"""
Cache coherence manager for distributed multi-level caching.

This module ensures consistency across different cache levels and nodes
in a distributed secure MPC transformer system.
"""

import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class CoherenceProtocol(Enum):
    """Cache coherence protocols."""
    WRITE_THROUGH = "write_through"  # Write to all levels synchronously
    WRITE_BACK = "write_back"       # Write to higher levels lazily
    WRITE_AROUND = "write_around"   # Skip intermediate levels
    MESI = "mesi"                   # Modified, Exclusive, Shared, Invalid


class CacheState(Enum):
    """Cache entry states for MESI protocol."""
    MODIFIED = "modified"    # Cache has the only valid copy and it's been modified
    EXCLUSIVE = "exclusive"  # Cache has the only valid copy and it's clean
    SHARED = "shared"       # Multiple caches may have valid copies
    INVALID = "invalid"     # Cache entry is invalid


@dataclass
class CoherenceConfig:
    """Configuration for cache coherence management."""
    protocol: CoherenceProtocol = CoherenceProtocol.WRITE_THROUGH
    enable_invalidation_broadcast: bool = True
    invalidation_timeout_ms: int = 5000
    consistency_check_interval_s: int = 300  # 5 minutes
    max_pending_invalidations: int = 1000
    enable_version_tracking: bool = True
    version_cleanup_interval_s: int = 600  # 10 minutes


@dataclass
class CacheEntryMetadata:
    """Metadata for cache entries in coherence tracking."""
    key: str
    version: int
    state: CacheState
    last_modified: float
    cache_levels: Set[str] = field(default_factory=set)
    pending_invalidations: Set[str] = field(default_factory=set)
    
    def is_valid(self) -> bool:
        """Check if cache entry is in valid state."""
        return self.state != CacheState.INVALID
    
    def mark_modified(self):
        """Mark entry as modified."""
        self.state = CacheState.MODIFIED
        self.last_modified = time.time()
        self.version += 1
    
    def mark_shared(self):
        """Mark entry as shared."""
        if self.state == CacheState.MODIFIED:
            # Can't transition directly from MODIFIED to SHARED
            # Must first invalidate other copies
            return False
        self.state = CacheState.SHARED
        return True


class InvalidationMessage:
    """Message for cache invalidation."""
    
    def __init__(self, key: str, version: int, sender_id: str, 
                 message_type: str = "invalidate"):
        self.key = key
        self.version = version
        self.sender_id = sender_id
        self.message_type = message_type
        self.timestamp = time.time()
        self.message_id = self._generate_message_id()
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        data = f"{self.key}:{self.version}:{self.sender_id}:{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for transmission."""
        return {
            'key': self.key,
            'version': self.version,
            'sender_id': self.sender_id,
            'message_type': self.message_type,
            'timestamp': self.timestamp,
            'message_id': self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvalidationMessage':
        """Create message from dictionary."""
        msg = cls(
            key=data['key'],
            version=data['version'],
            sender_id=data['sender_id'],
            message_type=data.get('message_type', 'invalidate')
        )
        msg.timestamp = data['timestamp']
        msg.message_id = data['message_id']
        return msg


class CacheCoherenceManager:
    """Manager for cache coherence across multiple levels and nodes."""
    
    def __init__(self, config: CoherenceConfig,
                 l1_cache=None, l2_cache=None, distributed_cache=None,
                 node_id: Optional[str] = None):
        
        self.config = config
        self.node_id = node_id or f"node_{int(time.time())}"
        
        # Cache references
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.distributed_cache = distributed_cache
        
        # Coherence state tracking
        self._entry_metadata: Dict[str, CacheEntryMetadata] = {}
        self._pending_invalidations: deque = deque(maxlen=config.max_pending_invalidations)
        self._version_vector: Dict[str, int] = {}  # Track versions per node
        
        # Communication
        self._invalidation_callbacks: List[Callable[[InvalidationMessage], None]] = []
        self._message_handlers: Dict[str, Callable] = {
            'invalidate': self._handle_invalidation,
            'update': self._handle_update,
            'query': self._handle_query
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_running = False
        
        # Statistics
        self._stats = {
            'cache_writes': 0,
            'cache_reads': 0,
            'invalidations_sent': 0,
            'invalidations_received': 0,
            'consistency_violations': 0,
            'version_conflicts': 0
        }
        
        # Start background maintenance
        self._start_maintenance_thread()
        
        logger.info(f"Cache coherence manager initialized with {config.protocol.value} protocol")
    
    def on_cache_write(self, key: str, value: Any, cache_level: str = "l1"):
        """Handle cache write operation."""
        with self._lock:
            self._stats['cache_writes'] += 1
            
            # Get or create metadata
            if key not in self._entry_metadata:
                self._entry_metadata[key] = CacheEntryMetadata(
                    key=key,
                    version=1,
                    state=CacheState.EXCLUSIVE,
                    last_modified=time.time()
                )
            else:
                self._entry_metadata[key].mark_modified()
            
            metadata = self._entry_metadata[key]
            metadata.cache_levels.add(cache_level)
            
            # Apply coherence protocol
            if self.config.protocol == CoherenceProtocol.WRITE_THROUGH:
                self._handle_write_through(key, value, metadata)
            
            elif self.config.protocol == CoherenceProtocol.WRITE_BACK:
                self._handle_write_back(key, value, metadata)
            
            elif self.config.protocol == CoherenceProtocol.WRITE_AROUND:
                self._handle_write_around(key, value, metadata)
            
            elif self.config.protocol == CoherenceProtocol.MESI:
                self._handle_mesi_write(key, value, metadata)
    
    def on_cache_read(self, key: str, cache_level: str = "l1") -> Optional[Any]:
        """Handle cache read operation."""
        with self._lock:
            self._stats['cache_reads'] += 1
            
            if key in self._entry_metadata:
                metadata = self._entry_metadata[key]
                
                # Check if entry is valid
                if not metadata.is_valid():
                    return None
                
                # Apply coherence protocol for reads
                if self.config.protocol == CoherenceProtocol.MESI:
                    return self._handle_mesi_read(key, metadata, cache_level)
                
                return self._get_from_cache_level(key, cache_level)
            
            return None
    
    def on_cache_invalidate(self, key: str):
        """Handle cache invalidation."""
        with self._lock:
            if key in self._entry_metadata:
                metadata = self._entry_metadata[key]
                metadata.state = CacheState.INVALID
                
                # Broadcast invalidation if enabled
                if self.config.enable_invalidation_broadcast:
                    self._broadcast_invalidation(key, metadata.version)
    
    def _handle_write_through(self, key: str, value: Any, metadata: CacheEntryMetadata):
        """Handle write-through coherence."""
        # Write to all cache levels synchronously
        if self.l1_cache:
            self.l1_cache.put(key, value)
        
        if self.l2_cache:
            self.l2_cache.put(key, value)
        
        if self.distributed_cache:
            self.distributed_cache.put(key, value)
        
        logger.debug(f"Write-through completed for key: {key}")
    
    def _handle_write_back(self, key: str, value: Any, metadata: CacheEntryMetadata):
        """Handle write-back coherence."""
        # Write to local cache immediately, sync to other levels later
        if self.l1_cache:
            self.l1_cache.put(key, value)
        
        # Mark for later synchronization
        metadata.pending_invalidations.add("sync_required")
        
        logger.debug(f"Write-back initiated for key: {key}")
    
    def _handle_write_around(self, key: str, value: Any, metadata: CacheEntryMetadata):
        """Handle write-around coherence."""
        # Skip L1 and L2, write directly to distributed cache
        if self.distributed_cache:
            self.distributed_cache.put(key, value)
        
        # Invalidate local caches
        if self.l1_cache:
            self.l1_cache.remove(key)
        
        if self.l2_cache:
            self.l2_cache.remove(key)
        
        logger.debug(f"Write-around completed for key: {key}")
    
    def _handle_mesi_write(self, key: str, value: Any, metadata: CacheEntryMetadata):
        """Handle MESI protocol write."""
        current_state = metadata.state
        
        if current_state == CacheState.MODIFIED:
            # Already have exclusive modified copy, just update
            if self.l1_cache:
                self.l1_cache.put(key, value)
        
        elif current_state == CacheState.EXCLUSIVE:
            # Have exclusive clean copy, can modify without notification
            metadata.mark_modified()
            if self.l1_cache:
                self.l1_cache.put(key, value)
        
        elif current_state == CacheState.SHARED:
            # Need to invalidate other copies first
            self._broadcast_invalidation(key, metadata.version)
            metadata.state = CacheState.MODIFIED
            if self.l1_cache:
                self.l1_cache.put(key, value)
        
        elif current_state == CacheState.INVALID:
            # Need to acquire exclusive access
            metadata.state = CacheState.EXCLUSIVE
            metadata.mark_modified()
            if self.l1_cache:
                self.l1_cache.put(key, value)
        
        logger.debug(f"MESI write completed for key: {key}, state: {metadata.state.value}")
    
    def _handle_mesi_read(self, key: str, metadata: CacheEntryMetadata, cache_level: str) -> Optional[Any]:
        """Handle MESI protocol read."""
        current_state = metadata.state
        
        if current_state in [CacheState.MODIFIED, CacheState.EXCLUSIVE, CacheState.SHARED]:
            # Valid data available
            return self._get_from_cache_level(key, cache_level)
        
        elif current_state == CacheState.INVALID:
            # Need to fetch from another cache or storage
            value = self._fetch_from_remote(key)
            if value is not None:
                metadata.state = CacheState.SHARED
                return value
        
        return None
    
    def _get_from_cache_level(self, key: str, cache_level: str) -> Optional[Any]:
        """Get value from specific cache level."""
        if cache_level == "l1" and self.l1_cache:
            return self.l1_cache.get(key)
        elif cache_level == "l2" and self.l2_cache:
            return self.l2_cache.get(key)
        elif cache_level == "distributed" and self.distributed_cache:
            return self.distributed_cache.get(key)
        
        return None
    
    def _fetch_from_remote(self, key: str) -> Optional[Any]:
        """Fetch value from remote cache or storage."""
        # Try distributed cache first
        if self.distributed_cache:
            value = self.distributed_cache.get(key)
            if value is not None:
                return value
        
        # Try L2 cache
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                return value
        
        return None
    
    def _broadcast_invalidation(self, key: str, version: int):
        """Broadcast invalidation message to other nodes."""
        message = InvalidationMessage(key, version, self.node_id)
        
        # Add to pending invalidations
        self._pending_invalidations.append(message)
        
        # Notify registered callbacks
        for callback in self._invalidation_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Invalidation callback failed: {e}")
        
        self._stats['invalidations_sent'] += 1
        logger.debug(f"Broadcasted invalidation for key: {key}, version: {version}")
    
    def register_invalidation_callback(self, callback: Callable[[InvalidationMessage], None]):
        """Register callback for invalidation messages."""
        self._invalidation_callbacks.append(callback)
    
    def handle_remote_message(self, message_data: Dict[str, Any]):
        """Handle incoming coherence message from remote node."""
        try:
            message = InvalidationMessage.from_dict(message_data)
            handler = self._message_handlers.get(message.message_type)
            
            if handler:
                handler(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Failed to handle remote message: {e}")
    
    def _handle_invalidation(self, message: InvalidationMessage):
        """Handle invalidation message from remote node."""
        with self._lock:
            key = message.key
            remote_version = message.version
            
            if key in self._entry_metadata:
                metadata = self._entry_metadata[key]
                
                # Check version
                if remote_version > metadata.version:
                    # Remote version is newer, invalidate local copy
                    metadata.state = CacheState.INVALID
                    
                    # Remove from local caches
                    if self.l1_cache:
                        self.l1_cache.remove(key)
                    if self.l2_cache:
                        self.l2_cache.remove(key)
                    
                    self._stats['invalidations_received'] += 1
                    logger.debug(f"Invalidated local copy of key: {key}")
                
                elif remote_version < metadata.version:
                    # Local version is newer, send update to remote
                    self._send_update_message(key, metadata.version, message.sender_id)
                    self._stats['version_conflicts'] += 1
    
    def _handle_update(self, message: InvalidationMessage):
        """Handle update message from remote node."""
        # Implementation for handling update messages
        logger.debug(f"Received update message for key: {message.key}")
    
    def _handle_query(self, message: InvalidationMessage):
        """Handle query message from remote node."""
        # Implementation for handling query messages
        logger.debug(f"Received query message for key: {message.key}")
    
    def _send_update_message(self, key: str, version: int, target_node: str):
        """Send update message to specific node."""
        message = InvalidationMessage(key, version, self.node_id, "update")
        # Implementation would send message to target_node
        logger.debug(f"Sending update message for key: {key} to node: {target_node}")
    
    def _start_maintenance_thread(self):
        """Start background maintenance thread."""
        self._maintenance_running = True
        self._maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self._maintenance_thread.start()
        logger.info("Cache coherence maintenance thread started")
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self._maintenance_running:
            try:
                # Perform consistency checks
                self._check_consistency()
                
                # Clean up old versions
                self._cleanup_old_versions()
                
                # Process pending write-backs
                self._process_pending_writebacks()
                
                time.sleep(self.config.consistency_check_interval_s)
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                time.sleep(30)  # Shorter sleep on error
    
    def _check_consistency(self):
        """Check cache consistency across levels."""
        inconsistencies = 0
        
        with self._lock:
            for key, metadata in self._entry_metadata.items():
                if not self._verify_entry_consistency(key, metadata):
                    inconsistencies += 1
                    self._stats['consistency_violations'] += 1
        
        if inconsistencies > 0:
            logger.warning(f"Found {inconsistencies} consistency violations")
    
    def _verify_entry_consistency(self, key: str, metadata: CacheEntryMetadata) -> bool:
        """Verify consistency of a single cache entry."""
        # Check if entry exists in claimed cache levels
        for level in metadata.cache_levels:
            if not self._entry_exists_in_level(key, level):
                logger.warning(f"Consistency violation: key {key} missing from {level}")
                return False
        
        return True
    
    def _entry_exists_in_level(self, key: str, level: str) -> bool:
        """Check if entry exists in specific cache level."""
        if level == "l1" and self.l1_cache:
            return key in self.l1_cache
        elif level == "l2" and self.l2_cache:
            return hasattr(self.l2_cache, '__contains__') and key in self.l2_cache
        elif level == "distributed" and self.distributed_cache:
            return self.distributed_cache.exists(key) if hasattr(self.distributed_cache, 'exists') else False
        
        return False
    
    def _cleanup_old_versions(self):
        """Clean up old version tracking data."""
        cutoff_time = time.time() - self.config.version_cleanup_interval_s
        
        with self._lock:
            # Remove old metadata
            expired_keys = [
                key for key, metadata in self._entry_metadata.items()
                if metadata.last_modified < cutoff_time and metadata.state == CacheState.INVALID
            ]
            
            for key in expired_keys:
                del self._entry_metadata[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} old cache metadata entries")
    
    def _process_pending_writebacks(self):
        """Process pending write-back operations."""
        if self.config.protocol != CoherenceProtocol.WRITE_BACK:
            return
        
        with self._lock:
            writeback_count = 0
            
            for key, metadata in self._entry_metadata.items():
                if "sync_required" in metadata.pending_invalidations:
                    # Perform delayed synchronization
                    if self.l1_cache and self.distributed_cache:
                        value = self.l1_cache.get(key)
                        if value is not None:
                            self.distributed_cache.put(key, value)
                            metadata.pending_invalidations.discard("sync_required")
                            writeback_count += 1
            
            if writeback_count > 0:
                logger.debug(f"Processed {writeback_count} write-back operations")
    
    def perform_maintenance(self):
        """Manually trigger maintenance operations."""
        self._check_consistency()
        self._cleanup_old_versions()
        self._process_pending_writebacks()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coherence manager statistics."""
        with self._lock:
            return {
                'node_id': self.node_id,
                'protocol': self.config.protocol.value,
                'tracked_entries': len(self._entry_metadata),
                'pending_invalidations': len(self._pending_invalidations),
                'cache_levels': {
                    'l1_enabled': self.l1_cache is not None,
                    'l2_enabled': self.l2_cache is not None,
                    'distributed_enabled': self.distributed_cache is not None
                },
                **self._stats,
                'state_distribution': self._get_state_distribution()
            }
    
    def _get_state_distribution(self) -> Dict[str, int]:
        """Get distribution of cache entry states."""
        distribution = defaultdict(int)
        
        for metadata in self._entry_metadata.values():
            distribution[metadata.state.value] += 1
        
        return dict(distribution)
    
    def shutdown(self):
        """Shutdown coherence manager."""
        logger.info("Shutting down cache coherence manager")
        
        self._maintenance_running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)
        
        logger.info("Cache coherence manager shutdown completed")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass