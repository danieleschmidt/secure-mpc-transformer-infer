"""
Distributed cache implementation with Redis/Memcached support.

This module provides distributed caching capabilities for multi-node
secure MPC transformer deployments.
"""

import time
import pickle
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import socket

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - distributed cache will be disabled")

# Try to import pymemcache
try:
    from pymemcache.client.base import Client as MemcacheClient
    from pymemcache import serde
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False
    logger.warning("Memcached client not available")


class DistributedCacheBackend(Enum):
    """Supported distributed cache backends."""
    REDIS = "redis"
    MEMCACHED = "memcached"


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)
    connection_pool_max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    decode_responses: bool = False  # We handle bytes manually
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    
    def to_connection_kwargs(self) -> Dict[str, Any]:
        """Convert config to Redis connection kwargs."""
        return {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'password': self.password,
            'socket_timeout': self.socket_timeout,
            'socket_connect_timeout': self.socket_connect_timeout,
            'socket_keepalive': self.socket_keepalive,
            'socket_keepalive_options': self.socket_keepalive_options,
            'retry_on_timeout': self.retry_on_timeout,
            'health_check_interval': self.health_check_interval,
            'decode_responses': self.decode_responses,
            'ssl': self.ssl,
            'ssl_cert_reqs': self.ssl_cert_reqs
        }


@dataclass
class MemcachedConfig:
    """Configuration for Memcached connection."""
    host: str = "localhost"
    port: int = 11211
    connect_timeout: float = 5.0
    timeout: float = 5.0
    no_delay: bool = True
    socket_module: str = socket.__name__
    key_prefix: str = "smpt:"  # Secure MPC Transformer prefix
    max_pool_size: int = 10
    
    def get_server_address(self) -> Tuple[str, int]:
        """Get server address tuple."""
        return (self.host, self.port)


class DistributedCacheEntry:
    """Entry for distributed cache with metadata."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None, 
                 compression: bool = True):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.compression = compression
        
        # Serialize and optionally compress
        self.serialized_data = self._serialize_value()
    
    def _serialize_value(self) -> bytes:
        """Serialize value for storage."""
        try:
            # Use pickle for serialization
            data = pickle.dumps(self.value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Optionally compress
            if self.compression and len(data) > 1024:  # Compress if > 1KB
                try:
                    import gzip
                    compressed_data = gzip.compress(data)
                    # Only use compression if it provides significant benefit
                    if len(compressed_data) < len(data) * 0.9:
                        return b'GZIP:' + compressed_data
                except ImportError:
                    pass
            
            return b'RAW:' + data
            
        except Exception as e:
            logger.error(f"Serialization failed for key {self.key}: {e}")
            raise
    
    @classmethod
    def deserialize_value(cls, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if data.startswith(b'GZIP:'):
                # Decompress first
                import gzip
                compressed_data = data[5:]  # Remove 'GZIP:' prefix
                raw_data = gzip.decompress(compressed_data)
            elif data.startswith(b'RAW:'):
                raw_data = data[4:]  # Remove 'RAW:' prefix
            else:
                # Backward compatibility
                raw_data = data
            
            return pickle.loads(raw_data)
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise


class RedisDistributedCache:
    """Redis-based distributed cache implementation."""
    
    def __init__(self, config: RedisConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available")
        
        self.config = config
        self._client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'connection_errors': 0
        }
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                **self.config.to_connection_kwargs(),
                max_connections=self.config.connection_pool_max_connections
            )
            
            # Create Redis client
            self._client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            self._client.ping()
            
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._stats['connection_errors'] += 1
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._client:
            return None
        
        try:
            with self._lock:
                data = self._client.get(key)
                
                if data is None:
                    self._stats['misses'] += 1
                    return None
                
                # Deserialize value
                value = DistributedCacheEntry.deserialize_value(data)
                self._stats['hits'] += 1
                
                logger.debug(f"Redis cache hit: {key}")
                return value
                
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            self._stats['errors'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in Redis cache."""
        if not self._client:
            return False
        
        try:
            # Create cache entry
            entry = DistributedCacheEntry(key, value, ttl)
            
            with self._lock:
                # Store in Redis
                if ttl:
                    result = self._client.setex(key, ttl, entry.serialized_data)
                else:
                    result = self._client.set(key, entry.serialized_data)
                
                if result:
                    self._stats['sets'] += 1
                    logger.debug(f"Redis cache set: {key}")
                    return True
                else:
                    self._stats['errors'] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def remove(self, key: str) -> bool:
        """Remove value from Redis cache."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                result = self._client.delete(key)
                
                if result > 0:
                    self._stats['deletes'] += 1
                    logger.debug(f"Redis cache delete: {key}")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def clear(self) -> bool:
        """Clear all keys from Redis cache."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                # Use FLUSHDB to clear current database
                result = self._client.flushdb()
                logger.info("Redis cache cleared")
                return result
                
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            self._stats['errors'] += 1
            return False
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys with given namespace prefix."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                pattern = f"{namespace}:*"
                keys = self._client.keys(pattern)
                
                if keys:
                    result = self._client.delete(*keys)
                    logger.info(f"Cleared {result} Redis keys from namespace: {namespace}")
                    return result > 0
                else:
                    return True
                    
        except Exception as e:
            logger.error(f"Redis namespace clear failed: {e}")
            self._stats['errors'] += 1
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                return bool(self._client.exists(key))
                
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        redis_info = {}
        
        if self._client:
            try:
                # Get Redis server info
                redis_info = self._client.info()
                
                # Extract relevant metrics
                memory_info = {
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B'),
                    'used_memory_peak': redis_info.get('used_memory_peak', 0),
                    'mem_fragmentation_ratio': redis_info.get('mem_fragmentation_ratio', 1.0)
                }
                
                performance_info = {
                    'total_connections_received': redis_info.get('total_connections_received', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'instantaneous_ops_per_sec': redis_info.get('instantaneous_ops_per_sec', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0)
                }
                
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
                memory_info = {}
                performance_info = {}
        
        return {
            'backend': 'redis',
            'client_stats': self._stats.copy(),
            'server_memory': memory_info,
            'server_performance': performance_info,
            'connection_config': {
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.db
            }
        }
    
    def ping(self) -> bool:
        """Test Redis connection."""
        if not self._client:
            return False
        
        try:
            return bool(self._client.ping())
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    def close(self):
        """Close Redis connection."""
        try:
            if self._connection_pool:
                self._connection_pool.disconnect()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


class MemcachedDistributedCache:
    """Memcached-based distributed cache implementation."""
    
    def __init__(self, config: MemcachedConfig):
        if not MEMCACHED_AVAILABLE:
            raise ImportError("Memcached client is not available")
        
        self.config = config
        self._client: Optional[MemcacheClient] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        self._connect()
    
    def _connect(self):
        """Establish Memcached connection."""
        try:
            # Create Memcached client with custom serialization
            self._client = MemcacheClient(
                self.config.get_server_address(),
                connect_timeout=self.config.connect_timeout,
                timeout=self.config.timeout,
                no_delay=self.config.no_delay,
                serde=serde.pickle_serde
            )
            
            # Test connection
            self._client.version()
            
            logger.info(f"Connected to Memcached at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Memcached: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key for Memcached."""
        return f"{self.config.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Memcached cache."""
        if not self._client:
            return None
        
        try:
            with self._lock:
                memcached_key = self._make_key(key)
                value = self._client.get(memcached_key)
                
                if value is None:
                    self._stats['misses'] += 1
                    return None
                
                self._stats['hits'] += 1
                logger.debug(f"Memcached cache hit: {key}")
                return value
                
        except Exception as e:
            logger.error(f"Memcached get failed for key {key}: {e}")
            self._stats['errors'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in Memcached cache."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                memcached_key = self._make_key(key)
                
                # Memcached TTL: 0 means no expiration, None uses default
                expire_time = ttl if ttl is not None else 0
                
                result = self._client.set(memcached_key, value, expire=expire_time)
                
                if result:
                    self._stats['sets'] += 1
                    logger.debug(f"Memcached cache set: {key}")
                    return True
                else:
                    self._stats['errors'] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"Memcached set failed for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def remove(self, key: str) -> bool:
        """Remove value from Memcached cache."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                memcached_key = self._make_key(key)
                result = self._client.delete(memcached_key)
                
                if result:
                    self._stats['deletes'] += 1
                    logger.debug(f"Memcached cache delete: {key}")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Memcached delete failed for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def clear(self) -> bool:
        """Clear all keys from Memcached cache."""
        if not self._client:
            return False
        
        try:
            with self._lock:
                result = self._client.flush_all()
                logger.info("Memcached cache cleared")
                return result
                
        except Exception as e:
            logger.error(f"Memcached clear failed: {e}")
            self._stats['errors'] += 1
            return False
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear namespace (not efficiently supported by Memcached)."""
        logger.warning("Namespace clearing not efficiently supported by Memcached")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Memcached cache statistics."""
        server_stats = {}
        
        if self._client:
            try:
                # Get Memcached server stats
                raw_stats = self._client.stats()
                
                if raw_stats:
                    server_stats = {
                        'curr_items': raw_stats.get(b'curr_items', b'0').decode(),
                        'total_items': raw_stats.get(b'total_items', b'0').decode(),
                        'bytes': raw_stats.get(b'bytes', b'0').decode(),
                        'curr_connections': raw_stats.get(b'curr_connections', b'0').decode(),
                        'total_connections': raw_stats.get(b'total_connections', b'0').decode(),
                        'cmd_get': raw_stats.get(b'cmd_get', b'0').decode(),
                        'cmd_set': raw_stats.get(b'cmd_set', b'0').decode(),
                        'get_hits': raw_stats.get(b'get_hits', b'0').decode(),
                        'get_misses': raw_stats.get(b'get_misses', b'0').decode(),
                    }
                
            except Exception as e:
                logger.error(f"Failed to get Memcached stats: {e}")
        
        return {
            'backend': 'memcached',
            'client_stats': self._stats.copy(),
            'server_stats': server_stats,
            'connection_config': {
                'host': self.config.host,
                'port': self.config.port,
                'key_prefix': self.config.key_prefix
            }
        }
    
    def close(self):
        """Close Memcached connection."""
        try:
            if self._client:
                self._client.close()
            logger.info("Memcached connection closed")
        except Exception as e:
            logger.error(f"Error closing Memcached connection: {e}")


class DistributedCache:
    """Main distributed cache interface supporting multiple backends."""
    
    def __init__(self, backend_type: DistributedCacheBackend = DistributedCacheBackend.REDIS,
                 redis_config: Optional[RedisConfig] = None,
                 memcached_config: Optional[MemcachedConfig] = None):
        
        self.backend_type = backend_type
        self._backend = None
        
        # Initialize backend
        if backend_type == DistributedCacheBackend.REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis is required but not available")
            config = redis_config or RedisConfig()
            self._backend = RedisDistributedCache(config)
            
        elif backend_type == DistributedCacheBackend.MEMCACHED:
            if not MEMCACHED_AVAILABLE:
                raise ImportError("Memcached client is required but not available")
            config = memcached_config or MemcachedConfig()
            self._backend = MemcachedDistributedCache(config)
            
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        logger.info(f"Distributed cache initialized with {backend_type.value} backend")
    
    # Convenience constructor methods
    @classmethod
    def redis(cls, config: Optional[RedisConfig] = None) -> 'DistributedCache':
        """Create Redis-based distributed cache."""
        return cls(DistributedCacheBackend.REDIS, redis_config=config)
    
    @classmethod
    def memcached(cls, config: Optional[MemcachedConfig] = None) -> 'DistributedCache':
        """Create Memcached-based distributed cache."""
        return cls(DistributedCacheBackend.MEMCACHED, memcached_config=config)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        return self._backend.get(key) if self._backend else None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in distributed cache."""
        return self._backend.put(key, value, ttl) if self._backend else False
    
    def remove(self, key: str) -> bool:
        """Remove value from distributed cache."""
        return self._backend.remove(key) if self._backend else False
    
    def clear(self) -> bool:
        """Clear all values from distributed cache."""
        return self._backend.clear() if self._backend else False
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys with given namespace prefix."""
        return self._backend.clear_namespace(namespace) if self._backend else False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in distributed cache."""
        if hasattr(self._backend, 'exists'):
            return self._backend.exists(key)
        else:
            # Fallback: try to get the value
            return self.get(key) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed cache statistics."""
        if self._backend:
            return self._backend.get_stats()
        else:
            return {'backend': 'none', 'error': 'No backend available'}
    
    def health_check(self) -> bool:
        """Perform health check on distributed cache."""
        if hasattr(self._backend, 'ping'):
            return self._backend.ping()
        else:
            # Fallback: try a simple operation
            test_key = f"health_check_{int(time.time())}"
            try:
                result = self.put(test_key, "test", ttl=1)
                if result:
                    self.remove(test_key)
                return result
            except:
                return False
    
    def close(self):
        """Close distributed cache connection."""
        if self._backend and hasattr(self._backend, 'close'):
            self._backend.close()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except:
            pass