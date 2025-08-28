"""Enhanced Model Service for Secure MPC Transformer System."""

import asyncio
import logging
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.secure_transformer import SecureTransformer

try:
    import torch
    import transformers
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model loading status."""
    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelLoadRequest:
    """Request for loading a model."""
    model_name: str
    protocol_config: Dict[str, Any]
    priority: int = 0
    timeout: float = 300.0
    request_id: str = ""
    
    def __post_init__(self):
        if not self.request_id:
            import uuid
            self.request_id = str(uuid.uuid4())


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_name: str
    memory_usage_mb: float
    load_time: float
    last_accessed: float
    access_count: int
    status: ModelStatus
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelCache:
    """Thread-safe LRU cache for models."""

    def __init__(self, max_models: int = 3, max_memory_mb: int = 8000):
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.access_order: List[str] = []
        self._lock = threading.RLock()

    def _get_total_memory(self) -> float:
        """Calculate total memory usage."""
        return sum(info.memory_usage_mb for info in self.model_info.values())

    def _remove_least_recently_used(self):
        """Remove the least recently used model."""
        if not self.access_order:
            return

        lru_key = self.access_order[0]
        self._remove_model(lru_key)

    def _remove_model(self, cache_key: str):
        """Remove a model from cache."""
        if cache_key in self.models:
            del self.models[cache_key]
            del self.model_info[cache_key]
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)

    def _update_access_order(self, cache_key: str):
        """Update access order for LRU."""
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

    def put(self, cache_key: str, model: "SecureTransformer", model_info: ModelInfo):
        """Add model to cache."""
        with self._lock:
            # Remove if already exists
            if cache_key in self.models:
                self._remove_model(cache_key)

            # Check if we need to make space
            while (len(self.models) >= self.max_models or
                   self._get_total_memory() + model_info.memory_usage_mb > self.max_memory_mb):
                if not self.models:
                    break
                self._remove_least_recently_used()

            # Add new model
            self.models[cache_key] = model
            self.model_info[cache_key] = model_info
            self.access_order.append(cache_key)

            logger.info(f"Model cached: {cache_key}")

    def get(self, cache_key: str) -> "SecureTransformer | None":
        """Get model from cache."""
        with self._lock:
            if cache_key in self.models:
                # Update access order
                self._update_access_order(cache_key)
                
                # Update access statistics
                self.model_info[cache_key].last_accessed = time.time()
                self.model_info[cache_key].access_count += 1
                
                return self.models[cache_key]
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_models": len(self.models),
                "max_models": self.max_models,
                "total_memory_mb": self._get_total_memory(),
                "max_memory_mb": self.max_memory_mb,
                "memory_utilization": (self._get_total_memory() / self.max_memory_mb) * 100,
                "models": [info.to_dict() for info in self.model_info.values()]
            }

    def clear(self):
        """Clear all cached models."""
        with self._lock:
            for cache_key in list(self.models.keys()):
                self._remove_model(cache_key)


class ModelLoader:
    """Asynchronous model loader with queue management."""

    def __init__(self, max_concurrent_loads: int = 2):
        self.max_concurrent_loads = max_concurrent_loads
        self.load_queue: List[ModelLoadRequest] = []
        self.loading_models: Dict[str, ModelLoadRequest] = {}
        self.load_results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def submit_load_request(self, request: ModelLoadRequest) -> str:
        """Submit a model load request."""
        with self._lock:
            # Check if already queued or loading
            for req in self.load_queue:
                if (req.model_name == request.model_name and 
                    req.protocol_config == request.protocol_config):
                    return req.request_id

            for req in self.loading_models.values():
                if (req.model_name == request.model_name and 
                    req.protocol_config == request.protocol_config):
                    return req.request_id

            # Add to queue
            self.load_queue.append(request)
            self.load_queue.sort(key=lambda x: x.priority, reverse=True)
            
            return request.request_id

    def get_load_status(self, request_id: str) -> Dict[str, Any]:
        """Get load status for a request."""
        with self._lock:
            # Check if loading
            for req in self.loading_models.values():
                if req.request_id == request_id:
                    return {"status": "loading", "request": asdict(req)}
            
            # Check queue
            for req in self.load_queue:
                if req.request_id == request_id:
                    return {"status": "queued", "request": asdict(req)}
            
            # Check results
            if request_id in self.load_results:
                return {"status": "completed", "result": self.load_results[request_id]}
            
            return {"status": "not_found"}

    def _process_load_queue(self):
        """Process the load queue (placeholder)."""
        # This would be implemented with actual model loading logic
        pass


class ModelHelper:
    """Helper utilities for model operations."""
    
    @staticmethod
    def estimate_model_memory(config: Dict[str, Any]) -> float:
        """Estimate model memory usage in MB."""
        # Simplified estimation - in practice this would be more sophisticated
        base_memory = 500  # Base memory for small models
        
        # Adjust based on model name or parameters
        model_name = config.get('model_name', '')
        if 'large' in model_name.lower():
            return base_memory * 4
        elif 'base' in model_name.lower():
            return base_memory * 2
        else:
            return base_memory

    @staticmethod
    def create_cache_key(model_name: str, protocol_config: Dict[str, Any]) -> str:
        """Create a unique cache key for model + config."""
        import hashlib
        config_str = str(sorted(protocol_config.items()))
        key_data = f"{model_name}:{config_str}"
        return hashlib.md5(key_data.encode()).hexdigest()


class ModelService:
    """Enhanced model service with caching and concurrent loading."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = ModelCache(
            max_models=config.get("max_cached_models", 3),
            max_memory_mb=config.get("max_cache_memory_mb", 8000)
        )
        self.loader = ModelLoader(
            max_concurrent_loads=config.get("max_concurrent_loads", 2)
        )
        self.auto_unload_timeout = config.get("auto_unload_timeout", 1800)  # 30 minutes
        
        self._setup_background_tasks()
        self._preload_models()

        logger.info("ModelService initialized")

    async def get_model(self, model_name: str, protocol_config: Dict[str, Any]) -> "SecureTransformer":
        """Get model, loading if necessary."""
        request = ModelLoadRequest(
            model_name=model_name,
            protocol_config=protocol_config
        )
        
        cache_key = ModelHelper.create_cache_key(model_name, protocol_config)
        
        # Check cache first
        cached_model = self.cache.get(cache_key)
        if cached_model:
            logger.info(f"Model cache hit: {model_name}")
            return cached_model
        
        # Submit load request
        request_id = self.loader.submit_load_request(request)
        
        # Wait for loading to complete (simplified)
        # In practice, this would use proper async waiting
        await asyncio.sleep(0.1)  # Placeholder
        
        # For now, return a mock model
        from ..models.secure_transformer import SecureTransformer
        mock_model = SecureTransformer.create_mock_model(model_name)
        
        # Cache the model
        model_info = ModelInfo(
            model_name=model_name,
            memory_usage_mb=ModelHelper.estimate_model_memory(protocol_config),
            load_time=1.0,
            last_accessed=time.time(),
            access_count=1,
            status=ModelStatus.LOADED
        )
        
        self.cache.put(cache_key, mock_model, model_info)
        
        return mock_model

    def list_models(self) -> Dict[str, Any]:
        """List all cached models and their information."""
        return {
            "cache_stats": self.cache.get_cache_stats(),
            "loader_stats": {
                "queue_size": len(self.loader.load_queue),
                "loading_count": len(self.loader.loading_models)
            }
        }

    def unload_model(self, model_name: str, protocol_config: Dict[str, Any]) -> bool:
        """Unload a specific model from cache."""
        cache_key = ModelHelper.create_cache_key(model_name, protocol_config)
        
        with self.cache._lock:
            if cache_key in self.cache.models:
                self.cache._remove_model(cache_key)
                logger.info(f"Model unloaded: {model_name}")
                return True
            return False

    def clear_cache(self):
        """Clear all cached models."""
        self.cache.clear()
        logger.info("Model cache cleared")

    def shutdown(self):
        """Shutdown the model service."""
        self.clear_cache()
        self.loader._stop_event.set()
        logger.info("ModelService shutdown")

    def _setup_background_tasks(self):
        """Setup background maintenance tasks."""
        # This would setup periodic cleanup tasks
        pass

    def _preload_models(self):
        """Preload frequently used models."""
        # This would preload common models based on configuration
        pass