"""Enhanced Model Service for Secure MPC Transformer System - Generation 1 Implementation."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import threading
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import torch
    import transformers
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model loading status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    status: ModelStatus
    load_time: Optional[float] = None
    memory_usage: Optional[int] = None  # in MB
    last_accessed: Optional[float] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, max_models: int = 3, max_memory_mb: int = 8000):
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_name in self.models:
                self._update_access_time(model_name)
                return self.models[model_name]
            return None
    
    def put(self, model_name: str, model: Any, info: ModelInfo) -> bool:
        """Put model in cache, evicting if necessary."""
        with self.lock:
            # Update access time
            info.last_accessed = time.time()
            
            # Check if we need to evict
            while len(self.models) >= self.max_models:
                if not self._evict_lru():
                    logger.warning("Could not evict models for new model")
                    return False
            
            # Store model and info
            self.models[model_name] = model
            self.model_info[model_name] = info
            
            # Update access order
            if model_name in self.access_order:
                self.access_order.remove(model_name)
            self.access_order.append(model_name)
            
            logger.info(f"Cached model: {model_name}")
            return True
    
    def remove(self, model_name: str) -> bool:
        """Remove model from cache."""
        with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_info[model_name]
                if model_name in self.access_order:
                    self.access_order.remove(model_name)
                logger.info(f"Removed model from cache: {model_name}")
                return True
            return False
    
    def _update_access_time(self, model_name: str):
        """Update access time and order."""
        if model_name in self.model_info:
            self.model_info[model_name].last_accessed = time.time()
        
        if model_name in self.access_order:
            self.access_order.remove(model_name)
        self.access_order.append(model_name)
    
    def _evict_lru(self) -> bool:
        """Evict least recently used model."""
        if not self.access_order:
            return False
        
        lru_model = self.access_order[0]
        return self.remove(lru_model)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_memory = sum(
                info.memory_usage or 0 
                for info in self.model_info.values()
            )
            
            return {
                "cached_models": len(self.models),
                "max_models": self.max_models,
                "total_memory_mb": total_memory,
                "max_memory_mb": self.max_memory_mb,
                "models": {
                    name: asdict(info) 
                    for name, info in self.model_info.items()
                }
            }


class ModelService:
    """Enhanced model service with caching and async loading."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache = ModelCache(
            max_models=self.config.get("max_cached_models", 3),
            max_memory_mb=self.config.get("max_cache_memory_mb", 8000)
        )
        
        # Loading state
        self.loading_tasks: Dict[str, asyncio.Task] = {}
        self.load_lock = asyncio.Lock()
        
        # Auto-unload timer
        self.auto_unload_timeout = self.config.get("auto_unload_timeout", 1800)  # 30 min
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Supported model types
        self.supported_models = {
            "bert-base-uncased",
            "bert-large-uncased", 
            "roberta-base",
            "roberta-large",
            "distilbert-base-uncased",
            "gpt2",
            "gpt2-medium",
            "gpt2-large"
        }
        
        logger.info("ModelService initialized")
        
        # Start cleanup task if event loop is running
        try:
            loop = asyncio.get_running_loop()
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop running yet, will start cleanup later
            pass
    
    async def load_model(self, model_name: str, **kwargs) -> Any:
        """Load model asynchronously with caching."""
        # Check cache first
        cached_model = self.cache.get(model_name)
        if cached_model is not None:
            logger.info(f"Model {model_name} loaded from cache")
            return cached_model
        
        # Initialize async lock if needed
        if not hasattr(self, 'load_lock') or self.load_lock is None:
            self.load_lock = asyncio.Lock()
        
        # Check if already loading
        async with self.load_lock:
            if model_name in self.loading_tasks:
                logger.info(f"Model {model_name} already loading, waiting...")
                return await self.loading_tasks[model_name]
            
            # Start loading
            task = asyncio.create_task(self._load_model_impl(model_name, **kwargs))
            self.loading_tasks[model_name] = task
            
            try:
                result = await task
                return result
            finally:
                # Clean up loading task
                if model_name in self.loading_tasks:
                    del self.loading_tasks[model_name]
    
    async def _load_model_impl(self, model_name: str, **kwargs) -> Any:
        """Implementation of model loading."""
        start_time = time.time()
        
        # Create model info
        info = ModelInfo(
            name=model_name,
            status=ModelStatus.LOADING,
            config=kwargs
        )
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            if not TORCH_AVAILABLE:
                # Fallback to mock model for testing
                await asyncio.sleep(1)  # Simulate loading time
                model = MockModel(model_name)
                info.memory_usage = 100  # MB
            else:
                # Load actual model
                model = await self._load_transformer_model(model_name, **kwargs)
                info.memory_usage = self._estimate_model_memory(model)
            
            # Update info
            info.status = ModelStatus.LOADED
            info.load_time = time.time() - start_time
            
            # Cache model
            success = self.cache.put(model_name, model, info)
            if not success:
                logger.warning(f"Failed to cache model: {model_name}")
            
            logger.info(f"Model {model_name} loaded in {info.load_time:.2f}s")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            info.status = ModelStatus.ERROR
            info.error_message = str(e)
            raise
    
    async def _load_transformer_model(self, model_name: str, **kwargs) -> Any:
        """Load actual transformer model."""
        def _load():
            try:
                # Load tokenizer and model
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                model = transformers.AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
                
                if torch.cuda.is_available():
                    model = model.cuda()
                
                return {"model": model, "tokenizer": tokenizer}
                
            except Exception as e:
                logger.error(f"Error loading transformer model: {e}")
                raise
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load)
    
    def _estimate_model_memory(self, model: Any) -> int:
        """Estimate model memory usage in MB."""
        if not TORCH_AVAILABLE:
            return 100  # Default estimate
        
        try:
            if isinstance(model, dict) and "model" in model:
                torch_model = model["model"]
                if hasattr(torch_model, 'parameters'):
                    total_params = sum(p.numel() for p in torch_model.parameters())
                    # Rough estimate: 4 bytes per parameter for float32, 2 for float16
                    bytes_per_param = 2 if torch_model.dtype == torch.float16 else 4
                    total_bytes = total_params * bytes_per_param
                    return int(total_bytes / (1024 * 1024))  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not estimate model memory: {e}")
        
        return 500  # Default estimate in MB
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload model from cache."""
        logger.info(f"Unloading model: {model_name}")
        
        # Cancel loading if in progress
        if model_name in self.loading_tasks:
            self.loading_tasks[model_name].cancel()
            del self.loading_tasks[model_name]
        
        # Remove from cache
        success = self.cache.remove(model_name)
        
        # Force garbage collection if using torch
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return success
    
    async def get_model(self, model_name: str, **kwargs) -> Any:
        """Get model, loading if necessary."""
        return await self.load_model(model_name, **kwargs)
    
    def list_models(self) -> Dict[str, Any]:
        """List available and cached models."""
        cache_stats = self.cache.get_stats()
        
        return {
            "supported_models": list(self.supported_models),
            "cache_stats": cache_stats,
            "loading_models": list(self.loading_tasks.keys())
        }
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        with self.cache.lock:
            if model_name in self.cache.model_info:
                return asdict(self.cache.model_info[model_name])
        return None
    
    async def _cleanup_loop(self):
        """Periodic cleanup of unused models."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                current_time = time.time()
                
                models_to_unload = []
                with self.cache.lock:
                    for name, info in self.cache.model_info.items():
                        if (info.last_accessed and 
                            current_time - info.last_accessed > self.auto_unload_timeout):
                            models_to_unload.append(name)
                
                for model_name in models_to_unload:
                    logger.info(f"Auto-unloading unused model: {model_name}")
                    await self.unload_model(model_name)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def shutdown(self):
        """Shutdown the model service."""
        logger.info("Shutting down ModelService")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Cancel all loading tasks
        for task in self.loading_tasks.values():
            task.cancel()
        self.loading_tasks.clear()
        
        # Clear cache
        with self.cache.lock:
            self.cache.models.clear()
            self.cache.model_info.clear()
            self.cache.access_order.clear()
        
        # Free GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ModelService shutdown complete")


class MockModel:
    """Mock model for testing when torch is not available."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = {"model_type": "mock", "hidden_size": 768}
    
    def __call__(self, *args, **kwargs):
        return {"logits": "mock_output"}
    
    def eval(self):
        return self
    
    def to(self, device):
        return self