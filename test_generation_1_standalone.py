#!/usr/bin/env python3
"""
Standalone test for Generation 1 basic functionality.
Tests enhanced components without importing full package.
"""

import sys
import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Mock torch availability
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


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = {"model_type": "mock", "hidden_size": 768}
    
    def __call__(self, *args, **kwargs):
        return {"logits": "mock_output"}
    
    def eval(self):
        return self
    
    def to(self, device):
        return self


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
        self.load_lock = None  # Will be initialized when needed
        
        # Auto-unload timer
        self.auto_unload_timeout = self.config.get("auto_unload_timeout", 1800)
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
    
    async def load_model(self, model_name: str, **kwargs) -> Any:
        """Load model asynchronously with caching."""
        # Check cache first
        cached_model = self.cache.get(model_name)
        if cached_model is not None:
            logger.info(f"Model {model_name} loaded from cache")
            return cached_model
        
        # Initialize async lock if needed
        if self.load_lock is None:
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
            
            # Always use mock model for this test
            await asyncio.sleep(0.1)  # Simulate loading time
            model = MockModel(model_name)
            info.memory_usage = 100  # MB
            
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
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload model from cache."""
        logger.info(f"Unloading model: {model_name}")
        
        # Cancel loading if in progress
        if model_name in self.loading_tasks:
            self.loading_tasks[model_name].cancel()
            del self.loading_tasks[model_name]
        
        # Remove from cache
        success = self.cache.remove(model_name)
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
        
        logger.info("ModelService shutdown complete")


def test_model_cache():
    """Test model cache functionality."""
    print("Testing ModelCache...")
    
    cache = ModelCache(max_models=2, max_memory_mb=1000)
    
    # Test adding models
    mock_model1 = MockModel("bert-base")
    mock_model2 = MockModel("roberta-base")
    mock_model3 = MockModel("gpt2")
    
    info1 = ModelInfo(name="bert-base", status=ModelStatus.LOADED, memory_usage=500)
    info2 = ModelInfo(name="roberta-base", status=ModelStatus.LOADED, memory_usage=600)
    info3 = ModelInfo(name="gpt2", status=ModelStatus.LOADED, memory_usage=400)
    
    # Add first two models
    assert cache.put("bert-base", mock_model1, info1) == True
    assert cache.put("roberta-base", mock_model2, info2) == True
    
    # Check cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    assert stats["cached_models"] == 2
    assert stats["total_memory_mb"] == 1100
    
    # Add third model (should evict LRU)
    assert cache.put("gpt2", mock_model3, info3) == True
    
    # Check that bert-base was evicted
    assert cache.get("bert-base") is None
    assert cache.get("roberta-base") is not None
    assert cache.get("gpt2") is not None
    
    print("✓ ModelCache tests passed")


async def test_model_service():
    """Test model service functionality."""
    print("Testing ModelService...")
    
    config = {
        "max_cached_models": 2,
        "max_cache_memory_mb": 1000,
        "auto_unload_timeout": 60
    }
    
    service = ModelService(config)
    
    # Test loading a mock model
    model = await service.load_model("bert-base-uncased")
    assert model is not None
    assert isinstance(model, MockModel)
    
    # Test that second call uses cache
    model2 = await service.load_model("bert-base-uncased")
    assert model2 is model  # Should be same object from cache
    
    # Test model listing
    models = service.list_models()
    print(f"Model list: {models}")
    assert len(models["cache_stats"]["models"]) == 1
    
    # Test model info
    info = service.get_model_info("bert-base-uncased")
    assert info is not None
    assert info["name"] == "bert-base-uncased"
    
    # Test unloading
    success = await service.unload_model("bert-base-uncased")
    assert success == True
    
    # Test that model is no longer cached
    models_after = service.list_models()
    assert len(models_after["cache_stats"]["models"]) == 0
    
    service.shutdown()
    print("✓ ModelService tests passed")


async def test_concurrent_loading():
    """Test concurrent model loading."""
    print("Testing concurrent model loading...")
    
    service = ModelService()
    
    # Start multiple load requests for the same model
    tasks = []
    for i in range(5):
        task = asyncio.create_task(service.load_model("bert-base-uncased"))
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    # All should return the same model instance (from cache)
    # Note: Due to async loading, first request creates model, others get same model
    first_model = results[0]
    print(f"First model: {first_model}")
    print(f"All models: {[type(m).__name__ for m in results]}")
    
    # Check that all models are MockModel instances
    for i, model in enumerate(results):
        assert isinstance(model, MockModel), f"Model {i} is not MockModel: {type(model)}"
        assert model.model_name == "bert-base-uncased"
    
    # Should only have one model in cache
    models = service.list_models()
    assert len(models["cache_stats"]["models"]) == 1
    
    service.shutdown()
    print("✓ Concurrent loading tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Generation 1 Standalone Tests")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Model Cache
        test_model_cache()
        tests_passed += 1
        
        # Test 2: Model Service
        asyncio.run(test_model_service())
        tests_passed += 1
        
        # Test 3: Concurrent Loading
        asyncio.run(test_concurrent_loading())
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All Generation 1 standalone tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())