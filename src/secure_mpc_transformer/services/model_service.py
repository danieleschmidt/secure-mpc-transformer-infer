"""Enhanced Model Service for Secure MPC Transformer System."""

import asyncio
import logging
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

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
    load_time: float | None = None
    memory_usage: int | None = None  # in MB
    last_accessed: float | None = None
    error_message: str | None = None
    config: dict[str, Any] | None = None


class ModelCache:
    """Thread-safe model cache with LRU eviction."""

    def __init__(self, max_models: int = 3, max_memory_mb: int = 8000):
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.models: dict[str, Any] = {}
        self.model_info: dict[str, ModelInfo] = {}
        self.access_order: list[str] = []
        self.lock = threading.RLock()

    def get(self, model_name: str) -> Any | None:
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

    def get_stats(self) -> dict[str, Any]:
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

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.cache = ModelCache(
            max_models=self.config.get("max_cached_models", 3),
            max_memory_mb=self.config.get("max_cache_memory_mb", 8000)
        )

        # Loading state
        self.loading_tasks: dict[str, asyncio.Task] = {}
        self.load_lock = asyncio.Lock()

        # Auto-unload timer
        self.auto_unload_timeout = self.config.get("auto_unload_timeout", 1800)  # 30 min
        self.cleanup_task: asyncio.Task | None = None

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

    def list_models(self) -> dict[str, Any]:
        """List available and cached models."""
        cache_stats = self.cache.get_stats()

        return {
            "supported_models": list(self.supported_models),
            "cache_stats": cache_stats,
            "loading_models": list(self.loading_tasks.keys())
        }

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
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



    def put(self, cache_key: str, model: SecureTransformer, model_info: ModelInfo):
        """Add model to cache."""
        with self._lock:
            # Remove if already exists
            if cache_key in self.models:
                self._remove_model(cache_key)

            # Check if we need to make space
            while (len(self.models) >= self.max_models or
                   self._get_total_memory() + model_info.memory_usage_mb > self.max_memory_mb):
                if not self.access_order:
                    break

                lru_key = self.access_order[0]
                self._remove_model(lru_key)

            # Add new model
            self.models[cache_key] = model
            self.model_info[cache_key] = model_info
            self.access_order.append(cache_key)

            logger.info(f"Model cached: {cache_key}")

    def get(self, cache_key: str) -> SecureTransformer | None:
        """Get model from cache."""
        with self._lock:
            if cache_key in self.models:
                # Update access order
                self.access_order.remove(cache_key)
                self.access_order.append(cache_key)

                # Update last used time
                self.model_info[cache_key].last_used = time.time()
                self.model_info[cache_key].request_count += 1

                return self.models[cache_key]

            return None

    def remove(self, cache_key: str) -> bool:
        """Remove model from cache."""
        with self._lock:
            return self._remove_model(cache_key)

    def _remove_model(self, cache_key: str) -> bool:
        """Internal method to remove model."""
        if cache_key in self.models:
            # Cleanup GPU memory
            del self.models[cache_key]
            del self.model_info[cache_key]
            self.access_order.remove(cache_key)

            # Force garbage collection for GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Model removed from cache: {cache_key}")
            return True

        return False

    def _get_total_memory(self) -> float:
        """Get total memory usage of cached models."""
        return sum(info.memory_usage_mb for info in self.model_info.values())

    def get_cache_stats(self) -> dict[str, Any]:
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
        self.load_queue: list[ModelLoadRequest] = []
        self.loading_models: dict[str, ModelLoadRequest] = {}
        self.load_results: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Start loader threads
        self.loader_threads = []
        for i in range(max_concurrent_loads):
            thread = threading.Thread(target=self._loader_worker, name=f"ModelLoader-{i}")
            thread.daemon = True
            thread.start()
            self.loader_threads.append(thread)

        logger.info(f"ModelLoader started with {max_concurrent_loads} workers")

    def submit_load_request(self, request: ModelLoadRequest) -> str:
        """Submit a model load request."""
        cache_key = request.get_cache_key()

        with self._lock:
            # Check if already loading or in queue
            if cache_key in self.loading_models:
                return cache_key

            # Check if already in queue
            for queued_request in self.load_queue:
                if queued_request.get_cache_key() == cache_key:
                    return cache_key

            # Add to queue (sort by priority)
            self.load_queue.append(request)
            self.load_queue.sort(key=lambda x: x.priority, reverse=True)

        logger.info(f"Model load request queued: {cache_key}")
        return cache_key

    def get_load_status(self, cache_key: str) -> dict[str, Any]:
        """Get status of a load request."""
        with self._lock:
            if cache_key in self.loading_models:
                return {"status": "loading", "request": asdict(self.loading_models[cache_key])}

            if cache_key in self.load_results:
                return {"status": "completed", "result": self.load_results[cache_key]}

            # Check if in queue
            for request in self.load_queue:
                if request.get_cache_key() == cache_key:
                    return {"status": "queued", "request": asdict(request)}

        return {"status": "not_found"}

    def _loader_worker(self):
        """Worker thread for loading models."""
        while not self._stop_event.is_set():
            request = None

            # Get next request from queue
            with self._lock:
                if self.load_queue:
                    request = self.load_queue.pop(0)
                    cache_key = request.get_cache_key()
                    self.loading_models[cache_key] = request

            if request is None:
                time.sleep(0.1)
                continue

            # Load the model
            try:
                result = self._load_model(request)

                with self._lock:
                    self.load_results[cache_key] = result
                    del self.loading_models[cache_key]

                logger.info(f"Model loaded successfully: {cache_key}")

            except Exception as e:
                logger.error(f"Model loading failed: {cache_key}, error: {str(e)}")

                with self._lock:
                    self.load_results[cache_key] = {"success": False, "error": str(e)}
                    del self.loading_models[cache_key]

    def _load_model(self, request: ModelLoadRequest) -> dict[str, Any]:
        """Load a model (blocking operation)."""
        start_time = time.time()

        try:
            # Create configuration
            config = TransformerConfig.from_pretrained(
                request.model_name,
                **request.protocol_config
            )

            # Load model
            model = SecureTransformer.from_pretrained(request.model_name, **request.protocol_config)

            # Estimate memory usage
            memory_estimate = ModelHelper.estimate_model_memory(asdict(config))

            # Create model info
            model_info = ModelInfo(
                model_id=request.get_cache_key(),
                model_name=request.model_name,
                config=config,
                load_timestamp=time.time(),
                last_used=time.time(),
                memory_usage_mb=memory_estimate["total_memory_mb"],
                request_count=0,
                status="ready"
            )

            load_time = time.time() - start_time

            return {
                "success": True,
                "model": model,
                "model_info": model_info,
                "load_time_ms": load_time * 1000,
                "memory_estimate": memory_estimate
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "load_time_ms": (time.time() - start_time) * 1000
            }

    def stop(self):
        """Stop the loader threads."""
        self._stop_event.set()
        for thread in self.loader_threads:
            thread.join(timeout=5)


class ModelService:
    """Main service for model management."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Initialize components
        self.cache = ModelCache(
            max_models=self.config.get('max_cached_models', 5),
            max_memory_mb=self.config.get('max_cache_memory_mb', 16000)
        )

        self.loader = ModelLoader(
            max_concurrent_loads=self.config.get('max_concurrent_loads', 2)
        )

        self.metrics = MetricsCollector()

        # Service configuration
        self.auto_unload_timeout = self.config.get('auto_unload_timeout', 3600)  # 1 hour
        self.preload_models = self.config.get('preload_models', [])

        # Start background tasks
        self._start_background_tasks()

        # Preload models if configured
        self._preload_models()

        logger.info("ModelService initialized")

    async def get_model(self, model_name: str, protocol_config: dict[str, Any]) -> SecureTransformer:
        """Get model, loading if necessary."""
        request = ModelLoadRequest(
            model_name=model_name,
            protocol_config=protocol_config
        )
        cache_key = request.get_cache_key()

        # Check cache first
        model = self.cache.get(cache_key)
        if model is not None:
            self.metrics.increment_counter("model_cache_hits")
            return model

        self.metrics.increment_counter("model_cache_misses")

        # Check if already loading
        load_status = self.loader.get_load_status(cache_key)

        if load_status["status"] == "not_found":
            # Submit new load request
            self.loader.submit_load_request(request)
            self.metrics.increment_counter("model_load_requests")

        # Wait for loading to complete
        max_wait_time = 300  # 5 minutes
        wait_interval = 0.5
        waited = 0

        while waited < max_wait_time:
            load_status = self.loader.get_load_status(cache_key)

            if load_status["status"] == "completed":
                result = load_status["result"]

                if result["success"]:
                    # Add to cache
                    self.cache.put(cache_key, result["model"], result["model_info"])

                    # Update metrics
                    self.metrics.observe_histogram("model_load_time_ms", result["load_time_ms"])
                    self.metrics.increment_counter("model_loads_success")

                    return result["model"]
                else:
                    self.metrics.increment_counter("model_loads_failed")
                    raise RuntimeError(f"Model loading failed: {result['error']}")

            elif load_status["status"] == "loading" or load_status["status"] == "queued":
                # Still loading, wait
                await asyncio.sleep(wait_interval)
                waited += wait_interval
            else:
                # Unknown status
                break

        raise TimeoutError(f"Model loading timeout: {cache_key}")

    def get_model_info(self, model_name: str, protocol_config: dict[str, Any]) -> ModelInfo | None:
        """Get information about a model."""
        request = ModelLoadRequest(model_name=model_name, protocol_config=protocol_config)
        cache_key = request.get_cache_key()

        if cache_key in self.cache.model_info:
            return self.cache.model_info[cache_key]

        return None

    def unload_model(self, model_name: str, protocol_config: dict[str, Any]) -> bool:
        """Unload a specific model."""
        request = ModelLoadRequest(model_name=model_name, protocol_config=protocol_config)
        cache_key = request.get_cache_key()

        success = self.cache.remove(cache_key)

        if success:
            self.metrics.increment_counter("model_unloads")
            logger.info(f"Model unloaded: {cache_key}")

        return success

    def list_models(self) -> dict[str, Any]:
        """List all loaded models."""
        cache_stats = self.cache.get_cache_stats()

        # Add loader queue information
        with self.loader._lock:
            queue_info = {
                "queued_requests": len(self.loader.load_queue),
                "loading_requests": len(self.loader.loading_models),
                "completed_requests": len(self.loader.load_results)
            }

        return {
            "cache_stats": cache_stats,
            "loader_stats": queue_info,
            "service_config": {
                "max_cached_models": self.cache.max_models,
                "max_cache_memory_mb": self.cache.max_memory_mb,
                "auto_unload_timeout": self.auto_unload_timeout
            }
        }

    def get_service_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            "timestamp": time.time(),
            "models": self.list_models(),
            "metrics": {
                "cache_hits": self.metrics.get_counter_value("model_cache_hits"),
                "cache_misses": self.metrics.get_counter_value("model_cache_misses"),
                "load_requests": self.metrics.get_counter_value("model_load_requests"),
                "loads_success": self.metrics.get_counter_value("model_loads_success"),
                "loads_failed": self.metrics.get_counter_value("model_loads_failed"),
                "unloads": self.metrics.get_counter_value("model_unloads")
            }
        }

    def _preload_models(self):
        """Preload configured models."""
        for model_config in self.preload_models:
            try:
                request = ModelLoadRequest(
                    model_name=model_config["model_name"],
                    protocol_config=model_config.get("protocol_config", {}),
                    priority=10  # High priority for preloads
                )

                self.loader.submit_load_request(request)
                logger.info(f"Preload request submitted: {model_config['model_name']}")

            except Exception as e:
                logger.error(f"Failed to submit preload request: {e}")

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_old_models()
                    self._update_memory_metrics()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_old_models(self):
        """Remove models that haven't been used recently."""
        current_time = time.time()
        models_to_remove = []

        for cache_key, model_info in self.cache.model_info.items():
            if current_time - model_info.last_used > self.auto_unload_timeout:
                models_to_remove.append(cache_key)

        for cache_key in models_to_remove:
            self.cache.remove(cache_key)
            logger.info(f"Auto-unloaded inactive model: {cache_key}")

    def _update_memory_metrics(self):
        """Update memory usage metrics."""
        cache_stats = self.cache.get_cache_stats()

        self.metrics.set_gauge("loaded_models", cache_stats["total_models"])
        self.metrics.set_gauge("model_cache_memory_mb", cache_stats["total_memory_mb"])
        self.metrics.set_gauge("model_cache_utilization", cache_stats["memory_utilization"])

    def shutdown(self):
        """Shutdown the model service."""
        logger.info("Shutting down ModelService")

        # Stop loader
        self.loader.stop()

        # Clear cache
        self.cache.clear()

        logger.info("ModelService shutdown complete")
