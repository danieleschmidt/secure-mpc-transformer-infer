"""Enhanced API Routes for Secure MPC Transformer System - Generation 1."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import asdict

try:
    from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback classes for when FastAPI is not available
    class APIRouter:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def delete(self, *args, **kwargs): return lambda f: f
    
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
    
    class JSONResponse:
        def __init__(self, *args, **kwargs): pass
    
    class BaseModel:
        pass
    
    class Field:
        @staticmethod
        def default(*args, **kwargs): return None
    
    def Depends(func): return func
    def Query(*args, **kwargs): return None
    
    class BackgroundTasks:
        def add_task(self, *args, **kwargs): pass
    
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request for model inference."""
    text: str = Field(..., description="Input text for inference")
    model_name: str = Field(default="bert-base-uncased", description="Model to use")
    max_length: Optional[int] = Field(default=512, description="Maximum sequence length")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")


class InferenceResponse(BaseModel):
    """Response from model inference."""
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    execution_time_ms: float
    timestamp: str


class ModelRequest(BaseModel):
    """Request to load/manage model."""
    model_name: str = Field(..., description="Model name")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model configuration")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: Dict[str, Any]
    system_info: Dict[str, Any]


class EnhancedInferenceRouter:
    """Enhanced inference router with improved error handling and features."""
    
    def __init__(self, inference_service=None, security_service=None, model_service=None):
        self.router = APIRouter()
        self.inference_service = inference_service
        self.security_service = security_service
        self.model_service = model_service
        
        # Setup routes
        self._setup_routes()
        
        # Metrics tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        
        logger.info("Enhanced InferenceRouter initialized")
    
    def _setup_routes(self):
        """Setup all inference routes."""
        
        @self.router.post("/inference", response_model=InferenceResponse if FASTAPI_AVAILABLE else dict)
        async def inference_endpoint(request: InferenceRequest, background_tasks: BackgroundTasks = None):
            """Enhanced inference endpoint with comprehensive error handling."""
            start_time = time.time()
            request_id = f"req_{int(time.time() * 1000)}"
            
            try:
                # Log request
                logger.info(f"[{request_id}] Inference request: model={request.model_name}, text_len={len(request.text)}")
                
                # Validate request
                self._validate_inference_request(request)
                
                # Security check if available
                if self.security_service:
                    security_result = await self._perform_security_check(request, request_id)
                    if not security_result["allowed"]:
                        raise HTTPException(status_code=403, detail=security_result["reason"])
                
                # Get or load model
                if self.model_service:
                    model = await self.model_service.get_model(request.model_name)
                    model_info = self.model_service.get_model_info(request.model_name)
                else:
                    # Fallback for testing
                    model = MockInferenceModel(request.model_name)
                    model_info = {"name": request.model_name, "status": "mock"}
                
                # Perform inference
                predictions = await self._run_inference(model, request, request_id)
                
                # Calculate metrics
                execution_time = (time.time() - start_time) * 1000
                self.request_count += 1
                self.total_latency += execution_time
                
                # Log success
                logger.info(f"[{request_id}] Inference completed in {execution_time:.2f}ms")
                
                # Background task for cleanup/metrics
                if background_tasks:
                    background_tasks.add_task(self._post_inference_cleanup, request_id, execution_time)
                
                response = {
                    "predictions": predictions,
                    "model_info": model_info or {"name": request.model_name},
                    "execution_time_ms": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                return JSONResponse(content=response) if FASTAPI_AVAILABLE else response
                
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                # Handle unexpected errors
                self.error_count += 1
                execution_time = (time.time() - start_time) * 1000
                
                logger.error(f"[{request_id}] Inference failed after {execution_time:.2f}ms: {e}")
                
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Inference failed",
                        "message": str(e),
                        "request_id": request_id,
                        "execution_time_ms": execution_time
                    }
                )
        
        @self.router.post("/inference/batch")
        async def batch_inference_endpoint(
            requests: List[InferenceRequest], 
            background_tasks: BackgroundTasks = None
        ):
            """Batch inference endpoint for multiple texts."""
            start_time = time.time()
            batch_id = f"batch_{int(time.time() * 1000)}"
            
            try:
                logger.info(f"[{batch_id}] Batch inference: {len(requests)} requests")
                
                # Validate batch size
                if len(requests) > 50:  # Configurable limit
                    raise HTTPException(status_code=400, detail="Batch size exceeds maximum (50)")
                
                # Process all requests concurrently
                tasks = []
                for i, req in enumerate(requests):
                    task = asyncio.create_task(self._process_single_inference(req, f"{batch_id}_{i}"))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Separate successful and failed results
                successful = []
                failed = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed.append({"index": i, "error": str(result)})
                    else:
                        successful.append({"index": i, "result": result})
                
                execution_time = (time.time() - start_time) * 1000
                
                logger.info(f"[{batch_id}] Batch completed: {len(successful)} success, {len(failed)} failed")
                
                return {
                    "batch_id": batch_id,
                    "successful": successful,
                    "failed": failed,
                    "execution_time_ms": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"[{batch_id}] Batch inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/inference/stats")
        async def inference_stats():
            """Get inference statistics."""
            avg_latency = self.total_latency / max(self.request_count, 1)
            
            return {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1),
                "average_latency_ms": avg_latency,
                "total_latency_ms": self.total_latency,
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_inference_request(self, request: InferenceRequest):
        """Validate inference request parameters."""
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        if len(request.text) > 10000:  # Configurable limit
            raise HTTPException(status_code=400, detail="Text input too long (max 10000 characters)")
        
        if request.max_length and (request.max_length < 1 or request.max_length > 2048):
            raise HTTPException(status_code=400, detail="Invalid max_length (must be 1-2048)")
    
    async def _perform_security_check(self, request: InferenceRequest, request_id: str) -> Dict[str, Any]:
        """Perform security validation on the request."""
        try:
            # Mock security check for now
            if self.security_service and hasattr(self.security_service, 'validate_input'):
                return await self.security_service.validate_input(request.text)
            
            # Basic security checks
            suspicious_patterns = ["<script>", "javascript:", "eval(", "exec("]
            for pattern in suspicious_patterns:
                if pattern.lower() in request.text.lower():
                    return {"allowed": False, "reason": f"Suspicious pattern detected: {pattern}"}
            
            return {"allowed": True, "reason": "Security check passed"}
            
        except Exception as e:
            logger.warning(f"[{request_id}] Security check failed: {e}")
            return {"allowed": True, "reason": "Security check bypassed due to error"}
    
    async def _run_inference(self, model: Any, request: InferenceRequest, request_id: str) -> List[Dict[str, Any]]:
        """Run model inference."""
        try:
            if hasattr(model, 'predict_async'):
                # Use async prediction if available
                result = await model.predict_async(request.text, **request.options)
            elif hasattr(model, 'predict'):
                # Use sync prediction in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, model.predict, request.text)
            else:
                # Fallback mock inference
                result = {
                    "text": request.text,
                    "predictions": [{"label": "MOCK", "confidence": 0.95}],
                    "model": request.model_name
                }
            
            # Ensure result is in expected format
            if isinstance(result, dict) and "predictions" in result:
                return result["predictions"]
            elif isinstance(result, list):
                return result
            else:
                return [{"output": str(result)}]
                
        except Exception as e:
            logger.error(f"[{request_id}] Model inference failed: {e}")
            raise
    
    async def _process_single_inference(self, request: InferenceRequest, request_id: str) -> Dict[str, Any]:
        """Process a single inference request (for batch processing)."""
        try:
            # Simplified version of main inference logic
            model = MockInferenceModel(request.model_name)
            predictions = await self._run_inference(model, request, request_id)
            
            return {
                "predictions": predictions,
                "model_name": request.model_name,
                "text_length": len(request.text)
            }
        except Exception as e:
            raise Exception(f"Inference failed: {str(e)}")
    
    async def _post_inference_cleanup(self, request_id: str, execution_time: float):
        """Background cleanup and metrics recording."""
        try:
            # Log metrics, cleanup temporary data, etc.
            logger.debug(f"[{request_id}] Post-inference cleanup completed")
        except Exception as e:
            logger.warning(f"[{request_id}] Cleanup failed: {e}")


class EnhancedModelRouter:
    """Enhanced model management router."""
    
    def __init__(self, model_service=None):
        self.router = APIRouter()
        self.model_service = model_service
        self._setup_routes()
        
        logger.info("Enhanced ModelRouter initialized")
    
    def _setup_routes(self):
        """Setup model management routes."""
        
        @self.router.get("/models")
        async def list_models():
            """List all available and cached models."""
            if self.model_service:
                return self.model_service.list_models()
            else:
                return {
                    "supported_models": ["bert-base-uncased", "roberta-base", "gpt2"],
                    "cache_stats": {"cached_models": 0, "max_models": 3}
                }
        
        @self.router.post("/models/load")
        async def load_model(request: ModelRequest):
            """Load a model into cache."""
            try:
                if self.model_service:
                    model = await self.model_service.load_model(request.model_name, **request.config)
                    info = self.model_service.get_model_info(request.model_name)
                    return {"message": "Model loaded successfully", "model_info": info}
                else:
                    return {"message": f"Mock model {request.model_name} loaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.delete("/models/{model_name}")
        async def unload_model(model_name: str):
            """Unload a model from cache."""
            try:
                if self.model_service:
                    success = await self.model_service.unload_model(model_name)
                    if success:
                        return {"message": f"Model {model_name} unloaded successfully"}
                    else:
                        raise HTTPException(status_code=404, detail="Model not found in cache")
                else:
                    return {"message": f"Mock model {model_name} unloaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/models/{model_name}/info")
        async def get_model_info(model_name: str):
            """Get information about a specific model."""
            if self.model_service:
                info = self.model_service.get_model_info(model_name)
                if info:
                    return info
                else:
                    raise HTTPException(status_code=404, detail="Model not found")
            else:
                return {"name": model_name, "status": "mock", "memory_usage": 100}


class EnhancedHealthRouter:
    """Enhanced health check router."""
    
    def __init__(self, services: Dict[str, Any] = None):
        self.router = APIRouter()
        self.services = services or {}
        self._setup_routes()
        
        logger.info("Enhanced HealthRouter initialized")
    
    def _setup_routes(self):
        """Setup health check routes."""
        
        @self.router.get("/health", response_model=HealthResponse if FASTAPI_AVAILABLE else dict)
        async def health_check():
            """Comprehensive health check."""
            start_time = time.time()
            
            # Check all services
            service_status = {}
            for name, service in self.services.items():
                try:
                    if hasattr(service, 'health_check'):
                        status = await service.health_check()
                    elif hasattr(service, 'list_models'):
                        # For model service, check if it's responsive
                        models = service.list_models()
                        status = "healthy"
                    else:
                        status = "healthy"  # Assume healthy if no check method
                    
                    service_status[name] = status
                except Exception as e:
                    service_status[name] = f"unhealthy: {str(e)}"
            
            # System information
            system_info = {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "health_check_duration_ms": (time.time() - start_time) * 1000
            }
            
            # Overall status
            overall_status = "healthy" if all(
                "unhealthy" not in str(status) for status in service_status.values()
            ) else "degraded"
            
            response = {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "services": service_status,
                "system_info": system_info
            }
            
            return JSONResponse(content=response) if FASTAPI_AVAILABLE else response
        
        @self.router.get("/health/ready")
        async def readiness_check():
            """Kubernetes-style readiness check."""
            # Check if critical services are ready
            critical_services = ["model_service"]
            
            for service_name in critical_services:
                if service_name in self.services:
                    service = self.services[service_name]
                    try:
                        if hasattr(service, 'list_models'):
                            service.list_models()  # Test basic functionality
                    except Exception as e:
                        raise HTTPException(status_code=503, detail=f"Service {service_name} not ready: {e}")
            
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
        @self.router.get("/health/live")
        async def liveness_check():
            """Kubernetes-style liveness check."""
            return {"status": "alive", "timestamp": datetime.now().isoformat()}


class MockInferenceModel:
    """Mock model for testing inference endpoints."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def predict_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Mock async prediction."""
        await asyncio.sleep(0.01)  # Simulate processing time
        return {
            "predictions": [
                {"label": "POSITIVE", "confidence": 0.85},
                {"label": "NEGATIVE", "confidence": 0.15}
            ],
            "text": text,
            "model": self.model_name
        }
    
    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """Mock sync prediction."""
        return {
            "predictions": [
                {"label": "POSITIVE", "confidence": 0.85},
                {"label": "NEGATIVE", "confidence": 0.15}
            ],
            "text": text,
            "model": self.model_name
        }


# Import fix for environments without some dependencies
import sys
if not FASTAPI_AVAILABLE:
    logger.warning("FastAPI not available - using mock implementations")