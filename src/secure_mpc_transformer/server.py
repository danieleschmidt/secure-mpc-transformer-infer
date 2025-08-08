"""FastAPI server for secure MPC transformer service."""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = object
    HTTPException = Exception

from .services import InferenceService, SecurityService, ModelService
from .api.routes import InferenceRouter, SecurityRouter, MetricsRouter
from .utils.metrics import MetricsCollector
from .config import SecurityConfig, load_config_from_file
from .integration import QuantumMPCIntegrator

logger = logging.getLogger(__name__)


class SecureMPCServer:
    """Main server class for secure MPC transformer service."""
    
    def __init__(self, config_path: Optional[str] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        # Load configuration
        if config_path:
            self.config = load_config_from_file(config_path)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.security_config = SecurityConfig(**self.config.get("security", {}))
        self.metrics = MetricsCollector()
        
        # Initialize services  
        self.security_service = SecurityService(self.config.get("security_service", {}))
        self.model_service = ModelService(self.config.get("model_service", {}))
        self.inference_service = InferenceService(
            model_service=self.model_service,
            security_service=self.security_service,
            config=self.config.get("inference_service", {})
        )
        
        # Initialize quantum-MPC integration
        self.quantum_integrator = QuantumMPCIntegrator(
            security_config=self.security_config,
            scheduler_config=None  # Will use defaults
        )
        
        # FastAPI app with lifespan management
        self.app = None
        self._initialize_app()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default server configuration."""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 1,
                "reload": False,
                "log_level": "info"
            },
            "security": {
                "protocol_name": "aby3",
                "party_id": 0,
                "num_parties": 3,
                "security_level": 128,
                "gpu_acceleration": True
            },
            "security_service": {
                "enable_threat_detection": True,
                "enable_audit_logging": True,
                "enable_privacy_accounting": True,
                "privacy_epsilon_budget": 1.0,
                "privacy_delta": 1e-5
            },
            "model_service": {
                "max_cached_models": 3,
                "max_cache_memory_mb": 8000,
                "max_concurrent_loads": 2,
                "auto_unload_timeout": 1800  # 30 minutes
            },
            "inference_service": {
                "max_batch_size": 16,
                "default_timeout": 300,
                "enable_caching": True
            },
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            }
        }
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Manage application lifespan (startup/shutdown)."""
        # Startup
        logger.info("Starting Secure MPC Transformer Server...")
        
        try:
            # Initialize quantum scheduler
            await self._initialize_quantum_integration()
            
            # Start metrics collection
            self.metrics.start_periodic_collection(interval=30.0)
            
            # Log startup info
            self._log_startup_info()
            
            logger.info("Server startup complete")
            yield
            
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            raise
        finally:
            # Shutdown
            logger.info("Shutting down Secure MPC Transformer Server...")
            
            try:
                # Cleanup services
                self.model_service.shutdown()
                self.quantum_integrator.cleanup()
                
                logger.info("Server shutdown complete")
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    async def _initialize_quantum_integration(self):
        """Initialize quantum-MPC integration."""
        try:
            # Initialize a default transformer for the integrator
            model_name = self.config.get("default_model", "bert-base-uncased")
            self.quantum_integrator.initialize_transformer(
                model_name=model_name,
                **self.security_config.__dict__
            )
            
            logger.info(f"Quantum-MPC integration initialized with model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not initialize quantum integration: {e}")
            # Continue without quantum integration
    
    def _initialize_app(self):
        """Initialize FastAPI application."""
        self.app = FastAPI(
            title="Secure MPC Transformer API",
            description="GPU-accelerated secure multi-party computation for transformer inference with quantum-inspired task planning",
            version="1.0.0",
            lifespan=self.lifespan
        )
        
        # Add CORS middleware
        cors_config = self.config.get("cors", {})
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("allow_origins", ["*"]),
            allow_credentials=cors_config.get("allow_credentials", True),
            allow_methods=cors_config.get("allow_methods", ["*"]),
            allow_headers=cors_config.get("allow_headers", ["*"])
        )
        
        # Setup routes
        self._setup_routes()
        
        # Add exception handlers
        self._setup_exception_handlers()
    
    def _setup_routes(self):
        """Setup API routes."""
        # Initialize routers
        inference_router = InferenceRouter(
            inference_service=self.inference_service,
            security_service=self.security_service
        )
        security_router = SecurityRouter(security_service=self.security_service)
        metrics_router = MetricsRouter(metrics_collector=self.metrics)
        
        # Include routers
        self.app.include_router(
            inference_router.router, 
            prefix="/api/v1",
            tags=["Inference"]
        )
        self.app.include_router(
            security_router.router,
            prefix="/api/v1", 
            tags=["Security"]
        )
        self.app.include_router(
            metrics_router.router,
            prefix="/api/v1",
            tags=["Metrics"]
        )
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": self.metrics.get_counter_value("startup_timestamp") or 0,
                "services": {
                    "inference": "active",
                    "security": self.security_service.get_security_status()["service_status"],
                    "models": len(self.model_service.list_models()["cache_stats"]["models"]),
                    "quantum_integration": "active" if hasattr(self.quantum_integrator, 'transformer') and self.quantum_integrator.transformer else "inactive"
                }
            }
        
        # Root endpoint
        @self.app.get("/")
        async def root():
            """Root endpoint with service information."""
            return {
                "service": "Secure MPC Transformer API",
                "version": "1.0.0",
                "description": "GPU-accelerated secure multi-party computation for transformer inference with quantum-inspired task planning",
                "endpoints": {
                    "health": "/health",
                    "inference": "/api/v1/inference",
                    "security": "/api/v1/security/status",
                    "metrics": "/api/v1/metrics",
                    "models": "/api/v1/models"
                },
                "documentation": "/docs"
            }
        
        # Quantum integration endpoints
        @self.app.post("/api/v1/quantum/inference")
        async def quantum_inference(request_data: Dict[str, Any]):
            """Quantum-optimized inference endpoint."""
            try:
                text_inputs = request_data.get("text_inputs", [])
                if not text_inputs:
                    raise HTTPException(status_code=400, detail="text_inputs required")
                
                if not hasattr(self.quantum_integrator, 'transformer') or not self.quantum_integrator.transformer:
                    raise HTTPException(status_code=503, detail="Quantum integration not available")
                
                result = await self.quantum_integrator.quantum_inference(
                    text_inputs=text_inputs,
                    priority=request_data.get("priority", "medium"),
                    optimize_schedule=request_data.get("optimize_schedule", True)
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Quantum inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/quantum/status")
        async def quantum_status():
            """Get quantum integration status."""
            try:
                if hasattr(self.quantum_integrator, 'transformer') and self.quantum_integrator.transformer:
                    performance_summary = self.quantum_integrator.get_performance_summary()
                    active_workflows = self.quantum_integrator.get_active_workflows()
                    
                    return {
                        "status": "active",
                        "performance_summary": performance_summary,
                        "active_workflows": len(active_workflows),
                        "workflows": list(active_workflows.keys())
                    }
                else:
                    return {
                        "status": "inactive",
                        "message": "Quantum integration not initialized"
                    }
                    
            except Exception as e:
                return {
                    "status": "error", 
                    "message": str(e)
                }
    
    def _setup_exception_handlers(self):
        """Setup custom exception handlers."""
        
        @self.app.exception_handler(ValueError)
        async def value_error_handler(request, exc):
            logger.error(f"ValueError: {exc}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid input", "detail": str(exc)}
            )
        
        @self.app.exception_handler(TimeoutError)
        async def timeout_error_handler(request, exc):
            logger.error(f"TimeoutError: {exc}")
            return JSONResponse(
                status_code=408,
                content={"error": "Request timeout", "detail": str(exc)}
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": "An unexpected error occurred"}
            )
    
    def _log_startup_info(self):
        """Log server startup information."""
        server_config = self.config.get("server", {})
        
        logger.info("=" * 60)
        logger.info("Secure MPC Transformer Server")
        logger.info("=" * 60)
        logger.info(f"Host: {server_config.get('host', '0.0.0.0')}")
        logger.info(f"Port: {server_config.get('port', 8080)}")
        logger.info(f"Security Protocol: {self.security_config.protocol_name}")
        logger.info(f"Security Level: {self.security_config.security_level}")
        logger.info(f"GPU Acceleration: {self.security_config.gpu_acceleration}")
        logger.info(f"Max Cached Models: {self.model_service.cache.max_models}")
        logger.info(f"Quantum Integration: {'Active' if hasattr(self.quantum_integrator, 'transformer') and self.quantum_integrator.transformer else 'Inactive'}")
        logger.info("=" * 60)
        
        # Record startup timestamp
        import time
        self.metrics.set_gauge("startup_timestamp", time.time())
    
    def run(self):
        """Run the server."""
        server_config = self.config.get("server", {})
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        uvicorn.run(
            self.app,
            host=server_config.get("host", "0.0.0.0"),
            port=server_config.get("port", 8080),
            workers=server_config.get("workers", 1),
            reload=server_config.get("reload", False),
            log_level=server_config.get("log_level", "info")
        )


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application (for ASGI servers)."""
    server = SecureMPCServer(config_path)
    return server.app


def main():
    """Main entry point for server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Secure MPC Transformer Server")
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        server = SecureMPCServer(args.config)
        
        # Override config with CLI args if provided
        if args.host != "0.0.0.0":
            server.config.setdefault("server", {})["host"] = args.host
        if args.port != 8080:
            server.config.setdefault("server", {})["port"] = args.port
        if args.log_level != "info":
            server.config.setdefault("server", {})["log_level"] = args.log_level
        
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()