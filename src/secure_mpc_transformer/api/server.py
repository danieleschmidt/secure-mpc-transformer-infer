"""FastAPI server for secure MPC transformer service."""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    FastAPI = None
    uvicorn = None
    logger = logging.getLogger(__name__)
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

from ..services import InferenceService, SecurityService, ModelService
from ..database.connection import DatabaseManager
from ..utils.metrics import MetricsCollector
from .middleware import SecurityMiddleware, RateLimitMiddleware
from .routes import InferenceRouter, SecurityRouter, MetricsRouter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting secure MPC transformer service")
    
    # Initialize services
    try:
        await app.state.db_manager.initialize()
        logger.info("Database connection established")
        
        # Start background tasks
        app.state.metrics_collector.start_periodic_collection()
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(app.state, 'db_manager'):
            await app.state.db_manager.close()
        
        if hasattr(app.state, 'model_service'):
            app.state.model_service.shutdown()
        
        logger.info("Service shutdown complete")


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create FastAPI application with all routes and middleware."""
    if FastAPI is None:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    config = config or {}
    
    # Create FastAPI app
    app = FastAPI(
        title="Secure MPC Transformer",
        description="GPU-accelerated secure multi-party computation for transformer inference",
        version="0.1.0",
        docs_url="/docs" if config.get("enable_docs", True) else None,
        redoc_url="/redoc" if config.get("enable_docs", True) else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services
    db_manager = DatabaseManager.from_env()
    metrics_collector = MetricsCollector()
    
    inference_service = InferenceService(config.get("inference", {}))
    security_service = SecurityService(config.get("security", {}))
    model_service = ModelService(config.get("model", {}))
    
    # Store services in app state
    app.state.db_manager = db_manager
    app.state.metrics_collector = metrics_collector
    app.state.inference_service = inference_service
    app.state.security_service = security_service
    app.state.model_service = model_service
    app.state.config = config
    
    # Add custom middleware
    app.add_middleware(SecurityMiddleware, security_service=security_service)
    app.add_middleware(RateLimitMiddleware, config=config.get("rate_limiting", {}))
    
    # Initialize routers
    inference_router = InferenceRouter(inference_service, security_service)
    security_router = SecurityRouter(security_service)
    metrics_router = MetricsRouter(metrics_collector)
    
    # Add routes
    app.include_router(inference_router.router, prefix="/api/v1", tags=["inference"])
    app.include_router(security_router.router, prefix="/api/v1", tags=["security"])
    app.include_router(metrics_router.router, prefix="/api/v1", tags=["metrics"])
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Service health check."""
        try:
            db_health = await app.state.db_manager.health_check()
            inference_health = await app.state.inference_service.health_check()
            security_status = app.state.security_service.get_security_status()
            
            return {
                "status": "healthy",
                "timestamp": asyncio.get_event_loop().time(),
                "services": {
                    "database": db_health,
                    "inference": inference_health,
                    "security": security_status
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unhealthy")
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        # Log security event
        if hasattr(request.app.state, 'security_service'):
            request.app.state.security_service.auditor.log_security_event(
                event_type="SYSTEM_ERROR",
                severity="HIGH",
                source_ip=request.client.host if request.client else "unknown",
                details={"error": str(exc), "endpoint": str(request.url)}
            )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, 'request_id', 'unknown')
            }
        )
    
    logger.info("FastAPI application created successfully")
    return app


class APIServer:
    """API server wrapper for easier management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.app = create_app(config)
        self.server = None
        
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the API server."""
        if uvicorn is None:
            raise ImportError("uvicorn not available. Install with: pip install uvicorn")
        
        server_config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=self.config.get("log_level", "info"),
            access_log=self.config.get("access_log", True),
            loop="asyncio"
        )
        
        self.server = uvicorn.Server(server_config)
        
        logger.info(f"Starting API server on {host}:{port}")
        await self.server.serve()
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the API server (blocking)."""
        if uvicorn is None:
            raise ImportError("uvicorn not available. Install with: pip install uvicorn")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=self.config.get("log_level", "info"),
            access_log=self.config.get("access_log", True)
        )
    
    async def stop(self):
        """Stop the API server."""
        if self.server:
            self.server.should_exit = True
            logger.info("API server stopped")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app