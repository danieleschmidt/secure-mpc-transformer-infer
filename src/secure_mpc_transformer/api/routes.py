"""API route handlers for secure MPC transformer service."""

import logging
import time
from typing import Any

try:
    from fastapi import APIRouter, Depends, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:
    APIRouter = object
    BaseModel = object
    HTTPException = None
    Depends = None
    Field = None

from ..services import InferenceService, SecurityService
from ..utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


# Pydantic models for request/response
if BaseModel != object:
    class InferenceRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10000, description="Input text for inference")
        model_name: str = Field(default="bert-base-uncased", description="Model name to use")
        max_length: int | None = Field(default=512, ge=1, le=2048, description="Maximum sequence length")
        return_attention: bool = Field(default=False, description="Return attention weights")
        return_hidden_states: bool = Field(default=False, description="Return hidden states")
        protocol_config: dict[str, Any] | None = Field(default=None, description="MPC protocol configuration")

    class InferenceResponse(BaseModel):
        request_id: str
        output_tensor: list[list[float]]
        latency_ms: float
        protocol_info: dict[str, Any]
        input_shape: list[int]
        output_shape: list[int]
        security_metrics: dict[str, Any] | None = None

    class BatchInferenceRequest(BaseModel):
        requests: list[InferenceRequest] = Field(..., min_items=1, max_items=10)

    class SecurityStatusResponse(BaseModel):
        timestamp: float
        service_status: str
        threat_detection_enabled: bool
        audit_logging_enabled: bool
        privacy_accounting_enabled: bool
        security_summary: dict[str, Any] | None = None
        privacy_summary: dict[str, Any] | None = None
        security_metrics: dict[str, Any]

    class MetricsResponse(BaseModel):
        timestamp: float
        counters: dict[str, float]
        gauges: dict[str, float]
        histograms: dict[str, dict[str, Any]]


class InferenceRouter:
    """Router for inference endpoints."""

    def __init__(self, inference_service: InferenceService, security_service: SecurityService):
        self.inference_service = inference_service
        self.security_service = security_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup inference routes."""

        @self.router.post("/inference", response_model=InferenceResponse if BaseModel != object else None)
        async def predict(request: InferenceRequest):
            """Perform secure inference on input text."""
            try:
                # Convert to service request format
                from ..services.inference_service import (
                    InferenceRequest as ServiceRequest,
                )

                service_request = ServiceRequest(
                    text=request.text,
                    model_name=request.model_name,
                    max_length=request.max_length,
                    return_attention=request.return_attention,
                    return_hidden_states=request.return_hidden_states,
                    protocol_config=request.protocol_config or {}
                )

                # Execute inference
                result = await self.inference_service.predict(service_request)

                return InferenceResponse(
                    request_id=result.request_id,
                    output_tensor=result.output_tensor.detach().cpu().numpy().tolist(),
                    latency_ms=result.latency_ms,
                    protocol_info=result.protocol_info,
                    input_shape=list(result.input_shape),
                    output_shape=list(result.output_shape),
                    security_metrics=result.security_metrics
                )

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

        @self.router.post("/inference/batch")
        async def predict_batch(request: BatchInferenceRequest):
            """Perform batch inference on multiple inputs."""
            try:
                # Convert to service requests
                from ..services.inference_service import (
                    InferenceRequest as ServiceRequest,
                )

                service_requests = [
                    ServiceRequest(
                        text=req.text,
                        model_name=req.model_name,
                        max_length=req.max_length,
                        return_attention=req.return_attention,
                        return_hidden_states=req.return_hidden_states,
                        protocol_config=req.protocol_config or {}
                    )
                    for req in request.requests
                ]

                # Execute batch inference
                results = await self.inference_service.predict_batch(service_requests)

                return {
                    "batch_id": f"batch_{int(time.time() * 1000)}",
                    "results": [
                        {
                            "request_id": result.request_id,
                            "output_tensor": result.output_tensor.detach().cpu().numpy().tolist() if hasattr(result.output_tensor, 'detach') else [],
                            "latency_ms": result.latency_ms,
                            "protocol_info": result.protocol_info,
                            "input_shape": list(result.input_shape),
                            "output_shape": list(result.output_shape),
                            "security_metrics": result.security_metrics
                        }
                        for result in results
                    ]
                }

            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

        @self.router.get("/inference/stats")
        async def get_inference_stats():
            """Get inference service statistics."""
            try:
                return self.inference_service.get_service_stats()
            except Exception as e:
                logger.error(f"Failed to get inference stats: {e}")
                raise HTTPException(status_code=500, detail="Failed to get stats")

        @self.router.get("/models")
        async def list_models():
            """List available models."""
            try:
                # This would typically fetch from model service
                return {
                    "available_models": [
                        {
                            "name": "bert-base-uncased",
                            "description": "BERT base model, 12-layer, 768-hidden, 12-heads, 110M parameters",
                            "supported": True
                        },
                        {
                            "name": "bert-large-uncased",
                            "description": "BERT large model, 24-layer, 1024-hidden, 16-heads, 340M parameters",
                            "supported": True
                        },
                        {
                            "name": "roberta-base",
                            "description": "RoBERTa base model, 12-layer, 768-hidden, 12-heads, 125M parameters",
                            "supported": True
                        },
                        {
                            "name": "distilbert-base-uncased",
                            "description": "DistilBERT base model, 6-layer, 768-hidden, 12-heads, 66M parameters",
                            "supported": True
                        }
                    ]
                }
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail="Failed to list models")


class SecurityRouter:
    """Router for security endpoints."""

    def __init__(self, security_service: SecurityService):
        self.security_service = security_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup security routes."""

        @self.router.get("/security/status", response_model=SecurityStatusResponse if BaseModel != object else None)
        async def get_security_status():
            """Get current security status."""
            try:
                status = self.security_service.get_security_status()

                if BaseModel != object:
                    return SecurityStatusResponse(**status)
                else:
                    return status

            except Exception as e:
                logger.error(f"Failed to get security status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get security status")

        @self.router.get("/security/report")
        async def get_security_report(hours: int = 24):
            """Generate security report for specified time period."""
            try:
                if not (1 <= hours <= 168):  # 1 hour to 1 week
                    raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")

                report = self.security_service.generate_security_report(hours)
                return report

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to generate security report: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate security report")

        @self.router.get("/security/events")
        async def get_security_events(hours: int = 24, risk_level: str | None = None):
            """Get security events."""
            try:
                if not (1 <= hours <= 168):
                    raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")

                if risk_level and risk_level not in ["low", "medium", "high", "critical"]:
                    raise HTTPException(status_code=400, detail="Invalid risk level")

                # Get events from auditor
                if risk_level:
                    events = await self.security_service.auditor.get_by_risk_level(risk_level)
                else:
                    events = await self.security_service.auditor.get_recent_logs(hours)

                return {
                    "events": [event.to_dict() for event in events],
                    "total_count": len(events),
                    "time_period_hours": hours,
                    "risk_level_filter": risk_level
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get security events: {e}")
                raise HTTPException(status_code=500, detail="Failed to get security events")

        @self.router.get("/security/privacy")
        async def get_privacy_status():
            """Get privacy accounting status."""
            try:
                return self.security_service.privacy_accountant.get_privacy_summary()
            except Exception as e:
                logger.error(f"Failed to get privacy status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get privacy status")

        @self.router.post("/security/emergency-lockdown")
        async def emergency_lockdown(reason: str):
            """Trigger emergency security lockdown."""
            try:
                if not reason or len(reason.strip()) < 10:
                    raise HTTPException(status_code=400, detail="Reason must be at least 10 characters")

                self.security_service.emergency_lockdown(reason)

                return {
                    "status": "lockdown_initiated",
                    "reason": reason,
                    "timestamp": time.time()
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to initiate emergency lockdown: {e}")
                raise HTTPException(status_code=500, detail="Failed to initiate lockdown")


class MetricsRouter:
    """Router for metrics endpoints."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup metrics routes."""

        @self.router.get("/metrics", response_model=MetricsResponse if BaseModel != object else None)
        async def get_metrics():
            """Get all metrics in JSON format."""
            try:
                metrics = self.metrics_collector.get_all_metrics()

                if BaseModel != object:
                    return MetricsResponse(
                        timestamp=time.time(),
                        counters=metrics.get("counters", {}),
                        gauges=metrics.get("gauges", {}),
                        histograms=metrics.get("histograms", {})
                    )
                else:
                    return {
                        "timestamp": time.time(),
                        **metrics
                    }

            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get metrics")

        @self.router.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Get metrics in Prometheus format."""
            try:
                prometheus_metrics = self.metrics_collector.export_prometheus_format()

                # Return as plain text
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(
                    content=prometheus_metrics,
                    media_type="text/plain; version=0.0.4; charset=utf-8"
                )

            except Exception as e:
                logger.error(f"Failed to get Prometheus metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to get Prometheus metrics")

        @self.router.get("/metrics/counters")
        async def get_counters():
            """Get counter metrics only."""
            try:
                metrics = self.metrics_collector.get_all_metrics()
                return {
                    "timestamp": time.time(),
                    "counters": metrics.get("counters", {})
                }
            except Exception as e:
                logger.error(f"Failed to get counters: {e}")
                raise HTTPException(status_code=500, detail="Failed to get counters")

        @self.router.get("/metrics/gauges")
        async def get_gauges():
            """Get gauge metrics only."""
            try:
                metrics = self.metrics_collector.get_all_metrics()
                return {
                    "timestamp": time.time(),
                    "gauges": metrics.get("gauges", {})
                }
            except Exception as e:
                logger.error(f"Failed to get gauges: {e}")
                raise HTTPException(status_code=500, detail="Failed to get gauges")

        @self.router.get("/metrics/histograms")
        async def get_histograms():
            """Get histogram metrics only."""
            try:
                metrics = self.metrics_collector.get_all_metrics()
                return {
                    "timestamp": time.time(),
                    "histograms": metrics.get("histograms", {})
                }
            except Exception as e:
                logger.error(f"Failed to get histograms: {e}")
                raise HTTPException(status_code=500, detail="Failed to get histograms")

        @self.router.post("/metrics/reset")
        async def reset_metrics():
            """Reset all metrics (admin operation)."""
            try:
                self.metrics_collector.reset_all_metrics()
                return {
                    "status": "metrics_reset",
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"Failed to reset metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to reset metrics")
