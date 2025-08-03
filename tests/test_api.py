"""Tests for API endpoints and middleware."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    TestClient = None
    FastAPI = None
    FASTAPI_AVAILABLE = False

from src.secure_mpc_transformer.api.server import create_app
from src.secure_mpc_transformer.services.inference_service import InferenceResult
from src.secure_mpc_transformer.services.security_service import SecurityService


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock all services."""
        services = {
            'inference_service': Mock(),
            'security_service': Mock(spec=SecurityService),
            'model_service': Mock(),
            'db_manager': Mock(),
            'metrics_collector': Mock()
        }
        
        # Setup common mock responses
        services['db_manager'].health_check = AsyncMock(return_value={
            "status": "healthy",
            "response_time_ms": 5.0
        })
        
        services['inference_service'].health_check = AsyncMock(return_value={
            "status": "healthy",
            "health_check_latency_ms": 100.0
        })
        
        services['security_service'].get_security_status.return_value = {
            "status": "active",
            "threat_detection_enabled": True,
            "audit_logging_enabled": True,
            "privacy_accounting_enabled": True
        }
        
        services['security_service'].validate_request.return_value = (True, [])
        
        return services
    
    @pytest.fixture
    def test_app(self, mock_services):
        """Create test FastAPI app with mocked services."""
        config = {
            "enable_docs": True,
            "cors_origins": ["*"]
        }
        
        app = create_app(config)
        
        # Replace services with mocks
        for name, service in mock_services.items():
            setattr(app.state, name, service)
        
        return app
    
    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        assert "database" in data["services"]
        assert "inference" in data["services"]
        assert "security" in data["services"]
    
    def test_inference_endpoint_success(self, client, test_app):
        """Test successful inference request."""
        # Mock inference result
        mock_result = InferenceResult(
            request_id="test_123",
            input_text="test input",
            model_name="bert-base-uncased",
            output_tensor=Mock(),
            secure_output=Mock(),
            latency_ms=150.0,
            protocol_info={"protocol_name": "test"},
            input_shape=(1, 5),
            output_shape=(1, 768)
        )
        
        # Setup mock to return tensor with detach/cpu/numpy methods
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [[1.0, 2.0, 3.0]]
        mock_result.output_tensor = mock_tensor
        
        test_app.state.inference_service.predict = AsyncMock(return_value=mock_result)
        
        request_data = {
            "text": "Hello world",
            "model_name": "bert-base-uncased",
            "max_length": 512
        }
        
        response = client.post("/api/v1/inference", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["request_id"] == "test_123"
        assert data["latency_ms"] == 150.0
        assert data["output_tensor"] == [[1.0, 2.0, 3.0]]
        assert "protocol_info" in data
    
    def test_inference_endpoint_validation_error(self, client, test_app):
        """Test inference with validation error."""
        request_data = {
            "text": "",  # Empty text should fail validation
            "model_name": "bert-base-uncased"
        }
        
        response = client.post("/api/v1/inference", json=request_data)
        
        # Should return validation error (422 for Pydantic validation)
        assert response.status_code == 422
    
    def test_batch_inference_endpoint(self, client, test_app):
        """Test batch inference endpoint."""
        # Mock inference results
        mock_results = [
            InferenceResult(
                request_id=f"test_{i}",
                input_text=f"input {i}",
                model_name="bert-base-uncased",
                output_tensor=Mock(),
                secure_output=Mock(),
                latency_ms=100.0 + i * 10,
                protocol_info={"protocol_name": "test"},
                input_shape=(1, 5),
                output_shape=(1, 768)
            )
            for i in range(2)
        ]
        
        # Setup mock tensors
        for i, result in enumerate(mock_results):
            mock_tensor = Mock()
            mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [[float(i)]]
            result.output_tensor = mock_tensor
        
        test_app.state.inference_service.predict_batch = AsyncMock(return_value=mock_results)
        
        request_data = {
            "requests": [
                {"text": "Hello world 1", "model_name": "bert-base-uncased"},
                {"text": "Hello world 2", "model_name": "bert-base-uncased"}
            ]
        }
        
        response = client.post("/api/v1/inference/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "batch_id" in data
        assert "results" in data
        assert len(data["results"]) == 2
    
    def test_inference_stats_endpoint(self, client, test_app):
        """Test inference stats endpoint."""
        mock_stats = {
            "active_requests": 2,
            "total_requests": 100,
            "cache_hits": 80,
            "cache_misses": 20
        }
        
        test_app.state.inference_service.get_service_stats.return_value = mock_stats
        
        response = client.get("/api/v1/inference/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["active_requests"] == 2
        assert data["total_requests"] == 100
    
    def test_security_status_endpoint(self, client, test_app):
        """Test security status endpoint."""
        mock_status = {
            "timestamp": 1234567890.0,
            "service_status": "active",
            "threat_detection_enabled": True,
            "audit_logging_enabled": True,
            "privacy_accounting_enabled": True,
            "security_metrics": {"validations_total": 100}
        }
        
        test_app.state.security_service.get_security_status.return_value = mock_status
        
        response = client.get("/api/v1/security/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service_status"] == "active"
        assert data["threat_detection_enabled"] is True
    
    def test_security_report_endpoint(self, client, test_app):
        """Test security report endpoint."""
        mock_report = {
            "report_timestamp": 1234567890.0,
            "time_period_hours": 24,
            "security_events": {"total_events": 10},
            "recommendations": ["Enable additional monitoring"]
        }
        
        test_app.state.security_service.generate_security_report.return_value = mock_report
        
        response = client.get("/api/v1/security/report?hours=24")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["time_period_hours"] == 24
        assert "security_events" in data
        assert "recommendations" in data
    
    def test_metrics_endpoint(self, client, test_app):
        """Test metrics endpoint."""
        mock_metrics = {
            "counters": {"requests_total": 100, "errors_total": 5},
            "gauges": {"active_connections": 10},
            "histograms": {"response_time": {"count": 100, "average": 150.0}}
        }
        
        test_app.state.metrics_collector.get_all_metrics.return_value = mock_metrics
        
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data
    
    def test_prometheus_metrics_endpoint(self, client, test_app):
        """Test Prometheus metrics endpoint."""
        mock_prometheus = """
# HELP requests_total Total requests
# TYPE requests_total counter
requests_total 100
"""
        
        test_app.state.metrics_collector.export_prometheus_format.return_value = mock_prometheus.strip()
        
        response = client.get("/api/v1/metrics/prometheus")
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "requests_total 100" in response.text
    
    def test_models_list_endpoint(self, client):
        """Test models list endpoint."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "available_models" in data
        assert len(data["available_models"]) > 0
        assert all("name" in model for model in data["available_models"])
    
    def test_unauthorized_access(self, client, test_app):
        """Test unauthorized access handling."""
        # Mock security service to reject request
        test_app.state.security_service.validate_request.return_value = (False, ["Unauthorized"])
        
        request_data = {
            "text": "Hello world",
            "model_name": "bert-base-uncased"
        }
        
        response = client.post("/api/v1/inference", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Security validation failed" in data["error"]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPIMiddleware:
    """Test API middleware components."""
    
    def test_rate_limiting_middleware(self):
        """Test rate limiting functionality."""
        from src.secure_mpc_transformer.api.middleware import RateLimitMiddleware
        
        # Mock app and request
        app = Mock()
        config = {"requests_per_minute": 2, "burst_limit": 1}
        middleware = RateLimitMiddleware(app, config)
        
        # Should work with low rate limiting config for testing
        assert middleware.requests_per_minute == 2
    
    def test_security_middleware_initialization(self):
        """Test security middleware initialization."""
        from src.secure_mpc_transformer.api.middleware import SecurityMiddleware
        
        app = Mock()
        security_service = Mock(spec=SecurityService)
        
        middleware = SecurityMiddleware(app, security_service)
        
        assert middleware.security_service == security_service
    
    def test_logging_middleware_initialization(self):
        """Test logging middleware initialization."""
        from src.secure_mpc_transformer.api.middleware import LoggingMiddleware
        
        app = Mock()
        config = {"log_requests": True, "log_responses": True}
        
        middleware = LoggingMiddleware(app, config)
        
        assert middleware.log_requests is True
        assert middleware.log_responses is True


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPIServer:
    """Test API server functionality."""
    
    def test_create_app(self):
        """Test FastAPI app creation."""
        config = {
            "enable_docs": True,
            "cors_origins": ["http://localhost:3000"]
        }
        
        with patch('src.secure_mpc_transformer.api.server.DatabaseManager'):
            with patch('src.secure_mpc_transformer.api.server.MetricsCollector'):
                with patch('src.secure_mpc_transformer.api.server.InferenceService'):
                    with patch('src.secure_mpc_transformer.api.server.SecurityService'):
                        with patch('src.secure_mpc_transformer.api.server.ModelService'):
                            app = create_app(config)
        
        assert isinstance(app, FastAPI)
        assert app.title == "Secure MPC Transformer"
        assert app.version == "0.1.0"
    
    def test_api_server_initialization(self):
        """Test API server wrapper initialization."""
        from src.secure_mpc_transformer.api.server import APIServer
        
        config = {"log_level": "debug"}
        
        with patch('src.secure_mpc_transformer.api.server.create_app') as mock_create:
            mock_app = Mock()
            mock_create.return_value = mock_app
            
            server = APIServer(config)
            
            assert server.config == config
            assert server.app == mock_app
            mock_create.assert_called_once_with(config)


class TestAPIWithoutFastAPI:
    """Test API components when FastAPI is not available."""
    
    def test_import_error_handling(self):
        """Test that imports handle missing FastAPI gracefully."""
        # These should not raise errors even if FastAPI is not installed
        from src.secure_mpc_transformer.api import server, routes, middleware
        
        # Basic imports should work
        assert server is not None
        assert routes is not None
        assert middleware is not None


if __name__ == "__main__":
    pytest.main([__file__])