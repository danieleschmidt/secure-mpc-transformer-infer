"""Tests for service layer components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import torch

from src.secure_mpc_transformer.services.inference_service import (
    InferenceService, InferenceRequest, InferenceResult
)
from src.secure_mpc_transformer.services.security_service import SecurityService
from src.secure_mpc_transformer.services.model_service import ModelService
from src.secure_mpc_transformer.utils.validators import InputValidator, ValidationError


class TestInferenceService:
    """Test InferenceService class."""
    
    @pytest.fixture
    def mock_validator(self):
        """Mock input validator."""
        validator = Mock(spec=InputValidator)
        validator.validate_inference_request.return_value = True
        return validator
    
    @pytest.fixture
    def mock_metrics(self):
        """Mock metrics collector."""
        metrics = Mock()
        metrics.increment_counter = Mock()
        metrics.observe_histogram = Mock()
        return metrics
    
    @pytest.fixture
    def inference_service(self, mock_validator, mock_metrics):
        """Create inference service with mocked dependencies."""
        service = InferenceService()
        service.validator = mock_validator
        service.metrics = mock_metrics
        return service
    
    @pytest.mark.asyncio
    async def test_predict_success(self, inference_service):
        """Test successful prediction."""
        # Mock model
        mock_model = Mock()
        mock_result = {
            'output_tensor': torch.tensor([[1.0, 2.0, 3.0]]),
            'secure_output': Mock(),
            'latency_ms': 100.0,
            'protocol_info': {'protocol_name': 'test'},
            'input_shape': (1, 5),
            'output_shape': (1, 3)
        }
        mock_model.predict_secure.return_value = mock_result
        
        # Mock _get_model method
        with patch.object(inference_service, '_get_model', new_callable=AsyncMock) as mock_get_model:
            mock_get_model.return_value = mock_model
            
            request = InferenceRequest(
                text="test input",
                model_name="test-model"
            )
            
            result = await inference_service.predict(request)
            
            assert isinstance(result, InferenceResult)
            assert result.input_text == "test input"
            assert result.model_name == "test-model"
            assert result.latency_ms == 100.0
            
            # Verify metrics were recorded
            inference_service.metrics.increment_counter.assert_called()
            inference_service.metrics.observe_histogram.assert_called()
    
    @pytest.mark.asyncio
    async def test_predict_validation_failure(self, inference_service):
        """Test prediction with validation failure."""
        # Mock validation failure
        inference_service.validator.validate_inference_request.side_effect = ValidationError(
            "text", "invalid input", "Input too long"
        )
        
        request = InferenceRequest(
            text="a" * 20000,  # Too long
            model_name="test-model"
        )
        
        with pytest.raises(ValidationError):
            await inference_service.predict(request)
    
    @pytest.mark.asyncio
    async def test_predict_batch(self, inference_service):
        """Test batch prediction."""
        # Mock model
        mock_model = Mock()
        mock_result = {
            'output_tensor': torch.tensor([[1.0, 2.0, 3.0]]),
            'secure_output': Mock(),
            'latency_ms': 100.0,
            'protocol_info': {'protocol_name': 'test'},
            'input_shape': (1, 5),
            'output_shape': (1, 3)
        }
        mock_model.predict_secure.return_value = mock_result
        
        with patch.object(inference_service, '_get_model', new_callable=AsyncMock) as mock_get_model:
            mock_get_model.return_value = mock_model
            
            requests = [
                InferenceRequest(text="input 1", model_name="test-model"),
                InferenceRequest(text="input 2", model_name="test-model")
            ]
            
            results = await inference_service.predict_batch(requests)
            
            assert len(results) == 2
            assert all(isinstance(r, InferenceResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_health_check(self, inference_service):
        """Test health check."""
        with patch.object(inference_service, 'predict', new_callable=AsyncMock) as mock_predict:
            mock_result = InferenceResult(
                request_id="test",
                input_text="test",
                model_name="test",
                output_tensor=torch.tensor([1.0]),
                secure_output=Mock(),
                latency_ms=50.0,
                protocol_info={},
                input_shape=(1,),
                output_shape=(1,)
            )
            mock_predict.return_value = mock_result
            
            health = await inference_service.health_check()
            
            assert health["status"] == "healthy"
            assert "health_check_latency_ms" in health
            assert health["health_check_latency_ms"] >= 0


class TestSecurityService:
    """Test SecurityService class."""
    
    @pytest.fixture
    def security_service(self):
        """Create security service."""
        return SecurityService()
    
    def test_validate_request_success(self, security_service):
        """Test successful request validation."""
        request_data = {
            "text": "Hello world",
            "model_name": "bert-base-uncased"
        }
        
        is_valid, errors = security_service.validate_request(
            request_data, "192.168.1.1"
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_request_suspicious_input(self, security_service):
        """Test request validation with suspicious input."""
        request_data = {
            "text": "<script>alert('xss')</script>",
            "model_name": "bert-base-uncased"
        }
        
        is_valid, errors = security_service.validate_request(
            request_data, "192.168.1.1"
        )
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_record_privacy_expenditure(self, security_service):
        """Test privacy budget recording."""
        security_service.record_privacy_expenditure(
            epsilon=0.1,
            operation="inference",
            details={"model": "bert-base"}
        )
        
        summary = security_service.privacy_accountant.get_privacy_summary()
        assert summary["epsilon_spent"] == 0.1
    
    def test_privacy_budget_exceeded(self, security_service):
        """Test privacy budget exceeded error."""
        # Set low budget
        security_service.privacy_accountant.epsilon_budget = 0.5
        
        # Spend budget
        security_service.record_privacy_expenditure(0.3, "test", {})
        security_service.record_privacy_expenditure(0.3, "test", {})
        
        # Should raise error on exceeding budget
        with pytest.raises(ValueError, match="Privacy budget exceeded"):
            security_service.record_privacy_expenditure(0.1, "test", {})
    
    def test_get_security_status(self, security_service):
        """Test security status retrieval."""
        status = security_service.get_security_status()
        
        assert "timestamp" in status
        assert "service_status" in status
        assert "threat_detection_enabled" in status
        assert "audit_logging_enabled" in status
        assert "privacy_accounting_enabled" in status
        assert "security_metrics" in status
    
    def test_generate_security_report(self, security_service):
        """Test security report generation."""
        report = security_service.generate_security_report(hours=24)
        
        assert "report_timestamp" in report
        assert "time_period_hours" in report
        assert "service_configuration" in report
        assert "recommendations" in report
    
    def test_emergency_lockdown(self, security_service):
        """Test emergency lockdown functionality."""
        reason = "Suspected security breach"
        
        security_service.emergency_lockdown(reason)
        
        # Should exhaust privacy budget
        summary = security_service.privacy_accountant.get_privacy_summary()
        assert summary["epsilon_remaining"] == 0
        
        # Should log critical event
        assert len(security_service.auditor.audit_log) > 0
        event = security_service.auditor.audit_log[-1]
        assert event.event_type.value == "EMERGENCY_LOCKDOWN"
        assert event.severity == "CRITICAL"


class TestModelService:
    """Test ModelService class."""
    
    @pytest.fixture
    def model_service(self):
        """Create model service."""
        return ModelService()
    
    @pytest.mark.asyncio
    async def test_get_model_cached(self, model_service):
        """Test getting cached model."""
        # Mock model in cache
        mock_model = Mock()
        cache_key = "test_model_{hash}"
        model_service.cache.models[cache_key] = mock_model
        
        with patch.object(model_service.cache, 'get', return_value=mock_model):
            result = await model_service.get_model(
                "test-model", 
                {"protocol_name": "test"}
            )
            
            assert result == mock_model
            model_service.metrics.increment_counter.assert_called_with("model_cache_hits")
    
    @pytest.mark.asyncio
    async def test_get_model_load_new(self, model_service):
        """Test loading new model."""
        # Mock cache miss
        model_service.cache.get = Mock(return_value=None)
        
        # Mock loader
        mock_model = Mock()
        load_result = {
            "success": True,
            "model": mock_model,
            "model_info": Mock(),
            "load_time_ms": 1000.0
        }
        
        with patch.object(model_service.loader, 'get_load_status') as mock_status:
            with patch.object(model_service.loader, 'submit_load_request') as mock_submit:
                mock_status.side_effect = [
                    {"status": "not_found"},
                    {"status": "completed", "result": load_result}
                ]
                
                result = await model_service.get_model(
                    "new-model",
                    {"protocol_name": "test"}
                )
                
                assert result == mock_model
                mock_submit.assert_called_once()
    
    def test_list_models(self, model_service):
        """Test listing models."""
        # Mock cache stats
        mock_stats = {
            "total_models": 2,
            "total_memory_mb": 1000.0,
            "models": []
        }
        model_service.cache.get_cache_stats = Mock(return_value=mock_stats)
        
        result = model_service.list_models()
        
        assert "cache_stats" in result
        assert "loader_stats" in result
        assert "service_config" in result
    
    def test_unload_model(self, model_service):
        """Test model unloading."""
        model_service.cache.remove = Mock(return_value=True)
        
        result = model_service.unload_model(
            "test-model",
            {"protocol_name": "test"}
        )
        
        assert result is True
        model_service.cache.remove.assert_called_once()
        model_service.metrics.increment_counter.assert_called_with("model_unloads")


class TestInputValidator:
    """Test InputValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create input validator."""
        return InputValidator()
    
    def test_validate_text_input_success(self, validator):
        """Test successful text validation."""
        request = InferenceRequest(
            text="This is a normal text input",
            model_name="bert-base-uncased"
        )
        
        # Should not raise exception
        result = validator.validate_inference_request(request)
        assert result is True
    
    def test_validate_text_input_too_long(self, validator):
        """Test text input too long."""
        request = InferenceRequest(
            text="a" * 20000,  # Exceeds max length
            model_name="bert-base-uncased"
        )
        
        with pytest.raises(ValidationError, match="Text length exceeds maximum"):
            validator.validate_inference_request(request)
    
    def test_validate_text_input_empty(self, validator):
        """Test empty text input."""
        request = InferenceRequest(
            text="",
            model_name="bert-base-uncased"
        )
        
        with pytest.raises(ValidationError, match="Input text cannot be empty"):
            validator.validate_inference_request(request)
    
    def test_validate_suspicious_patterns(self, validator):
        """Test detection of suspicious patterns."""
        request = InferenceRequest(
            text="SELECT * FROM users WHERE 1=1 --",
            model_name="bert-base-uncased"
        )
        
        with pytest.raises(ValidationError, match="Input contains suspicious patterns"):
            validator.validate_inference_request(request)
    
    def test_validate_max_length(self, validator):
        """Test max length validation."""
        request = InferenceRequest(
            text="test",
            model_name="bert-base-uncased",
            max_length=5000  # Exceeds limit
        )
        
        with pytest.raises(ValidationError, match="Max length exceeds limit"):
            validator.validate_inference_request(request)
    
    def test_validate_batch_size(self, validator):
        """Test batch size validation."""
        request = InferenceRequest(
            text="test",
            model_name="bert-base-uncased",
            batch_size=100  # Too large
        )
        
        with pytest.raises(ValidationError, match="Batch size too large"):
            validator.validate_inference_request(request)
    
    def test_validate_protocol_config(self, validator):
        """Test protocol configuration validation."""
        request = InferenceRequest(
            text="test",
            model_name="bert-base-uncased",
            protocol_config={
                "security_level": 50,  # Too low
                "num_parties": 15  # Too high
            }
        )
        
        with pytest.raises(ValidationError):
            validator.validate_inference_request(request)
    
    def test_sanitize_text_input(self, validator):
        """Test text input sanitization."""
        malicious_text = "<script>alert('xss')</script>Hello\x00World\n\n\n  "
        
        sanitized = validator.sanitize_text_input(malicious_text)
        
        assert "<script>" not in sanitized
        assert "\x00" not in sanitized
        assert sanitized.strip() == "HelloWorld"
    
    def test_validate_tensor_input(self, validator):
        """Test tensor input validation."""
        # Valid tensor
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = validator.validate_tensor_input(tensor, expected_shape=(2, 2))
        assert result is True
        
        # Invalid shape
        with pytest.raises(ValidationError, match="Expected shape"):
            validator.validate_tensor_input(tensor, expected_shape=(3, 3))
        
        # NaN values
        tensor_nan = torch.tensor([[1.0, float('nan')]])
        with pytest.raises(ValidationError, match="Tensor contains NaN values"):
            validator.validate_tensor_input(tensor_nan)
        
        # Infinite values
        tensor_inf = torch.tensor([[1.0, float('inf')]])
        with pytest.raises(ValidationError, match="Tensor contains infinite values"):
            validator.validate_tensor_input(tensor_inf)


if __name__ == "__main__":
    pytest.main([__file__])