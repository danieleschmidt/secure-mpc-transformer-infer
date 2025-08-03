"""Integration tests for the secure MPC transformer system."""

import pytest
import asyncio
import torch
from unittest.mock import Mock, patch, AsyncMock

from src.secure_mpc_transformer.models.secure_transformer import SecureTransformer, TransformerConfig
from src.secure_mpc_transformer.services.inference_service import InferenceService, InferenceRequest
from src.secure_mpc_transformer.services.security_service import SecurityService
from src.secure_mpc_transformer.database.connection import DatabaseManager, DatabaseType
from src.secure_mpc_transformer.database.repositories import SessionRepository, ResultRepository
from src.secure_mpc_transformer.protocols.base import SecureValue


class TestEndToEndInference:
    """Test end-to-end inference pipeline."""
    
    @pytest.fixture
    def mock_protocol(self):
        """Create a mock protocol for testing."""
        protocol = Mock()
        protocol.party_id = 0
        protocol.num_parties = 3
        protocol.device = torch.device("cpu")
        
        def mock_share_value(tensor):
            return SecureValue(
                shares=[tensor, tensor.clone(), tensor.clone()],
                party_id=0,
                is_public=False
            )
        
        def mock_reconstruct_value(secure_value):
            return secure_value.shares[0]
        
        def mock_secure_add(a, b):
            result_shares = [a.shares[i] + b.shares[i] for i in range(len(a.shares))]
            return SecureValue(shares=result_shares, party_id=a.party_id, is_public=False)
        
        def mock_secure_matmul(a, b):
            result = torch.matmul(a.shares[0], b.shares[0])
            return SecureValue(shares=[result, result.clone(), result.clone()], party_id=a.party_id, is_public=False)
        
        protocol.share_value = Mock(side_effect=mock_share_value)
        protocol.reconstruct_value = Mock(side_effect=mock_reconstruct_value)
        protocol.secure_add = Mock(side_effect=mock_secure_add)
        protocol.secure_matmul = Mock(side_effect=mock_secure_matmul)
        protocol.secure_relu = Mock(side_effect=lambda x: x)
        protocol.secure_softmax = Mock(side_effect=lambda x, dim: x)
        protocol.initialize = Mock()
        
        protocol.get_protocol_info.return_value = {
            "protocol_name": "MockProtocol",
            "party_id": 0,
            "num_parties": 3,
            "device": "cpu",
            "initialized": True
        }
        
        return protocol
    
    @pytest.mark.asyncio
    async def test_complete_inference_pipeline(self, mock_protocol):
        """Test complete inference pipeline from request to result."""
        # Setup configuration
        config = TransformerConfig(
            model_name="test-model",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=128
        )
        
        # Mock dependencies
        with patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory') as mock_factory:
            with patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer') as mock_tokenizer:
                mock_factory.create.return_value = mock_protocol
                
                # Mock tokenizer
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
                }
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                
                # Create model
                model = SecureTransformer(config)
                
                # Mock model's forward pass to return predictable output
                with patch.object(model, 'forward') as mock_forward:
                    mock_output = SecureValue(
                        shares=[
                            torch.randn(1, 5, config.hidden_size),
                            torch.randn(1, 5, config.hidden_size),
                            torch.randn(1, 5, config.hidden_size)
                        ],
                        party_id=0,
                        is_public=False
                    )
                    mock_forward.return_value = mock_output
                    
                    # Create inference service
                    inference_service = InferenceService()
                    
                    # Mock the _get_model method to return our model
                    inference_service._get_model = AsyncMock(return_value=model)
                    
                    # Create inference request
                    request = InferenceRequest(
                        text="Hello world, this is a test input.",
                        model_name="test-model",
                        max_length=128
                    )
                    
                    # Execute inference
                    result = await inference_service.predict(request)
                    
                    # Verify results
                    assert result.model_name == "test-model"
                    assert result.input_text == "Hello world, this is a test input."
                    assert result.latency_ms > 0
                    assert result.protocol_info["protocol_name"] == "MockProtocol"
                    assert result.output_tensor is not None
                    assert len(result.input_shape) > 0
                    assert len(result.output_shape) > 0
    
    @pytest.mark.asyncio
    async def test_security_validation_integration(self):
        """Test security validation integration with inference."""
        security_service = SecurityService()
        inference_service = InferenceService()
        
        # Test valid request
        valid_request_data = {
            "text": "This is a valid input",
            "model_name": "bert-base-uncased"
        }
        
        is_valid, errors = security_service.validate_request(
            valid_request_data, 
            "192.168.1.1"
        )
        
        assert is_valid
        assert len(errors) == 0
        
        # Test malicious request
        malicious_request_data = {
            "text": "<script>alert('xss')</script>",
            "model_name": "bert-base-uncased"
        }
        
        is_valid, errors = security_service.validate_request(
            malicious_request_data,
            "192.168.1.1"
        )
        
        assert not is_valid
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_privacy_budget_integration(self):
        """Test privacy budget tracking integration."""
        security_service = SecurityService(config={"privacy_epsilon_budget": 1.0})
        
        # Record privacy expenditure
        security_service.record_privacy_expenditure(
            epsilon=0.3,
            operation="inference",
            details={"model": "bert-base", "input_length": 50}
        )
        
        # Check privacy summary
        summary = security_service.privacy_accountant.get_privacy_summary()
        assert summary["epsilon_spent"] == 0.3
        assert summary["epsilon_remaining"] == 0.7
        
        # Record more expenditure
        security_service.record_privacy_expenditure(
            epsilon=0.5,
            operation="inference",
            details={"model": "bert-base", "input_length": 100}
        )
        
        # Check updated summary
        summary = security_service.privacy_accountant.get_privacy_summary()
        assert summary["epsilon_spent"] == 0.8
        assert summary["epsilon_remaining"] == 0.2
        
        # Try to exceed budget
        with pytest.raises(ValueError, match="Privacy budget exceeded"):
            security_service.record_privacy_expenditure(
                epsilon=0.5,
                operation="inference",
                details={"model": "bert-base", "input_length": 200}
            )


class TestDatabaseIntegration:
    """Test database integration with services."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.database_type = DatabaseType.SQLITE
        db_manager.initialize = AsyncMock()
        db_manager.close = AsyncMock()
        db_manager.health_check = AsyncMock(return_value={
            "status": "healthy",
            "response_time_ms": 5.0
        })
        return db_manager
    
    @pytest.mark.asyncio
    async def test_session_repository_integration(self, mock_db_manager):
        """Test session repository integration."""
        from src.secure_mpc_transformer.database.models import ComputationSession, SessionStatus
        
        # Mock database operations
        mock_db_manager.get_connection = AsyncMock()
        
        session_repo = SessionRepository(mock_db_manager)
        
        # Create test session
        session = ComputationSession(
            model_name="bert-base-uncased",
            input_text="test input",
            sequence_length=10
        )
        
        # Mock create operation
        with patch.object(session_repo, '_create_sqlite', new_callable=AsyncMock) as mock_create:
            await session_repo.create(session)
            mock_create.assert_called_once()
        
        # Mock get operation
        with patch.object(session_repo, '_get_by_id_sqlite', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = session
            result = await session_repo.get_by_id(session.session_id)
            assert result == session
    
    @pytest.mark.asyncio
    async def test_result_repository_integration(self, mock_db_manager):
        """Test result repository integration."""
        from src.secure_mpc_transformer.database.models import InferenceResult
        
        result_repo = ResultRepository(mock_db_manager)
        
        # Create test result
        result = InferenceResult(
            session_id="test_session",
            output_text="test output",
            computation_time_ms=150.0,
            total_operations=100
        )
        
        # Mock create operation
        with patch.object(result_repo, '_create_sqlite', new_callable=AsyncMock) as mock_create:
            await result_repo.create(result)
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audit_repository_integration(self, mock_db_manager):
        """Test audit repository integration."""
        from src.secure_mpc_transformer.database.models import AuditLog, AuditEventType
        
        audit_repo = ResultRepository(mock_db_manager)  # Using same base methods
        
        # Create test audit log
        audit_log = AuditLog.create_security_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            description="Test security event",
            session_id="test_session",
            risk_level="medium"
        )
        
        # Mock create operation
        with patch.object(audit_repo, '_create_sqlite', new_callable=AsyncMock) as mock_create:
            await audit_repo.create(audit_log)
            mock_create.assert_called_once()


class TestMultiPartySimulation:
    """Test multi-party computation simulation."""
    
    @pytest.mark.asyncio
    async def test_three_party_protocol_simulation(self):
        """Simulate 3-party protocol execution."""
        from src.secure_mpc_transformer.protocols.aby3 import ABY3Protocol
        
        # Create three party instances
        parties = []
        for party_id in range(3):
            with patch.object(ABY3Protocol, '__init__', return_value=None):
                with patch.object(ABY3Protocol, 'initialize'):
                    party = ABY3Protocol(party_id=party_id, num_parties=3)
                    party.party_id = party_id
                    party.num_parties = 3
                    party.device = torch.device("cpu")
                    party._initialized = True
                    parties.append(party)
        
        # Mock secure value sharing between parties
        test_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        for party in parties:
            with patch.object(party, 'share_value') as mock_share:
                mock_share.return_value = SecureValue(
                    shares=[test_tensor, test_tensor, test_tensor],
                    party_id=party.party_id,
                    is_public=False
                )
                
                # Test sharing
                shared_value = party.share_value(test_tensor)
                assert isinstance(shared_value, SecureValue)
                assert len(shared_value.shares) == 3
    
    @pytest.mark.asyncio
    async def test_protocol_communication_simulation(self):
        """Simulate communication between protocol parties."""
        from src.secure_mpc_transformer.protocols.base import Protocol
        
        # Mock network communication
        communication_log = []
        
        def mock_send_shares(shares, recipient):
            communication_log.append({
                "sender": 0,
                "recipient": recipient,
                "shares_count": len(shares)
            })
        
        def mock_receive_shares(sender):
            return [torch.tensor([1.0, 2.0])]
        
        # Create mock protocol
        protocol = Mock(spec=Protocol)
        protocol.send_shares = Mock(side_effect=mock_send_shares)
        protocol.receive_shares = Mock(side_effect=mock_receive_shares)
        
        # Simulate communication
        test_shares = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        
        # Party 0 sends to parties 1 and 2
        protocol.send_shares(test_shares, 1)
        protocol.send_shares(test_shares, 2)
        
        # Verify communication occurred
        assert len(communication_log) == 2
        assert communication_log[0]["recipient"] == 1
        assert communication_log[1]["recipient"] == 2


class TestPerformanceIntegration:
    """Test performance monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self):
        """Test metrics collection during operations."""
        from src.secure_mpc_transformer.utils.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        # Simulate various operations
        metrics.increment_counter("inference_requests_total")
        metrics.increment_counter("inference_requests_success")
        metrics.observe_histogram("inference_latency_ms", 150.0)
        metrics.set_gauge("active_requests", 5)
        
        # Get metrics summary
        all_metrics = metrics.get_all_metrics()
        
        assert all_metrics["counters"]["inference_requests_total"] == 1
        assert all_metrics["counters"]["inference_requests_success"] == 1
        assert all_metrics["gauges"]["active_requests"] == 5
        assert "inference_latency_ms" in all_metrics["histograms"]
    
    def test_prometheus_export_integration(self):
        """Test Prometheus metrics export."""
        from src.secure_mpc_transformer.utils.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        # Add some metrics
        metrics.increment_counter("requests_total")
        metrics.set_gauge("memory_usage_mb", 1024.5)
        
        # Export to Prometheus format
        prometheus_output = metrics.export_prometheus_format()
        
        assert "requests_total 1" in prometheus_output
        assert "memory_usage_mb 1024.5" in prometheus_output
        assert "# TYPE requests_total counter" in prometheus_output
        assert "# TYPE memory_usage_mb gauge" in prometheus_output


class TestErrorHandlingIntegration:
    """Test error handling across system components."""
    
    @pytest.mark.asyncio
    async def test_inference_error_handling(self):
        """Test error handling in inference pipeline."""
        inference_service = InferenceService()
        
        # Mock model loading failure
        inference_service._get_model = AsyncMock(side_effect=Exception("Model loading failed"))
        
        request = InferenceRequest(
            text="test input",
            model_name="nonexistent-model"
        )
        
        with pytest.raises(Exception, match="Model loading failed"):
            await inference_service.predict(request)
    
    def test_security_error_handling(self):
        """Test security error handling."""
        security_service = SecurityService()
        
        # Test with invalid input that should raise validation error
        invalid_request = {
            "text": "a" * 50000,  # Too long
            "model_name": "test"
        }
        
        is_valid, errors = security_service.validate_request(
            invalid_request,
            "192.168.1.1"
        )
        
        assert not is_valid
        assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_db_manager):
        """Test database error handling."""
        # Mock database connection failure
        mock_db_manager.get_connection = AsyncMock(side_effect=Exception("Database connection failed"))
        
        session_repo = SessionRepository(mock_db_manager)
        
        with pytest.raises(Exception, match="Database connection failed"):
            await session_repo.count()


if __name__ == "__main__":
    pytest.main([__file__])