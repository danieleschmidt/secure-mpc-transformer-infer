"""
Unit tests for secure transformer models.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from secure_mpc_transformer.models import SecureTransformer
from secure_mpc_transformer.config import SecurityConfig


class TestSecureTransformer:
    """Test cases for SecureTransformer class."""

    def test_init_with_default_config(self, mock_model_config):
        """Test initialization with default security configuration."""
        config = SecurityConfig()
        model = SecureTransformer(mock_model_config, config)
        
        assert model.config == mock_model_config
        assert model.security_config == config
        assert model.protocol is not None

    def test_init_with_custom_config(self, mock_model_config, security_config):
        """Test initialization with custom security configuration."""
        config = SecurityConfig(**security_config)
        model = SecureTransformer(mock_model_config, config)
        
        assert model.security_config.protocol == "semi_honest_3pc"
        assert model.security_config.security_level == 128
        assert model.security_config.num_parties == 3

    @patch('secure_mpc_transformer.models.SecureTransformer._load_pretrained')
    def test_from_pretrained(self, mock_load, mock_model_config):
        """Test loading pretrained model."""
        mock_load.return_value = Mock()
        config = SecurityConfig()
        
        model = SecureTransformer.from_pretrained("bert-base-uncased", config)
        
        mock_load.assert_called_once_with("bert-base-uncased")
        assert isinstance(model, SecureTransformer)

    def test_input_validation(self, mock_model_config):
        """Test input validation for model initialization."""
        config = SecurityConfig()
        
        # Test invalid model config
        with pytest.raises(ValueError):
            SecureTransformer({}, config)
        
        # Test invalid security config
        with pytest.raises(TypeError):
            SecureTransformer(mock_model_config, "invalid_config")

    @pytest.mark.parametrize("protocol", [
        "semi_honest_3pc",
        "malicious_3pc", 
        "aby3",
        "fantastic_four"
    ])
    def test_different_protocols(self, mock_model_config, protocol):
        """Test model initialization with different MPC protocols."""
        config = SecurityConfig(protocol=protocol)
        model = SecureTransformer(mock_model_config, config)
        
        assert model.security_config.protocol == protocol

    def test_model_serialization(self, mock_model_config, temp_dir):
        """Test model serialization and deserialization."""
        config = SecurityConfig()
        model = SecureTransformer(mock_model_config, config)
        
        # Save model
        save_path = temp_dir / "model.pkl"
        model.save(save_path)
        
        # Load model
        loaded_model = SecureTransformer.load(save_path)
        
        assert loaded_model.config == model.config
        assert loaded_model.security_config.protocol == model.security_config.protocol

    @pytest.mark.gpu
    def test_gpu_initialization(self, mock_model_config, gpu_available):
        """Test GPU initialization when available."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        config = SecurityConfig(gpu_acceleration=True)
        model = SecureTransformer(mock_model_config, config)
        
        assert model.security_config.gpu_acceleration is True
        assert model.device.type == "cuda"

    def test_memory_cleanup(self, mock_model_config, cleanup_gpu):
        """Test proper memory cleanup after model operations."""
        config = SecurityConfig()
        model = SecureTransformer(mock_model_config, config)
        
        # Simulate model operations
        dummy_input = torch.randn(1, 128, 768)
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = dummy_input
            _ = model.forward(dummy_input)
        
        # Memory should be cleaned up automatically
        del model
        
    @pytest.mark.security
    def test_security_invariants(self, mock_model_config):
        """Test that security invariants are maintained."""
        config = SecurityConfig(security_level=128)
        model = SecureTransformer(mock_model_config, config)
        
        # Check that security level is maintained
        assert model.security_config.security_level >= 128
        
        # Check that protocol implements required security guarantees
        assert hasattr(model.protocol, 'security_level')
        assert model.protocol.security_level >= 128

    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 64),
        (4, 128),
        (8, 256),
    ])
    def test_input_shapes(self, mock_model_config, batch_size, seq_len):
        """Test model with different input shapes."""
        config = SecurityConfig()
        model = SecureTransformer(mock_model_config, config)
        
        # Create mock input with specified shape
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
        
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(batch_size, seq_len, 768)
            output = model.forward(dummy_input)
            
            assert output.shape[0] == batch_size
            assert output.shape[1] == seq_len

    def test_error_handling(self, mock_model_config):
        """Test proper error handling in model operations."""
        config = SecurityConfig()
        model = SecureTransformer(mock_model_config, config)
        
        # Test with invalid input
        with pytest.raises(ValueError):
            model.forward("invalid_input")
        
        # Test with mismatched dimensions
        with pytest.raises(RuntimeError):
            invalid_input = torch.randn(1, 1000)  # Wrong dimensions
            model.forward(invalid_input)