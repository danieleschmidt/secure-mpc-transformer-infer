"""Tests for secure transformer models."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.secure_mpc_transformer.models.secure_transformer import (
    SecureTransformer, TransformerConfig
)
from src.secure_mpc_transformer.protocols.base import SecureValue, Protocol
from src.secure_mpc_transformer.protocols.factory import ProtocolFactory


class TestTransformerConfig:
    """Test TransformerConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TransformerConfig()
        
        assert config.model_name == "bert-base-uncased"
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.max_position_embeddings == 512
        assert config.protocol_name == "aby3"
        assert config.security_level == 128
        assert config.num_parties == 3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TransformerConfig(
            model_name="roberta-base",
            hidden_size=1024,
            num_hidden_layers=24,
            protocol_name="semi_honest_3pc",
            security_level=256
        )
        
        assert config.model_name == "roberta-base"
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.protocol_name == "semi_honest_3pc"
        assert config.security_level == 256
    
    @patch('src.secure_mpc_transformer.models.secure_transformer.AutoConfig')
    def test_from_pretrained(self, mock_auto_config):
        """Test loading config from pretrained model."""
        # Mock HuggingFace config
        mock_hf_config = Mock()
        mock_hf_config.vocab_size = 50265
        mock_hf_config.hidden_size = 768
        mock_hf_config.num_hidden_layers = 12
        mock_hf_config.num_attention_heads = 12
        mock_hf_config.intermediate_size = 3072
        mock_hf_config.max_position_embeddings = 514
        mock_hf_config.type_vocab_size = 1
        mock_hf_config.layer_norm_eps = 1e-5
        
        mock_auto_config.from_pretrained.return_value = mock_hf_config
        
        config = TransformerConfig.from_pretrained("roberta-base")
        
        assert config.model_name == "roberta-base"
        assert config.vocab_size == 50265
        assert config.hidden_size == 768
        assert config.max_position_embeddings == 514
        assert config.type_vocab_size == 1
        assert config.layer_norm_eps == 1e-5


class TestSecureTransformer:
    """Test SecureTransformer class."""
    
    @pytest.fixture
    def mock_protocol(self):
        """Create a mock protocol for testing."""
        protocol = Mock(spec=Protocol)
        protocol.party_id = 0
        protocol.num_parties = 3
        protocol.device = torch.device("cpu")
        
        # Mock protocol methods
        def mock_share_value(tensor):
            return SecureValue(
                shares=[tensor, tensor.clone(), tensor.clone()],
                party_id=0,
                is_public=False
            )
        
        def mock_reconstruct_value(secure_value):
            return secure_value.shares[0]
        
        def mock_secure_add(a, b):
            result_shares = [
                a.shares[i] + b.shares[i] for i in range(len(a.shares))
            ]
            return SecureValue(
                shares=result_shares,
                party_id=a.party_id,
                is_public=a.is_public and b.is_public
            )
        
        def mock_secure_matmul(a, b):
            result = torch.matmul(a.shares[0], b.shares[0])
            return SecureValue(
                shares=[result, result.clone(), result.clone()],
                party_id=a.party_id,
                is_public=False
            )
        
        protocol.share_value = Mock(side_effect=mock_share_value)
        protocol.reconstruct_value = Mock(side_effect=mock_reconstruct_value)
        protocol.secure_add = Mock(side_effect=mock_secure_add)
        protocol.secure_matmul = Mock(side_effect=mock_secure_matmul)
        protocol.secure_relu = Mock(side_effect=lambda x: x)  # Simplified
        protocol.secure_softmax = Mock(side_effect=lambda x, dim: x)  # Simplified
        
        protocol.get_protocol_info.return_value = {
            "protocol_name": "MockProtocol",
            "party_id": 0,
            "num_parties": 3,
            "device": "cpu",
            "initialized": True
        }
        
        return protocol
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TransformerConfig(
            model_name="test-model",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=128,
            type_vocab_size=2
        )
    
    @patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory')
    @patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer')
    def test_secure_transformer_init(self, mock_tokenizer, mock_factory, config, mock_protocol):
        """Test SecureTransformer initialization."""
        mock_factory.create.return_value = mock_protocol
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        model = SecureTransformer(config)
        
        assert model.config == config
        assert model.protocol == mock_protocol
        assert len(model.layers) == config.num_hidden_layers
        mock_factory.create.assert_called_once()
        mock_protocol.initialize.assert_called_once()
    
    @patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory')
    @patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer')
    def test_forward_pass(self, mock_tokenizer, mock_factory, config, mock_protocol):
        """Test forward pass through secure transformer."""
        mock_factory.create.return_value = mock_protocol
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        model = SecureTransformer(config)
        
        # Create test input
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        
        # Mock embeddings
        with patch.object(model.embeddings, 'forward') as mock_embeddings:
            mock_embeddings.return_value = SecureValue(
                shares=[
                    torch.randn(batch_size, seq_length, config.hidden_size),
                    torch.randn(batch_size, seq_length, config.hidden_size),
                    torch.randn(batch_size, seq_length, config.hidden_size)
                ],
                party_id=0,
                is_public=False
            )
            
            # Mock layer forward passes
            for layer in model.layers:
                with patch.object(layer, 'forward') as mock_layer:
                    mock_layer.return_value = mock_embeddings.return_value
            
            # Run forward pass
            result = model.forward(input_ids, attention_mask)
            
            assert isinstance(result, SecureValue)
            assert len(result.shares) == 3
            assert result.shares[0].shape == (batch_size, seq_length, config.hidden_size)
    
    @patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory')
    @patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer')
    def test_predict_secure(self, mock_tokenizer, mock_factory, config, mock_protocol):
        """Test secure prediction method."""
        mock_factory.create.return_value = mock_protocol
        
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        model = SecureTransformer(config)
        
        # Mock forward pass
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
            
            # Test prediction
            result = model.predict_secure("test input text")
            
            assert "secure_output" in result
            assert "output_tensor" in result
            assert "latency_ms" in result
            assert "protocol_info" in result
            assert "input_shape" in result
            assert "output_shape" in result
            
            assert isinstance(result["latency_ms"], float)
            assert result["latency_ms"] >= 0
    
    @patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory')
    @patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer')
    def test_from_pretrained(self, mock_tokenizer, mock_factory, mock_protocol):
        """Test loading from pretrained model."""
        mock_factory.create.return_value = mock_protocol
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        with patch('src.secure_mpc_transformer.models.secure_transformer.TransformerConfig.from_pretrained') as mock_config:
            mock_config.return_value = TransformerConfig(model_name="test-model")
            
            model = SecureTransformer.from_pretrained("test-model")
            
            assert isinstance(model, SecureTransformer)
            assert model.config.model_name == "test-model"
            mock_config.assert_called_once_with("test-model")
    
    def test_get_model_info(self, config, mock_protocol):
        """Test model info retrieval."""
        with patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory') as mock_factory:
            with patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer'):
                mock_factory.create.return_value = mock_protocol
                
                model = SecureTransformer(config)
                info = model.get_model_info()
                
                assert "config" in info
                assert "protocol_info" in info
                assert "total_parameters" in info
                assert "trainable_parameters" in info
                assert "device" in info
                assert "model_size_mb" in info
                
                assert isinstance(info["total_parameters"], int)
                assert isinstance(info["trainable_parameters"], int)
                assert isinstance(info["model_size_mb"], float)


class TestSecureValue:
    """Test SecureValue class."""
    
    def test_secure_value_creation(self):
        """Test SecureValue creation and properties."""
        shares = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        ]
        
        secure_val = SecureValue(shares=shares, party_id=0, is_public=False)
        
        assert len(secure_val.shares) == 3
        assert secure_val.party_id == 0
        assert not secure_val.is_public
        assert secure_val.shape == torch.Size([2, 2])
        assert secure_val.dtype == torch.float32
    
    def test_secure_value_addition(self):
        """Test SecureValue addition."""
        shares1 = [torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0]), torch.tensor([0.5, 1.0])]
        shares2 = [torch.tensor([2.0, 3.0]), torch.tensor([1.0, 1.5]), torch.tensor([1.0, 1.5])]
        
        val1 = SecureValue(shares=shares1, party_id=0)
        val2 = SecureValue(shares=shares2, party_id=0)
        
        result = val1 + val2
        
        assert len(result.shares) == 3
        assert torch.allclose(result.shares[0], torch.tensor([3.0, 5.0]))
        assert torch.allclose(result.shares[1], torch.tensor([1.5, 2.5]))
        assert torch.allclose(result.shares[2], torch.tensor([1.5, 2.5]))
    
    def test_secure_value_scalar_multiplication(self):
        """Test SecureValue scalar multiplication."""
        shares = [torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0])]
        secure_val = SecureValue(shares=shares, party_id=0)
        
        result = secure_val * 2.0
        
        assert len(result.shares) == 2
        assert torch.allclose(result.shares[0], torch.tensor([2.0, 4.0]))
        assert torch.allclose(result.shares[1], torch.tensor([1.0, 2.0]))
    
    def test_secure_value_device_transfer(self):
        """Test moving SecureValue to different device."""
        shares = [torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0])]
        secure_val = SecureValue(shares=shares, party_id=0)
        
        # Test moving to same device (should work)
        result = secure_val.to("cpu")
        
        assert result.device == torch.device("cpu")
        assert len(result.shares) == 2
        assert torch.allclose(result.shares[0], torch.tensor([1.0, 2.0]))
    
    def test_secure_value_validation(self):
        """Test SecureValue validation."""
        # Test empty shares
        with pytest.raises(ValueError, match="SecureValue must have at least one share"):
            SecureValue(shares=[], party_id=0)


class TestProtocolIntegration:
    """Test protocol integration with transformer models."""
    
    @pytest.fixture
    def protocol_factory_mock(self):
        """Mock protocol factory."""
        with patch('src.secure_mpc_transformer.models.secure_transformer.ProtocolFactory') as mock:
            yield mock
    
    def test_protocol_initialization(self, protocol_factory_mock):
        """Test protocol initialization in transformer."""
        mock_protocol = Mock(spec=Protocol)
        mock_protocol.device = torch.device("cpu")
        protocol_factory_mock.create.return_value = mock_protocol
        
        config = TransformerConfig(protocol_name="test_protocol")
        
        with patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer'):
            model = SecureTransformer(config)
        
        protocol_factory_mock.create.assert_called_once_with(
            "test_protocol",
            party_id=0,
            num_parties=3,
            device=torch.device("cpu")
        )
        mock_protocol.initialize.assert_called_once()
    
    def test_gpu_device_selection(self, protocol_factory_mock):
        """Test GPU device selection when available."""
        mock_protocol = Mock(spec=Protocol)
        mock_protocol.device = torch.device("cpu")  # Mock as CPU for testing
        protocol_factory_mock.create.return_value = mock_protocol
        
        config = TransformerConfig(gpu_acceleration=True)
        
        with patch('src.secure_mpc_transformer.models.secure_transformer.AutoTokenizer'):
            with patch('torch.cuda.is_available', return_value=False):  # Mock CUDA not available
                model = SecureTransformer(config)
        
        # Should fall back to CPU when CUDA not available
        call_args = protocol_factory_mock.create.call_args
        assert call_args[1]['device'] == torch.device("cpu")


if __name__ == "__main__":
    pytest.main([__file__])