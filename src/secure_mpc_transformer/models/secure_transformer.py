"""Secure transformer implementation for MPC inference."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer
import logging

from ..protocols.base import Protocol, SecureValue
from ..protocols.factory import ProtocolFactory
from .attention import SecureAttention
from .feedforward import SecureFeedForward
from .embeddings import SecureEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for secure transformer models."""
    
    model_name: str = "bert-base-uncased"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    dropout_prob: float = 0.1
    
    # MPC-specific settings
    protocol_name: str = "aby3"
    security_level: int = 128
    gpu_acceleration: bool = True
    party_id: int = 0
    num_parties: int = 3
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "TransformerConfig":
        """Load configuration from pre-trained model."""
        try:
            hf_config = AutoConfig.from_pretrained(model_name)
            
            return cls(
                model_name=model_name,
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                intermediate_size=hf_config.intermediate_size,
                max_position_embeddings=hf_config.max_position_embeddings,
                type_vocab_size=getattr(hf_config, 'type_vocab_size', 2),
                layer_norm_eps=getattr(hf_config, 'layer_norm_eps', 1e-12),
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Could not load config for {model_name}: {e}")
            return cls(model_name=model_name, **kwargs)


class SecureTransformerLayer(nn.Module):
    """Single transformer layer with secure operations."""
    
    def __init__(self, config: TransformerConfig, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        
        self.attention = SecureAttention(config, protocol)
        self.feedforward = SecureFeedForward(config, protocol)
        
        # Layer normalization parameters (kept as plaintext for efficiency)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: SecureValue, attention_mask: Optional[torch.Tensor] = None) -> SecureValue:
        """Forward pass through transformer layer."""
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        
        # Add residual connection and apply layer norm
        # Note: In practice, secure layer norm would be implemented
        attention_output = self._secure_layer_norm(
            self.protocol.secure_add(hidden_states, attention_output),
            self.attention_norm
        )
        
        # Feed-forward with residual connection
        ff_output = self.feedforward(attention_output)
        
        # Final residual connection and layer norm
        output = self._secure_layer_norm(
            self.protocol.secure_add(attention_output, ff_output),
            self.output_norm
        )
        
        return output
    
    def _secure_layer_norm(self, x: SecureValue, norm_layer: nn.LayerNorm) -> SecureValue:
        """Approximate secure layer normalization."""
        # Simplified implementation - in practice would use secure statistics
        # For now, reconstruct, normalize, and re-share
        plaintext = self.protocol.reconstruct_value(x)
        normalized = norm_layer(plaintext)
        return self.protocol.share_value(normalized)


class SecureTransformer(nn.Module):
    """Secure transformer model for MPC inference."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Initialize MPC protocol
        self.protocol = ProtocolFactory.create(
            config.protocol_name,
            party_id=config.party_id,
            num_parties=config.num_parties,
            device=torch.device("cuda" if config.gpu_acceleration and torch.cuda.is_available() else "cpu")
        )
        self.protocol.initialize()
        
        # Model components
        self.embeddings = SecureEmbeddings(config, self.protocol)
        self.layers = nn.ModuleList([
            SecureTransformerLayer(config, self.protocol) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        except:
            logger.warning(f"Could not load tokenizer for {config.model_name}")
            self.tokenizer = None
            
        self.device = self.protocol.device
        
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "SecureTransformer":
        """Load secure transformer from pre-trained model."""
        config = TransformerConfig.from_pretrained(model_name, **kwargs)
        model = cls(config)
        
        # Load and convert pre-trained weights
        try:
            weights_dict = model._load_pretrained_weights(model_name)
            model._convert_weights_to_secure(weights_dict)
            logger.info(f"Loaded pretrained weights for {model_name} ({len(weights_dict)} tensors)")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights for {model_name}: {e}")
            logger.info("Using randomly initialized weights")
        
        return model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> SecureValue:
        """Forward pass through secure transformer."""
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states
    
    def predict_secure(self, text: str) -> Dict[str, Any]:
        """Secure inference on text input."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for this model")
        
        # Tokenize input
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Secure forward pass
        import time
        start_time = time.time()
        
        with torch.no_grad():
            secure_output = self.forward(input_ids, attention_mask)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # For demonstration, reconstruct output (in practice would depend on use case)
        output_tensor = self.protocol.reconstruct_value(secure_output)
        
        return {
            "secure_output": secure_output,
            "output_tensor": output_tensor,
            "latency_ms": latency_ms,
            "protocol_info": self.protocol.get_protocol_info(),
            "input_shape": input_ids.shape,
            "output_shape": output_tensor.shape
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "config": self.config.__dict__,
            "protocol_info": self.protocol.get_protocol_info(),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        super().to(device)
        self.device = device
        return self
    
    def _load_pretrained_weights(self, model_name: str) -> Dict[str, torch.Tensor]:
        """Load pretrained model weights."""
        try:
            # Try to load from HuggingFace transformers first
            from transformers import AutoModel
            pretrained_model = AutoModel.from_pretrained(model_name)
            state_dict = pretrained_model.state_dict()
            
            # Filter weights that match our architecture
            filtered_weights = {}
            
            # Map common weight names to our secure model structure
            weight_mapping = {
                # Embeddings
                "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
                "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight", 
                "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
                "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
                "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
            }
            
            # Add transformer layer mappings
            for i in range(self.config.num_hidden_layers):
                layer_prefix = f"encoder.layer.{i}"
                our_prefix = f"layers.{i}"
                
                weight_mapping.update({
                    f"{layer_prefix}.attention.self.query.weight": f"{our_prefix}.attention.query.weight",
                    f"{layer_prefix}.attention.self.query.bias": f"{our_prefix}.attention.query.bias",
                    f"{layer_prefix}.attention.self.key.weight": f"{our_prefix}.attention.key.weight", 
                    f"{layer_prefix}.attention.self.key.bias": f"{our_prefix}.attention.key.bias",
                    f"{layer_prefix}.attention.self.value.weight": f"{our_prefix}.attention.value.weight",
                    f"{layer_prefix}.attention.self.value.bias": f"{our_prefix}.attention.value.bias",
                    f"{layer_prefix}.attention.output.dense.weight": f"{our_prefix}.attention.output.weight",
                    f"{layer_prefix}.attention.output.dense.bias": f"{our_prefix}.attention.output.bias",
                    f"{layer_prefix}.attention.output.LayerNorm.weight": f"{our_prefix}.attention_norm.weight",
                    f"{layer_prefix}.attention.output.LayerNorm.bias": f"{our_prefix}.attention_norm.bias",
                    f"{layer_prefix}.intermediate.dense.weight": f"{our_prefix}.feed_forward.intermediate.weight",
                    f"{layer_prefix}.intermediate.dense.bias": f"{our_prefix}.feed_forward.intermediate.bias",
                    f"{layer_prefix}.output.dense.weight": f"{our_prefix}.feed_forward.output.weight",
                    f"{layer_prefix}.output.dense.bias": f"{our_prefix}.feed_forward.output.bias", 
                    f"{layer_prefix}.output.LayerNorm.weight": f"{our_prefix}.output_norm.weight",
                    f"{layer_prefix}.output.LayerNorm.bias": f"{our_prefix}.output_norm.bias",
                })
            
            # Extract matching weights
            for pretrained_name, tensor in state_dict.items():
                if pretrained_name in weight_mapping:
                    our_name = weight_mapping[pretrained_name]
                    filtered_weights[our_name] = tensor.clone()
            
            logger.info(f"Mapped {len(filtered_weights)} weight tensors from {model_name}")
            return filtered_weights
            
        except ImportError:
            logger.warning("transformers library not available, cannot load pretrained weights")
            return {}
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            return {}
    
    def _convert_weights_to_secure(self, weights_dict: Dict[str, torch.Tensor]):
        """Convert loaded weights to secure shared format."""
        if not weights_dict:
            return
            
        # Load weights into the model and then convert to secure format
        missing_keys = []
        unexpected_keys = []
        
        model_state = self.state_dict()
        
        for name, tensor in weights_dict.items():
            if name in model_state:
                # Check shape compatibility
                if tensor.shape == model_state[name].shape:
                    model_state[name].copy_(tensor)
                else:
                    logger.warning(f"Shape mismatch for {name}: {tensor.shape} vs {model_state[name].shape}")
                    missing_keys.append(name)
            else:
                unexpected_keys.append(name)
        
        # Find missing keys
        for name in model_state:
            if name not in weights_dict:
                missing_keys.append(name)
        
        if missing_keys:
            logger.info(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys: {unexpected_keys}")
        
        # Load the updated state dict
        self.load_state_dict(model_state)
        
        # For secure MPC, we would normally convert these weights to secret shares
        # For now, we keep them as plaintext since the sharing happens during inference
        logger.info("Weights loaded and ready for secure computation")
    
    def save_secure_checkpoint(self, checkpoint_path: str):
        """Save secure model checkpoint."""
        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            "config": self.config.__dict__,
            "model_state_dict": self.state_dict(),
            "protocol_info": self.protocol.get_protocol_info() if hasattr(self.protocol, 'get_protocol_info') else {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Secure checkpoint saved to {checkpoint_path}")
    
    @classmethod 
    def load_secure_checkpoint(cls, checkpoint_path: str, **kwargs) -> "SecureTransformer":
        """Load secure model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Merge config from checkpoint with any overrides
        config_dict = checkpoint["config"]
        config_dict.update(kwargs)
        config = TransformerConfig(**config_dict)
        
        # Create model and load weights
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Secure model loaded from checkpoint: {checkpoint_path}")
        return model