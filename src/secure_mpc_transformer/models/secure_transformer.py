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
        
        # In practice, would load and convert pre-trained weights
        logger.info(f"Initialized secure transformer: {model_name}")
        
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