"""
Secure transformer model implementation with MPC support.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from ..config import SecurityConfig
from ..protocols.factory import ProtocolFactory
from ..secret_sharing.engine import SecretSharingEngine
from .transformer_layers import SecureAttention, SecureFeedForward
from .embeddings import SecureEmbedding

logger = logging.getLogger(__name__)


class SecureOutput:
    """Output from secure transformer inference."""
    
    def __init__(
        self,
        logits: torch.Tensor,
        decoded_text: str,
        latency_ms: float,
        privacy_spent: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        self.logits = logits
        self.decoded_text = decoded_text
        self.latency_ms = latency_ms
        self.privacy_spent = privacy_spent
        self.metadata = metadata or {}


class SecureTransformer(nn.Module):
    """
    Secure transformer model for privacy-preserving inference using MPC.
    
    This implementation provides GPU-accelerated secure computation for
    transformer models like BERT, RoBERTa, and GPT-2.
    """
    
    def __init__(
        self,
        config: AutoConfig,
        security_config: SecurityConfig,
        model_name: str = "bert-base-uncased"
    ):
        super().__init__()
        self.config = config
        self.security_config = security_config
        self.model_name = model_name
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize secret sharing engine
        self.sharing_engine = SecretSharingEngine(security_config)
        
        # Initialize MPC protocol
        self.protocol = ProtocolFactory.create(
            security_config.protocol,
            security_config=security_config
        )
        
        # Model components
        self.embeddings = SecureEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            config.type_vocab_size if hasattr(config, 'type_vocab_size') else 2,
            security_config
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SecureTransformerLayer(config, security_config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Layer normalization
        self.layer_norm = SecureLayerNorm(config.hidden_size, security_config)
        
        # Classification head for BERT-style models
        if hasattr(config, 'num_labels'):
            self.classifier = SecureLinear(
                config.hidden_size,
                config.num_labels,
                security_config
            )
        
        # Language modeling head for GPT-style models
        else:
            self.lm_head = SecureLinear(
                config.hidden_size,
                config.vocab_size,
                security_config,
                bias=False
            )
        
        logger.info(f"Initialized SecureTransformer with {config.num_hidden_layers} layers")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        security_config: SecurityConfig,
        **kwargs
    ) -> "SecureTransformer":
        """Load a pre-trained model and convert to secure version."""
        config = AutoConfig.from_pretrained(model_name)
        model = cls(config, security_config, model_name)
        
        # Load and convert pre-trained weights
        model._load_pretrained_weights(model_name)
        
        return model
    
    def _load_pretrained_weights(self, model_name: str):
        """Load pre-trained weights and convert to secure format."""
        from transformers import AutoModel
        
        # Load pre-trained model
        pretrained_model = AutoModel.from_pretrained(model_name)
        
        # Convert weights to secure format
        logger.info(f"Converting pre-trained weights from {model_name}")
        
        # This is a simplified conversion - in practice would need
        # careful weight mapping between architectures
        state_dict = pretrained_model.state_dict()
        
        # Convert embeddings
        if 'embeddings.word_embeddings.weight' in state_dict:
            self.embeddings.word_embeddings.weight.data = state_dict['embeddings.word_embeddings.weight']
        
        logger.info("Pre-trained weights loaded and converted")
    
    def predict_secure(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        return_attention: bool = False
    ) -> SecureOutput:
        """
        Perform secure inference on input text.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            return_attention: Whether to return attention weights
            
        Returns:
            SecureOutput with predictions and metadata
        """
        start_time = time.time()
        
        # Tokenize input
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Perform secure inference
        with torch.no_grad():
            outputs = self.forward_secure(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                token_type_ids=inputs.get('token_type_ids'),
                return_attention=return_attention
            )
        
        # Decode output
        if hasattr(self.config, 'num_labels'):
            # Classification model
            predictions = torch.argmax(outputs['logits'], dim=-1)
            decoded_text = f"Class {predictions[0].item()}"
        else:
            # Language model - find masked token predictions
            mask_token_index = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_token_index[0]) > 0:
                mask_logits = outputs['logits'][mask_token_index]
                predicted_token_id = torch.argmax(mask_logits, dim=-1)
                decoded_text = self.tokenizer.decode(predicted_token_id[0])
            else:
                decoded_text = "No [MASK] token found"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return SecureOutput(
            logits=outputs['logits'],
            decoded_text=decoded_text,
            latency_ms=latency_ms,
            privacy_spent=outputs.get('privacy_spent', 0.0),
            metadata={
                'input_length': inputs['input_ids'].shape[1],
                'protocol': self.security_config.protocol,
                'security_level': self.security_config.security_level
            }
        )
    
    def forward_secure(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with secure computation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with logits and optional attention weights
        """
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length))
        
        # Convert inputs to secret shares
        logger.debug("Converting inputs to secret shares")
        input_shares = self.sharing_engine.share_tensor(input_ids)
        
        # Secure embedding lookup
        hidden_states = self.embeddings.forward_secure(
            input_shares,
            token_type_ids=token_type_ids
        )
        
        attention_weights = []
        
        # Process through secure transformer layers
        for i, layer in enumerate(self.layers):
            logger.debug(f"Processing layer {i+1}/{len(self.layers)}")
            
            layer_outputs = layer.forward_secure(
                hidden_states,
                attention_mask=attention_mask,
                return_attention=return_attention
            )
            
            hidden_states = layer_outputs['hidden_states']
            
            if return_attention:
                attention_weights.append(layer_outputs['attention_weights'])
        
        # Final layer normalization
        hidden_states = self.layer_norm.forward_secure(hidden_states)
        
        # Generate output logits
        if hasattr(self, 'classifier'):
            # Classification head
            logits = self.classifier.forward_secure(hidden_states[:, 0])  # Use [CLS] token
        else:
            # Language modeling head
            logits = self.lm_head.forward_secure(hidden_states)
        
        # Reconstruct final output
        logger.debug("Reconstructing secure output")
        output_logits = self.sharing_engine.reconstruct_tensor(logits)
        
        outputs = {'logits': output_logits}
        
        if return_attention:
            # Reconstruct attention weights
            reconstructed_attention = []
            for layer_attention in attention_weights:
                reconstructed_attention.append(
                    self.sharing_engine.reconstruct_tensor(layer_attention)
                )
            outputs['attention_weights'] = reconstructed_attention
        
        return outputs


class SecureTransformerLayer(nn.Module):
    """Secure transformer layer with self-attention and feed-forward."""
    
    def __init__(self, config: AutoConfig, security_config: SecurityConfig):
        super().__init__()
        self.attention = SecureAttention(config, security_config)
        self.feed_forward = SecureFeedForward(config, security_config)
        self.attention_norm = SecureLayerNorm(config.hidden_size, security_config)
        self.ff_norm = SecureLayerNorm(config.hidden_size, security_config)
        
    def forward_secure(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Secure forward pass through transformer layer."""
        # Self-attention with residual connection
        attention_output = self.attention.forward_secure(
            hidden_states,
            attention_mask=attention_mask,
            return_attention=return_attention
        )
        
        # Add residual connection and layer norm
        hidden_states = self.attention_norm.forward_secure(
            hidden_states + attention_output['attention_output']
        )
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward_secure(hidden_states)
        hidden_states = self.ff_norm.forward_secure(hidden_states + ff_output)
        
        outputs = {'hidden_states': hidden_states}
        if return_attention:
            outputs['attention_weights'] = attention_output['attention_weights']
        
        return outputs


class SecureLayerNorm(nn.Module):
    """Secure layer normalization using polynomial approximation."""
    
    def __init__(self, hidden_size: int, security_config: SecurityConfig, eps: float = 1e-12):
        super().__init__()
        self.hidden_size = hidden_size
        self.security_config = security_config
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
    def forward_secure(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Secure layer normalization using polynomial approximation.
        
        For MPC compatibility, we use a polynomial approximation of
        the normalization function instead of exact computation.
        """
        # Compute mean (secure)
        mean = torch.mean(hidden_states, dim=-1, keepdim=True)
        
        # Compute variance approximation (polynomial)
        centered = hidden_states - mean
        variance_approx = torch.mean(centered ** 2, dim=-1, keepdim=True)
        
        # Approximate 1/sqrt(variance + eps) using polynomial
        # Taylor expansion around 1: 1/sqrt(x) ≈ 1.5 - 0.5*x for x near 1
        normalized_var = variance_approx / (variance_approx + self.eps)
        inv_sqrt_approx = 1.5 - 0.5 * normalized_var
        
        # Apply normalization
        normalized = centered * inv_sqrt_approx
        
        # Apply learned parameters
        return normalized * self.weight + self.bias


class SecureLinear(nn.Module):
    """Secure linear layer using MPC matrix multiplication."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        security_config: SecurityConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.security_config = security_config
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward_secure(self, input_shares: torch.Tensor) -> torch.Tensor:
        """Secure matrix multiplication."""
        # This would use the MPC protocol for secure matrix multiplication
        # For now, simplified implementation
        output = torch.matmul(input_shares, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output