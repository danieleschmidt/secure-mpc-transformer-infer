"""
Secure transformer layer implementations for MPC computation.
"""

import math
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SecurityConfig

logger = logging.getLogger(__name__)


class SecureAttention(nn.Module):
    """
    Secure multi-head self-attention mechanism.
    
    Implements privacy-preserving attention using polynomial approximations
    for softmax and other non-linear operations.
    """
    
    def __init__(self, config, security_config: SecurityConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.security_config = security_config
        
        # Linear projections for Q, K, V
        self.query = SecureLinear(config.hidden_size, self.all_head_size, security_config)
        self.key = SecureLinear(config.hidden_size, self.all_head_size, security_config)
        self.value = SecureLinear(config.hidden_size, self.all_head_size, security_config)
        
        # Output projection
        self.output = SecureLinear(self.all_head_size, config.hidden_size, security_config)
        
        # Dropout (simplified for secure computation)
        self.dropout_prob = getattr(config, 'attention_probs_dropout_prob', 0.1)
        
        # Scale factor for attention
        self.scale = math.sqrt(self.attention_head_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_size)
    
    def secure_softmax(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Polynomial approximation of softmax for secure computation.
        
        Uses a polynomial approximation: softmax(x) ≈ sigmoid(x) normalized
        where sigmoid(x) ≈ 0.5 + 0.25*x - 0.01*x^3 for |x| < 3
        """
        # Clip scores to prevent overflow
        clipped_scores = torch.clamp(attention_scores, min=-3.0, max=3.0)
        
        # Polynomial approximation of sigmoid
        sigmoid_approx = 0.5 + 0.25 * clipped_scores - 0.01 * (clipped_scores ** 3)
        sigmoid_approx = torch.clamp(sigmoid_approx, min=0.01, max=0.99)
        
        # Normalize to sum to 1 (approximate softmax)
        attention_probs = sigmoid_approx / (torch.sum(sigmoid_approx, dim=-1, keepdim=True) + 1e-8)
        
        return attention_probs
    
    def forward_secure(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Secure forward pass through attention layer.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with attention output and optional attention weights
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Generate Q, K, V using secure linear layers
        query_layer = self.query.forward_secure(hidden_states)
        key_layer = self.key.forward_secure(hidden_states)
        value_layer = self.value.forward_secure(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)  # [batch, heads, seq, head_size]
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Compute attention scores
        # Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, self.num_attention_heads, seq_length, seq_length
            )
            
            # Apply mask (set masked positions to very negative values)
            attention_scores = attention_scores + (1.0 - extended_attention_mask) * -10000.0
        
        # Secure softmax approximation
        attention_probs = self.secure_softmax(attention_scores)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Final output projection
        attention_output = self.output.forward_secure(context_layer)
        
        outputs = {'attention_output': attention_output}
        
        if return_attention:
            outputs['attention_weights'] = attention_probs
        
        return outputs


class SecureFeedForward(nn.Module):
    """
    Secure feed-forward network with polynomial activation.
    
    Replaces GELU/ReLU with polynomial approximations for MPC compatibility.
    """
    
    def __init__(self, config, security_config: SecurityConfig):
        super().__init__()
        self.security_config = security_config
        
        # Feed-forward dimensions
        self.intermediate_size = getattr(config, 'intermediate_size', 4 * config.hidden_size)
        
        # Linear layers
        self.dense_1 = SecureLinear(
            config.hidden_size,
            self.intermediate_size,
            security_config
        )
        self.dense_2 = SecureLinear(
            self.intermediate_size,
            config.hidden_size,
            security_config
        )
        
        # Activation function type
        self.activation_type = getattr(config, 'hidden_act', 'gelu')
        
    def secure_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Polynomial approximation of activation functions.
        
        For GELU: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        Simplified polynomial: x * (0.5 + 0.25*x - 0.01*x^3) for |x| < 2
        """
        if self.activation_type.lower() in ['gelu', 'gelu_new']:
            # Polynomial approximation of GELU
            clipped_x = torch.clamp(x, min=-2.0, max=2.0)
            return clipped_x * (0.5 + 0.25 * clipped_x - 0.01 * (clipped_x ** 3))
        
        elif self.activation_type.lower() == 'relu':
            # Polynomial approximation of ReLU: max(0, x) ≈ 0.5*x + 0.5*|x|
            # Using x^2 approximation: x * sigmoid(10*x) ≈ ReLU for practical ranges
            return x * torch.sigmoid(10 * x)
        
        elif self.activation_type.lower() == 'swish':
            # Swish: x * sigmoid(x)
            sigmoid_approx = torch.sigmoid(x)  # Could be further approximated
            return x * sigmoid_approx
        
        else:
            # Default to polynomial GELU
            clipped_x = torch.clamp(x, min=-2.0, max=2.0)
            return clipped_x * (0.5 + 0.25 * clipped_x - 0.01 * (clipped_x ** 3))
    
    def forward_secure(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Secure forward pass through feed-forward network.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            
        Returns:
            Output hidden states [batch, seq_len, hidden_size]
        """
        # First linear transformation
        intermediate_output = self.dense_1.forward_secure(hidden_states)
        
        # Secure activation function
        intermediate_output = self.secure_activation(intermediate_output)
        
        # Second linear transformation
        output = self.dense_2.forward_secure(intermediate_output)
        
        return output


class SecureLinear(nn.Module):
    """
    Secure linear layer using MPC-compatible operations.
    
    Implements matrix multiplication that can be computed securely
    across multiple parties.
    """
    
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
        
        # Initialize weights using Xavier initialization
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward_secure(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Secure matrix multiplication.
        
        In a full MPC implementation, this would use secure protocols
        for matrix multiplication. For now, we implement the basic operation
        with awareness of the secure computation context.
        
        Args:
            input_tensor: Input tensor [batch, seq_len, in_features]
            
        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        # Secure matrix multiplication: input @ weight.T
        # In actual MPC, this would be computed using secure protocols
        output = torch.matmul(input_tensor, self.weight.t())
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def get_secure_operation_count(self, input_shape: Tuple[int, ...]) -> Dict[str, int]:
        """
        Calculate the number of secure operations required.
        
        This is useful for benchmarking and optimization in MPC protocols.
        """
        batch_size, seq_len = input_shape[:2]
        
        # Matrix multiplication operations
        mult_ops = batch_size * seq_len * self.in_features * self.out_features
        
        # Addition operations (bias)
        add_ops = batch_size * seq_len * self.out_features if self.bias is not None else 0
        
        return {
            'multiplications': mult_ops,
            'additions': add_ops,
            'total_operations': mult_ops + add_ops
        }


class SecureDropout(nn.Module):
    """
    Secure dropout layer for MPC computation.
    
    In secure computation, traditional dropout with random masks
    can leak information. This implements a deterministic alternative.
    """
    
    def __init__(self, dropout_prob: float = 0.1, security_config: SecurityConfig = None):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.security_config = security_config
        
    def forward_secure(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Secure dropout implementation.
        
        Instead of random dropout, we use a deterministic pattern
        or simply scale the activations during training.
        """
        if self.training and self.dropout_prob > 0:
            # For secure computation, we avoid random dropout
            # Instead, scale all activations uniformly
            scale_factor = 1.0 / (1.0 - self.dropout_prob)
            return hidden_states * scale_factor
        else:
            return hidden_states


class SecurePositionalEncoding(nn.Module):
    """
    Secure positional encoding for transformer models.
    
    Uses deterministic sinusoidal encodings that don't leak position information.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 512,
        security_config: SecurityConfig = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.security_config = security_config
        
        # Create positional encoding matrix
        pe = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(0, max_position_embeddings).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            -(math.log(10000.0) / hidden_size)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward_secure(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add positional encoding to hidden states.
        
        Args:
            hidden_states: Input embeddings [batch, seq_len, hidden_size]
            position_ids: Position IDs [batch, seq_len]
            
        Returns:
            Hidden states with positional encoding added
        """
        seq_len = hidden_states.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        # Get positional encodings
        pos_embeddings = self.pe[:, :seq_len, :]
        
        # Add to hidden states
        return hidden_states + pos_embeddings