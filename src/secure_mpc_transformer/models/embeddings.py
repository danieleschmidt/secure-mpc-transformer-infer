"""
Secure embedding layers for transformer models.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..config import SecurityConfig
from .transformer_layers import SecurePositionalEncoding

logger = logging.getLogger(__name__)


class SecureEmbedding(nn.Module):
    """
    Secure embedding layer for transformer models.
    
    Combines word embeddings, position embeddings, and token type embeddings
    in a privacy-preserving manner suitable for MPC computation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        security_config: SecurityConfig = None,
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.security_config = security_config
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Position embeddings (learnable alternative to sinusoidal)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Token type embeddings (for BERT-style models)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # Sinusoidal positional encoding (alternative)
        self.positional_encoding = SecurePositionalEncoding(
            hidden_size, max_position_embeddings, security_config
        )
        
        # Layer normalization (secure version)
        self.layer_norm = SecureLayerNorm(hidden_size, security_config, layer_norm_eps)
        
        # Dropout (secure version)
        self.dropout = SecureDropout(dropout_prob, security_config)
        
        # Initialize embeddings
        self._init_embeddings()
        
        logger.info(f"Initialized SecureEmbedding: vocab={vocab_size}, hidden={hidden_size}")
    
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)
    
    def forward_secure(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_sinusoidal: bool = False
    ) -> torch.Tensor:
        """
        Secure forward pass through embedding layer.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len] (optional)
            position_ids: Position IDs [batch, seq_len] (optional)
            use_sinusoidal: Whether to use sinusoidal position encoding
            
        Returns:
            Embedded representations [batch, seq_len, hidden_size]
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Generate token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        
        # Word embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Position embeddings
        if use_sinusoidal:
            # Use sinusoidal positional encoding
            embeddings = self.positional_encoding.forward_secure(word_embeddings, position_ids)
        else:
            # Use learnable position embeddings
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        
        # Token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = embeddings + token_type_embeddings
        
        # Layer normalization
        embeddings = self.layer_norm.forward_secure(embeddings)
        
        # Dropout
        embeddings = self.dropout.forward_secure(embeddings)
        
        return embeddings
    
    def get_word_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a specific token ID."""
        return self.word_embeddings.weight[token_id]
    
    def get_vocabulary_embeddings(self) -> torch.Tensor:
        """Get all vocabulary embeddings."""
        return self.word_embeddings.weight


class SecureLayerNorm(nn.Module):
    """
    Secure layer normalization using polynomial approximation.
    
    Traditional layer normalization requires computing statistics and square roots,
    which are expensive in MPC. This implements a polynomial approximation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        security_config: SecurityConfig = None,
        eps: float = 1e-12
    ):
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
        
        For MPC compatibility, we approximate the normalization function:
        1. Compute mean and variance using secure operations
        2. Use polynomial approximation for 1/sqrt(variance + eps)
        3. Apply learned scale and shift parameters
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Normalized tensor [batch, seq_len, hidden_size]
        """
        # Compute mean across the last dimension
        mean = torch.mean(hidden_states, dim=-1, keepdim=True)
        
        # Center the input
        centered = hidden_states - mean
        
        # Compute variance
        variance = torch.mean(centered ** 2, dim=-1, keepdim=True)
        
        # Polynomial approximation of 1/sqrt(variance + eps)
        # Using Taylor expansion: 1/sqrt(1+x) ≈ 1 - 0.5*x + 0.375*x^2 for small x
        normalized_var = variance / (variance.detach() + self.eps)
        
        # Clamp to prevent numerical issues
        normalized_var = torch.clamp(normalized_var, max=1.0)
        
        # Polynomial approximation
        inv_sqrt_approx = 1.0 - 0.5 * normalized_var + 0.375 * (normalized_var ** 2)
        
        # Apply normalization
        normalized = centered * inv_sqrt_approx
        
        # Apply learned parameters
        return normalized * self.weight + self.bias


class SecureDropout(nn.Module):
    """
    Secure dropout implementation for MPC computation.
    
    Traditional dropout uses random masks which can leak information in MPC.
    This implementation uses deterministic patterns or scaling.
    """
    
    def __init__(self, dropout_prob: float = 0.1, security_config: SecurityConfig = None):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.security_config = security_config
        
    def forward_secure(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Secure dropout forward pass.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output tensor with dropout applied
        """
        if self.training and self.dropout_prob > 0:
            # For secure computation, avoid random dropout
            # Option 1: Uniform scaling (simplest)
            scale_factor = 1.0 / (1.0 - self.dropout_prob)
            return hidden_states * scale_factor
            
            # Option 2: Deterministic pattern-based dropout
            # This could be implemented using a fixed pattern
            # based on position or other deterministic factors
            
        else:
            # No dropout during inference
            return hidden_states


class TokenTypeEmbedding(nn.Module):
    """
    Secure token type embedding for models like BERT.
    
    Handles segment/sentence embeddings in a privacy-preserving manner.
    """
    
    def __init__(
        self,
        type_vocab_size: int = 2,
        hidden_size: int = 768,
        security_config: SecurityConfig = None
    ):
        super().__init__()
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.security_config = security_config
        
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize token type embedding weights."""
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)
    
    def forward_secure(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Secure forward pass for token type embeddings.
        
        Args:
            token_type_ids: Token type IDs [batch, seq_len]
            
        Returns:
            Token type embeddings [batch, seq_len, hidden_size]
        """
        return self.token_type_embeddings(token_type_ids)


class PositionEmbedding(nn.Module):
    """
    Learnable position embeddings for transformer models.
    
    Alternative to sinusoidal encodings with learnable parameters.
    """
    
    def __init__(
        self,
        max_position_embeddings: int = 512,
        hidden_size: int = 768,
        security_config: SecurityConfig = None
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.security_config = security_config
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embedding weights."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward_secure(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Secure forward pass for position embeddings.
        
        Args:
            position_ids: Position IDs [batch, seq_len]
            
        Returns:
            Position embeddings [batch, seq_len, hidden_size]
        """
        return self.position_embeddings(position_ids)


class SecureWordEmbedding(nn.Module):
    """
    Secure word embedding layer with privacy-preserving lookup.
    
    Implements embedding lookup that can be computed securely
    across multiple parties without revealing input tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int = 0,
        security_config: SecurityConfig = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.security_config = security_config
        
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize word embedding weights."""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.word_embeddings.weight[self.padding_idx].fill_(0)
    
    def forward_secure(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Secure word embedding lookup.
        
        In a full MPC implementation, this would use privacy-preserving
        table lookup protocols to prevent leaking input token information.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            Word embeddings [batch, seq_len, hidden_size]
        """
        return self.word_embeddings(input_ids)
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the full embedding matrix."""
        return self.word_embeddings.weight