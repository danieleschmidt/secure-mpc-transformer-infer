"""Secure embedding layers for MPC transformer inference."""

import torch
import torch.nn as nn
from typing import Optional

from ..protocols.base import Protocol, SecureValue


class SecureEmbeddings(nn.Module):
    """Secure embedding layer for transformer input processing."""
    
    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        
        # Embedding layers
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> SecureValue:
        """Forward pass through secure embeddings."""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Generate token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings (these are plaintext operations)
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Sum embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds
        
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        # Apply dropout (in training mode)
        if self.training:
            embeddings = self.dropout(embeddings)
        
        # Convert to secure shares
        secure_embeddings = self.protocol.share_value(embeddings)
        
        return secure_embeddings


class SecurePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for secure transformers."""
    
    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        self.hidden_size = config.hidden_size
        self.max_length = config.max_position_embeddings
        
        # Pre-compute positional encodings
        self.register_buffer('pe', self._create_positional_encoding())
        
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(self.max_length, self.hidden_size)
        position = torch.arange(0, self.max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / self.hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x: SecureValue, position_ids: Optional[torch.Tensor] = None) -> SecureValue:
        """Add positional encoding to input embeddings."""
        batch_size, seq_length, hidden_size = x.shape
        
        if position_ids is None:
            # Use sequential positions
            pos_encoding = self.pe[:, :seq_length, :]
        else:
            # Use specified positions
            pos_encoding = self.pe[:, position_ids, :]
        
        # Expand to match batch size
        pos_encoding = pos_encoding.expand(batch_size, -1, -1)
        
        # Convert positional encoding to secure shares
        secure_pos_encoding = self.protocol.share_value(pos_encoding)
        
        # Add to input embeddings
        return self.protocol.secure_add(x, secure_pos_encoding)


class SecureLearnableEmbeddings(nn.Module):
    """Learnable embeddings with secure operations."""
    
    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        
        # Learnable embedding tables
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Additional embedding types for specialized tokens
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=0.02)
        
        # Zero out padding token embedding
        if self.word_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.word_embeddings.weight[self.word_embeddings.padding_idx].fill_(0)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None) -> SecureValue:
        """Forward pass with learnable embeddings."""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Generate default position and token type IDs
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Lookup embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_embeds + position_embeds + segment_embeds
        
        # Normalize and dropout
        embeddings = self.layer_norm(embeddings)
        if self.training:
            embeddings = self.dropout(embeddings)
        
        # Convert to secure representation
        return self.protocol.share_value(embeddings)


class SecureRelativePositionalEmbedding(nn.Module):
    """Relative positional embeddings for secure transformers."""
    
    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_relative_positions = 2 * config.max_position_embeddings + 1
        
        # Relative position embeddings
        self.relative_positions_embeddings = nn.Embedding(
            self.max_relative_positions, 
            self.head_dim
        )
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize relative position embeddings."""
        nn.init.normal_(self.relative_positions_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, seq_length: int) -> torch.Tensor:
        """Generate relative position embeddings."""
        # Create relative position matrix
        range_vec = torch.arange(seq_length)
        range_mat = range_vec.unsqueeze(0).expand(seq_length, -1)
        distance_mat = range_mat - range_mat.t()
        
        # Clip distances to maximum range
        max_distance = self.max_relative_positions // 2
        distance_mat_clipped = torch.clamp(distance_mat, -max_distance, max_distance)
        
        # Shift to positive indices
        final_mat = distance_mat_clipped + max_distance
        
        # Get embeddings
        embeddings = self.relative_positions_embeddings(final_mat)
        
        return embeddings
    
    def get_secure_relative_embeddings(self, seq_length: int) -> SecureValue:
        """Get secure relative position embeddings."""
        embeddings = self.forward(seq_length)
        return self.protocol.share_value(embeddings)


class SecureAdaptiveEmbedding(nn.Module):
    """Adaptive embeddings with different dimensions for frequent/rare tokens."""
    
    def __init__(self, config, protocol: Protocol, cutoffs: list = None):
        super().__init__()
        self.config = config
        self.protocol = protocol
        
        # Default cutoffs for adaptive embedding
        if cutoffs is None:
            cutoffs = [config.vocab_size // 4, config.vocab_size // 2, config.vocab_size]
        
        self.cutoffs = cutoffs
        self.div_val = 4  # Dimension division factor
        
        # Create embedding layers with different dimensions
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ModuleList()
        
        for i, (cutoff, next_cutoff) in enumerate(zip([0] + cutoffs[:-1], cutoffs)):
            vocab_size = next_cutoff - cutoff
            emb_dim = config.hidden_size // (self.div_val ** i)
            
            self.emb_layers.append(nn.Embedding(vocab_size, emb_dim))
            
            if emb_dim != config.hidden_size:
                self.emb_projs.append(nn.Linear(emb_dim, config.hidden_size))
            else:
                self.emb_projs.append(None)
    
    def forward(self, input_ids: torch.Tensor) -> SecureValue:
        """Forward pass with adaptive embeddings."""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, seq_length, self.config.hidden_size, device=device)
        
        # Process each cutoff range
        for i, (cutoff, next_cutoff) in enumerate(zip([0] + self.cutoffs[:-1], self.cutoffs)):
            # Find tokens in this range
            mask = (input_ids >= cutoff) & (input_ids < next_cutoff)
            
            if mask.any():
                # Adjust token IDs for this embedding layer
                masked_ids = input_ids[mask] - cutoff
                
                # Get embeddings
                emb = self.emb_layers[i](masked_ids)
                
                # Project to full dimension if needed
                if self.emb_projs[i] is not None:
                    emb = self.emb_projs[i](emb)
                
                # Place embeddings in correct positions
                embeddings[mask] = emb
        
        # Convert to secure representation
        return self.protocol.share_value(embeddings)