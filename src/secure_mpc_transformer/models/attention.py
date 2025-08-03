"""Secure attention mechanisms for MPC transformer inference."""

import torch
import torch.nn as nn
import math
from typing import Optional

from ..protocols.base import Protocol, SecureValue


class SecureAttention(nn.Module):
    """Secure multi-head self-attention mechanism."""
    
    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections for queries, keys, and values
        # In practice, these would be loaded from pre-trained weights
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.scale = 1.0 / math.sqrt(self.attention_head_size)
        
    def forward(self, hidden_states: SecureValue, attention_mask: Optional[torch.Tensor] = None) -> SecureValue:
        """Secure multi-head attention forward pass."""
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Generate Q, K, V through secure linear transformations
        query_layer = self._secure_linear_projection(hidden_states, self.query)
        key_layer = self._secure_linear_projection(hidden_states, self.key)
        value_layer = self._secure_linear_projection(hidden_states, self.value)
        
        # Reshape for multi-head attention
        query_layer = self._transpose_for_scores(query_layer)
        key_layer = self._transpose_for_scores(key_layer)
        value_layer = self._transpose_for_scores(value_layer)
        
        # Compute attention scores
        attention_scores = self._secure_attention_scores(query_layer, key_layer)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = self._apply_attention_mask(attention_scores, attention_mask)
        
        # Apply softmax (approximated securely)
        attention_probs = self.protocol.secure_softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context_layer = self._secure_attention_output(attention_probs, value_layer)
        
        # Reshape back to original dimensions
        context_layer = self._transpose_from_scores(context_layer)
        
        # Final output projection
        output = self._secure_linear_projection(context_layer, self.output)
        
        return output
    
    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        # Convert weights to secure shares
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = self.protocol.share_value(linear_layer.bias) if linear_layer.bias is not None else None
        
        # Secure matrix multiplication
        output = self.protocol.secure_matmul(input_tensor, weight_shared)
        
        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)
        
        return output
    
    def _transpose_for_scores(self, x: SecureValue) -> SecureValue:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_length, _ = x.shape
        
        # Reshape shares individually
        reshaped_shares = []
        for share in x.shares:
            new_shape = (batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            reshaped = share.view(new_shape)
            # Transpose to (batch_size, num_heads, seq_length, head_size)
            transposed = reshaped.permute(0, 2, 1, 3)
            reshaped_shares.append(transposed)
        
        return SecureValue(
            shares=reshaped_shares,
            party_id=x.party_id,
            is_public=x.is_public
        )
    
    def _transpose_from_scores(self, x: SecureValue) -> SecureValue:
        """Reshape tensor back from multi-head attention format."""
        batch_size, num_heads, seq_length, head_size = x.shares[0].shape
        
        # Transpose and reshape shares
        reshaped_shares = []
        for share in x.shares:
            # Transpose back to (batch_size, seq_length, num_heads, head_size)
            transposed = share.permute(0, 2, 1, 3)
            # Reshape to (batch_size, seq_length, hidden_size)
            reshaped = transposed.contiguous().view(batch_size, seq_length, self.all_head_size)
            reshaped_shares.append(reshaped)
        
        return SecureValue(
            shares=reshaped_shares,
            party_id=x.party_id,
            is_public=x.is_public
        )
    
    def _secure_attention_scores(self, query: SecureValue, key: SecureValue) -> SecureValue:
        """Compute attention scores securely."""
        # Secure matrix multiplication: Q * K^T
        # Need to transpose key tensor
        key_transposed = self._transpose_key(key)
        
        # Secure matrix multiplication
        attention_scores = self.protocol.secure_matmul(query, key_transposed)
        
        # Scale by sqrt(d_k)
        attention_scores = attention_scores * self.scale
        
        return attention_scores
    
    def _transpose_key(self, key: SecureValue) -> SecureValue:
        """Transpose key tensor for attention computation."""
        transposed_shares = []
        for share in key.shares:
            # Transpose last two dimensions
            transposed = share.transpose(-2, -1)
            transposed_shares.append(transposed)
        
        return SecureValue(
            shares=transposed_shares,
            party_id=key.party_id,
            is_public=key.is_public
        )
    
    def _apply_attention_mask(self, attention_scores: SecureValue, attention_mask: torch.Tensor) -> SecureValue:
        """Apply attention mask to scores."""
        # Convert mask to large negative values for masked positions
        mask_value = torch.tensor(-10000.0, device=attention_mask.device)
        
        # Expand mask to match attention scores shape
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
        expanded_mask = expanded_mask.expand(-1, self.num_attention_heads, -1, -1)
        
        # Create additive mask (0 for valid positions, large negative for masked)
        additive_mask = (1.0 - expanded_mask) * mask_value
        
        # Share the mask and add to attention scores
        mask_shared = self.protocol.share_value(additive_mask)
        masked_scores = self.protocol.secure_add(attention_scores, mask_shared)
        
        return masked_scores
    
    def _secure_attention_output(self, attention_probs: SecureValue, value: SecureValue) -> SecureValue:
        """Apply attention weights to values."""
        # Secure matrix multiplication: attention_probs * V
        context = self.protocol.secure_matmul(attention_probs, value)
        return context


class SecureMultiHeadAttention(nn.Module):
    """Alternative implementation with explicit multi-head structure."""
    
    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Individual attention heads
        self.attention_heads = nn.ModuleList([
            SecureSingleHeadAttention(config, protocol, self.head_dim)
            for _ in range(self.num_heads)
        ])
        
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, hidden_states: SecureValue, attention_mask: Optional[torch.Tensor] = None) -> SecureValue:
        """Forward pass through multi-head attention."""
        head_outputs = []
        
        # Process each head independently
        for head in self.attention_heads:
            head_output = head(hidden_states, attention_mask)
            head_outputs.append(head_output)
        
        # Concatenate head outputs
        concatenated = self._concatenate_heads(head_outputs)
        
        # Final output projection
        output = self._secure_linear_projection(concatenated, self.output_projection)
        
        return output
    
    def _concatenate_heads(self, head_outputs: list) -> SecureValue:
        """Concatenate outputs from multiple attention heads."""
        # Concatenate shares along the last dimension
        concatenated_shares = []
        for i, share in enumerate(head_outputs[0].shares):
            head_shares_i = [head.shares[i] for head in head_outputs]
            concatenated_share = torch.cat(head_shares_i, dim=-1)
            concatenated_shares.append(concatenated_share)
        
        return SecureValue(
            shares=concatenated_shares,
            party_id=head_outputs[0].party_id,
            is_public=head_outputs[0].is_public
        )
    
    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = self.protocol.share_value(linear_layer.bias) if linear_layer.bias is not None else None
        
        output = self.protocol.secure_matmul(input_tensor, weight_shared)
        
        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)
        
        return output


class SecureSingleHeadAttention(nn.Module):
    """Single attention head for use in multi-head attention."""
    
    def __init__(self, config, protocol: Protocol, head_dim: int):
        super().__init__()
        self.config = config
        self.protocol = protocol
        self.head_dim = head_dim
        
        self.query = nn.Linear(config.hidden_size, head_dim)
        self.key = nn.Linear(config.hidden_size, head_dim)
        self.value = nn.Linear(config.hidden_size, head_dim)
        
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, hidden_states: SecureValue, attention_mask: Optional[torch.Tensor] = None) -> SecureValue:
        """Forward pass for single attention head."""
        # Generate Q, K, V
        q = self._secure_linear_projection(hidden_states, self.query)
        k = self._secure_linear_projection(hidden_states, self.key)
        v = self._secure_linear_projection(hidden_states, self.value)
        
        # Compute attention
        scores = self.protocol.secure_matmul(q, self._transpose_tensor(k))
        scores = scores * self.scale
        
        if attention_mask is not None:
            scores = self._apply_mask(scores, attention_mask)
        
        probs = self.protocol.secure_softmax(scores, dim=-1)
        output = self.protocol.secure_matmul(probs, v)
        
        return output
    
    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = self.protocol.share_value(linear_layer.bias) if linear_layer.bias is not None else None
        
        output = self.protocol.secure_matmul(input_tensor, weight_shared)
        
        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)
        
        return output
    
    def _transpose_tensor(self, x: SecureValue) -> SecureValue:
        """Transpose last two dimensions of tensor."""
        transposed_shares = [share.transpose(-2, -1) for share in x.shares]
        return SecureValue(
            shares=transposed_shares,
            party_id=x.party_id,
            is_public=x.is_public
        )
    
    def _apply_mask(self, scores: SecureValue, mask: torch.Tensor) -> SecureValue:
        """Apply attention mask."""
        mask_value = torch.tensor(-10000.0, device=mask.device)
        additive_mask = (1.0 - mask.unsqueeze(-1)) * mask_value
        mask_shared = self.protocol.share_value(additive_mask)
        return self.protocol.secure_add(scores, mask_shared)