"""Secure feed-forward networks for MPC transformer inference."""


import torch
import torch.nn as nn

from ..protocols.base import Protocol, SecureValue


class SecureFeedForward(nn.Module):
    """Secure feed-forward network with ReLU activation."""

    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol

        # Feed-forward layers
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: SecureValue) -> SecureValue:
        """Forward pass through secure feed-forward network."""
        # First linear transformation
        intermediate_output = self._secure_linear_projection(hidden_states, self.intermediate)

        # Secure ReLU activation
        activated_output = self.protocol.secure_relu(intermediate_output)

        # Second linear transformation
        output = self._secure_linear_projection(activated_output, self.output)

        return output

    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        # Convert weights to secure shares
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = None
        if linear_layer.bias is not None:
            bias_shared = self.protocol.share_value(linear_layer.bias)

        # Secure matrix multiplication
        output = self.protocol.secure_matmul(input_tensor, weight_shared)

        # Add bias if present
        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)

        return output


class SecureGELUFeedForward(nn.Module):
    """Secure feed-forward network with GELU activation approximation."""

    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: SecureValue) -> SecureValue:
        """Forward pass with secure GELU approximation."""
        # First linear transformation
        intermediate_output = self._secure_linear_projection(hidden_states, self.intermediate)

        # Secure GELU activation
        activated_output = self._secure_gelu(intermediate_output)

        # Second linear transformation
        output = self._secure_linear_projection(activated_output, self.output)

        return output

    def _secure_gelu(self, x: SecureValue) -> SecureValue:
        """
        Secure GELU activation using polynomial approximation.
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        """
        # Simplified polynomial approximation for GELU
        # GELU(x) ≈ x * σ(1.702 * x) where σ is sigmoid

        # Coefficients for approximation
        coeff = 1.702

        # x * 1.702
        scaled_x = x * coeff

        # Sigmoid approximation using tanh: σ(x) ≈ 0.5 * (1 + tanh(x/2))
        sigmoid_approx = self._secure_sigmoid_approx(scaled_x)

        # x * sigmoid(1.702 * x)
        result = self.protocol.secure_multiply(x, sigmoid_approx)

        return result

    def _secure_sigmoid_approx(self, x: SecureValue) -> SecureValue:
        """Polynomial approximation of sigmoid function."""
        # Using polynomial approximation: σ(x) ≈ 0.5 + 0.25x - 0.0208x³

        # Compute x³
        x_squared = self.protocol.secure_multiply(x, x)
        x_cubed = self.protocol.secure_multiply(x_squared, x)

        # 0.5 (constant term)
        half = self.protocol.share_value(torch.full_like(x.shares[0], 0.5))

        # 0.25x
        linear_term = x * 0.25

        # -0.0208x³
        cubic_term = x_cubed * (-0.0208)

        # Combine terms
        result = self.protocol.secure_add(half, linear_term)
        result = self.protocol.secure_add(result, cubic_term)

        return result

    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = None
        if linear_layer.bias is not None:
            bias_shared = self.protocol.share_value(linear_layer.bias)

        output = self.protocol.secure_matmul(input_tensor, weight_shared)

        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)

        return output


class SecureResidualFeedForward(nn.Module):
    """Feed-forward network with explicit residual connections."""

    def __init__(self, config, protocol: Protocol, activation_type: str = "relu"):
        super().__init__()
        self.config = config
        self.protocol = protocol
        self.activation_type = activation_type

        # Multi-layer feed-forward
        self.layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.Linear(config.intermediate_size, config.intermediate_size),
            nn.Linear(config.intermediate_size, config.hidden_size)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: SecureValue) -> SecureValue:
        """Forward pass with residual connections."""
        # Store input for residual connection
        residual = hidden_states

        # First layer + activation
        output = self._secure_linear_projection(hidden_states, self.layers[0])
        output = self._apply_activation(output)

        # Second layer + activation
        output = self._secure_linear_projection(output, self.layers[1])
        output = self._apply_activation(output)

        # Final layer
        output = self._secure_linear_projection(output, self.layers[2])

        # Residual connection
        output = self.protocol.secure_add(residual, output)

        # Layer normalization (simplified)
        output = self._secure_layer_norm(output)

        return output

    def _apply_activation(self, x: SecureValue) -> SecureValue:
        """Apply the specified activation function."""
        if self.activation_type == "relu":
            return self.protocol.secure_relu(x)
        elif self.activation_type == "gelu":
            return self._secure_gelu(x)
        else:
            # Default to ReLU
            return self.protocol.secure_relu(x)

    def _secure_gelu(self, x: SecureValue) -> SecureValue:
        """Secure GELU activation."""
        # Simplified GELU approximation
        coeff = 1.702
        scaled_x = x * coeff
        sigmoid_approx = self._secure_sigmoid_approx(scaled_x)
        return self.protocol.secure_multiply(x, sigmoid_approx)

    def _secure_sigmoid_approx(self, x: SecureValue) -> SecureValue:
        """Polynomial approximation of sigmoid."""
        x_squared = self.protocol.secure_multiply(x, x)
        x_cubed = self.protocol.secure_multiply(x_squared, x)

        half = self.protocol.share_value(torch.full_like(x.shares[0], 0.5))
        linear_term = x * 0.25
        cubic_term = x_cubed * (-0.0208)

        result = self.protocol.secure_add(half, linear_term)
        result = self.protocol.secure_add(result, cubic_term)

        return result

    def _secure_layer_norm(self, x: SecureValue) -> SecureValue:
        """Simplified secure layer normalization."""
        # In practice, would implement secure statistics computation
        # For now, reconstruct, normalize, and re-share
        plaintext = self.protocol.reconstruct_value(x)
        normalized = self.layer_norm(plaintext)
        return self.protocol.share_value(normalized)

    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = None
        if linear_layer.bias is not None:
            bias_shared = self.protocol.share_value(linear_layer.bias)

        output = self.protocol.secure_matmul(input_tensor, weight_shared)

        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)

        return output


class SecureSwishFeedForward(nn.Module):
    """Feed-forward network with Swish activation."""

    def __init__(self, config, protocol: Protocol):
        super().__init__()
        self.config = config
        self.protocol = protocol

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: SecureValue) -> SecureValue:
        """Forward pass with secure Swish activation."""
        # First linear transformation
        intermediate_output = self._secure_linear_projection(hidden_states, self.intermediate)

        # Secure Swish activation: x * sigmoid(x)
        activated_output = self._secure_swish(intermediate_output)

        # Second linear transformation
        output = self._secure_linear_projection(activated_output, self.output)

        return output

    def _secure_swish(self, x: SecureValue) -> SecureValue:
        """Secure Swish activation: x * sigmoid(x)."""
        sigmoid_x = self._secure_sigmoid_approx(x)
        return self.protocol.secure_multiply(x, sigmoid_x)

    def _secure_sigmoid_approx(self, x: SecureValue) -> SecureValue:
        """Polynomial approximation of sigmoid function."""
        x_squared = self.protocol.secure_multiply(x, x)
        x_cubed = self.protocol.secure_multiply(x_squared, x)

        half = self.protocol.share_value(torch.full_like(x.shares[0], 0.5))
        linear_term = x * 0.25
        cubic_term = x_cubed * (-0.0208)

        result = self.protocol.secure_add(half, linear_term)
        result = self.protocol.secure_add(result, cubic_term)

        return result

    def _secure_linear_projection(self, input_tensor: SecureValue, linear_layer: nn.Linear) -> SecureValue:
        """Apply linear transformation securely."""
        weight_shared = self.protocol.share_value(linear_layer.weight.T)
        bias_shared = None
        if linear_layer.bias is not None:
            bias_shared = self.protocol.share_value(linear_layer.bias)

        output = self.protocol.secure_matmul(input_tensor, weight_shared)

        if bias_shared is not None:
            output = self.protocol.secure_add(output, bias_shared)

        return output
