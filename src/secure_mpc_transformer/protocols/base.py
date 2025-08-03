"""Base classes for MPC protocols."""

from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional, Any
import numpy as np
import torch
from dataclasses import dataclass


class ProtocolError(Exception):
    """Exception raised for protocol-related errors."""
    pass


@dataclass
class SecureValue:
    """Represents a secret-shared value in MPC."""
    
    shares: List[torch.Tensor]
    party_id: int
    is_public: bool = False
    
    def __post_init__(self):
        if not self.shares:
            raise ValueError("SecureValue must have at least one share")
    
    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self.shares[0].shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensor."""
        return self.shares[0].dtype
    
    @property
    def device(self) -> torch.device:
        """Device of the underlying tensor."""
        return self.shares[0].device
    
    def to(self, device: Union[str, torch.device]) -> "SecureValue":
        """Move shares to specified device."""
        return SecureValue(
            shares=[share.to(device) for share in self.shares],
            party_id=self.party_id,
            is_public=self.is_public
        )
    
    def __add__(self, other: "SecureValue") -> "SecureValue":
        """Element-wise addition of secure values."""
        if len(self.shares) != len(other.shares):
            raise ValueError("Cannot add SecureValues with different number of shares")
        
        return SecureValue(
            shares=[s1 + s2 for s1, s2 in zip(self.shares, other.shares)],
            party_id=self.party_id,
            is_public=self.is_public and other.is_public
        )
    
    def __mul__(self, scalar: Union[float, int, torch.Tensor]) -> "SecureValue":
        """Scalar multiplication."""
        return SecureValue(
            shares=[share * scalar for share in self.shares],
            party_id=self.party_id,
            is_public=self.is_public
        )


class Protocol(ABC):
    """Abstract base class for MPC protocols."""
    
    def __init__(self, party_id: int, num_parties: int, device: Optional[torch.device] = None):
        self.party_id = party_id
        self.num_parties = num_parties
        self.device = device or torch.device("cpu")
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize protocol-specific parameters."""
        pass
    
    @abstractmethod
    def share_value(self, value: torch.Tensor) -> SecureValue:
        """Convert plaintext value to secret shares."""
        pass
    
    @abstractmethod
    def reconstruct_value(self, secure_value: SecureValue) -> torch.Tensor:
        """Reconstruct plaintext from secret shares."""
        pass
    
    @abstractmethod
    def secure_add(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure addition of two shared values."""
        pass
    
    @abstractmethod
    def secure_multiply(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure multiplication of two shared values."""
        pass
    
    @abstractmethod
    def secure_matmul(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure matrix multiplication."""
        pass
    
    def secure_linear(self, input_val: SecureValue, weight: SecureValue, 
                     bias: Optional[SecureValue] = None) -> SecureValue:
        """Secure linear transformation: y = xW + b."""
        result = self.secure_matmul(input_val, weight)
        if bias is not None:
            result = self.secure_add(result, bias)
        return result
    
    def secure_relu(self, x: SecureValue) -> SecureValue:
        """Secure ReLU activation using polynomial approximation."""
        # Polynomial approximation: max(0, x) ≈ 0.5 * (x + |x|)
        # |x| ≈ x * tanh(kx) for large k (approximation)
        # For simplicity, using quadratic approximation
        zero = self.share_value(torch.zeros_like(x.shares[0]))
        
        # Simple quadratic approximation for positive region
        x_squared = self.secure_multiply(x, x)
        x_scaled = x_squared * 0.25  # Scale factor
        
        return self.secure_add(x_scaled, x * 0.5)
    
    def secure_softmax(self, x: SecureValue, dim: int = -1) -> SecureValue:
        """Secure softmax using polynomial approximation."""
        # Simplified softmax approximation
        # In practice, would use more sophisticated techniques
        x_exp_approx = self._polynomial_exp_approx(x)
        
        # Sum along specified dimension
        sum_exp = self._secure_sum(x_exp_approx, dim=dim, keepdim=True)
        
        # Division approximation (simplified)
        return self._secure_divide_approx(x_exp_approx, sum_exp)
    
    def _polynomial_exp_approx(self, x: SecureValue) -> SecureValue:
        """Polynomial approximation of exp(x)."""
        # Taylor series: exp(x) ≈ 1 + x + x²/2 + x³/6
        one = self.share_value(torch.ones_like(x.shares[0]))
        x_squared = self.secure_multiply(x, x)
        x_cubed = self.secure_multiply(x_squared, x)
        
        result = one
        result = self.secure_add(result, x)
        result = self.secure_add(result, x_squared * 0.5)
        result = self.secure_add(result, x_cubed * (1.0/6))
        
        return result
    
    def _secure_sum(self, x: SecureValue, dim: int, keepdim: bool = False) -> SecureValue:
        """Secure sum along dimension."""
        summed_shares = [share.sum(dim=dim, keepdim=keepdim) for share in x.shares]
        return SecureValue(shares=summed_shares, party_id=x.party_id, is_public=x.is_public)
    
    def _secure_divide_approx(self, numerator: SecureValue, denominator: SecureValue) -> SecureValue:
        """Approximate secure division using Newton-Raphson method."""
        # Simplified division - in practice would use iterative methods
        # For demonstration, using approximation
        return self.secure_multiply(numerator, denominator)  # Placeholder
    
    @abstractmethod
    def send_shares(self, shares: List[torch.Tensor], recipient: int) -> None:
        """Send shares to another party."""
        pass
    
    @abstractmethod
    def receive_shares(self, sender: int) -> List[torch.Tensor]:
        """Receive shares from another party."""
        pass
    
    def validate_shares(self, secure_value: SecureValue) -> bool:
        """Validate that shares are consistent."""
        if not secure_value.shares:
            return False
        
        reference_shape = secure_value.shares[0].shape
        reference_dtype = secure_value.shares[0].dtype
        
        for share in secure_value.shares[1:]:
            if share.shape != reference_shape or share.dtype != reference_dtype:
                return False
        
        return True
    
    def get_protocol_info(self) -> dict:
        """Get protocol information and statistics."""
        return {
            "protocol_name": self.__class__.__name__,
            "party_id": self.party_id,
            "num_parties": self.num_parties,
            "device": str(self.device),
            "initialized": self._initialized
        }
