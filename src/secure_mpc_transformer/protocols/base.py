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
        # Better polynomial approximation: ReLU(x) ≈ 0.5 * (x + sqrt(x^2 + ε))
        # Or using degree-2 polynomial: ReLU(x) ≈ max(0, 0.5*x + 0.25*x - 0.0625*x^2) for x in [-1,1]
        
        # Simplified approach: ReLU(x) ≈ 0.5 * x + 0.5 * |x|
        # Where |x| is approximated using x^2 / (|x| + δ) ≈ x * sign(x)
        
        # For secure computation, use polynomial approximation of |x|:
        # |x| ≈ x * tanh(k*x) for large k, or polynomial approximation
        
        # Degree-3 polynomial approximation for |x|: |x| ≈ x^2 / (sqrt(x^2 + 0.01))
        # Simplified to: |x| ≈ x * (0.5 + 0.5*tanh(4*x))
        
        x_squared = self.secure_multiply(x, x)
        
        # Approximate |x| using x^2 / (sqrt(x^2) + small_constant)
        # Simplified: |x| ≈ sqrt(x^2) ≈ x^2 / (|x| + ε) iteratively
        # For simplicity: |x| ≈ x * sigmoid(4*x) * 2 - x
        
        # Even simpler approximation: max(0, x) ≈ x * sigmoid(k*x) where k is large
        # sigmoid(k*x) ≈ 0.5 + 0.5*tanh(k*x/2) ≈ 0.5 + k*x/4 - (k*x)^3/48 for small k*x
        
        k = 4.0  # Steepness parameter
        kx = x * k
        
        # Polynomial approximation of sigmoid: σ(x) ≈ 0.5 + x/4 - x³/48
        # For sigmoid(k*x): σ(k*x) ≈ 0.5 + kx/4 - (kx)³/48
        kx_cubed = self.secure_multiply(self.secure_multiply(kx, kx), kx)
        
        sigmoid_approx_shares = []
        for i, share in enumerate(x.shares):
            # σ(kx) ≈ 0.5 + kx/4 - (kx)³/48
            half = torch.full_like(share, 0.5)
            linear_term = kx.shares[i] / 4.0
            cubic_term = kx_cubed.shares[i] / 48.0
            
            sigmoid_share = half + linear_term - cubic_term
            sigmoid_approx_shares.append(sigmoid_share)
        
        sigmoid_approx = SecureValue(
            shares=sigmoid_approx_shares,
            party_id=x.party_id,
            is_public=x.is_public
        )
        
        # ReLU(x) ≈ x * sigmoid(k*x)
        return self.secure_multiply(x, sigmoid_approx)
    
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
        """Polynomial approximation of exp(x) with higher degree for better accuracy."""
        # Taylor series: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
        # For better accuracy over [-2, 2] range
        
        one = self.share_value(torch.ones_like(x.shares[0]))
        x_squared = self.secure_multiply(x, x)
        x_cubed = self.secure_multiply(x_squared, x)
        x_fourth = self.secure_multiply(x_cubed, x)
        x_fifth = self.secure_multiply(x_fourth, x)
        
        # Build terms with proper coefficients
        term1 = one  # 1
        term2 = x    # x  
        
        # x²/2 - need to scale shares
        term3_shares = []
        for share in x_squared.shares:
            term3_shares.append(share * 0.5)
        term3 = SecureValue(shares=term3_shares, party_id=x.party_id, is_public=x.is_public)
        
        # x³/6
        term4_shares = []
        for share in x_cubed.shares:
            term4_shares.append(share * (1.0/6))
        term4 = SecureValue(shares=term4_shares, party_id=x.party_id, is_public=x.is_public)
        
        # x⁴/24
        term5_shares = []
        for share in x_fourth.shares:
            term5_shares.append(share * (1.0/24))
        term5 = SecureValue(shares=term5_shares, party_id=x.party_id, is_public=x.is_public)
        
        # x⁵/120
        term6_shares = []
        for share in x_fifth.shares:
            term6_shares.append(share * (1.0/120))
        term6 = SecureValue(shares=term6_shares, party_id=x.party_id, is_public=x.is_public)
        
        # Sum all terms
        result = self.secure_add(term1, term2)
        result = self.secure_add(result, term3)
        result = self.secure_add(result, term4) 
        result = self.secure_add(result, term5)
        result = self.secure_add(result, term6)
        
        return result
    
    def _secure_sum(self, x: SecureValue, dim: int, keepdim: bool = False) -> SecureValue:
        """Secure sum along dimension."""
        summed_shares = [share.sum(dim=dim, keepdim=keepdim) for share in x.shares]
        return SecureValue(shares=summed_shares, party_id=x.party_id, is_public=x.is_public)
    
    def _secure_divide_approx(self, numerator: SecureValue, denominator: SecureValue) -> SecureValue:
        """Approximate secure division using Newton-Raphson method."""
        # Simplified division - using reciprocal approximation
        # In practice would use iterative Newton-Raphson: x_{n+1} = x_n * (2 - a * x_n)
        
        # For now, use simplified approximation assuming denominator >> 0
        # Real implementation would compute secure reciprocal first
        
        # Create approximate reciprocal (simplified)
        reciprocal_shares = []
        for denom_share in denominator.shares:
            # Simple reciprocal approximation - in practice would be more sophisticated
            reciprocal_share = torch.ones_like(denom_share) / (denom_share + 1e-8)
            reciprocal_shares.append(reciprocal_share)
        
        reciprocal = SecureValue(
            shares=reciprocal_shares,
            party_id=denominator.party_id,
            is_public=denominator.is_public
        )
        
        # Multiply numerator by reciprocal
        return self.secure_multiply(numerator, reciprocal)
    
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
