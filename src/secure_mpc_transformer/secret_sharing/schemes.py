"""
Different secret sharing schemes for MPC protocols.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SecretSharingScheme(ABC):
    """Abstract base class for secret sharing schemes."""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
    
    @abstractmethod
    def share(self, secret: torch.Tensor) -> List[torch.Tensor]:
        """Split secret into shares."""
        pass
    
    @abstractmethod
    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct secret from shares."""
        pass
    
    @abstractmethod
    def validate_shares(self, shares: List[torch.Tensor]) -> bool:
        """Validate that shares are consistent."""
        pass
    
    def multiply_shares(
        self, 
        shares_a: List[torch.Tensor], 
        shares_b: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Default multiplication (may need protocol-specific implementation)."""
        if len(shares_a) != len(shares_b):
            raise ValueError("Share lists must have same length")
        
        return [a * b for a, b in zip(shares_a, shares_b)]
    
    def matmul_shares(
        self, 
        shares_a: List[torch.Tensor], 
        shares_b: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Default matrix multiplication of shares."""
        if len(shares_a) != len(shares_b):
            raise ValueError("Share lists must have same length")
        
        return [torch.matmul(a, b) for a, b in zip(shares_a, shares_b)]


class ShamirSecretSharing(SecretSharingScheme):
    """
    Shamir's threshold secret sharing scheme.
    
    Allows reconstruction with any subset of t+1 shares out of n total shares,
    where t is the threshold parameter.
    """
    
    def __init__(
        self, 
        threshold: int, 
        num_parties: int, 
        field_size: int = 2**31 - 1,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__(device)
        
        if threshold >= num_parties:
            raise ValueError("Threshold must be less than number of parties")
        if threshold < 1:
            raise ValueError("Threshold must be at least 1")
        
        self.threshold = threshold
        self.num_parties = num_parties
        self.field_size = field_size
        
        # Pre-compute evaluation points and Lagrange coefficients
        self.eval_points = list(range(1, num_parties + 1))
        self.lagrange_coeffs = self._precompute_lagrange_coefficients()
        
        logger.info(f"Initialized Shamir sharing: t={threshold}, n={num_parties}")
    
    def _precompute_lagrange_coefficients(self) -> torch.Tensor:
        """Pre-compute Lagrange interpolation coefficients."""
        coeffs = torch.zeros(self.num_parties, self.num_parties, device=self.device)
        
        for subset_size in range(self.threshold + 1, self.num_parties + 1):
            for i in range(subset_size):
                coeff = 1.0
                for j in range(subset_size):
                    if i != j:
                        coeff *= (0 - self.eval_points[j]) / (self.eval_points[i] - self.eval_points[j])
                coeffs[subset_size - 1][i] = coeff
        
        return coeffs
    
    def share(self, secret: torch.Tensor) -> List[torch.Tensor]:
        """
        Share secret using Shamir's scheme.
        
        Creates a polynomial of degree t with secret as constant term,
        then evaluates at different points to create shares.
        """
        secret = secret.to(self.device)
        original_shape = secret.shape
        
        # Flatten secret for polynomial operations
        secret_flat = secret.flatten()
        
        # Generate random coefficients for polynomial of degree t
        coefficients = []
        coefficients.append(secret_flat)  # Constant term is the secret
        
        for _ in range(self.threshold):
            coeff = torch.randint(
                0, self.field_size, 
                secret_flat.shape, 
                device=self.device, 
                dtype=secret_flat.dtype
            )
            coefficients.append(coeff)
        
        # Evaluate polynomial at different points to create shares
        shares = []
        for i, x in enumerate(self.eval_points):
            share_value = coefficients[0].clone()  # Start with constant term
            
            x_power = torch.tensor(x, device=self.device, dtype=secret_flat.dtype)
            for coeff in coefficients[1:]:
                share_value += coeff * x_power
                x_power *= x
            
            # Reshape back to original shape
            share_value = share_value.view(original_shape)
            shares.append(share_value)
        
        return shares
    
    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct secret using Lagrange interpolation.
        
        Requires at least threshold+1 shares for reconstruction.
        """
        if len(shares) < self.threshold + 1:
            raise ValueError(f"Need at least {self.threshold + 1} shares, got {len(shares)}")
        
        # Use first threshold+1 shares
        used_shares = shares[:self.threshold + 1]
        used_points = self.eval_points[:self.threshold + 1]
        
        # Perform Lagrange interpolation to find constant term
        result = torch.zeros_like(used_shares[0])
        
        for i, (share, x_i) in enumerate(zip(used_shares, used_points)):
            # Compute Lagrange coefficient for point x_i
            coeff = 1.0
            for j, x_j in enumerate(used_points):
                if i != j:
                    coeff *= (0 - x_j) / (x_i - x_j)
            
            result += share * coeff
        
        return result
    
    def validate_shares(self, shares: List[torch.Tensor]) -> bool:
        """Validate Shamir shares."""
        if len(shares) < self.threshold + 1:
            return False
        
        # Check if reconstruction with different subsets gives same result
        if len(shares) > self.threshold + 1:
            result1 = self.reconstruct(shares[:self.threshold + 1])
            result2 = self.reconstruct(shares[1:self.threshold + 2])
            
            # Allow small numerical differences
            return torch.allclose(result1, result2, rtol=1e-5, atol=1e-8)
        
        return True


class AdditiveSharing(SecretSharingScheme):
    """
    Additive secret sharing scheme.
    
    Splits secret into random shares that sum to the original value.
    Simple and efficient for many MPC protocols.
    """
    
    def __init__(self, num_parties: int, device: torch.device = torch.device('cpu')):
        super().__init__(device)
        self.num_parties = num_parties
        
        logger.info(f"Initialized additive sharing with {num_parties} parties")
    
    def share(self, secret: torch.Tensor) -> List[torch.Tensor]:
        """
        Share secret using additive sharing.
        
        Generates n-1 random shares, with the last share ensuring
        the sum equals the original secret.
        """
        secret = secret.to(self.device)
        shares = []
        
        # Generate n-1 random shares
        for _ in range(self.num_parties - 1):
            random_share = torch.randn_like(secret, device=self.device) * 0.1
            shares.append(random_share)
        
        # Last share ensures sum equals secret
        last_share = secret - sum(shares)
        shares.append(last_share)
        
        return shares
    
    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct secret by summing all shares."""
        if not shares:
            raise ValueError("Cannot reconstruct from empty shares")
        
        return sum(shares)
    
    def validate_shares(self, shares: List[torch.Tensor]) -> bool:
        """Validate additive shares (always valid if same shape/dtype)."""
        if len(shares) != self.num_parties:
            return False
        
        if not shares:
            return False
        
        reference = shares[0]
        return all(
            share.shape == reference.shape and 
            share.dtype == reference.dtype and
            share.device == reference.device
            for share in shares[1:]
        )


class ReplicatedSharing(SecretSharingScheme):
    """
    Replicated secret sharing for 3-party computation.
    
    Each party holds two shares in a specific pattern that allows
    efficient secure computation without communication for linear operations.
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__(device)
        
        logger.info("Initialized replicated 3-party sharing")
    
    def share(self, secret: torch.Tensor) -> List[torch.Tensor]:
        """
        Share secret using replicated sharing.
        
        Creates shares such that:
        - Party 0 gets (r0, r1)
        - Party 1 gets (r1, r2)
        - Party 2 gets (r2, r0)
        Where secret = r0 + r1 + r2
        """
        secret = secret.to(self.device)
        
        # Generate two random shares
        r1 = torch.randn_like(secret, device=self.device) * 0.1
        r2 = torch.randn_like(secret, device=self.device) * 0.1
        r0 = secret - r1 - r2
        
        # Return shares for each party
        party_shares = [
            [r0, r1],  # Party 0
            [r1, r2],  # Party 1
            [r2, r0]   # Party 2
        ]
        
        return party_shares
    
    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct secret from replicated shares.
        
        Args:
            shares: Either list of party share lists, or flattened list of unique shares
        """
        if not shares:
            raise ValueError("Cannot reconstruct from empty shares")
        
        # Handle different input formats
        if isinstance(shares[0], list):
            # Input is list of party shares
            if len(shares) != 3:
                raise ValueError("Replicated sharing requires exactly 3 parties")
            
            # Extract unique shares: r0, r1, r2
            r0 = shares[0][0]  # From party 0
            r1 = shares[0][1]  # From party 0 (same as shares[1][0])
            r2 = shares[1][1]  # From party 1
            
            return r0 + r1 + r2
        
        else:
            # Input is flattened list of shares
            if len(shares) == 3:
                # Assume shares are [r0, r1, r2]
                return sum(shares)
            else:
                raise ValueError(f"Invalid number of shares for reconstruction: {len(shares)}")
    
    def validate_shares(self, shares: List[torch.Tensor]) -> bool:
        """Validate replicated shares."""
        if isinstance(shares[0], list):
            # Validate party share format
            if len(shares) != 3:
                return False
            
            # Each party should have exactly 2 shares
            for party_shares in shares:
                if len(party_shares) != 2:
                    return False
                
                # Check share consistency
                if (party_shares[0].shape != party_shares[1].shape or
                    party_shares[0].dtype != party_shares[1].dtype):
                    return False
            
            # Check that shared values are consistent across parties
            # shares[0][1] should equal shares[1][0] (both are r1)
            # shares[1][1] should equal shares[2][0] (both are r2)  
            # shares[2][1] should equal shares[0][0] (both are r0)
            
            tolerance = 1e-6
            if not (torch.allclose(shares[0][1], shares[1][0], atol=tolerance) and
                    torch.allclose(shares[1][1], shares[2][0], atol=tolerance) and
                    torch.allclose(shares[2][1], shares[0][0], atol=tolerance)):\n                return False\n        \n        return True\n    \n    def multiply_shares(\n        self, \n        shares_a: List[torch.Tensor], \n        shares_b: List[torch.Tensor]\n    ) -> List[torch.Tensor]:\n        \"\"\"\n        Multiply replicated shares (requires resharing for security).\n        \n        This is more complex than linear operations and typically\n        requires communication between parties.\n        \"\"\"\n        if (len(shares_a) != 3 or len(shares_b) != 3 or\n            not isinstance(shares_a[0], list) or not isinstance(shares_b[0], list)):\n            raise ValueError(\"Replicated multiplication requires 3 parties with 2 shares each\")\n        \n        # Simplified multiplication (in practice needs degree reduction)\n        result_shares = []\n        \n        for i in range(3):\n            party_products = []\n            for j in range(2):\n                for k in range(2):\n                    product = shares_a[i][j] * shares_b[i][k]\n                    party_products.append(product)\n            \n            # Sum cross-products (simplified degree reduction)\n            reduced_shares = [\n                party_products[0] + party_products[1],\n                party_products[2] + party_products[3]\n            ]\n            result_shares.append(reduced_shares)\n        \n        return result_shares\n\n\nclass PackedSharing(SecretSharingScheme):\n    \"\"\"\n    Packed secret sharing for improved efficiency.\n    \n    Allows sharing multiple secrets in a single share,\n    improving throughput for batch operations.\n    \"\"\"\n    \n    def __init__(\n        self, \n        num_parties: int, \n        pack_size: int = 1,\n        device: torch.device = torch.device('cpu')\n    ):\n        super().__init__(device)\n        self.num_parties = num_parties\n        self.pack_size = pack_size\n        \n        logger.info(f\"Initialized packed sharing: {num_parties} parties, pack size {pack_size}\")\n    \n    def share(self, secret: torch.Tensor) -> List[torch.Tensor]:\n        \"\"\"\n        Share tensor using packed sharing.\n        \n        Packs multiple values into each share for efficiency.\n        \"\"\"\n        secret = secret.to(self.device)\n        \n        # If tensor is smaller than pack size, use regular additive sharing\n        if secret.numel() < self.pack_size:\n            return self._simple_additive_share(secret)\n        \n        # Pack multiple secrets into shares\n        return self._packed_share(secret)\n    \n    def _simple_additive_share(self, secret: torch.Tensor) -> List[torch.Tensor]:\n        \"\"\"Simple additive sharing for small tensors.\"\"\"\n        shares = []\n        for _ in range(self.num_parties - 1):\n            random_share = torch.randn_like(secret, device=self.device) * 0.1\n            shares.append(random_share)\n        \n        last_share = secret - sum(shares)\n        shares.append(last_share)\n        \n        return shares\n    \n    def _packed_share(self, secret: torch.Tensor) -> List[torch.Tensor]:\n        \"\"\"Packed sharing for larger tensors.\"\"\"\n        # Reshape secret to pack multiple values\n        original_shape = secret.shape\n        flat_secret = secret.flatten()\n        \n        # Pad to multiple of pack_size\n        if flat_secret.numel() % self.pack_size != 0:\n            padding_size = self.pack_size - (flat_secret.numel() % self.pack_size)\n            padding = torch.zeros(padding_size, device=self.device, dtype=secret.dtype)\n            flat_secret = torch.cat([flat_secret, padding])\n        \n        # Reshape for packing\n        packed_secret = flat_secret.view(-1, self.pack_size)\n        \n        # Generate shares for each pack\n        packed_shares = [[] for _ in range(self.num_parties)]\n        \n        for pack in packed_secret:\n            pack_shares = self._simple_additive_share(pack)\n            for i, share in enumerate(pack_shares):\n                packed_shares[i].append(share)\n        \n        # Concatenate and reshape\n        final_shares = []\n        for party_shares in packed_shares:\n            if party_shares:\n                concatenated = torch.stack(party_shares).flatten()\n                # Remove padding and reshape to original\n                truncated = concatenated[:secret.numel()]\n                reshaped = truncated.view(original_shape)\n                final_shares.append(reshaped)\n            else:\n                final_shares.append(torch.zeros_like(secret))\n        \n        return final_shares\n    \n    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:\n        \"\"\"Reconstruct from packed shares.\"\"\"\n        if not shares:\n            raise ValueError(\"Cannot reconstruct from empty shares\")\n        \n        return sum(shares)\n    \n    def validate_shares(self, shares: List[torch.Tensor]) -> bool:\n        \"\"\"Validate packed shares.\"\"\"\n        if len(shares) != self.num_parties:\n            return False\n        \n        if not shares:\n            return False\n        \n        reference = shares[0]\n        return all(\n            share.shape == reference.shape and \n            share.dtype == reference.dtype and\n            share.device == reference.device\n            for share in shares[1:]\n        )"