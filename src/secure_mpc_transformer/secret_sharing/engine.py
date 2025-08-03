"""
Secret sharing engine for coordinating different sharing schemes.
"""

import logging
from typing import Dict, List, Optional, Union

import torch

from ..config import SecurityConfig
from .schemes import ShamirSecretSharing, AdditiveSharing, ReplicatedSharing

logger = logging.getLogger(__name__)


class SecretSharingEngine:
    """
    Unified interface for different secret sharing schemes.
    
    Provides high-level operations for sharing and reconstructing secrets
    while abstracting the underlying sharing scheme details.
    """
    
    def __init__(self, security_config: SecurityConfig):
        self.security_config = security_config
        self.device = torch.device(security_config.device)
        
        # Initialize appropriate sharing scheme based on protocol
        self.sharing_scheme = self._create_sharing_scheme()
        
        # Cache for frequently used shares
        self._share_cache: Dict[str, List[torch.Tensor]] = {}
        self._cache_max_size = 1000
        
        logger.info(f"Initialized SecretSharingEngine with {type(self.sharing_scheme).__name__}")
    
    def _create_sharing_scheme(self):
        """Create the appropriate sharing scheme based on protocol."""
        protocol = self.security_config.protocol.lower()
        
        if protocol in ['shamir', 'threshold']:
            return ShamirSecretSharing(
                threshold=self.security_config.threshold,
                num_parties=self.security_config.num_parties,
                field_size=self.security_config.field_size,
                device=self.device
            )
        
        elif protocol in ['replicated_3pc', 'aby3']:
            if self.security_config.num_parties != 3:
                raise ValueError("Replicated sharing requires exactly 3 parties")
            return ReplicatedSharing(device=self.device)
        
        elif protocol in ['additive', 'bgw', 'semi_honest_3pc']:
            return AdditiveSharing(
                num_parties=self.security_config.num_parties,
                device=self.device
            )
        
        else:
            # Default to additive sharing
            logger.warning(f"Unknown protocol {protocol}, defaulting to additive sharing")
            return AdditiveSharing(
                num_parties=self.security_config.num_parties,
                device=self.device
            )
    
    def share_tensor(
        self, 
        tensor: torch.Tensor, 
        party_id: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Share a tensor using the configured sharing scheme.
        
        Args:
            tensor: Tensor to share
            party_id: ID of the party that owns this tensor (optional)
            
        Returns:
            List of shares, one for each party
        """
        tensor = tensor.to(self.device)
        
        # Check cache first
        cache_key = self._get_cache_key(tensor)
        if cache_key in self._share_cache:
            logger.debug(f"Using cached shares for tensor {cache_key}")
            return self._share_cache[cache_key]
        
        # Generate shares
        shares = self.sharing_scheme.share(tensor)
        
        # Cache the shares if tensor is small enough
        if self._should_cache_tensor(tensor):
            self._add_to_cache(cache_key, shares)
        
        return shares
    
    def reconstruct_tensor(
        self, 
        shares: Union[List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Reconstruct a tensor from its shares.
        
        Args:
            shares: List of shares or single share tensor
            
        Returns:
            Reconstructed tensor
        """
        if isinstance(shares, torch.Tensor):
            # Already reconstructed or public value
            return shares
        
        if not shares:
            raise ValueError("Cannot reconstruct from empty shares")
        
        return self.sharing_scheme.reconstruct(shares)
    
    def share_batch(
        self, 
        tensors: List[torch.Tensor]
    ) -> List[List[torch.Tensor]]:
        """
        Efficiently share a batch of tensors.
        
        Args:
            tensors: List of tensors to share
            
        Returns:
            List of share lists, one per tensor
        """
        return [self.share_tensor(tensor) for tensor in tensors]
    
    def reconstruct_batch(
        self, 
        share_lists: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Efficiently reconstruct a batch of tensors.
        
        Args:
            share_lists: List of share lists to reconstruct
            
        Returns:
            List of reconstructed tensors
        """
        return [self.reconstruct_tensor(shares) for shares in share_lists]
    
    def add_shares(
        self, 
        shares_a: List[torch.Tensor], 
        shares_b: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Add two sets of shares element-wise.
        
        Addition can be performed locally on shares without communication.
        
        Args:
            shares_a: First set of shares
            shares_b: Second set of shares
            
        Returns:
            Shares of the sum
        """
        if len(shares_a) != len(shares_b):
            raise ValueError("Share lists must have the same length")
        
        return [share_a + share_b for share_a, share_b in zip(shares_a, shares_b)]
    
    def multiply_shares_by_scalar(
        self, 
        shares: List[torch.Tensor], 
        scalar: Union[float, int, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Multiply shares by a public scalar.
        
        Args:
            shares: Shares to multiply
            scalar: Public scalar value
            
        Returns:
            Shares of the scaled value
        """
        return [share * scalar for share in shares]
    
    def multiply_shares(
        self, 
        shares_a: List[torch.Tensor], 
        shares_b: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Multiply two sets of shares (requires protocol-specific handling).
        
        This is a placeholder - actual multiplication depends on the specific
        sharing scheme and may require communication between parties.
        
        Args:\n            shares_a: First set of shares\n            shares_b: Second set of shares\n            \n        Returns:\n            Shares of the product\n        \"\"\"\n        return self.sharing_scheme.multiply_shares(shares_a, shares_b)\n    \n    def linear_combination(\n        self, \n        share_lists: List[List[torch.Tensor]], \n        coefficients: List[Union[float, int, torch.Tensor]]\n    ) -> List[torch.Tensor]:\n        \"\"\"\n        Compute linear combination of multiple shared values.\n        \n        result = sum(coeff_i * shares_i for i in range(len(share_lists)))\n        \n        Args:\n            share_lists: List of share lists\n            coefficients: Coefficients for linear combination\n            \n        Returns:\n            Shares of the linear combination\n        \"\"\"\n        if len(share_lists) != len(coefficients):\n            raise ValueError(\"Number of share lists must match number of coefficients\")\n        \n        if not share_lists:\n            raise ValueError(\"Cannot compute linear combination of empty list\")\n        \n        # Initialize result with first term\n        result = self.multiply_shares_by_scalar(share_lists[0], coefficients[0])\n        \n        # Add remaining terms\n        for shares, coeff in zip(share_lists[1:], coefficients[1:]):\n            scaled_shares = self.multiply_shares_by_scalar(shares, coeff)\n            result = self.add_shares(result, scaled_shares)\n        \n        return result\n    \n    def matmul_shares(\n        self, \n        shares_a: List[torch.Tensor], \n        shares_b: List[torch.Tensor]\n    ) -> List[torch.Tensor]:\n        \"\"\"\n        Matrix multiplication of shared values.\n        \n        Args:\n            shares_a: Shares of first matrix\n            shares_b: Shares of second matrix\n            \n        Returns:\n            Shares of the matrix product\n        \"\"\"\n        return self.sharing_scheme.matmul_shares(shares_a, shares_b)\n    \n    def get_share_for_party(\n        self, \n        shares: List[torch.Tensor], \n        party_id: int\n    ) -> torch.Tensor:\n        \"\"\"\n        Get the share for a specific party.\n        \n        Args:\n            shares: List of all shares\n            party_id: ID of the party\n            \n        Returns:\n            Share for the specified party\n        \"\"\"\n        if party_id < 0 or party_id >= len(shares):\n            raise ValueError(f\"Invalid party ID {party_id} for {len(shares)} parties\")\n        \n        return shares[party_id]\n    \n    def validate_shares(self, shares: List[torch.Tensor]) -> bool:\n        \"\"\"\n        Validate that shares are consistent and well-formed.\n        \n        Args:\n            shares: List of shares to validate\n            \n        Returns:\n            True if shares are valid, False otherwise\n        \"\"\"\n        if not shares:\n            return False\n        \n        # Check that all shares have the same shape and dtype\n        reference_shape = shares[0].shape\n        reference_dtype = shares[0].dtype\n        reference_device = shares[0].device\n        \n        for share in shares[1:]:\n            if (share.shape != reference_shape or \n                share.dtype != reference_dtype or \n                share.device != reference_device):\n                return False\n        \n        # Check scheme-specific validation\n        return self.sharing_scheme.validate_shares(shares)\n    \n    def get_sharing_info(self) -> Dict[str, Union[str, int]]:\n        \"\"\"\n        Get information about the current sharing configuration.\n        \n        Returns:\n            Dictionary with sharing scheme details\n        \"\"\"\n        return {\n            'scheme_type': type(self.sharing_scheme).__name__,\n            'num_parties': self.security_config.num_parties,\n            'threshold': getattr(self.security_config, 'threshold', None),\n            'field_size': getattr(self.security_config, 'field_size', None),\n            'device': str(self.device),\n            'cache_size': len(self._share_cache)\n        }\n    \n    def _get_cache_key(self, tensor: torch.Tensor) -> str:\n        \"\"\"\n        Generate a cache key for a tensor.\n        \n        Args:\n            tensor: Tensor to generate key for\n            \n        Returns:\n            Cache key string\n        \"\"\"\n        # Use tensor properties to create a unique key\n        # In practice, might use hash of tensor values for small tensors\n        return f\"shape_{tensor.shape}_dtype_{tensor.dtype}_device_{tensor.device}\"\n    \n    def _should_cache_tensor(self, tensor: torch.Tensor) -> bool:\n        \"\"\"\n        Determine if a tensor should be cached.\n        \n        Args:\n            tensor: Tensor to check\n            \n        Returns:\n            True if tensor should be cached\n        \"\"\"\n        # Cache small tensors and avoid caching if cache is full\n        return (tensor.numel() < 1000 and \n                len(self._share_cache) < self._cache_max_size)\n    \n    def _add_to_cache(\n        self, \n        cache_key: str, \n        shares: List[torch.Tensor]\n    ):\n        \"\"\"\n        Add shares to cache.\n        \n        Args:\n            cache_key: Key for caching\n            shares: Shares to cache\n        \"\"\"\n        if len(self._share_cache) >= self._cache_max_size:\n            # Remove oldest entry (simple FIFO)\n            oldest_key = next(iter(self._share_cache))\n            del self._share_cache[oldest_key]\n        \n        # Deep copy shares to avoid modifications\n        cached_shares = [share.clone().detach() for share in shares]\n        self._share_cache[cache_key] = cached_shares\n    \n    def clear_cache(self):\n        \"\"\"Clear the share cache.\"\"\"\n        self._share_cache.clear()\n        logger.info(\"Share cache cleared\")\n    \n    def get_cache_stats(self) -> Dict[str, int]:\n        \"\"\"\n        Get cache statistics.\n        \n        Returns:\n            Dictionary with cache statistics\n        \"\"\"\n        return {\n            'cache_size': len(self._share_cache),\n            'max_cache_size': self._cache_max_size,\n            'cache_hit_rate': 0  # Would track in real implementation\n        }"