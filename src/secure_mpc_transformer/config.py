"""
Configuration classes for secure MPC transformer.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """
    Configuration for secure computation parameters.
    
    Defines security level, protocol choice, and cryptographic parameters
    for the secure MPC transformer system.
    """
    
    # Protocol configuration
    protocol: str = "semi_honest_3pc"  # MPC protocol to use
    num_parties: int = 3  # Number of parties in computation
    party_id: int = 0  # ID of this party (0-indexed)
    
    # Security parameters
    security_level: int = 128  # Security level in bits
    field_size: int = 2**31 - 1  # Prime field size for arithmetic
    threshold: int = 1  # Threshold for secret sharing (t-out-of-n)
    
    # GPU configuration
    gpu_acceleration: bool = True  # Enable GPU acceleration
    device: str = "cuda"  # Device to use ("cuda" or "cpu")
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use
    
    # Network configuration
    network_config: Dict[str, Union[str, int]] = field(default_factory=lambda: {
        "host": "localhost",
        "base_port": 50000,
        "timeout": 30,
        "use_tls": True
    })
    
    # Privacy parameters
    differential_privacy: bool = False  # Enable differential privacy
    epsilon: float = 3.0  # Privacy budget (ε)
    delta: float = 1e-5  # Privacy parameter (δ)
    
    # Performance parameters
    batch_size: int = 32  # Default batch size
    max_sequence_length: int = 512  # Maximum sequence length
    
    # Randomness
    seed: Optional[int] = None  # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_protocol()
        self._validate_security_params()
        self._validate_gpu_config()
        self._validate_privacy_params()
        
        logger.info(f"Initialized SecurityConfig: {self.protocol} with {self.num_parties} parties")
    
    def _validate_protocol(self):
        """Validate protocol configuration."""
        supported_protocols = [
            "semi_honest_3pc", "malicious_3pc", "aby3", 
            "replicated_3pc", "shamir", "additive", "bgw"
        ]
        
        if self.protocol not in supported_protocols:
            raise ValueError(f"Unsupported protocol: {self.protocol}. "
                           f"Supported: {supported_protocols}")
        
        if self.party_id < 0 or self.party_id >= self.num_parties:
            raise ValueError(f"Invalid party_id {self.party_id} for {self.num_parties} parties")
        
        # Protocol-specific validations
        if "3pc" in self.protocol and self.num_parties != 3:
            raise ValueError(f"Protocol {self.protocol} requires exactly 3 parties")
        
        if self.protocol == "shamir" and self.threshold >= self.num_parties:
            raise ValueError(f"Shamir threshold {self.threshold} must be < num_parties {self.num_parties}")
    
    def _validate_security_params(self):
        """Validate security parameters."""
        if self.security_level < 80:
            logger.warning(f"Security level {self.security_level} is below recommended minimum of 80 bits")
        
        if self.security_level > 256:
            logger.warning(f"Security level {self.security_level} is higher than typical maximum of 256 bits")
        
        if self.field_size <= 0:
            raise ValueError("Field size must be positive")
        
        # Check if field_size is reasonable for security level
        min_field_bits = max(self.security_level, 31)
        if self.field_size.bit_length() < min_field_bits:
            logger.warning(f"Field size may be too small for security level {self.security_level}")
    
    def _validate_gpu_config(self):
        """Validate GPU configuration."""
        if self.gpu_acceleration and not torch.cuda.is_available():
            logger.warning("GPU acceleration requested but CUDA not available, falling back to CPU")
            self.gpu_acceleration = False
            self.device = "cpu"
        
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {self.device}")
        
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError(f"GPU memory fraction {self.gpu_memory_fraction} must be between 0.1 and 1.0")
    
    def _validate_privacy_params(self):
        """Validate differential privacy parameters."""
        if self.differential_privacy:
            if self.epsilon <= 0:
                raise ValueError(f"Privacy budget epsilon {self.epsilon} must be positive")
            
            if self.delta < 0 or self.delta >= 1:
                raise ValueError(f"Privacy parameter delta {self.delta} must be in [0, 1)")
            
            if self.epsilon > 10:
                logger.warning(f"Large epsilon {self.epsilon} may not provide meaningful privacy")
    
    def get_device(self) -> torch.device:
        """Get the configured PyTorch device."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def get_network_address(self, party_id: Optional[int] = None) -> str:
        """Get network address for a specific party."""
        target_party = party_id if party_id is not None else self.party_id
        port = self.network_config["base_port"] + target_party
        return f"{self.network_config['host']}:{port}"
