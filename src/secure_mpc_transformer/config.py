"""Security configuration for MPC transformer inference."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class ProtocolType(Enum):
    """Supported MPC protocols."""
    SEMI_HONEST_3PC = "semi_honest_3pc"
    MALICIOUS_3PC = "malicious_3pc"
    ABY3 = "aby3"
    BGW = "bgw"
    REPLICATED_3PC = "replicated_3pc"
    FANTASTIC_FOUR = "fantastic_four"


class SecurityLevel(Enum):
    """Security levels in bits."""
    BITS_80 = 80
    BITS_128 = 128
    BITS_256 = 256


@dataclass
class SecurityConfig:
    """Configuration for secure computation."""
    
    protocol: ProtocolType = ProtocolType.SEMI_HONEST_3PC
    security_level: SecurityLevel = SecurityLevel.BITS_128
    gpu_acceleration: bool = True
    party_id: int = 0
    num_parties: int = 3
    
    # Network configuration
    host: str = "localhost"
    port: int = 50051
    use_tls: bool = True
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    
    # Performance tuning
    batch_size: int = 32
    num_threads: int = 8
    gpu_memory_fraction: float = 0.9
    
    # Privacy parameters
    differential_privacy: bool = False
    epsilon: float = 3.0
    delta: float = 1e-5
    
    # Protocol-specific parameters
    protocol_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.protocol_params is None:
            self.protocol_params = {}
            
        # Set default protocol parameters
        if self.protocol == ProtocolType.ABY3:
            self.protocol_params.setdefault("ring_size", 2**64)
            self.protocol_params.setdefault("mac_key_size", 128)
            self.protocol_params.setdefault("preprocessing_mode", "online")
        elif self.protocol == ProtocolType.FANTASTIC_FOUR:
            self.protocol_params.setdefault("shares_per_party", 2)
            self.protocol_params.setdefault("communication_rounds", 3)
            self.protocol_params.setdefault("packing_factor", 128)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_parties < 2:
            raise ValueError("Need at least 2 parties for MPC")
        
        if self.party_id >= self.num_parties:
            raise ValueError(f"Party ID {self.party_id} must be < {self.num_parties}")
        
        if self.gpu_memory_fraction <= 0 or self.gpu_memory_fraction > 1:
            raise ValueError("GPU memory fraction must be in (0, 1]")
        
        if self.differential_privacy and (self.epsilon <= 0 or self.delta <= 0):
            raise ValueError("Invalid differential privacy parameters")
    
    @classmethod
    def for_protocol(cls, protocol_name: str, **kwargs) -> "SecurityConfig":
        """Create configuration for specific protocol."""
        protocol = ProtocolType(protocol_name)
        return cls(protocol=protocol, **kwargs)
    
    @classmethod
    def production_config(cls, party_id: int, num_parties: int = 3) -> "SecurityConfig":
        """Production-ready configuration."""
        return cls(
            protocol=ProtocolType.MALICIOUS_3PC,
            security_level=SecurityLevel.BITS_128,
            gpu_acceleration=True,
            party_id=party_id,
            num_parties=num_parties,
            use_tls=True,
            differential_privacy=True,
            batch_size=64,
            num_threads=16
        )
    
    @classmethod
    def development_config(cls, party_id: int = 0) -> "SecurityConfig":
        """Development/testing configuration."""
        return cls(
            protocol=ProtocolType.SEMI_HONEST_3PC,
            security_level=SecurityLevel.BITS_80,
            gpu_acceleration=False,
            party_id=party_id,
            num_parties=3,
            use_tls=False,
            differential_privacy=False,
            batch_size=16,
            num_threads=4
        )
