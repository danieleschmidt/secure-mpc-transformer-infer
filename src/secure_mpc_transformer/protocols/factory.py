"""Factory for creating MPC protocol instances."""

from typing import Optional, Dict, Any
import torch
from ..config import SecurityConfig, ProtocolType
from .base import Protocol
from .semi_honest_3pc import SemiHonest3PC
from .malicious_3pc import Malicious3PC
from .aby3 import ABY3Protocol


class ProtocolFactory:
    """Factory class for creating MPC protocol instances."""
    
    _protocol_registry: Dict[ProtocolType, type] = {
        ProtocolType.SEMI_HONEST_3PC: SemiHonest3PC,
        ProtocolType.MALICIOUS_3PC: Malicious3PC,
        ProtocolType.ABY3: ABY3Protocol,
        ProtocolType.REPLICATED_3PC: SemiHonest3PC,  # Alias
    }
    
    @classmethod
    def create(cls, protocol_name: str, party_id: int, num_parties: int = 3,
               device: Optional[torch.device] = None, **kwargs) -> Protocol:
        """Create protocol instance by name.
        
        Args:
            protocol_name: Name of the protocol
            party_id: ID of this party
            num_parties: Total number of parties
            device: Compute device (CPU/GPU)
            **kwargs: Additional protocol-specific parameters
            
        Returns:
            Protocol instance
            
        Raises:
            ValueError: If protocol name is not supported
        """
        try:
            protocol_type = ProtocolType(protocol_name)
        except ValueError:
            raise ValueError(f"Unsupported protocol: {protocol_name}. "
                           f"Supported protocols: {list(cls._protocol_registry.keys())}")
        
        if protocol_type not in cls._protocol_registry:
            raise ValueError(f"Protocol {protocol_type} not implemented")
        
        protocol_class = cls._protocol_registry[protocol_type]
        
        # Create instance with appropriate parameters
        if protocol_type == ProtocolType.ABY3:
            ring_size = kwargs.get("ring_size", 2**64)
            return protocol_class(party_id, num_parties, device, ring_size=ring_size)
        elif protocol_type == ProtocolType.MALICIOUS_3PC:
            mac_key_size = kwargs.get("mac_key_size", 128)
            return protocol_class(party_id, num_parties, device, mac_key_size=mac_key_size)
        else:
            return protocol_class(party_id, num_parties, device)
    
    @classmethod
    def create_from_config(cls, config: SecurityConfig) -> Protocol:
        """Create protocol instance from security configuration.
        
        Args:
            config: Security configuration
            
        Returns:
            Protocol instance configured according to config
        """
        config.validate()
        
        device = torch.device("cuda" if config.gpu_acceleration and torch.cuda.is_available() else "cpu")
        
        # Extract protocol-specific parameters
        protocol_kwargs = config.protocol_params.copy() if config.protocol_params else {}
        
        protocol = cls.create(
            protocol_name=config.protocol.value,
            party_id=config.party_id,
            num_parties=config.num_parties,
            device=device,
            **protocol_kwargs
        )
        
        # Initialize the protocol
        protocol.initialize()
        
        return protocol
    
    @classmethod
    def register_protocol(cls, protocol_type: ProtocolType, protocol_class: type) -> None:
        """Register a new protocol implementation.
        
        Args:
            protocol_type: Protocol type enum
            protocol_class: Protocol implementation class
        """
        if not issubclass(protocol_class, Protocol):
            raise ValueError("Protocol class must inherit from Protocol base class")
        
        cls._protocol_registry[protocol_type] = protocol_class
    
    @classmethod
    def get_available_protocols(cls) -> list:
        """Get list of available protocol types.
        
        Returns:
            List of available protocol type names
        """
        return [protocol_type.value for protocol_type in cls._protocol_registry.keys()]
    
    @classmethod
    def get_protocol_info(cls, protocol_name: str) -> Dict[str, Any]:
        """Get information about a specific protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            Dictionary with protocol information
        """
        try:
            protocol_type = ProtocolType(protocol_name)
        except ValueError:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        if protocol_type not in cls._protocol_registry:
            raise ValueError(f"Protocol {protocol_type} not implemented")
        
        protocol_class = cls._protocol_registry[protocol_type]
        
        info = {
            "name": protocol_name,
            "class": protocol_class.__name__,
            "description": protocol_class.__doc__ or "No description available",
            "supported_parties": 3,  # Most protocols support 3 parties
        }
        
        # Add protocol-specific information
        if protocol_type == ProtocolType.SEMI_HONEST_3PC:
            info.update({
                "security_model": "semi-honest",
                "communication_rounds": "low",
                "performance": "high"
            })
        elif protocol_type == ProtocolType.MALICIOUS_3PC:
            info.update({
                "security_model": "malicious",
                "communication_rounds": "medium",
                "performance": "medium",
                "features": ["MAC authentication", "zero-knowledge proofs"]
            })
        elif protocol_type == ProtocolType.ABY3:
            info.update({
                "security_model": "semi-honest",
                "sharing_types": ["arithmetic", "boolean", "yao"],
                "performance": "high",
                "features": ["mixed protocols", "efficient conversions"]
            })
        
        return info
    
    @classmethod
    def benchmark_protocols(cls, protocols: Optional[list] = None, 
                          test_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark multiple protocols.
        
        Args:
            protocols: List of protocol names to benchmark (None for all)
            test_size: Size of test tensors
            
        Returns:
            Dictionary mapping protocol names to performance metrics
        """
        if protocols is None:
            protocols = cls.get_available_protocols()
        
        results = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for protocol_name in protocols:
            try:
                # Create protocol instance
                protocol = cls.create(protocol_name, party_id=0, device=device)
                
                # Run benchmark if available
                if hasattr(protocol, "benchmark_operations"):
                    benchmark_results = protocol.benchmark_operations(num_ops=10)
                    results[protocol_name] = benchmark_results
                else:
                    # Basic benchmark
                    import time
                    test_data = torch.randn(test_size, test_size, device=device)
                    
                    start_time = time.time()
                    shared_value = protocol.share_value(test_data)
                    share_time = time.time() - start_time
                    
                    start_time = time.time()
                    reconstructed = protocol.reconstruct_value(shared_value)
                    reconstruct_time = time.time() - start_time
                    
                    results[protocol_name] = {
                        "share_ms": share_time * 1000,
                        "reconstruct_ms": reconstruct_time * 1000
                    }
                    
            except Exception as e:
                results[protocol_name] = {"error": str(e)}
        
        return results
    
    @classmethod
    def recommend_protocol(cls, security_requirements: Dict[str, Any]) -> str:
        """Recommend protocol based on security requirements.
        
        Args:
            security_requirements: Dictionary specifying requirements
            
        Returns:
            Recommended protocol name
        """
        adversary_model = security_requirements.get("adversary_model", "semi-honest")
        performance_priority = security_requirements.get("performance_priority", "medium")
        num_parties = security_requirements.get("num_parties", 3)
        
        if num_parties != 3:
            raise ValueError("Only 3-party protocols currently supported")
        
        if adversary_model == "malicious":
            return ProtocolType.MALICIOUS_3PC.value
        elif performance_priority == "high":
            return ProtocolType.ABY3.value
        else:
            return ProtocolType.SEMI_HONEST_3PC.value
    
    @classmethod
    def create_optimized_for_model(cls, model_type: str, party_id: int,
                                 security_level: str = "semi-honest") -> Protocol:
        """Create protocol optimized for specific model type.
        
        Args:
            model_type: Type of model ("bert", "gpt", etc.)
            party_id: Party ID
            security_level: Security level required
            
        Returns:
            Optimized protocol instance
        """
        # Choose optimal protocol based on model characteristics
        if model_type.lower() in ["bert", "roberta", "distilbert"]:
            # BERT-like models benefit from mixed protocols
            protocol_name = ProtocolType.ABY3.value
            protocol_kwargs = {"ring_size": 2**32}  # Smaller ring for speed
        elif model_type.lower() in ["gpt", "gpt2", "transformer"]:
            # GPT-like models need higher precision
            if security_level == "malicious":
                protocol_name = ProtocolType.MALICIOUS_3PC.value
                protocol_kwargs = {"mac_key_size": 128}
            else:
                protocol_name = ProtocolType.ABY3.value
                protocol_kwargs = {"ring_size": 2**64}
        else:
            # Default choice
            protocol_name = ProtocolType.SEMI_HONEST_3PC.value
            protocol_kwargs = {}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        protocol = cls.create(protocol_name, party_id, device=device, **protocol_kwargs)
        
        # Apply model-specific optimizations
        if hasattr(protocol, "optimize_for_model"):
            protocol.optimize_for_model(model_type)
        
        return protocol
