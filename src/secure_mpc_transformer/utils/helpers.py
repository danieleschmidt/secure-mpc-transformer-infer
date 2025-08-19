"""Helper utilities for secure MPC transformer operations."""

import hashlib
import json
import logging
import secrets
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SecurityHelper:
    """Helper functions for security operations."""

    @staticmethod
    def generate_random_seed(bit_length: int = 256) -> int:
        """Generate cryptographically secure random seed."""
        return secrets.randbits(bit_length)

    @staticmethod
    def generate_party_keys(num_parties: int, key_length: int = 32) -> list[bytes]:
        """Generate unique keys for each party."""
        keys = []
        for i in range(num_parties):
            key = secrets.token_bytes(key_length)
            keys.append(key)
        return keys

    @staticmethod
    def hash_tensor(tensor: torch.Tensor, algorithm: str = "sha256") -> str:
        """Compute hash of tensor data."""
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()

        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "sha512":
            hasher = hashlib.sha512()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hasher.update(tensor_bytes)
        return hasher.hexdigest()

    @staticmethod
    def verify_tensor_integrity(tensor: torch.Tensor, expected_hash: str,
                               algorithm: str = "sha256") -> bool:
        """Verify tensor integrity using hash."""
        computed_hash = SecurityHelper.hash_tensor(tensor, algorithm)
        return secrets.compare_digest(computed_hash, expected_hash)

    @staticmethod
    def generate_commitment(value: torch.Tensor, nonce: bytes | None = None) -> tuple[str, bytes]:
        """Generate cryptographic commitment to a value."""
        if nonce is None:
            nonce = secrets.token_bytes(32)

        value_bytes = value.detach().cpu().numpy().tobytes()
        commitment_input = value_bytes + nonce

        hasher = hashlib.sha256()
        hasher.update(commitment_input)
        commitment = hasher.hexdigest()

        return commitment, nonce

    @staticmethod
    def verify_commitment(value: torch.Tensor, commitment: str, nonce: bytes) -> bool:
        """Verify cryptographic commitment."""
        expected_commitment, _ = SecurityHelper.generate_commitment(value, nonce)
        return secrets.compare_digest(commitment, expected_commitment)

    @staticmethod
    def secure_compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor,
                              tolerance: float = 1e-6) -> bool:
        """Securely compare two tensors for equality."""
        if tensor1.shape != tensor2.shape:
            return False

        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()

        return max_diff <= tolerance

    @staticmethod
    def sanitize_model_path(path: str) -> Path:
        """Sanitize and validate model path."""
        path_obj = Path(path).resolve()

        # Check for path traversal attempts
        if ".." in str(path_obj):
            raise ValueError("Path traversal detected in model path")

        # Ensure path is within allowed directories
        allowed_dirs = [Path.cwd(), Path.home() / ".cache", Path("/tmp")]

        is_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                path_obj.relative_to(allowed_dir.resolve())
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            raise ValueError(f"Model path not in allowed directories: {path_obj}")

        return path_obj

    @staticmethod
    def validate_network_address(address: str) -> bool:
        """Validate network address format."""
        import re

        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}:\d{1,5}$'

        # IPv6 pattern (simplified)
        ipv6_pattern = r'^\[([0-9a-fA-F:]+)\]:\d{1,5}$'

        # Hostname pattern
        hostname_pattern = r'^[a-zA-Z0-9.-]+:\d{1,5}$'

        patterns = [ipv4_pattern, ipv6_pattern, hostname_pattern]

        for pattern in patterns:
            if re.match(pattern, address):
                return True

        return False


class ModelHelper:
    """Helper functions for model operations."""

    @staticmethod
    def estimate_model_memory(config_dict: dict[str, Any]) -> dict[str, float]:
        """Estimate memory requirements for a model."""
        hidden_size = config_dict.get('hidden_size', 768)
        num_layers = config_dict.get('num_hidden_layers', 12)
        vocab_size = config_dict.get('vocab_size', 30522)
        intermediate_size = config_dict.get('intermediate_size', 3072)

        # Estimate parameter counts
        # Embeddings: vocab_size * hidden_size
        embedding_params = vocab_size * hidden_size

        # Each transformer layer:
        # - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
        # - Feed-forward: 2 * hidden_size * intermediate_size
        # - Layer norms: 2 * hidden_size (small, can ignore)
        layer_params = num_layers * (4 * hidden_size**2 + 2 * hidden_size * intermediate_size)

        total_params = embedding_params + layer_params

        # Memory estimates (in MB)
        # Assuming float32 (4 bytes per parameter)
        model_size_mb = total_params * 4 / (1024 * 1024)

        # Additional memory for activations (rough estimate)
        # Depends on batch size and sequence length
        activation_memory_mb = model_size_mb * 0.5  # Conservative estimate

        # Total memory with overhead
        total_memory_mb = model_size_mb * 1.5 + activation_memory_mb

        return {
            "parameters": total_params,
            "model_size_mb": model_size_mb,
            "activation_memory_mb": activation_memory_mb,
            "total_memory_mb": total_memory_mb,
            "recommended_gpu_memory_mb": total_memory_mb * 2  # Safety factor
        }

    @staticmethod
    def get_model_complexity_score(config_dict: dict[str, Any]) -> float:
        """Calculate complexity score for a model."""
        hidden_size = config_dict.get('hidden_size', 768)
        num_layers = config_dict.get('num_hidden_layers', 12)
        num_heads = config_dict.get('num_attention_heads', 12)
        intermediate_size = config_dict.get('intermediate_size', 3072)

        # Factors that contribute to complexity
        attention_complexity = num_layers * num_heads * hidden_size
        feedforward_complexity = num_layers * intermediate_size

        # Normalize to a 0-100 scale based on common model sizes
        total_complexity = attention_complexity + feedforward_complexity

        # Reference point: BERT-base ≈ 100M parameters ≈ score of 50
        # GPT-3 ≈ 175B parameters ≈ score of 100
        normalized_score = min(100, (total_complexity / 2000000) * 50)

        return normalized_score

    @staticmethod
    def suggest_protocol_for_model(config_dict: dict[str, Any]) -> str:
        """Suggest optimal MPC protocol based on model characteristics."""
        complexity_score = ModelHelper.get_model_complexity_score(config_dict)
        memory_estimate = ModelHelper.estimate_model_memory(config_dict)

        # Simple heuristics for protocol selection
        if complexity_score < 30 and memory_estimate["total_memory_mb"] < 1000:
            return "malicious_3pc"  # Can afford stronger security
        elif complexity_score < 60:
            return "aby3"  # Good balance
        else:
            return "semi_honest_3pc"  # Prioritize performance

    @staticmethod
    def optimize_batch_size(model_memory_mb: float, available_memory_mb: float,
                           sequence_length: int = 512) -> int:
        """Suggest optimal batch size based on memory constraints."""
        # Rough estimate: each sequence uses model_memory * factor
        memory_per_sequence = model_memory_mb * 0.1  # Conservative factor

        # Account for sequence length impact
        memory_per_sequence *= (sequence_length / 512)  # Normalize to 512 tokens

        # Leave 20% memory as buffer
        usable_memory = available_memory_mb * 0.8

        suggested_batch_size = max(1, int(usable_memory / memory_per_sequence))

        # Cap at reasonable limits
        return min(suggested_batch_size, 32)

    @staticmethod
    def validate_model_config(config_dict: dict[str, Any]) -> list[str]:
        """Validate model configuration and return warnings."""
        warnings = []

        # Check for reasonable parameter ranges
        hidden_size = config_dict.get('hidden_size', 768)
        if hidden_size % 64 != 0:
            warnings.append(f"Hidden size {hidden_size} not divisible by 64, may be inefficient")

        num_heads = config_dict.get('num_attention_heads', 12)
        if hidden_size % num_heads != 0:
            warnings.append(f"Hidden size {hidden_size} not divisible by num_heads {num_heads}")

        intermediate_size = config_dict.get('intermediate_size', 3072)
        expected_intermediate = hidden_size * 4
        if abs(intermediate_size - expected_intermediate) / expected_intermediate > 0.2:
            warnings.append(f"Unusual intermediate size {intermediate_size}, expected around {expected_intermediate}")

        # Check for very large models
        memory_estimate = ModelHelper.estimate_model_memory(config_dict)
        if memory_estimate["total_memory_mb"] > 24000:  # 24GB
            warnings.append("Model may be too large for typical GPU memory")

        return warnings


class DataHelper:
    """Helper functions for data processing and manipulation."""

    @staticmethod
    def pad_tensor_to_multiple(tensor: torch.Tensor, multiple: int, dim: int = -1) -> torch.Tensor:
        """Pad tensor to nearest multiple along specified dimension."""
        size = tensor.size(dim)
        target_size = ((size + multiple - 1) // multiple) * multiple

        if target_size == size:
            return tensor

        pad_size = target_size - size

        # Create padding configuration
        pad_config = [0, 0] * tensor.dim()
        pad_config[-(dim + 1) * 2 - 1] = pad_size

        return torch.nn.functional.pad(tensor, pad_config[::-1])

    @staticmethod
    def split_tensor_for_parties(tensor: torch.Tensor, num_parties: int,
                                dim: int = 0) -> list[torch.Tensor]:
        """Split tensor into chunks for multiple parties."""
        size = tensor.size(dim)
        chunk_size = size // num_parties
        remainder = size % num_parties

        chunks = []
        start_idx = 0

        for i in range(num_parties):
            # Distribute remainder among first few parties
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size

            # Extract chunk
            indices = torch.arange(start_idx, end_idx, device=tensor.device)
            chunk = torch.index_select(tensor, dim, indices)
            chunks.append(chunk)

            start_idx = end_idx

        return chunks

    @staticmethod
    def combine_tensor_chunks(chunks: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Combine tensor chunks back into single tensor."""
        return torch.cat(chunks, dim=dim)

    @staticmethod
    def mask_sensitive_data(tensor: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
        """Mask random elements in tensor for privacy."""
        mask = torch.rand_like(tensor) < mask_ratio
        masked_tensor = tensor.clone()
        masked_tensor[mask] = 0.0
        return masked_tensor

    @staticmethod
    def add_differential_privacy_noise(tensor: torch.Tensor, epsilon: float,
                                      sensitivity: float = 1.0) -> torch.Tensor:
        """Add Laplace noise for differential privacy."""
        scale = sensitivity / epsilon
        noise = torch.distributions.Laplace(0, scale).sample(tensor.shape)
        noise = noise.to(tensor.device)
        return tensor + noise

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize tensor to specified bit precision."""
        if bits == 32:
            return tensor  # No quantization needed

        # Simple linear quantization
        min_val = tensor.min()
        max_val = tensor.max()

        if min_val == max_val:
            return tensor

        # Calculate scale and zero point
        qmin = 0
        qmax = (1 << bits) - 1

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)

        # Dequantize
        dequantized = (quantized - zero_point) * scale

        return dequantized


class ConfigHelper:
    """Helper functions for configuration management."""

    @staticmethod
    def load_config_from_file(config_path: str | Path) -> dict[str, Any]:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            if config_path.suffix.lower() == '.json':
                return json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @staticmethod
    def save_config_to_file(config: dict[str, Any], config_path: str | Path):
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    @staticmethod
    def merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigHelper.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def validate_config_schema(config: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        """Validate configuration against schema (simplified validation)."""
        errors = []

        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check field types
        field_types = schema.get('types', {})
        for field, expected_type in field_types.items():
            if field in config:
                value = config[field]
                if expected_type == 'int' and not isinstance(value, int):
                    errors.append(f"Field {field} must be integer, got {type(value)}")
                elif expected_type == 'float' and not isinstance(value, (int, float)):
                    errors.append(f"Field {field} must be number, got {type(value)}")
                elif expected_type == 'str' and not isinstance(value, str):
                    errors.append(f"Field {field} must be string, got {type(value)}")
                elif expected_type == 'bool' and not isinstance(value, bool):
                    errors.append(f"Field {field} must be boolean, got {type(value)}")

        return errors

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """Get default configuration for the application."""
        return {
            "model": {
                "name": "bert-base-uncased",
                "max_sequence_length": 512,
                "batch_size": 1
            },
            "protocol": {
                "name": "aby3",
                "security_level": 128,
                "num_parties": 3,
                "gpu_acceleration": True
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "max_concurrent_requests": 10,
                "request_timeout": 300
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "security": {
                "enable_input_validation": True,
                "max_input_length": 10000,
                "rate_limiting": True,
                "audit_logging": True
            }
        }
