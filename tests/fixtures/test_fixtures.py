"""
Comprehensive test fixtures for Secure MPC Transformer testing.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock

import pytest
import torch
import numpy as np


class MockSecurityConfig:
    """Mock security configuration for testing."""
    
    def __init__(self, **kwargs):
        self.protocol = kwargs.get("protocol", "3pc_semi_honest")
        self.security_level = kwargs.get("security_level", 128)
        self.gpu_acceleration = kwargs.get("gpu_acceleration", False)
        self.debug_mode = kwargs.get("debug_mode", True)
        self.num_parties = kwargs.get("num_parties", 3)


class MockTransformerModel:
    """Mock transformer model for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.get("gpu", False) else "cpu"
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mock forward pass."""
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.get("hidden_size", 768)
        return torch.randn(batch_size, seq_len, hidden_size, device=self.device)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MockMPCParty:
    """Mock MPC party for testing."""
    
    def __init__(self, party_id: int, num_parties: int = 3):
        self.party_id = party_id
        self.num_parties = num_parties
        self.is_connected = True
        self.shared_data = {}
    
    def share_secret(self, data: Any) -> List[Any]:
        """Mock secret sharing."""
        return [f"share_{i}_{data}" for i in range(self.num_parties)]
    
    def reconstruct_secret(self, shares: List[Any]) -> Any:
        """Mock secret reconstruction."""
        return shares[0].split("_", 2)[2] if shares else None


class MockNetworkManager:
    """Mock network manager for testing."""
    
    def __init__(self):
        self.connected_parties = {}
        self.message_log = []
    
    def connect_to_party(self, party_id: int, host: str, port: int) -> bool:
        """Mock party connection."""
        self.connected_parties[party_id] = {"host": host, "port": port}
        return True
    
    def send_message(self, party_id: int, message: Dict[str, Any]) -> bool:
        """Mock message sending."""
        self.message_log.append({
            "to": party_id,
            "message": message,
            "timestamp": "mock_timestamp"
        })
        return True


@pytest.fixture
def mock_security_config():
    """Mock security configuration fixture."""
    return MockSecurityConfig(
        protocol="3pc_semi_honest",
        security_level=128,
        gpu_acceleration=False,
        debug_mode=True
    )


@pytest.fixture
def mock_transformer_model():
    """Mock transformer model fixture."""
    config = {
        "model_name": "bert-base-uncased",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512
    }
    return MockTransformerModel(config)


@pytest.fixture
def mock_mpc_parties():
    """Mock MPC parties fixture."""
    return [MockMPCParty(i) for i in range(3)]


@pytest.fixture
def mock_network_manager():
    """Mock network manager fixture."""
    return MockNetworkManager()


@pytest.fixture
def sample_input_data():
    """Sample input data for testing."""
    return {
        "texts": [
            "The capital of France is [MASK].",
            "Secure computation enables private [MASK].",
            "GPU acceleration provides significant [MASK]."
        ],
        "input_ids": torch.randint(0, 30522, (3, 128)),
        "attention_mask": torch.ones(3, 128),
        "labels": ["Paris", "inference", "speedup"]
    }


@pytest.fixture
def sample_cryptographic_data():
    """Sample cryptographic data for testing."""
    return {
        "public_key": "mock_public_key_data",
        "private_key": "mock_private_key_data",
        "ciphertext": "mock_encrypted_data",
        "plaintext": "original_data",
        "shares": ["share_0", "share_1", "share_2"],
        "polynomial_coefficients": [1, 2, 3, 4]
    }


@pytest.fixture
def gpu_test_data():
    """GPU-specific test data."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    return {
        "device": "cuda",
        "tensor_gpu": torch.randn(100, 768, device="cuda"),
        "large_tensor": torch.randn(1000, 1000, device="cuda"),
        "memory_stats": torch.cuda.memory_stats()
    }


@pytest.fixture
def performance_test_data():
    """Performance testing data with various sizes."""
    return {
        "small_batch": {
            "input_ids": torch.randint(0, 30522, (4, 64)),
            "expected_time": 1.0  # seconds
        },
        "medium_batch": {
            "input_ids": torch.randint(0, 30522, (16, 128)),
            "expected_time": 3.0
        },
        "large_batch": {
            "input_ids": torch.randint(0, 30522, (32, 256)),
            "expected_time": 10.0
        }
    }


@pytest.fixture
def security_test_vectors():
    """Security test vectors for cryptographic operations."""
    return {
        "timing_attack_data": [
            {"input": "normal_input", "expected_time_range": (0.1, 0.2)},
            {"input": "adversarial_input", "expected_time_range": (0.1, 0.2)},
        ],
        "side_channel_data": {
            "power_consumption": np.random.normal(100, 10, 1000),
            "timing_measurements": np.random.normal(0.15, 0.02, 1000)
        },
        "malicious_inputs": [
            {"type": "overflow", "data": "A" * 10000},
            {"type": "injection", "data": "'; DROP TABLE users; --"},
            {"type": "encoding", "data": "\x00\x01\x02\x03"}
        ]
    }


@pytest.fixture
def benchmark_config():
    """Benchmark configuration for performance tests."""
    return {
        "iterations": 10,
        "warmup_iterations": 3,
        "timeout_seconds": 300,
        "memory_limit_mb": 8192,
        "acceptable_variance": 0.1,  # 10% variance
        "baseline_times": {
            "inference_cpu": 5.0,
            "inference_gpu": 0.5,
            "protocol_setup": 0.1,
            "secret_sharing": 0.05
        }
    }


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "parties": [
            {"id": 0, "role": "data_owner", "host": "localhost", "port": 9000},
            {"id": 1, "role": "compute", "host": "localhost", "port": 9001},
            {"id": 2, "role": "compute", "host": "localhost", "port": 9002}
        ],
        "protocols_to_test": ["3pc_semi_honest", "3pc_malicious"],
        "models_to_test": ["bert-base-uncased"],
        "timeout_seconds": 600
    }


@pytest.fixture
def test_certificates(tmp_path):
    """Generate test certificates for TLS testing."""
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    
    # Mock certificate data (in real implementation, generate actual certs)
    cert_data = {
        "ca_cert": "-----BEGIN CERTIFICATE-----\nMOCK_CA_CERT\n-----END CERTIFICATE-----",
        "server_cert": "-----BEGIN CERTIFICATE-----\nMOCK_SERVER_CERT\n-----END CERTIFICATE-----",
        "server_key": "-----BEGIN PRIVATE KEY-----\nMOCK_SERVER_KEY\n-----END PRIVATE KEY-----",
        "client_cert": "-----BEGIN CERTIFICATE-----\nMOCK_CLIENT_CERT\n-----END CERTIFICATE-----",
        "client_key": "-----BEGIN PRIVATE KEY-----\nMOCK_CLIENT_KEY\n-----END PRIVATE KEY-----"
    }
    
    for name, content in cert_data.items():
        (cert_dir / f"{name}.pem").write_text(content)
    
    return cert_dir


@pytest.fixture
def mock_model_weights(tmp_path):
    """Mock model weights for testing."""
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    
    # Create mock weight files
    weights = {
        "embeddings.weight": torch.randn(30522, 768),
        "encoder.layer.0.attention.self.query.weight": torch.randn(768, 768),
        "encoder.layer.0.attention.self.key.weight": torch.randn(768, 768),
        "encoder.layer.0.attention.self.value.weight": torch.randn(768, 768),
        "pooler.dense.weight": torch.randn(768, 768)
    }
    
    for name, weight in weights.items():
        torch.save(weight, weights_dir / f"{name}.pt")
    
    return weights_dir


@pytest.fixture
def api_test_client():
    """Mock API client for testing."""
    class MockAPIClient:
        def __init__(self):
            self.base_url = "http://localhost:8080"
            self.headers = {"Content-Type": "application/json"}
            self.session_id = "mock_session"
        
        def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
            # Mock successful response
            return {
                "status_code": 200,
                "data": {"result": "mock_result", "session_id": self.session_id},
                "headers": self.headers
            }
        
        def get(self, endpoint: str) -> Dict[str, Any]:
            return {
                "status_code": 200,
                "data": {"status": "healthy"},
                "headers": self.headers
            }
    
    return MockAPIClient()


@pytest.fixture
def database_test_data(tmp_path):
    """Test database with sample data."""
    db_file = tmp_path / "test.db"
    
    # Mock database data
    test_data = {
        "sessions": [
            {"id": 1, "session_id": "test_session_1", "status": "active"},
            {"id": 2, "session_id": "test_session_2", "status": "completed"}
        ],
        "results": [
            {"id": 1, "session_id": "test_session_1", "result": "test_result"},
            {"id": 2, "session_id": "test_session_2", "result": "another_result"}
        ]
    }
    
    # Save as JSON (in real implementation, use proper database)
    with open(db_file, 'w') as f:
        json.dump(test_data, f)
    
    return db_file


class TestMetricsCollector:
    """Utility class for collecting test metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing a operation."""
        import time
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        import time
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[f"{name}_duration"] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def record_metric(self, name: str, value: Any):
        """Record a metric value."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()


@pytest.fixture
def test_metrics_collector():
    """Test metrics collector fixture."""
    return TestMetricsCollector()


# Utility functions for test data generation

def generate_random_text(length: int = 100) -> str:
    """Generate random text for testing."""
    import random
    import string
    words = ['secure', 'computation', 'privacy', 'transformer', 'inference', 
             'cryptography', 'protocol', 'neural', 'network', 'machine', 'learning']
    
    text = []
    current_length = 0
    while current_length < length:
        word = random.choice(words)
        if current_length + len(word) + 1 <= length:
            text.append(word)
            current_length += len(word) + 1
        else:
            break
    
    return ' '.join(text)


def generate_test_tensor(shape: tuple, device: str = "cpu") -> torch.Tensor:
    """Generate test tensor with specific properties."""
    tensor = torch.randn(shape, device=device)
    # Ensure tensor has some structure for testing
    tensor = torch.clamp(tensor, -1.0, 1.0)
    return tensor


def create_mock_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create attention mask from input IDs."""
    return (input_ids != pad_token_id).long()
