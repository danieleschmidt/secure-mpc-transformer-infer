"""
Shared pytest configuration and fixtures for secure MPC transformer tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator, Any

import pytest
import torch


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run GPU-enabled tests"
    )
    parser.addoption(
        "--slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--protocol",
        action="store",
        default="semi_honest_3pc",
        help="MPC protocol to use for tests",
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmark tests"
    )


def pytest_configure(config):
    """Configure test markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "security: mark test as security-focused")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")
    config.addinivalue_line("markers", "protocol: specify MPC protocol for test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")

    for item in items:
        if "gpu" in item.keywords and not config.getoption("--gpu"):
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not config.getoption("--slow"):
            item.add_marker(skip_slow)
        if "benchmark" in item.keywords and not config.getoption("--benchmark"):
            item.add_marker(skip_benchmark)


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def test_protocol(request) -> str:
    """Get the MPC protocol to use for testing."""
    return request.config.getoption("--protocol")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Mock model configuration for testing."""
    return {
        "model_name": "bert-base-uncased",
        "max_sequence_length": 128,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
    }


@pytest.fixture
def security_config() -> Dict[str, Any]:
    """Security configuration for MPC tests."""
    return {
        "protocol": "semi_honest_3pc",
        "security_level": 128,
        "num_parties": 3,
        "gpu_acceleration": False,
    }


@pytest.fixture
def sample_text_data() -> list[str]:
    """Sample text data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Secure multi-party computation enables privacy-preserving ML.",
        "BERT model inference with homomorphic encryption.",
        "GPU acceleration for cryptographic protocols.",
    ]


@pytest.fixture(scope="session")
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "batch_sizes": [1, 4, 8],
        "sequence_lengths": [64, 128, 256],
        "iterations": 5,
        "warmup_iterations": 2,
        "timeout_seconds": 300,
    }


@pytest.fixture
def cleanup_gpu():
    """Cleanup GPU memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_network_config() -> Dict[str, Any]:
    """Mock network configuration for multi-party tests."""
    return {
        "parties": [
            {"id": 0, "host": "localhost", "port": 9000},
            {"id": 1, "host": "localhost", "port": 9001},
            {"id": 2, "host": "localhost", "port": 9002},
        ],
        "timeout": 30,
        "max_retries": 3,
    }


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    os.environ["MPC_TEST_MODE"] = "1"
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class GPUMemoryTracker:
    """Track GPU memory usage during tests."""

    def __init__(self):
        self.initial_memory = 0
        self.peak_memory = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()

    def stop(self) -> Dict[str, int]:
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated()
            return {
                "initial_memory_mb": self.initial_memory // (1024 * 1024),
                "peak_memory_mb": self.peak_memory // (1024 * 1024),
                "memory_delta_mb": (self.peak_memory - self.initial_memory)
                // (1024 * 1024),
            }
        return {}


@pytest.fixture
def gpu_memory_tracker():
    """Track GPU memory usage during test execution."""
    tracker = GPUMemoryTracker()
    tracker.start()
    yield tracker
    stats = tracker.stop()
    if stats:
        print(f"GPU Memory Stats: {stats}")