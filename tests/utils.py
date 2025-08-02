"""
Test utilities and helper functions for secure MPC transformer tests.
"""

import time
import functools
from typing import Callable, Any, Dict, List
from contextlib import contextmanager

import pytest
import torch


def requires_gpu(func: Callable) -> Callable:
    """Decorator to skip tests if GPU is not available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return func(*args, **kwargs)
    return wrapper


def timeout(seconds: int):
    """Decorator to timeout long-running tests."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator


@contextmanager
def timing_context():
    """Context manager to time code execution."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


def assert_secure_computation_properties(result: Any, expected_type: type = None):
    """Assert that secure computation results have expected properties."""
    assert result is not None, "Secure computation result should not be None"
    
    if expected_type:
        assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"
    
    # Add more security-specific assertions as needed
    if hasattr(result, 'security_level'):
        assert result.security_level >= 128, "Security level should be at least 128 bits"


def create_test_tensor(shape: tuple, device: str = "cpu") -> torch.Tensor:
    """Create a test tensor with deterministic values."""
    torch.manual_seed(42)  # Ensure reproducible tests
    return torch.randn(shape, device=device)


def verify_mpc_protocol_invariants(protocol_result: Dict[str, Any]):
    """Verify that MPC protocol results maintain required invariants."""
    required_fields = ['shares', 'computation_time', 'communication_rounds']
    
    for field in required_fields:
        assert field in protocol_result, f"Missing required field: {field}"
    
    assert protocol_result['computation_time'] > 0, "Computation time should be positive"
    assert protocol_result['communication_rounds'] >= 0, "Communication rounds should be non-negative"
    assert len(protocol_result['shares']) > 0, "Should have at least one share"


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        self.metrics[name] = value
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        if self.start_time:
            elapsed = time.perf_counter() - self.start_time
            self.metrics['total_time'] = elapsed
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return self.metrics.copy()


def generate_test_protocol_configs() -> List[Dict[str, Any]]:
    """Generate test configurations for different MPC protocols."""
    return [
        {
            "name": "semi_honest_3pc",
            "security": "semi-honest",
            "num_parties": 3,
            "expected_rounds": 1,
        },
        {
            "name": "malicious_3pc", 
            "security": "malicious",
            "num_parties": 3,
            "expected_rounds": 2,
        },
        {
            "name": "4pc_gpu",
            "security": "semi-honest",
            "num_parties": 4,
            "expected_rounds": 1,
            "gpu_required": True,
        }
    ]


def assert_privacy_preservation(original_data: Any, processed_data: Any):
    """Assert that privacy is preserved during computation."""
    # Ensure processed data doesn't leak original information
    if isinstance(original_data, torch.Tensor) and isinstance(processed_data, torch.Tensor):
        # Data should be transformed (not identical)
        assert not torch.equal(original_data, processed_data), \
            "Processed data should not be identical to original"
        
        # Should maintain same shape for most operations
        if original_data.shape == processed_data.shape:
            correlation = torch.corrcoef(torch.stack([
                original_data.flatten(), 
                processed_data.flatten()
            ]))[0, 1]
            
            # Correlation should be low for secure transformations
            assert abs(correlation) < 0.1, \
                f"Correlation too high ({correlation:.3f}), may leak information"


@pytest.fixture
def performance_monitor():
    """Fixture providing performance monitoring capabilities."""
    monitor = PerformanceMonitor()
    yield monitor
    metrics = monitor.stop()
    if metrics:
        print(f"Performance metrics: {metrics}")