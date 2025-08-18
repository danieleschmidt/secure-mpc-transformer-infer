"""
Comprehensive test utilities for Secure MPC Transformer testing.
"""

import asyncio
import contextlib
import functools
import io
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch

import pytest
import torch
import numpy as np


class TestTimer:
    """Context manager for timing test operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"⏱️  {self.name} took {self.duration:.4f} seconds")


class MemoryMonitor:
    """Monitor memory usage during tests."""
    
    def __init__(self, gpu: bool = True):
        self.gpu = gpu and torch.cuda.is_available()
        self.initial_memory = None
        self.peak_memory = None
        
    def __enter__(self):
        if self.gpu:
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
        else:
            import psutil
            self.initial_memory = psutil.Process().memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu:
            self.peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (self.peak_memory - self.initial_memory) / 1024 / 1024
            print(f"🧠 GPU Memory used: {memory_used:.1f} MB")
        else:
            import psutil
            current_memory = psutil.Process().memory_info().rss
            memory_used = (current_memory - self.initial_memory) / 1024 / 1024
            print(f"🧠 CPU Memory used: {memory_used:.1f} MB")


class CaptureOutput:
    """Capture stdout/stderr during tests."""
    
    def __init__(self, capture_stdout: bool = True, capture_stderr: bool = True):
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr
        self.stdout = None
        self.stderr = None
        self.old_stdout = None
        self.old_stderr = None
    
    def __enter__(self):
        if self.capture_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = self.stdout = io.StringIO()
        
        if self.capture_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.stderr = io.StringIO()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stdout:
            sys.stdout = self.old_stdout
        
        if self.capture_stderr:
            sys.stderr = self.old_stderr
    
    def get_output(self) -> Tuple[Optional[str], Optional[str]]:
        """Get captured output."""
        stdout_val = self.stdout.getvalue() if self.stdout else None
        stderr_val = self.stderr.getvalue() if self.stderr else None
        return stdout_val, stderr_val


class TempDirectory:
    """Enhanced temporary directory context manager."""
    
    def __init__(self, prefix: str = "mpc_test_", cleanup: bool = True):
        self.prefix = prefix
        self.cleanup = cleanup
        self.path = None
        self.temp_dir = None
    
    def __enter__(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix=self.prefix)
        self.path = Path(self.temp_dir.name)
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup and self.temp_dir:
            self.temp_dir.cleanup()


class NetworkSimulator:
    """Simulate network conditions for testing."""
    
    def __init__(self, latency_ms: float = 0, packet_loss: float = 0, bandwidth_mbps: float = 1000):
        self.latency_ms = latency_ms
        self.packet_loss = packet_loss
        self.bandwidth_mbps = bandwidth_mbps
    
    async def simulate_network_delay(self):
        """Simulate network latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
    
    def should_drop_packet(self) -> bool:
        """Simulate packet loss."""
        import random
        return random.random() < self.packet_loss
    
    def calculate_transfer_time(self, size_bytes: int) -> float:
        """Calculate transfer time based on bandwidth."""
        size_mbits = (size_bytes * 8) / (1024 * 1024)
        return size_mbits / self.bandwidth_mbps


class MockGPUEnvironment:
    """Mock GPU environment for testing without actual GPU."""
    
    def __init__(self, num_gpus: int = 1, memory_per_gpu_mb: int = 8192):
        self.num_gpus = num_gpus
        self.memory_per_gpu_mb = memory_per_gpu_mb
        self.allocated_memory = [0] * num_gpus
    
    def __enter__(self):
        # Mock CUDA functions
        self.original_is_available = torch.cuda.is_available
        self.original_device_count = torch.cuda.device_count
        self.original_get_device_name = torch.cuda.get_device_name
        
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: self.num_gpus
        torch.cuda.get_device_name = lambda device=0: f"Mock GPU {device}"
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original functions
        torch.cuda.is_available = self.original_is_available
        torch.cuda.device_count = self.original_device_count
        torch.cuda.get_device_name = self.original_get_device_name


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions: Tuple = (Exception,)):
    """Decorator to retry test functions on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⚠️  Test failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"❌ Test failed after {max_retries + 1} attempts")
            raise last_exception
        return wrapper
    return decorator


def timeout_test(timeout_seconds: float):
    """Decorator to add timeout to test functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target():
                return func(*args, **kwargs)
            
            import threading
            result = [None]
            exception = [None]
            
            def run_target():
                try:
                    result[0] = target()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=run_target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Test timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator


def skip_if_no_gpu(func):
    """Decorator to skip tests if GPU is not available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return func(*args, **kwargs)
    return wrapper


def require_environment_variable(var_name: str):
    """Decorator to skip tests if environment variable is not set."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.environ.get(var_name):
                pytest.skip(f"Environment variable {var_name} not set")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class AssertionHelper:
    """Helper class for common test assertions."""
    
    @staticmethod
    def assert_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8):
        """Assert that two tensors are equal within tolerance."""
        assert tensor1.shape == tensor2.shape, f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), "Tensors are not equal within tolerance"
    
    @staticmethod
    def assert_timing_within_bounds(duration: float, min_time: float, max_time: float):
        """Assert that operation timing is within expected bounds."""
        assert min_time <= duration <= max_time, f"Timing {duration:.4f}s not in range [{min_time}, {max_time}]"
    
    @staticmethod
    def assert_memory_usage_reasonable(memory_mb: float, max_memory_mb: float):
        """Assert that memory usage is reasonable."""
        assert memory_mb <= max_memory_mb, f"Memory usage {memory_mb:.1f}MB exceeds limit {max_memory_mb}MB"
    
    @staticmethod
    def assert_security_property(test_func: Callable, error_message: str):
        """Assert that a security property holds."""
        assert test_func(), error_message
    
    @staticmethod
    def assert_no_information_leakage(secret_data: Any, public_output: Any):
        """Assert that no information about secret data is leaked in public output."""
        # This is a simplified check - real implementation would be more sophisticated
        secret_str = str(secret_data).lower()
        output_str = str(public_output).lower()
        
        # Check for obvious leakage patterns
        for word in secret_str.split():
            if len(word) > 3 and word in output_str:
                raise AssertionError(f"Potential information leakage detected: '{word}' found in output")


class TestDataValidator:
    """Validate test data integrity and consistency."""
    
    @staticmethod
    def validate_model_output(output: Dict[str, Any], expected_keys: List[str]):
        """Validate that model output has expected structure."""
        assert isinstance(output, dict), "Output must be a dictionary"
        
        for key in expected_keys:
            assert key in output, f"Missing expected key: {key}"
        
        # Check for common output validation
        if "predictions" in output:
            assert isinstance(output["predictions"], (list, torch.Tensor)), "Predictions must be list or tensor"
        
        if "confidence" in output:
            assert 0 <= output["confidence"] <= 1, "Confidence must be between 0 and 1"
    
    @staticmethod
    def validate_cryptographic_output(ciphertext: str, min_entropy: float = 4.0):
        """Validate that cryptographic output has sufficient entropy."""
        import math
        from collections import Counter
        
        # Calculate Shannon entropy
        char_counts = Counter(ciphertext)
        length = len(ciphertext)
        entropy = -sum((count / length) * math.log2(count / length) for count in char_counts.values())
        
        assert entropy >= min_entropy, f"Insufficient entropy: {entropy:.2f} < {min_entropy}"
    
    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, float], thresholds: Dict[str, float]):
        """Validate that performance metrics meet thresholds."""
        for metric, threshold in thresholds.items():
            if metric in metrics:
                assert metrics[metric] <= threshold, f"Performance metric {metric} ({metrics[metric]}) exceeds threshold ({threshold})"


class BenchmarkHelper:
    """Helper for benchmark tests."""
    
    def __init__(self, name: str, iterations: int = 10, warmup: int = 3):
        self.name = name
        self.iterations = iterations
        self.warmup = warmup
        self.times = []
    
    def run_benchmark(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Run benchmark and return statistics."""
        # Warmup runs
        for _ in range(self.warmup):
            func(*args, **kwargs)
        
        # Actual benchmark runs
        self.times = []
        for _ in range(self.iterations):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            self.times.append(end_time - start_time)
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get benchmark statistics."""
        times_array = np.array(self.times)
        return {
            "mean": float(np.mean(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "median": float(np.median(times_array)),
            "p95": float(np.percentile(times_array, 95)),
            "p99": float(np.percentile(times_array, 99))
        }


def create_test_environment_marker():
    """Create a marker to identify test environment."""
    import uuid
    return f"test_env_{uuid.uuid4().hex[:8]}"


def cleanup_test_artifacts(directory: Path):
    """Clean up test artifacts from directory."""
    patterns_to_clean = [
        "*.tmp",
        "*.log",
        "test_*",
        "mock_*",
        "__pycache__",
        "*.pyc",
        ".pytest_cache"
    ]
    
    import glob
    for pattern in patterns_to_clean:
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)


@contextlib.contextmanager
def suppress_warnings():
    """Context manager to suppress warnings during tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def compare_model_outputs(output1: torch.Tensor, output2: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """Compare model outputs with tolerance for numerical differences."""
    if output1.shape != output2.shape:
        return False
    
    return torch.allclose(output1, output2, rtol=tolerance, atol=tolerance)


def generate_deterministic_seed(test_name: str) -> int:
    """Generate deterministic seed from test name for reproducible tests."""
    import hashlib
    hash_object = hashlib.md5(test_name.encode())
    return int(hash_object.hexdigest()[:8], 16) % (2**31)


class TestReporter:
    """Collect and report test results and metrics."""
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
        self.start_time = time.time()
    
    def add_result(self, test_name: str, result: Dict[str, Any]):
        """Add test result."""
        self.results[test_name] = result
    
    def add_metric(self, metric_name: str, value: float):
        """Add performance metric."""
        self.metrics[metric_name] = value
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "num_tests": len(self.results),
            "results": self.results,
            "metrics": self.metrics,
            "summary": {
                "passed": sum(1 for r in self.results.values() if r.get("passed", False)),
                "failed": sum(1 for r in self.results.values() if not r.get("passed", True)),
                "avg_duration": np.mean([r.get("duration", 0) for r in self.results.values()])
            }
        }
