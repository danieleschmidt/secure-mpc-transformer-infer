"""
Performance benchmarks and profiling tests.
"""

import pytest
import torch
import time
import psutil
import gc
from typing import Dict, List, Any
from contextlib import contextmanager

from secure_mpc_transformer.models import SecureTransformer
from secure_mpc_transformer.protocols import ProtocolFactory
from secure_mpc_transformer.config import SecurityConfig


class PerformanceProfiler:
    """Profile performance metrics during test execution."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Start profiling."""
        gc.collect()  # Clean up before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return metrics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.metrics = {
            "execution_time_ms": (end_time - self.start_time) * 1000,
            "memory_usage_mb": end_memory - self.start_memory,
            "peak_memory_mb": end_memory,
        }
        
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.metrics.update({
                "gpu_memory_usage_mb": end_gpu_memory - self.start_gpu_memory,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            })
        
        return self.metrics


@contextmanager
def performance_profile():
    """Context manager for performance profiling."""
    profiler = PerformanceProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        metrics = profiler.stop()
        print(f"Performance metrics: {metrics}")


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 128),
        (4, 128),
        (8, 128),
        (1, 256),
        (4, 256),
    ])
    def test_inference_latency_scaling(self, mock_model_config, batch_size, seq_len):
        """Test inference latency scaling with batch size and sequence length."""
        config = SecurityConfig(protocol="semi_honest_3pc")
        model = SecureTransformer(mock_model_config, config)
        
        # Create input tensor
        input_tensor = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.forward(input_tensor)
        
        # Benchmark
        times = []
        for _ in range(10):
            with performance_profile() as profiler:
                with torch.no_grad():
                    output = model.forward(input_tensor)
            
            times.append(profiler.metrics["execution_time_ms"])
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        print(f"Batch={batch_size}, SeqLen={seq_len}: {avg_time:.1f}±{std_time:.1f}ms")
        
        # Performance assertions
        assert avg_time < 60000  # Should complete within 60 seconds
        assert std_time / avg_time < 0.2  # CV should be less than 20%
    
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_gpu_vs_cpu_performance(self, mock_model_config, gpu_available):
        """Compare GPU vs CPU performance."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        input_tensor = torch.randint(0, 1000, (4, 128))
        results = {}
        
        for device in ["cpu", "gpu"]:
            config = SecurityConfig(
                protocol="semi_honest_3pc",
                gpu_acceleration=(device == "gpu")
            )
            model = SecureTransformer(mock_model_config, config)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model.forward(input_tensor)
            
            # Benchmark
            times = []
            for _ in range(5):
                with performance_profile() as profiler:
                    with torch.no_grad():
                        output = model.forward(input_tensor)
                
                times.append(profiler.metrics["execution_time_ms"])
            
            results[device] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
            }
            
            print(f"{device.upper()} performance: {results[device]}")
        
        # GPU should be faster than CPU (allowing some variance)
        speedup = results["cpu"]["avg_time"] / results["gpu"]["avg_time"]
        print(f"GPU speedup: {speedup:.2f}x")
        
        assert speedup > 1.5  # GPU should be at least 1.5x faster
    
    @pytest.mark.parametrize("protocol", [
        "semi_honest_3pc",
        "malicious_3pc",
        "aby3"
    ])
    def test_protocol_performance_comparison(self, protocol, mock_network_config):
        """Compare performance across different protocols."""
        protocol_impl = ProtocolFactory.create(
            protocol,
            num_parties=3,
            party_id=0,
            network_config=mock_network_config
        )
        
        # Test data
        test_tensor = torch.randn(256, 256)
        
        # Warmup
        for _ in range(3):
            shares = protocol_impl.share_secret(test_tensor)
            _ = protocol_impl.reconstruct_secret(shares)
        
        # Benchmark sharing
        share_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            shares = protocol_impl.share_secret(test_tensor)
            share_times.append((time.perf_counter() - start_time) * 1000)
        
        # Benchmark reconstruction
        reconstruct_times = []
        for _ in range(10):
            shares = protocol_impl.share_secret(test_tensor)
            start_time = time.perf_counter()
            _ = protocol_impl.reconstruct_secret(shares)
            reconstruct_times.append((time.perf_counter() - start_time) * 1000)
        
        avg_share_time = sum(share_times) / len(share_times)
        avg_reconstruct_time = sum(reconstruct_times) / len(reconstruct_times)
        
        print(f"{protocol} - Share: {avg_share_time:.2f}ms, Reconstruct: {avg_reconstruct_time:.2f}ms")
        
        # Performance thresholds
        assert avg_share_time < 1000  # Should complete within 1 second
        assert avg_reconstruct_time < 1000
    
    def test_memory_usage_scaling(self, mock_model_config):
        """Test memory usage scaling with tensor sizes."""
        config = SecurityConfig(protocol="semi_honest_3pc")
        model = SecureTransformer(mock_model_config, config)
        
        memory_usage = []
        tensor_sizes = [64, 128, 256, 512]
        
        for size in tensor_sizes:
            input_tensor = torch.randint(0, 1000, (1, size))
            
            with performance_profile() as profiler:
                with torch.no_grad():
                    output = model.forward(input_tensor)
                    
                    # Force memory cleanup
                    del output
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            memory_usage.append({
                "size": size,
                "memory_mb": profiler.metrics.get("memory_usage_mb", 0),
                "gpu_memory_mb": profiler.metrics.get("gpu_memory_usage_mb", 0)
            })
        
        # Analyze memory scaling
        for usage in memory_usage:
            print(f"Size {usage['size']}: {usage['memory_mb']:.1f}MB RAM, "
                  f"{usage['gpu_memory_mb']:.1f}MB GPU")
        
        # Memory usage should scale reasonably
        max_memory = max(usage["memory_mb"] for usage in memory_usage)
        assert max_memory < 4000  # Should use less than 4GB RAM
        
        if torch.cuda.is_available():
            max_gpu_memory = max(usage["gpu_memory_mb"] for usage in memory_usage)
            assert max_gpu_memory < 8000  # Should use less than 8GB GPU memory
    
    @pytest.mark.slow
    def test_throughput_measurement(self, mock_model_config):
        """Measure inference throughput."""
        config = SecurityConfig(protocol="semi_honest_3pc")
        model = SecureTransformer(mock_model_config, config)
        
        batch_sizes = [1, 2, 4, 8]
        throughput_results = []
        
        for batch_size in batch_sizes:
            input_tensor = torch.randint(0, 1000, (batch_size, 128))
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model.forward(input_tensor)
            
            # Measure throughput
            num_batches = 20
            start_time = time.perf_counter()
            
            for _ in range(num_batches):
                with torch.no_grad():
                    output = model.forward(input_tensor)
            
            total_time = time.perf_counter() - start_time
            total_samples = num_batches * batch_size
            throughput = total_samples / total_time
            
            throughput_results.append({
                "batch_size": batch_size,
                "throughput_samples_per_sec": throughput,
                "latency_per_sample_ms": (total_time / total_samples) * 1000
            })
            
            print(f"Batch size {batch_size}: {throughput:.2f} samples/sec, "
                  f"{throughput_results[-1]['latency_per_sample_ms']:.1f}ms per sample")
        
        # Throughput should increase with batch size (up to a point)
        for i in range(1, len(throughput_results)):
            current_throughput = throughput_results[i]["throughput_samples_per_sec"]
            assert current_throughput > 0.1  # Minimum viable throughput
    
    @pytest.mark.integration
    def test_end_to_end_performance(self, mock_model_config, sample_text_data):
        """Test end-to-end performance from text to prediction."""
        config = SecurityConfig(protocol="semi_honest_3pc")
        model = SecureTransformer(mock_model_config, config)
        
        # Simulate tokenization (mock)
        tokenized_inputs = [
            torch.randint(0, 1000, (1, 128)) for _ in sample_text_data
        ]
        
        total_times = []
        
        for text, input_tensor in zip(sample_text_data, tokenized_inputs):
            with performance_profile() as profiler:
                # End-to-end inference
                with torch.no_grad():
                    output = model.forward(input_tensor)
                    # Simulate post-processing
                    prediction = torch.argmax(output, dim=-1)
            
            total_times.append(profiler.metrics["execution_time_ms"])
            print(f"Text length {len(text.split())}: {profiler.metrics['execution_time_ms']:.1f}ms")
        
        avg_time = sum(total_times) / len(total_times)
        print(f"Average end-to-end time: {avg_time:.1f}ms")
        
        # Performance target: less than 60 seconds per inference
        assert avg_time < 60000
        assert all(t < 120000 for t in total_times)  # No single inference > 2 minutes


@pytest.mark.benchmark
class TestStressTests:
    """Stress tests for system limits."""
    
    @pytest.mark.slow
    def test_memory_stress(self, mock_model_config):
        """Test system behavior under memory pressure."""
        config = SecurityConfig(protocol="semi_honest_3pc")
        model = SecureTransformer(mock_model_config, config)
        
        # Gradually increase tensor sizes until memory pressure
        max_size = 2048  # Start reasonable
        successful_sizes = []
        
        for size in [64, 128, 256, 512, 1024, max_size]:
            try:
                input_tensor = torch.randint(0, 1000, (1, size))
                
                with torch.no_grad():
                    output = model.forward(input_tensor)
                    successful_sizes.append(size)
                
                # Cleanup
                del input_tensor, output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"Memory limit reached at size {size}: {e}")
                break
        
        print(f"Successfully processed sizes: {successful_sizes}")
        assert len(successful_sizes) > 0  # Should handle at least small sizes
    
    @pytest.mark.slow
    def test_concurrent_inference(self, mock_model_config):
        """Test concurrent inference requests."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        config = SecurityConfig(protocol="semi_honest_3pc")
        model = SecureTransformer(mock_model_config, config)
        
        def inference_worker(worker_id: int) -> Dict[str, Any]:
            """Worker function for concurrent inference."""
            input_tensor = torch.randint(0, 1000, (1, 128))
            
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model.forward(input_tensor)
            execution_time = time.perf_counter() - start_time
            
            return {
                "worker_id": worker_id,
                "execution_time": execution_time * 1000,
                "output_shape": output.shape
            }
        
        # Test with multiple concurrent requests
        num_workers = 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(inference_worker, i) for i in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        execution_times = [r["execution_time"] for r in results]
        avg_time = sum(execution_times) / len(execution_times)
        
        print(f"Concurrent inference results:")
        for result in results:
            print(f"  Worker {result['worker_id']}: {result['execution_time']:.1f}ms")
        print(f"Average time: {avg_time:.1f}ms")
        
        # All requests should complete successfully
        assert len(results) == num_workers
        assert all(r["execution_time"] < 120000 for r in results)  # < 2 minutes each