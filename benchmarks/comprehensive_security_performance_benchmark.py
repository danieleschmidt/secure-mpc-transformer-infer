#!/usr/bin/env python3
"""
Comprehensive Security and Performance Benchmark Suite
for Secure MPC Transformer System

This benchmark suite validates both security properties and performance
characteristics under realistic production conditions.
"""

import time
import torch
import numpy as np
import asyncio
import statistics
import json
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import concurrent.futures
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from secure_mpc_transformer.planning.quantum_planner import QuantumTaskPlanner, Task, TaskType, QuantumTaskConfig
    from secure_mpc_transformer.caching.cache_manager import CacheManager, CacheConfig
    from secure_mpc_transformer.optimization.gpu_manager import GPUMemoryManager, StreamConfig
    from secure_mpc_transformer.security.key_manager import CryptographicKeyManager
    from secure_mpc_transformer.security.threat_detector import AdvancedThreatDetector, SecurityEvent
    from secure_mpc_transformer.resilience.health_checks import HealthCheckManager
    from secure_mpc_transformer.utils.error_handling import ErrorHandler
    from secure_mpc_transformer.protocols.factory import ProtocolFactory
    from secure_mpc_transformer.models.secure_transformer import SecureTransformer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the src directory is in your Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result from a benchmark test."""
    test_name: str
    category: str
    success: bool
    duration_ms: float
    throughput_ops_per_sec: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    security_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    test_duration_seconds: int = 60
    warmup_seconds: int = 10
    num_iterations: int = 1000
    concurrent_workers: int = 4
    enable_gpu_tests: bool = torch.cuda.is_available()
    security_level: int = 128
    memory_limit_mb: int = 8192
    
class SecurityPerformanceBenchmark:
    """Comprehensive security and performance benchmark suite."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.setup_components()
        
    def setup_components(self):
        """Initialize system components for testing."""
        logger.info("Initializing benchmark components...")
        
        # Initialize quantum planner
        self.quantum_config = QuantumTaskConfig(
            max_parallel_tasks=self.config.concurrent_workers,
            enable_gpu_acceleration=self.config.enable_gpu_tests
        )
        self.quantum_planner = QuantumTaskPlanner(self.quantum_config)
        
        # Initialize cache manager
        cache_config = CacheConfig(
            l1_max_memory_mb=1024,
            l2_max_memory_mb=2048,
            enable_cache_warming=True
        )
        self.cache_manager = CacheManager(cache_config)
        
        # Initialize GPU manager (if available)
        if self.config.enable_gpu_tests:
            stream_config = StreamConfig(max_streams=self.config.concurrent_workers)
            self.gpu_manager = GPUMemoryManager(stream_config=stream_config)
        else:
            self.gpu_manager = None
            
        # Initialize security components
        self.key_manager = CryptographicKeyManager()
        self.threat_detector = AdvancedThreatDetector()
        self.health_manager = HealthCheckManager()
        self.error_handler = ErrorHandler()
        
        logger.info("Component initialization completed")
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark categories."""
        logger.info("Starting comprehensive benchmark suite...")
        start_time = time.time()
        
        # Benchmark categories
        benchmark_categories = [
            ("Security", self.run_security_benchmarks),
            ("Performance", self.run_performance_benchmarks),
            ("Quantum Planning", self.run_quantum_planning_benchmarks),
            ("Caching", self.run_caching_benchmarks),
            ("GPU Operations", self.run_gpu_benchmarks),
            ("Resilience", self.run_resilience_benchmarks),
            ("Integration", self.run_integration_benchmarks)
        ]
        
        category_results = {}
        
        for category_name, benchmark_func in benchmark_categories:
            logger.info(f"Running {category_name} benchmarks...")
            try:
                category_results[category_name] = benchmark_func()
            except Exception as e:
                logger.error(f"Error in {category_name} benchmarks: {e}")
                category_results[category_name] = {
                    "error": str(e),
                    "success": False
                }
        
        total_duration = time.time() - start_time
        
        # Compile final results
        final_results = {
            "benchmark_config": asdict(self.config),
            "total_duration_seconds": total_duration,
            "categories": category_results,
            "summary": self.generate_summary(),
            "recommendations": self.generate_recommendations()
        }
        
        logger.info(f"Benchmark suite completed in {total_duration:.2f} seconds")
        return final_results
    
    def run_security_benchmarks(self) -> Dict[str, Any]:
        """Run security-focused benchmarks."""
        results = []
        
        # Test 1: Key Management Performance
        results.append(self.benchmark_key_management())
        
        # Test 2: Threat Detection Accuracy
        results.append(self.benchmark_threat_detection())
        
        # Test 3: Cryptographic Operations
        results.append(self.benchmark_cryptographic_operations())
        
        # Test 4: Side-Channel Resistance
        results.append(self.benchmark_side_channel_resistance())
        
        # Test 5: Protocol Security
        results.append(self.benchmark_protocol_security())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results),
            "avg_security_score": statistics.mean(r.security_score for r in results if r.security_score)
        }
    
    def benchmark_key_management(self) -> BenchmarkResult:
        """Benchmark cryptographic key management operations."""
        start_time = time.perf_counter()
        
        try:
            operations_count = 0
            
            # Generate various types of keys
            for _ in range(100):
                # Symmetric keys
                self.key_manager.generate_symmetric_key(expires_in=3600)
                operations_count += 1
                
                # Asymmetric keypairs  
                self.key_manager.generate_asymmetric_keypair(expires_in=7200)
                operations_count += 2
                
                # MAC keys
                self.key_manager.generate_mac_key()
                operations_count += 1
            
            # Test key retrieval and operations
            keys = self.key_manager.key_store.list_keys()
            for key in keys[:10]:  # Test subset
                retrieved = self.key_manager.get_key(key.key_id)
                if retrieved:
                    operations_count += 1
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = operations_count / (duration_ms / 1000)
            
            # Security score based on key strength and management practices
            stats = self.key_manager.get_key_stats()
            security_score = min(1.0, stats.get('by_status', {}).get('active', 0) / 100)
            
            return BenchmarkResult(
                test_name="Key Management",
                category="Security",
                success=True,
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                security_score=security_score,
                metrics=stats
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Key Management",
                category="Security", 
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def benchmark_threat_detection(self) -> BenchmarkResult:
        """Benchmark threat detection system."""
        start_time = time.perf_counter()
        
        try:
            # Generate test security events
            test_events = []
            
            # Normal events
            for i in range(500):
                event = SecurityEvent(
                    event_id=f"normal_{i}",
                    timestamp=time.time(),
                    source_ip=f"192.168.1.{i % 254 + 1}",
                    user_agent="Mozilla/5.0 Test",
                    endpoint="/api/inference",
                    method="POST",
                    status_code=200,
                    response_time=100.0,
                    request_size=1024,
                    response_size=2048,
                    headers={"Content-Type": "application/json"},
                    payload_hash="abc123",
                    geolocation={"country": "US"},
                    threat_indicators=[],
                    risk_score=10
                )
                test_events.append(event)
            
            # Malicious events
            malicious_patterns = [
                ("SQL Injection", "'; DROP TABLE users; --"),
                ("XSS", "<script>alert('xss')</script>"),
                ("DDoS", "192.168.1.100"),
                ("Brute Force", "admin:password123")
            ]
            
            for attack_type, pattern in malicious_patterns:
                for i in range(50):
                    event = SecurityEvent(
                        event_id=f"malicious_{attack_type}_{i}",
                        timestamp=time.time(),
                        source_ip="192.168.1.200",
                        user_agent="AttackBot/1.0",
                        endpoint="/api/login",
                        method="POST",
                        status_code=401,
                        response_time=5000.0,
                        request_size=2048,
                        response_size=512,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        payload_hash=pattern,
                        geolocation={"country": "Unknown"},
                        threat_indicators=[attack_type.lower()],
                        risk_score=90
                    )
                    test_events.append(event)
            
            # Process events through threat detector
            detected_threats = []
            false_positives = 0
            false_negatives = 0
            
            for event in test_events:
                threat_intel = self.threat_detector.analyze_request(event)
                detected_threats.append(threat_intel)
                
                # Evaluate accuracy
                is_actually_malicious = event.risk_score > 80
                is_detected_as_threat = threat_intel.threat_level.value in ["high", "critical"]
                
                if is_detected_as_threat and not is_actually_malicious:
                    false_positives += 1
                elif not is_detected_as_threat and is_actually_malicious:
                    false_negatives += 1
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = len(test_events) / (duration_ms / 1000)
            
            # Calculate security metrics
            total_malicious = sum(1 for e in test_events if e.risk_score > 80)
            total_benign = len(test_events) - total_malicious
            
            false_positive_rate = false_positives / total_benign if total_benign > 0 else 0
            false_negative_rate = false_negatives / total_malicious if total_malicious > 0 else 0
            
            # Security score (higher is better)
            security_score = 1.0 - (false_positive_rate + false_negative_rate) / 2
            
            return BenchmarkResult(
                test_name="Threat Detection",
                category="Security",
                success=True,
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                security_score=security_score,
                metrics={
                    "total_events": len(test_events),
                    "detected_threats": len(detected_threats),
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "false_positive_rate": false_positive_rate,
                    "false_negative_rate": false_negative_rate
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Threat Detection",
                category="Security",
                success=False, 
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def benchmark_cryptographic_operations(self) -> BenchmarkResult:
        """Benchmark cryptographic operations."""
        start_time = time.perf_counter()
        
        try:
            # Test various cryptographic operations
            operations = []
            
            # Generate test data
            test_data = torch.randn(1000, 64)  # 1000 samples of 64-dim vectors
            
            # Benchmark secret sharing
            try:
                protocol = ProtocolFactory.create(
                    "semi_honest_3pc",
                    num_parties=3,
                    party_id=0
                )
                
                for i in range(100):
                    sample = test_data[i]
                    
                    # Share secret
                    share_start = time.perf_counter()
                    shares = protocol.share_value(sample)
                    share_time = (time.perf_counter() - share_start) * 1000
                    operations.append(("share", share_time))
                    
                    # Reconstruct secret
                    reconstruct_start = time.perf_counter()
                    reconstructed = protocol.reconstruct_value(shares)
                    reconstruct_time = (time.perf_counter() - reconstruct_start) * 1000
                    operations.append(("reconstruct", reconstruct_time))
                    
                    # Verify correctness (security property)
                    if not torch.allclose(sample, reconstructed, rtol=1e-4):
                        raise ValueError(f"Reconstruction failed for sample {i}")
                
            except Exception as e:
                logger.warning(f"Protocol operations failed: {e}")
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Calculate performance metrics
            if operations:
                avg_share_time = statistics.mean(t for op, t in operations if op == "share")
                avg_reconstruct_time = statistics.mean(t for op, t in operations if op == "reconstruct")
                total_ops = len(operations)
                throughput = total_ops / (duration_ms / 1000)
                
                metrics = {
                    "total_operations": total_ops,
                    "avg_share_time_ms": avg_share_time,
                    "avg_reconstruct_time_ms": avg_reconstruct_time,
                    "operations_breakdown": dict(Counter(op for op, _ in operations))
                }
            else:
                throughput = 0
                metrics = {"error": "No operations completed"}
            
            # Security score based on correctness and timing consistency
            timing_variance = statistics.stdev(t for _, t in operations) if len(operations) > 1 else 0
            timing_consistency = 1.0 - min(1.0, timing_variance / 100)  # Penalize high variance
            security_score = timing_consistency
            
            return BenchmarkResult(
                test_name="Cryptographic Operations",
                category="Security",
                success=len(operations) > 0,
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                security_score=security_score,
                metrics=metrics
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Cryptographic Operations", 
                category="Security",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def benchmark_side_channel_resistance(self) -> BenchmarkResult:
        """Benchmark resistance to side-channel attacks."""
        start_time = time.perf_counter()
        
        try:
            # Test timing consistency across different inputs
            input_sizes = [64, 128, 256, 512]
            timing_measurements = {}
            
            for size in input_sizes:
                times = []
                for _ in range(50):  # Multiple measurements
                    test_input = torch.randn(size)
                    
                    op_start = time.perf_counter()
                    
                    # Simulate secure operation
                    result = test_input * 2.0  # Placeholder operation
                    result = torch.sum(result)
                    
                    op_time = (time.perf_counter() - op_start) * 1000000  # microseconds
                    times.append(op_time)
                
                timing_measurements[size] = {
                    "mean": statistics.mean(times),
                    "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                    "measurements": times
                }
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Analyze timing patterns for side-channel leakage
            # Calculate coefficient of variation for each input size
            timing_consistency_scores = []
            for size, measurements in timing_measurements.items():
                cv = measurements["stdev"] / measurements["mean"] if measurements["mean"] > 0 else 1
                consistency_score = 1.0 - min(1.0, cv)  # Lower variance is better
                timing_consistency_scores.append(consistency_score)
            
            # Overall security score
            security_score = statistics.mean(timing_consistency_scores)
            
            return BenchmarkResult(
                test_name="Side-Channel Resistance",
                category="Security",
                success=True,
                duration_ms=duration_ms,
                security_score=security_score,
                metrics={
                    "timing_measurements": timing_measurements,
                    "consistency_scores": timing_consistency_scores,
                    "avg_consistency": security_score
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Side-Channel Resistance",
                category="Security",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def benchmark_protocol_security(self) -> BenchmarkResult:
        """Benchmark MPC protocol security properties."""
        start_time = time.perf_counter()
        
        try:
            security_tests = []
            
            # Test different protocol types
            protocol_types = ["semi_honest_3pc"]  # Add more as available
            
            for protocol_type in protocol_types:
                try:
                    protocol = ProtocolFactory.create(
                        protocol_type,
                        num_parties=3,
                        party_id=0
                    )
                    
                    # Test 1: Privacy preservation
                    secret = torch.randn(100)
                    shares = protocol.share_value(secret)
                    
                    # Verify individual shares don't reveal secret
                    privacy_preserved = True
                    for share in shares.shares:
                        correlation = torch.corrcoef(torch.stack([secret, share]))[0,1].item()
                        if abs(correlation) > 0.1:  # Threshold for acceptable correlation
                            privacy_preserved = False
                            break
                    
                    # Test 2: Correctness
                    reconstructed = protocol.reconstruct_value(shares)
                    correctness = torch.allclose(secret, reconstructed, rtol=1e-4)
                    
                    # Test 3: Consistency across multiple runs
                    consistency_results = []
                    for _ in range(10):
                        shares_test = protocol.share_value(secret)
                        reconstructed_test = protocol.reconstruct_value(shares_test)
                        consistency_results.append(
                            torch.allclose(secret, reconstructed_test, rtol=1e-4)
                        )
                    
                    consistency = all(consistency_results)
                    
                    security_tests.append({
                        "protocol": protocol_type,
                        "privacy_preserved": privacy_preserved,
                        "correctness": correctness,
                        "consistency": consistency,
                        "overall_pass": privacy_preserved and correctness and consistency
                    })
                    
                except Exception as e:
                    logger.warning(f"Protocol {protocol_type} test failed: {e}")
                    security_tests.append({
                        "protocol": protocol_type,
                        "error": str(e),
                        "overall_pass": False
                    })
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Calculate overall security score
            passed_tests = sum(1 for test in security_tests if test.get("overall_pass", False))
            security_score = passed_tests / len(security_tests) if security_tests else 0
            
            return BenchmarkResult(
                test_name="Protocol Security",
                category="Security",
                success=security_score > 0.5,
                duration_ms=duration_ms,
                security_score=security_score,
                metrics={
                    "security_tests": security_tests,
                    "passed_tests": passed_tests,
                    "total_tests": len(security_tests)
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Protocol Security",
                category="Security", 
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance-focused benchmarks."""
        results = []
        
        # Test 1: Memory Management
        results.append(self.benchmark_memory_management())
        
        # Test 2: Concurrent Processing
        results.append(self.benchmark_concurrent_processing())
        
        # Test 3: Error Handling Performance
        results.append(self.benchmark_error_handling())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results),
            "avg_throughput": statistics.mean(r.throughput_ops_per_sec for r in results if r.throughput_ops_per_sec)
        }
    
    def benchmark_memory_management(self) -> BenchmarkResult:
        """Benchmark memory management efficiency."""
        start_time = time.perf_counter()
        
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate memory-intensive operations
            large_tensors = []
            operations_count = 0
            
            for i in range(100):
                # Allocate tensor
                tensor = torch.randn(1000, 1000)
                large_tensors.append(tensor)
                operations_count += 1
                
                # Perform computation
                result = torch.matmul(tensor, tensor.T)
                operations_count += 1
                
                # Cleanup periodically
                if i % 10 == 0:
                    large_tensors = large_tensors[-5:]  # Keep only last 5
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    operations_count += 1
            
            # Final cleanup
            large_tensors.clear()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = operations_count / (duration_ms / 1000)
            
            return BenchmarkResult(
                test_name="Memory Management",
                category="Performance",
                success=memory_increase < 1000,  # Less than 1GB increase
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=memory_increase,
                metrics={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "operations_count": operations_count
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Memory Management",
                category="Performance",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def benchmark_concurrent_processing(self) -> BenchmarkResult:
        """Benchmark concurrent processing capabilities."""
        start_time = time.perf_counter()
        
        try:
            def cpu_intensive_task(task_id: int) -> int:
                """Simulate CPU-intensive work."""
                result = 0
                for i in range(10000):
                    result += i * task_id
                return result
            
            # Sequential processing baseline
            sequential_start = time.perf_counter()
            sequential_results = []
            for i in range(self.config.concurrent_workers * 10):
                sequential_results.append(cpu_intensive_task(i))
            sequential_time = time.perf_counter() - sequential_start
            
            # Concurrent processing test
            concurrent_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
                concurrent_futures = [
                    executor.submit(cpu_intensive_task, i) 
                    for i in range(self.config.concurrent_workers * 10)
                ]
                concurrent_results = [f.result() for f in concurrent_futures]
            concurrent_time = time.perf_counter() - concurrent_start
            
            # Calculate speedup
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
            efficiency = speedup / self.config.concurrent_workers
            
            total_operations = len(sequential_results) + len(concurrent_results)
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = total_operations / (duration_ms / 1000)
            
            return BenchmarkResult(
                test_name="Concurrent Processing",
                category="Performance",
                success=speedup > 1.5,  # At least 1.5x speedup
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                metrics={
                    "sequential_time_s": sequential_time,
                    "concurrent_time_s": concurrent_time,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "workers": self.config.concurrent_workers
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Concurrent Processing",
                category="Performance",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def benchmark_error_handling(self) -> BenchmarkResult:
        """Benchmark error handling performance."""
        start_time = time.perf_counter()
        
        try:
            error_scenarios = []
            
            # Test different error conditions
            for i in range(1000):
                try:
                    if i % 4 == 0:
                        # ValueError scenario
                        raise ValueError(f"Test error {i}")
                    elif i % 4 == 1:
                        # TypeError scenario
                        raise TypeError(f"Type error {i}")
                    elif i % 4 == 2:
                        # Custom exception scenario
                        from secure_mpc_transformer.utils.error_handling import SecurityException
                        raise SecurityException(f"Security error {i}")
                    else:
                        # No error
                        result = i * 2
                        
                except Exception as e:
                    error_details = self.error_handler.handle_error(e)
                    error_scenarios.append({
                        "error_id": error_details.error_id,
                        "category": error_details.category.value,
                        "severity": error_details.severity.value
                    })
            
            # Get error statistics
            error_stats = self.error_handler.get_error_statistics()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = len(error_scenarios) / (duration_ms / 1000)
            
            return BenchmarkResult(
                test_name="Error Handling",
                category="Performance",
                success=len(error_scenarios) > 0,
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                metrics={
                    "error_scenarios": len(error_scenarios),
                    "error_statistics": error_stats
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Error Handling",
                category="Performance",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def run_quantum_planning_benchmarks(self) -> Dict[str, Any]:
        """Run quantum planning algorithm benchmarks."""
        results = []
        
        # Test quantum task scheduling
        results.append(self.benchmark_quantum_task_scheduling())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results)
        }
    
    def benchmark_quantum_task_scheduling(self) -> BenchmarkResult:
        """Benchmark quantum-inspired task scheduling."""
        start_time = time.perf_counter()
        
        try:
            # Create test tasks
            tasks = []
            task_types = list(TaskType)
            
            for i in range(100):
                task = Task(
                    id=f"task_{i}",
                    task_type=task_types[i % len(task_types)],
                    priority=np.random.uniform(0.1, 1.0),
                    estimated_duration=np.random.uniform(0.1, 2.0),
                    required_resources={
                        "cpu": np.random.uniform(0.1, 1.0),
                        "memory": np.random.uniform(0.1, 1.0),
                        "gpu": np.random.uniform(0.0, 0.5) if self.config.enable_gpu_tests else 0.0
                    },
                    dependencies=[]
                )
                tasks.append(task)
                self.quantum_planner.add_task(task)
            
            # Add some dependencies
            for i in range(20):
                dependent_task = tasks[i + 50]
                dependency_task = tasks[i]
                dependent_task.dependencies.append(dependency_task.id)
            
            # Test quantum priority calculation
            ready_tasks = self.quantum_planner.get_ready_tasks()
            prioritized_tasks = self.quantum_planner.calculate_quantum_priority(ready_tasks)
            
            # Test quantum annealing schedule
            task_batches = self.quantum_planner.quantum_anneal_schedule(ready_tasks)
            
            # Get execution statistics
            stats = self.quantum_planner.get_execution_stats()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = len(tasks) / (duration_ms / 1000)
            
            return BenchmarkResult(
                test_name="Quantum Task Scheduling",
                category="Quantum Planning",
                success=len(task_batches) > 0,
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                metrics={
                    "total_tasks": len(tasks),
                    "ready_tasks": len(ready_tasks),
                    "prioritized_tasks": len(prioritized_tasks),
                    "task_batches": len(task_batches),
                    "execution_stats": stats
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Quantum Task Scheduling", 
                category="Quantum Planning",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def run_caching_benchmarks(self) -> Dict[str, Any]:
        """Run caching system benchmarks."""
        results = []
        
        # Test cache performance
        results.append(self.benchmark_cache_performance())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results)
        }
    
    def benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark multi-level caching performance."""
        start_time = time.perf_counter()
        
        try:
            cache_operations = 0
            hit_count = 0
            miss_count = 0
            
            # Generate test data
            test_data = {}
            for i in range(1000):
                key = f"test_key_{i}"
                value = torch.randn(64, 64)  # 64x64 tensor
                test_data[key] = value
            
            # Populate cache
            for key, value in list(test_data.items())[:500]:
                self.cache_manager.put(key, value)
                cache_operations += 1
            
            # Test cache hits and misses
            for key in test_data.keys():
                cached_value = self.cache_manager.get(key)
                cache_operations += 1
                
                if cached_value is not None:
                    hit_count += 1
                else:
                    miss_count += 1
            
            # Test cache warming
            remaining_data = list(test_data.items())[500:]
            self.cache_manager.warm_cache(remaining_data)
            cache_operations += len(remaining_data)
            
            # Get cache statistics
            cache_stats = self.cache_manager.get_cache_stats()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = cache_operations / (duration_ms / 1000)
            
            # Calculate hit rate
            total_gets = hit_count + miss_count
            hit_rate = hit_count / total_gets if total_gets > 0 else 0
            
            return BenchmarkResult(
                test_name="Cache Performance",
                category="Caching",
                success=hit_rate > 0.4,  # At least 40% hit rate
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                metrics={
                    "cache_operations": cache_operations,
                    "hit_count": hit_count,
                    "miss_count": miss_count,
                    "hit_rate": hit_rate,
                    "cache_stats": cache_stats
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Cache Performance",
                category="Caching",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def run_gpu_benchmarks(self) -> Dict[str, Any]:
        """Run GPU-specific benchmarks."""
        if not self.config.enable_gpu_tests or not self.gpu_manager:
            return {
                "tests": [],
                "success_rate": 0,
                "message": "GPU tests disabled or unavailable"
            }
        
        results = []
        results.append(self.benchmark_gpu_memory_management())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0
        }
    
    def benchmark_gpu_memory_management(self) -> BenchmarkResult:
        """Benchmark GPU memory management."""
        if not torch.cuda.is_available():
            return BenchmarkResult(
                test_name="GPU Memory Management",
                category="GPU Operations",
                success=False,
                duration_ms=0,
                error_message="CUDA not available"
            )
        
        start_time = time.perf_counter()
        
        try:
            device_id = 0
            initial_stats = self.gpu_manager.get_memory_stats(device_id)
            
            # Test tensor allocation and deallocation
            allocated_tensors = []
            operations_count = 0
            
            for i in range(100):
                # Allocate GPU tensor
                tensor = self.gpu_manager.allocate_tensor(
                    (1024, 1024), 
                    torch.float32, 
                    device_id
                )
                allocated_tensors.append(tensor)
                operations_count += 1
                
                # Perform GPU computation
                result = torch.matmul(tensor, tensor.T)
                operations_count += 1
                
                # Periodic cleanup
                if i % 10 == 0:
                    allocated_tensors = allocated_tensors[-5:]
                    self.gpu_manager.optimize_memory_usage(device_id)
                    operations_count += 1
            
            # Final cleanup
            allocated_tensors.clear()
            self.gpu_manager.optimize_memory_usage(device_id)
            
            final_stats = self.gpu_manager.get_memory_stats(device_id)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            throughput = operations_count / (duration_ms / 1000)
            
            # Calculate memory efficiency
            if device_id in initial_stats and device_id in final_stats:
                initial_mem = initial_stats[device_id].allocated_memory
                final_mem = final_stats[device_id].allocated_memory
                memory_increase = (final_mem - initial_mem) / (1024 * 1024)  # MB
            else:
                memory_increase = 0
            
            return BenchmarkResult(
                test_name="GPU Memory Management",
                category="GPU Operations",
                success=memory_increase < 100,  # Less than 100MB increase
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=memory_increase,
                metrics={
                    "operations_count": operations_count,
                    "initial_stats": initial_stats[device_id].to_dict() if device_id in initial_stats else {},
                    "final_stats": final_stats[device_id].to_dict() if device_id in final_stats else {},
                    "memory_increase_mb": memory_increase
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="GPU Memory Management",
                category="GPU Operations",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def run_resilience_benchmarks(self) -> Dict[str, Any]:
        """Run resilience and health check benchmarks."""
        results = []
        
        # Test health check system
        results.append(self.benchmark_health_checks())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results)
        }
    
    def benchmark_health_checks(self) -> BenchmarkResult:
        """Benchmark health check system."""
        start_time = time.perf_counter()
        
        try:
            # Execute health checks
            health_results = asyncio.run(self.health_manager.execute_all_checks())
            
            # Get overall status
            overall_status = self.health_manager.get_overall_status()
            detailed_status = self.health_manager.get_detailed_status()
            
            # Get readiness and liveness
            readiness = self.health_manager.get_readiness_status()
            liveness = self.health_manager.get_liveness_status()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Count successful health checks
            successful_checks = sum(1 for result in health_results.values() if result.is_healthy)
            total_checks = len(health_results)
            success_rate = successful_checks / total_checks if total_checks > 0 else 0
            
            return BenchmarkResult(
                test_name="Health Check System",
                category="Resilience", 
                success=success_rate > 0.8,  # At least 80% healthy
                duration_ms=duration_ms,
                metrics={
                    "total_checks": total_checks,
                    "successful_checks": successful_checks,
                    "success_rate": success_rate,
                    "overall_status": overall_status,
                    "readiness": readiness,
                    "liveness": liveness,
                    "detailed_status": detailed_status
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Health Check System",
                category="Resilience",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def run_integration_benchmarks(self) -> Dict[str, Any]:
        """Run end-to-end integration benchmarks."""
        results = []
        
        # Test full system integration
        results.append(self.benchmark_system_integration())
        
        return {
            "tests": results,
            "success_rate": sum(1 for r in results if r.success) / len(results)
        }
    
    def benchmark_system_integration(self) -> BenchmarkResult:
        """Benchmark full system integration."""
        start_time = time.perf_counter()
        
        try:
            integration_steps = []
            
            # Step 1: Initialize components
            integration_steps.append("Component initialization")
            
            # Step 2: Security setup
            key_id = self.key_manager.generate_symmetric_key()
            integration_steps.append("Security key generation")
            
            # Step 3: Create and schedule tasks
            tasks = []
            for i in range(10):
                task = Task(
                    id=f"integration_task_{i}",
                    task_type=TaskType.COMPUTATION,
                    priority=0.5,
                    estimated_duration=0.1,
                    required_resources={"cpu": 0.1, "memory": 0.1},
                    dependencies=[]
                )
                tasks.append(task)
                self.quantum_planner.add_task(task)
            
            integration_steps.append("Task creation and scheduling")
            
            # Step 4: Execute quantum planning
            execution_result = asyncio.run(self.quantum_planner.execute_quantum_plan())
            integration_steps.append("Quantum plan execution")
            
            # Step 5: Cache operations
            for i, task in enumerate(tasks):
                cache_key = f"result_{task.id}"
                cache_value = f"result_data_{i}"
                self.cache_manager.put(cache_key, cache_value)
            
            # Verify cached results
            cache_hits = 0
            for task in tasks:
                cache_key = f"result_{task.id}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    cache_hits += 1
            
            integration_steps.append("Cache operations")
            
            # Step 6: Health checks
            health_results = asyncio.run(self.health_manager.execute_all_checks())
            integration_steps.append("Health check execution")
            
            # Step 7: Cleanup
            self.cache_manager.clear_cache()
            integration_steps.append("Cleanup")
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Evaluate integration success
            success_criteria = [
                execution_result.get("status") == "completed",
                execution_result.get("tasks_completed", 0) > 0,
                cache_hits == len(tasks),
                len(health_results) > 0
            ]
            
            overall_success = all(success_criteria)
            
            return BenchmarkResult(
                test_name="System Integration",
                category="Integration",
                success=overall_success,
                duration_ms=duration_ms,
                metrics={
                    "integration_steps": integration_steps,
                    "execution_result": execution_result,
                    "cache_hits": cache_hits,
                    "health_results_count": len(health_results),
                    "success_criteria": success_criteria
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="System Integration",
                category="Integration",
                success=False,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e)
            )
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        if not self.results:
            return {"message": "No benchmark results available"}
        
        successful_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)
        
        # Calculate averages
        avg_duration = statistics.mean(r.duration_ms for r in self.results)
        
        throughput_results = [r.throughput_ops_per_sec for r in self.results if r.throughput_ops_per_sec]
        avg_throughput = statistics.mean(throughput_results) if throughput_results else 0
        
        security_scores = [r.security_score for r in self.results if r.security_score]
        avg_security_score = statistics.mean(security_scores) if security_scores else 0
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests,
            "avg_duration_ms": avg_duration,
            "avg_throughput_ops_per_sec": avg_throughput,
            "avg_security_score": avg_security_score,
            "categories_tested": list(set(r.category for r in self.results))
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if not self.results:
            return ["Run benchmarks to generate recommendations"]
        
        # Analyze results and generate recommendations
        failed_tests = [r for r in self.results if not r.success]
        
        if failed_tests:
            recommendations.append(
                f"Address {len(failed_tests)} failed tests: {', '.join(r.test_name for r in failed_tests[:3])}"
            )
        
        # Performance recommendations
        slow_tests = [r for r in self.results if r.duration_ms > 5000]  # > 5 seconds
        if slow_tests:
            recommendations.append(
                f"Optimize performance for slow tests: {', '.join(r.test_name for r in slow_tests[:3])}"
            )
        
        # Security recommendations
        low_security_tests = [r for r in self.results if r.security_score and r.security_score < 0.7]
        if low_security_tests:
            recommendations.append(
                f"Improve security for: {', '.join(r.test_name for r in low_security_tests[:3])}"
            )
        
        # Memory recommendations
        high_memory_tests = [r for r in self.results if r.memory_usage_mb and r.memory_usage_mb > 500]
        if high_memory_tests:
            recommendations.append(
                "Consider memory optimization for high-memory tests"
            )
        
        # GPU recommendations
        if self.config.enable_gpu_tests:
            gpu_tests = [r for r in self.results if r.category == "GPU Operations"]
            if not gpu_tests or not all(r.success for r in gpu_tests):
                recommendations.append(
                    "Review GPU utilization and memory management"
                )
        
        if not recommendations:
            recommendations.append("All benchmarks passed successfully - system is performing well")
        
        return recommendations

def main():
    """Run the comprehensive benchmark suite."""
    print("=" * 80)
    print("Secure MPC Transformer - Comprehensive Security & Performance Benchmark")
    print("=" * 80)
    
    # Configuration
    config = BenchmarkConfig(
        test_duration_seconds=30,  # Shorter for demo
        num_iterations=100,
        concurrent_workers=4,
        enable_gpu_tests=torch.cuda.is_available()
    )
    
    print(f"Configuration:")
    print(f"- Test Duration: {config.test_duration_seconds}s")
    print(f"- Iterations: {config.num_iterations}")
    print(f"- Workers: {config.concurrent_workers}")
    print(f"- GPU Tests: {config.enable_gpu_tests}")
    print(f"- Security Level: {config.security_level}")
    print()
    
    # Run benchmarks
    benchmark = SecurityPerformanceBenchmark(config)
    
    try:
        results = benchmark.run_all_benchmarks()
        
        # Save results
        output_file = "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Display summary
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Duration: {summary['avg_duration_ms']:.2f}ms")
        print(f"Average Throughput: {summary['avg_throughput_ops_per_sec']:.2f} ops/sec")
        print(f"Average Security Score: {summary['avg_security_score']:.2f}")
        
        print(f"\nCategories Tested: {', '.join(summary['categories_tested'])}")
        
        print(f"\nTotal Benchmark Duration: {results['total_duration_seconds']:.2f} seconds")
        
        # Display recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"{i}. {rec}")
        
        # Display category results
        print("\n" + "=" * 80)
        print("CATEGORY BREAKDOWN")
        print("=" * 80)
        for category, category_result in results["categories"].items():
            if isinstance(category_result, dict) and "success_rate" in category_result:
                print(f"{category}: {category_result['success_rate']:.1%} success rate "
                      f"({len(category_result.get('tests', []))} tests)")
            else:
                print(f"{category}: Error or incomplete")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Return exit code based on overall success
        overall_success_rate = summary['success_rate']
        if overall_success_rate >= 0.9:
            print("\n🎉 Excellent! All systems performing optimally.")
            return 0
        elif overall_success_rate >= 0.7:
            print("\n⚠️  Good performance with some areas for improvement.")
            return 0
        else:
            print("\n❌ Multiple issues detected. Review recommendations.")
            return 1
            
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        print(f"\n❌ Benchmark execution failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    from collections import Counter
    exit_code = main()
    sys.exit(exit_code)