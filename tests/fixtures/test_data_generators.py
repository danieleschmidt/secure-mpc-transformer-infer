"""
Test data generators for various testing scenarios.
"""

import json
import random
import string
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch


class TextDataGenerator:
    """Generate text data for testing transformer models."""
    
    def __init__(self, vocab_size: int = 30522, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.common_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "through", "during",
            "secure", "computation", "privacy", "transformer", "inference",
            "cryptography", "protocol", "neural", "network", "machine", "learning",
            "artificial", "intelligence", "deep", "model", "training", "data"
        ]
    
    def generate_random_text(self, min_words: int = 5, max_words: int = 50) -> str:
        """Generate random text with realistic word patterns."""
        num_words = random.randint(min_words, max_words)
        words = random.choices(self.common_words, k=num_words)
        return " ".join(words)
    
    def generate_masked_text(self, mask_probability: float = 0.15) -> Tuple[str, List[int]]:
        """Generate text with [MASK] tokens."""
        text = self.generate_random_text()
        words = text.split()
        masked_positions = []
        
        for i, word in enumerate(words):
            if random.random() < mask_probability:
                words[i] = "[MASK]"
                masked_positions.append(i)
        
        return " ".join(words), masked_positions
    
    def generate_input_ids(self, text: str, pad_to_length: Optional[int] = None) -> torch.Tensor:
        """Generate mock input IDs from text."""
        # Simple mock tokenization - each word gets a random ID
        words = text.split()
        input_ids = [random.randint(1, self.vocab_size - 1) for _ in words]
        
        # Add special tokens
        input_ids = [101] + input_ids + [102]  # [CLS] + tokens + [SEP]
        
        if pad_to_length:
            if len(input_ids) > pad_to_length:
                input_ids = input_ids[:pad_to_length]
            else:
                input_ids.extend([0] * (pad_to_length - len(input_ids)))
        
        return torch.tensor(input_ids, dtype=torch.long)
    
    def generate_batch(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of input data."""
        input_ids = torch.randint(1, self.vocab_size, (batch_size, sequence_length))
        attention_mask = torch.ones(batch_size, sequence_length)
        
        # Add some padding tokens randomly
        for i in range(batch_size):
            pad_start = random.randint(sequence_length // 2, sequence_length - 1)
            input_ids[i, pad_start:] = 0
            attention_mask[i, pad_start:] = 0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class CryptographicDataGenerator:
    """Generate cryptographic test data."""
    
    def __init__(self, security_level: int = 128):
        self.security_level = security_level
        self.field_size = 2 ** security_level
    
    def generate_secret_shares(self, secret: int, num_parties: int = 3) -> List[int]:
        """Generate secret shares using mock Shamir's secret sharing."""
        # Simple additive secret sharing for testing
        shares = []
        total = 0
        for i in range(num_parties - 1):
            share = random.randint(0, self.field_size - 1)
            shares.append(share)
            total = (total + share) % self.field_size
        
        # Last share ensures reconstruction
        last_share = (secret - total) % self.field_size
        shares.append(last_share)
        
        return shares
    
    def generate_polynomial_coefficients(self, degree: int) -> List[int]:
        """Generate random polynomial coefficients."""
        return [random.randint(0, self.field_size - 1) for _ in range(degree + 1)]
    
    def generate_encryption_keys(self) -> Dict[str, str]:
        """Generate mock encryption keys."""
        return {
            "public_key": ''.join(random.choices(string.ascii_letters + string.digits, k=64)),
            "private_key": ''.join(random.choices(string.ascii_letters + string.digits, k=64)),
            "key_id": ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        }
    
    def generate_ciphertext(self, plaintext: str, key: str) -> str:
        """Generate mock ciphertext."""
        # Simple XOR-based mock encryption for testing
        result = []
        key_chars = [ord(c) for c in key]
        for i, char in enumerate(plaintext):
            key_char = key_chars[i % len(key_chars)]
            encrypted_char = ord(char) ^ key_char
            result.append(hex(encrypted_char)[2:].zfill(2))
        return ''.join(result)
    
    def generate_homomorphic_operations(self, num_operations: int = 10) -> List[Dict[str, Any]]:
        """Generate homomorphic operation test cases."""
        operations = []
        for _ in range(num_operations):
            op_type = random.choice(["add", "multiply", "subtract"])
            operand1 = random.randint(1, 1000)
            operand2 = random.randint(1, 1000)
            
            if op_type == "add":
                result = operand1 + operand2
            elif op_type == "multiply":
                result = operand1 * operand2
            else:  # subtract
                result = operand1 - operand2
            
            operations.append({
                "operation": op_type,
                "operand1": operand1,
                "operand2": operand2,
                "expected_result": result
            })
        
        return operations


class NetworkDataGenerator:
    """Generate network-related test data."""
    
    def __init__(self):
        self.message_types = ["share", "result", "heartbeat", "error", "ack"]
    
    def generate_network_message(self, sender_id: int, receiver_id: int) -> Dict[str, Any]:
        """Generate a network message."""
        return {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "message_type": random.choice(self.message_types),
            "timestamp": random.randint(1600000000, 1700000000),
            "sequence_number": random.randint(1, 10000),
            "payload": {
                "data": ''.join(random.choices(string.ascii_letters, k=100)),
                "checksum": random.randint(0, 65535)
            }
        }
    
    def generate_network_topology(self, num_parties: int) -> Dict[str, Any]:
        """Generate network topology for testing."""
        parties = []
        for i in range(num_parties):
            party = {
                "id": i,
                "host": f"party{i}.example.com",
                "port": 9000 + i,
                "role": "data_owner" if i == 0 else "compute",
                "public_key": ''.join(random.choices(string.ascii_letters, k=32))
            }
            parties.append(party)
        
        return {
            "parties": parties,
            "protocol": "3pc_malicious",
            "network_timeout": 30,
            "max_retries": 3
        }
    
    def generate_latency_data(self, num_measurements: int = 100) -> List[float]:
        """Generate realistic network latency data."""
        # Generate latency with normal distribution around 50ms
        base_latency = 0.05  # 50ms
        latencies = np.random.normal(base_latency, base_latency * 0.1, num_measurements)
        # Ensure no negative latencies
        latencies = np.maximum(latencies, 0.001)
        return latencies.tolist()


class PerformanceDataGenerator:
    """Generate performance testing data."""
    
    def __init__(self):
        self.operation_types = ["inference", "encryption", "secret_sharing", "reconstruction"]
    
    def generate_benchmark_config(self) -> Dict[str, Any]:
        """Generate benchmark configuration."""
        return {
            "batch_sizes": [1, 4, 8, 16, 32],
            "sequence_lengths": [64, 128, 256, 512],
            "models": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
            "protocols": ["3pc_semi_honest", "3pc_malicious"],
            "iterations": 10,
            "warmup_iterations": 3,
            "timeout_seconds": 300
        }
    
    def generate_performance_metrics(self, operation: str, num_samples: int = 50) -> Dict[str, Any]:
        """Generate performance metrics for an operation."""
        # Base times for different operations (in seconds)
        base_times = {
            "inference": 2.0,
            "encryption": 0.1,
            "secret_sharing": 0.05,
            "reconstruction": 0.03
        }
        
        base_time = base_times.get(operation, 1.0)
        
        # Generate normally distributed times
        times = np.random.normal(base_time, base_time * 0.1, num_samples)
        times = np.maximum(times, 0.001)  # Ensure positive times
        
        return {
            "operation": operation,
            "num_samples": num_samples,
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "percentile_95": float(np.percentile(times, 95)),
            "percentile_99": float(np.percentile(times, 99)),
            "raw_times": times.tolist()
        }
    
    def generate_memory_usage_data(self, operation: str) -> Dict[str, Any]:
        """Generate memory usage data."""
        # Base memory usage in MB
        base_memory = {
            "inference": 2048,
            "encryption": 512,
            "secret_sharing": 256,
            "reconstruction": 128
        }
        
        base = base_memory.get(operation, 1024)
        
        return {
            "operation": operation,
            "initial_memory_mb": base,
            "peak_memory_mb": base + random.randint(100, 500),
            "final_memory_mb": base + random.randint(0, 100),
            "memory_efficiency": random.uniform(0.7, 0.95)
        }


class SecurityTestDataGenerator:
    """Generate security-specific test data."""
    
    def __init__(self):
        self.attack_types = ["timing", "side_channel", "malicious_input", "replay"]
    
    def generate_timing_attack_data(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Generate timing attack test data."""
        # Normal operation times
        normal_times = np.random.normal(0.1, 0.01, num_samples // 2)
        
        # Potentially vulnerable times (slightly different distribution)
        vulnerable_times = np.random.normal(0.12, 0.015, num_samples // 2)
        
        return {
            "normal_operations": normal_times.tolist(),
            "vulnerable_operations": vulnerable_times.tolist(),
            "threshold": 0.11,
            "expected_detection": True
        }
    
    def generate_malicious_inputs(self) -> List[Dict[str, Any]]:
        """Generate malicious input test cases."""
        return [
            {
                "type": "buffer_overflow",
                "input": "A" * 10000,
                "expected_behavior": "reject_input"
            },
            {
                "type": "sql_injection",
                "input": "'; DROP TABLE users; --",
                "expected_behavior": "sanitize_input"
            },
            {
                "type": "null_byte",
                "input": "normal_input\x00malicious_data",
                "expected_behavior": "reject_input"
            },
            {
                "type": "unicode_bypass",
                "input": "script\u2028alert('xss')",
                "expected_behavior": "sanitize_input"
            },
            {
                "type": "large_integer",
                "input": str(2**1000),
                "expected_behavior": "handle_gracefully"
            }
        ]
    
    def generate_side_channel_data(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Generate side-channel attack test data."""
        # Simulate power consumption data
        normal_power = np.random.normal(100, 5, num_samples // 2)
        crypto_power = np.random.normal(110, 8, num_samples // 2)
        
        return {
            "power_consumption": {
                "normal_operations": normal_power.tolist(),
                "crypto_operations": crypto_power.tolist(),
                "sampling_rate_hz": 1000000
            },
            "electromagnetic_emissions": {
                "frequency_spectrum": np.random.normal(0, 1, 1024).tolist(),
                "time_domain": np.random.normal(0, 0.1, 10000).tolist()
            }
        }


class GPUTestDataGenerator:
    """Generate GPU-specific test data."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def generate_gpu_tensors(self, shapes: List[Tuple[int, ...]], dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
        """Generate tensors on GPU for testing."""
        if not torch.cuda.is_available():
            return [torch.randn(shape, dtype=dtype) for shape in shapes]
        
        return [torch.randn(shape, dtype=dtype, device="cuda") for shape in shapes]
    
    def generate_memory_stress_data(self) -> Dict[str, Any]:
        """Generate data for GPU memory stress testing."""
        if not torch.cuda.is_available():
            return {"message": "GPU not available"}
        
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "total_memory_bytes": total_memory,
            "stress_tensor_size": total_memory // 4,  # Use 25% of memory
            "batch_sizes_to_test": [1, 4, 8, 16, 32, 64],
            "expected_oom_threshold": total_memory * 0.9
        }
    
    def generate_cuda_kernel_test_data(self) -> Dict[str, Any]:
        """Generate test data for CUDA kernels."""
        return {
            "matrix_sizes": [(64, 64), (128, 128), (256, 256), (512, 512)],
            "vector_sizes": [1024, 2048, 4096, 8192],
            "data_types": ["float32", "float16", "int32"],
            "grid_sizes": [(1, 1), (8, 8), (16, 16), (32, 32)],
            "block_sizes": [(16, 16), (32, 32)]
        }


# Factory function to get appropriate generator
def get_data_generator(generator_type: str, **kwargs):
    """Factory function to get data generators."""
    generators = {
        "text": TextDataGenerator,
        "crypto": CryptographicDataGenerator,
        "network": NetworkDataGenerator,
        "performance": PerformanceDataGenerator,
        "security": SecurityTestDataGenerator,
        "gpu": GPUTestDataGenerator
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    return generators[generator_type](**kwargs)


# Utility function to save test data
def save_test_data(data: Dict[str, Any], filepath: str) -> None:
    """Save test data to file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# Utility function to load test data
def load_test_data(filepath: str) -> Dict[str, Any]:
    """Load test data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)
