#!/usr/bin/env python3
"""
BERT inference benchmark for Secure MPC Transformer.

This script benchmarks BERT model inference performance across different
configurations, protocols, and hardware setups.
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install required packages: pip install torch transformers")
    sys.exit(1)

class BERTBenchmark:
    """BERT inference benchmark runner."""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 batch_size: int = 1,
                 sequence_length: int = 128,
                 use_gpu: bool = True,
                 mpc_protocol: Optional[str] = None):
        """
        Initialize BERT benchmark.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
            sequence_length: Input sequence length
            use_gpu: Whether to use GPU acceleration
            mpc_protocol: MPC protocol to use (None for plaintext)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.mpc_protocol = mpc_protocol
        
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.tokenizer = None
        self.test_inputs = None
        
    def setup(self):
        """Setup model, tokenizer, and test inputs."""
        print(f"Setting up BERT benchmark:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  MPC protocol: {self.mpc_protocol or 'plaintext'}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        if self.mpc_protocol:
            # In a real implementation, this would load the MPC-enabled model
            # For now, we simulate MPC overhead
            self.model = AutoModel.from_pretrained(self.model_name)
            print("  Note: MPC protocol simulation enabled")
        else:
            self.model = AutoModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare test inputs
        self._prepare_test_inputs()
        
        # Warmup
        print("  Running warmup...")
        for _ in range(5):
            self._run_single_inference()
        
        if self.use_gpu:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        print("  Setup complete")
    
    def _prepare_test_inputs(self):
        """Prepare test input tensors."""
        # Generate sample text
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "To be or not to be, that is the question.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "It was the best of times, it was the worst of times."
        ] * (self.batch_size // 5 + 1)
        
        sample_texts = sample_texts[:self.batch_size]
        
        # Tokenize inputs
        encoded = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt"
        )
        
        self.test_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }
    
    def _run_single_inference(self) -> Dict[str, float]:
        """Run a single inference and return metrics."""
        if self.use_gpu:
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Memory before inference
        if self.use_gpu:
            gpu_memory_before = torch.cuda.memory_allocated(self.device)
        
        # Run inference
        with torch.no_grad():
            if self.mpc_protocol:
                # Simulate MPC overhead
                mpc_overhead = self._simulate_mpc_overhead()
                time.sleep(mpc_overhead)
            
            outputs = self.model(**self.test_inputs)
        
        if self.use_gpu:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        
        metrics = {
            'latency_ms': latency_ms,
            'throughput_qps': self.batch_size / (latency_ms / 1000),
        }
        
        # Memory metrics
        if self.use_gpu:
            gpu_memory_after = torch.cuda.memory_allocated(self.device)
            metrics['gpu_memory_mb'] = (gpu_memory_after - gpu_memory_before) / (1024 * 1024)
            metrics['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        
        return metrics
    
    def _simulate_mpc_overhead(self) -> float:
        """Simulate MPC protocol overhead."""
        # Simulate network communication and cryptographic operations
        protocol_overhead = {
            'semi_honest_3pc': 0.01,  # 10ms overhead
            'malicious_3pc': 0.05,   # 50ms overhead
            'aby3': 0.03,            # 30ms overhead
            '4pc_gpu': 0.02          # 20ms overhead
        }
        
        return protocol_overhead.get(self.mmp_protocol, 0.01)
    
    def run_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Run benchmark for specified iterations.
        
        Args:
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results dictionary
        """
        print(f"Running benchmark for {iterations} iterations...")
        
        all_metrics = []
        
        for i in range(iterations):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{iterations}")
            
            metrics = self._run_single_inference()
            all_metrics.append(metrics)
        
        # Calculate statistics
        results = self._calculate_statistics(all_metrics)
        
        # Add configuration info
        results['config'] = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'device': self.device,
            'mpc_protocol': self.mpc_protocol,
            'iterations': iterations
        }
        
        # Add system info
        results['system_info'] = self._get_system_info()
        
        print("Benchmark complete!")
        return results
    
    def _calculate_statistics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate statistics from metrics list."""
        results = {}
        
        # Get all metric names
        metric_names = metrics_list[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            
            results[metric_name] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info

def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description='BERT inference benchmark')
    
    parser.add_argument('--model', default='bert-base-uncased',
                       help='HuggingFace model name')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--sequence-length', type=int, default=128,
                       help='Input sequence length')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--mpc-protocol', 
                       choices=['semi_honest_3pc', 'malicious_3pc', 'aby3', '4pc_gpu'],
                       help='MPC protocol to benchmark')
    parser.add_argument('--output', type=str,
                       help='Output file path (JSON format)')
    parser.add_argument('--format', choices=['json', 'table'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = BERTBenchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        use_gpu=not args.no_gpu,
        mpc_protocol=args.mmp_protocol
    )
    
    # Setup and run
    benchmark.setup()
    results = benchmark.run_benchmark(args.iterations)
    
    # Add metadata
    results['metadata'] = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'benchmark_script': 'benchmark_bert.py',
        'args': vars(args)
    }
    
    # Output results
    if args.format == 'table':
        print_results_table(results)
    else:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))

def print_results_table(results: Dict[str, Any]):
    """Print results in table format."""
    print("\n" + "="*60)
    print("BERT INFERENCE BENCHMARK RESULTS")
    print("="*60)
    
    config = results['config']
    print(f"Model: {config['model_name']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Sequence Length: {config['sequence_length']}")
    print(f"Device: {config['device']}")
    print(f"MPC Protocol: {config['mmp_protocol'] or 'plaintext'}")
    print(f"Iterations: {config['iterations']}")
    
    print("\n" + "-"*60)
    print("PERFORMANCE METRICS")
    print("-"*60)
    
    for metric_name, stats in results.items():
        if metric_name in ['config', 'system_info', 'metadata']:
            continue
        
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Mean:   {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std:    {stats['std']:.2f}")
        print(f"  Min:    {stats['min']:.2f}")
        print(f"  Max:    {stats['max']:.2f}")
        print(f"  P95:    {stats['p95']:.2f}")
        print(f"  P99:    {stats['p99']:.2f}")

if __name__ == '__main__':
    main()