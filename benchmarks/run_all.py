#!/usr/bin/env python3
"""
Comprehensive benchmark suite runner for Secure MPC Transformer.

This script runs all available benchmarks and generates consolidated results.
"""

import argparse
import json
import time
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    """Comprehensive benchmark suite runner."""
    
    def __init__(self, 
                 output_dir: str = "results",
                 quick_mode: bool = False,
                 gpu: bool = True,
                 parallel: bool = False):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to store benchmark results
            quick_mode: Run reduced iterations for faster execution
            gpu: Enable GPU benchmarks
            parallel: Run benchmarks in parallel where possible
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quick_mode = quick_mode
        self.gpu = gpu
        self.parallel = parallel
        
        # Benchmark configurations
        self.iterations = 20 if quick_mode else 100
        self.models = ["bert-base-uncased"] if quick_mode else [
            "bert-base-uncased",
            "distilbert-base-uncased"
        ]
        self.protocols = ["semi_honest_3pc"] if quick_mode else [
            "semi_honest_3pc",
            "malicious_3pc",
            "aby3"
        ]
        
        self.results = {}
        
    def run_benchmark_script(self, 
                           script: str, 
                           args: List[str], 
                           benchmark_id: str) -> Dict[str, Any]:
        """
        Run a single benchmark script.
        
        Args:
            script: Script name to run
            args: Additional arguments for the script
            benchmark_id: Unique identifier for this benchmark
            
        Returns:
            Benchmark results dictionary
        """
        cmd = [sys.executable, f"benchmarks/{script}"] + args
        output_file = self.output_dir / f"{benchmark_id}.json"
        
        logger.info(f"Running benchmark: {benchmark_id}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            # Add output file to command
            cmd.extend(["--output", str(output_file)])
            
            # Run benchmark
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                logger.error(f"Benchmark {benchmark_id} failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "benchmark_id": benchmark_id,
                    "status": "failed",
                    "error": result.stderr,
                    "returncode": result.returncode
                }
            
            # Read results
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                results["benchmark_id"] = benchmark_id
                results["status"] = "success"
                return results
            else:
                logger.error(f"Output file not found: {output_file}")
                return {
                    "benchmark_id": benchmark_id,
                    "status": "failed",
                    "error": "Output file not found"
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark {benchmark_id} timed out")
            return {
                "benchmark_id": benchmark_id,
                "status": "timeout",
                "error": "Benchmark execution timed out"
            }
        except Exception as e:
            logger.error(f"Benchmark {benchmark_id} crashed: {e}")
            return {
                "benchmark_id": benchmark_id,
                "status": "crashed",
                "error": str(e)
            }
    
    def run_bert_benchmarks(self) -> List[Dict[str, Any]]:
        """Run BERT model benchmarks."""
        logger.info("Running BERT benchmarks...")
        
        benchmarks = []
        
        # Different batch sizes
        batch_sizes = [1, 8] if self.quick_mode else [1, 4, 8, 16, 32]
        
        # Different sequence lengths
        seq_lengths = [128, 256] if self.quick_mode else [128, 256, 512]
        
        for model in self.models:
            for batch_size in batch_sizes:
                for seq_length in seq_lengths:
                    # CPU benchmark
                    benchmark_id = f"bert_{model.replace('-', '_')}_cpu_b{batch_size}_s{seq_length}"
                    args = [
                        "--model", model,
                        "--batch-size", str(batch_size),
                        "--sequence-length", str(seq_length),
                        "--iterations", str(self.iterations),
                        "--no-gpu"
                    ]
                    benchmarks.append(("benchmark_bert.py", args, benchmark_id))
                    
                    # GPU benchmark (if enabled)
                    if self.gpu:
                        benchmark_id = f"bert_{model.replace('-', '_')}_gpu_b{batch_size}_s{seq_length}"
                        args = [
                            "--model", model,
                            "--batch-size", str(batch_size),
                            "--sequence-length", str(seq_length),
                            "--iterations", str(self.iterations)
                        ]
                        benchmarks.append(("benchmark_bert.py", args, benchmark_id))
        
        return self._run_benchmarks(benchmarks)
    
    def run_protocol_benchmarks(self) -> List[Dict[str, Any]]:
        """Run MPC protocol benchmarks."""
        logger.info("Running MPC protocol benchmarks...")
        
        benchmarks = []
        
        for protocol in self.protocols:
            for model in self.models:
                # CPU protocol benchmark
                benchmark_id = f"protocol_{protocol}_{model.replace('-', '_')}_cpu"
                args = [
                    "--model", model,
                    "--mpc-protocol", protocol,
                    "--iterations", str(self.iterations),
                    "--no-gpu"
                ]
                benchmarks.append(("benchmark_bert.py", args, benchmark_id))
                
                # GPU protocol benchmark (if enabled)
                if self.gpu:
                    benchmark_id = f"protocol_{protocol}_{model.replace('-', '_')}_gpu"
                    args = [
                        "--model", model,
                        "--mpc-protocol", protocol,
                        "--iterations", str(self.iterations)
                    ]
                    benchmarks.append(("benchmark_bert.py", args, benchmark_id))
        
        return self._run_benchmarks(benchmarks)
    
    def run_scalability_benchmarks(self) -> List[Dict[str, Any]]:
        """Run scalability benchmarks."""
        if self.quick_mode:
            logger.info("Skipping scalability benchmarks in quick mode")
            return []
        
        logger.info("Running scalability benchmarks...")
        
        benchmarks = []
        
        # Test different batch sizes for scalability
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            benchmark_id = f"scalability_bert_batch_{batch_size}"
            args = [
                "--model", "bert-base-uncased",
                "--batch-size", str(batch_size),
                "--iterations", str(max(10, self.iterations // batch_size))
            ]
            
            if not self.gpu:
                args.append("--no-gpu")
            
            benchmarks.append(("benchmark_bert.py", args, benchmark_id))
        
        return self._run_benchmarks(benchmarks)
    
    def _run_benchmarks(self, benchmarks: List[tuple]) -> List[Dict[str, Any]]:
        """Run a list of benchmarks, optionally in parallel."""
        results = []
        
        if self.parallel and len(benchmarks) > 1:
            # Run benchmarks in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_benchmark = {
                    executor.submit(self.run_benchmark_script, script, args, benchmark_id): benchmark_id
                    for script, args, benchmark_id in benchmarks
                }
                
                for future in as_completed(future_to_benchmark):
                    benchmark_id = future_to_benchmark[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed benchmark: {benchmark_id}")
                    except Exception as e:
                        logger.error(f"Benchmark {benchmark_id} failed: {e}")
                        results.append({
                            "benchmark_id": benchmark_id,
                            "status": "failed",
                            "error": str(e)
                        })
        else:
            # Run benchmarks sequentially
            for script, args, benchmark_id in benchmarks:
                result = self.run_benchmark_script(script, args, benchmark_id)
                results.append(result)
        
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks and return consolidated results."""
        logger.info("Starting comprehensive benchmark suite...")
        
        suite_start_time = time.time()
        
        all_results = {}
        
        # Run different benchmark categories
        try:
            bert_results = self.run_bert_benchmarks()
            all_results["bert_benchmarks"] = bert_results
        except Exception as e:
            logger.error(f"BERT benchmarks failed: {e}")
            all_results["bert_benchmarks"] = {"error": str(e)}
        
        try:
            protocol_results = self.run_protocol_benchmarks()
            all_results["protocol_benchmarks"] = protocol_results
        except Exception as e:
            logger.error(f"Protocol benchmarks failed: {e}")
            all_results["protocol_benchmarks"] = {"error": str(e)}
        
        try:
            scalability_results = self.run_scalability_benchmarks()
            all_results["scalability_benchmarks"] = scalability_results
        except Exception as e:
            logger.error(f"Scalability benchmarks failed: {e}")
            all_results["scalability_benchmarks"] = {"error": str(e)}
        
        suite_end_time = time.time()
        
        # Add suite metadata
        all_results["suite_metadata"] = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "total_duration_seconds": suite_end_time - suite_start_time,
            "quick_mode": self.quick_mode,
            "gpu_enabled": self.gpu,
            "parallel_execution": self.parallel,
            "iterations_per_benchmark": self.iterations,
            "output_directory": str(self.output_dir)
        }
        
        # Generate summary statistics
        all_results["summary"] = self._generate_summary(all_results)
        
        logger.info(f"Benchmark suite completed in {suite_end_time - suite_start_time:.2f} seconds")
        
        return all_results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for all benchmarks."""
        summary = {
            "total_benchmarks": 0,
            "successful_benchmarks": 0,
            "failed_benchmarks": 0,
            "categories": {}
        }
        
        for category, category_results in results.items():
            if category in ["suite_metadata", "summary"]:
                continue
            
            if isinstance(category_results, list):
                category_summary = {
                    "total": len(category_results),
                    "successful": sum(1 for r in category_results if r.get("status") == "success"),
                    "failed": sum(1 for r in category_results if r.get("status") != "success")
                }
                
                summary["categories"][category] = category_summary
                summary["total_benchmarks"] += category_summary["total"]
                summary["successful_benchmarks"] += category_summary["successful"]
                summary["failed_benchmarks"] += category_summary["failed"]
        
        summary["success_rate"] = (
            summary["successful_benchmarks"] / summary["total_benchmarks"] 
            if summary["total_benchmarks"] > 0 else 0
        )
        
        return summary

def main():
    """Main benchmark suite runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive benchmark suite')
    
    parser.add_argument('--output-dir', default='results',
                       help='Directory to store benchmark results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmarks with reduced iterations')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU benchmarks')
    parser.add_argument('--parallel', action='store_true',
                       help='Run benchmarks in parallel')
    parser.add_argument('--output', type=str,
                       help='Output file for consolidated results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        output_dir=args.output_dir,
        quick_mode=args.quick,
        gpu=not args.no_gpu,
        parallel=args.parallel
    )
    
    # Run all benchmarks
    results = suite.run_all()
    
    # Save consolidated results
    output_file = args.output or str(suite.output_dir / "consolidated_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Consolidated results saved to {output_file}")
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*60)
    print("BENCHMARK SUITE SUMMARY")
    print("="*60)
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Successful: {summary['successful_benchmarks']}")
    print(f"Failed: {summary['failed_benchmarks']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    for category, stats in summary["categories"].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Total: {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")

if __name__ == '__main__':
    main()