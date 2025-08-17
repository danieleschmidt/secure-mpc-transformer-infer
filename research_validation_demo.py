#!/usr/bin/env python3
"""
Research Validation Demo - Comprehensive Benchmark Study

This script demonstrates the novel quantum-enhanced MPC algorithms with rigorous
academic-quality validation and statistical analysis.

Usage:
    python research_validation_demo.py --experiments all --output results/
    python research_validation_demo.py --experiments performance --parallel 8
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from secure_mpc_transformer.research.advanced_quantum_mpc import (
    VariationalQuantumEigenvalueSolver,
    AdaptiveQuantumMPCOrchestrator,
    QuantumMPCConfig,
    QuantumOptimizationMethod
)

from secure_mpc_transformer.research.comparative_benchmark_framework import (
    ComparativeBenchmarkFramework,
    ExperimentConfig,
    BenchmarkCategory,
    MetricType,
    QuantumMPCBenchmarkExecutor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ResearchValidationDemo:
    """
    Comprehensive research validation demonstration.
    
    Conducts rigorous comparative studies of quantum-enhanced MPC algorithms
    with statistical analysis and publication-ready results.
    """
    
    def __init__(self, output_dir: str = "research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark framework
        self.benchmark_framework = ComparativeBenchmarkFramework()
        
        # Define experiment configurations
        self.experiments = self._define_experiments()
        
    def _define_experiments(self) -> dict:
        """Define comprehensive experiment configurations."""
        
        # Common algorithms to test
        algorithms = [
            "classical_baseline",
            "quantum_vqe", 
            "adaptive_quantum",
            "hybrid_quantum_classical",
            "post_quantum_secure"
        ]
        
        # Common datasets
        datasets = [
            "synthetic_small",
            "synthetic_medium",
            "synthetic_large",
            "transformer_inference"
        ]
        
        # Common metrics
        metrics = [
            MetricType.LATENCY,
            MetricType.THROUGHPUT,
            MetricType.MEMORY_USAGE,
            MetricType.ACCURACY,
            MetricType.SECURITY_SCORE,
            MetricType.QUANTUM_ADVANTAGE
        ]
        
        experiments = {
            "performance_comparison": ExperimentConfig(
                experiment_name="quantum_mpc_performance_comparison",
                category=BenchmarkCategory.PERFORMANCE_COMPARISON,
                algorithms=algorithms,
                datasets=datasets,
                parameter_ranges={
                    "quantum_depth": [4, 6, 8],
                    "entanglement_layers": [2, 3, 4],
                    "variational_steps": [50, 100, 200],
                    "security_level": [128, 256]
                },
                metrics=metrics,
                repetitions=15,  # Increased for statistical power
                significance_level=0.01,  # More stringent
                max_runtime_seconds=300
            ),
            
            "scalability_analysis": ExperimentConfig(
                experiment_name="quantum_mpc_scalability_study",
                category=BenchmarkCategory.SCALABILITY_ANALYSIS,
                algorithms=algorithms,
                datasets=["synthetic_small", "synthetic_medium", "synthetic_large"],
                parameter_ranges={
                    "quantum_depth": [6],
                    "entanglement_layers": [3],
                    "variational_steps": [100],
                    "security_level": [128],
                    "party_count": [3, 5, 8, 10]
                },
                metrics=[MetricType.LATENCY, MetricType.THROUGHPUT, MetricType.MEMORY_USAGE],
                repetitions=12,
                significance_level=0.05,
                max_runtime_seconds=600
            ),
            
            "security_validation": ExperimentConfig(
                experiment_name="quantum_mpc_security_analysis",
                category=BenchmarkCategory.SECURITY_VALIDATION,
                algorithms=algorithms,
                datasets=["synthetic_medium", "transformer_inference"],
                parameter_ranges={
                    "security_level": [128, 192, 256],
                    "post_quantum": [True, False],
                    "malicious_secure": [True, False],
                    "quantum_depth": [6, 8]
                },
                metrics=[MetricType.SECURITY_SCORE, MetricType.LATENCY, MetricType.ACCURACY],
                repetitions=10,
                significance_level=0.05,
                max_runtime_seconds=400
            ),
            
            "algorithm_convergence": ExperimentConfig(
                experiment_name="quantum_optimization_convergence",
                category=BenchmarkCategory.ALGORITHM_CONVERGENCE,
                algorithms=["quantum_vqe", "adaptive_quantum", "hybrid_quantum_classical"],
                datasets=["synthetic_medium"],
                parameter_ranges={
                    "variational_steps": [25, 50, 100, 200, 400],
                    "learning_rate": [0.001, 0.01, 0.1],
                    "quantum_depth": [4, 6, 8],
                    "convergence_threshold": [1e-4, 1e-5, 1e-6]
                },
                metrics=[MetricType.QUANTUM_ADVANTAGE, MetricType.LATENCY, MetricType.ACCURACY],
                repetitions=8,
                significance_level=0.05,
                max_runtime_seconds=200
            )
        }
        
        return experiments
    
    async def run_performance_comparison(self) -> None:
        """Run comprehensive performance comparison study."""
        
        logger.info("Starting Performance Comparison Study")
        
        experiment = self.experiments["performance_comparison"]
        
        reports = await self.benchmark_framework.run_benchmark_suite(
            [experiment],
            parallel_jobs=6,
            save_results=True,
            results_dir=str(self.output_dir / "performance_comparison")
        )
        
        # Analyze results
        report = reports[0]
        
        logger.info(f"Performance Comparison Results:")
        logger.info(f"  Total runs: {report.total_experiments}")
        logger.info(f"  Successful runs: {report.successful_experiments}")
        logger.info(f"  Success rate: {report.successful_experiments/report.total_experiments:.1%}")
        logger.info(f"  Execution time: {report.execution_time:.1f} seconds")
        
        # Print key findings
        logger.info("Key Performance Findings:")
        for rec in report.recommendations[:5]:
            logger.info(f"  • {rec}")
        
        # Print statistical significance results
        significant_comparisons = [comp for comp in report.statistical_comparisons if comp.significant]
        logger.info(f"Found {len(significant_comparisons)} statistically significant differences")
        
        for comp in significant_comparisons[:10]:  # Top 10
            logger.info(f"  {comp.algorithm_a} vs {comp.algorithm_b} on {comp.metric}: "
                       f"p={comp.p_value:.4f}, effect_size={comp.effect_size:.3f}")
    
    async def run_scalability_analysis(self) -> None:
        """Run scalability analysis study."""
        
        logger.info("Starting Scalability Analysis Study")
        
        experiment = self.experiments["scalability_analysis"]
        
        reports = await self.benchmark_framework.run_benchmark_suite(
            [experiment],
            parallel_jobs=4,
            save_results=True,
            results_dir=str(self.output_dir / "scalability_analysis")
        )
        
        report = reports[0]
        
        logger.info(f"Scalability Analysis Results:")
        logger.info(f"  Total runs: {report.total_experiments}")
        logger.info(f"  Successful runs: {report.successful_experiments}")
        logger.info(f"  Execution time: {report.execution_time:.1f} seconds")
        
        # Analyze scalability trends
        self._analyze_scalability_trends(report)
    
    def _analyze_scalability_trends(self, report) -> None:
        """Analyze scalability trends from benchmark results."""
        
        # Group results by party count
        party_performance = {}
        
        for result in report.results:
            if result.success and "party_count" in result.parameters:
                party_count = result.parameters["party_count"]
                algorithm = result.algorithm
                
                if party_count not in party_performance:
                    party_performance[party_count] = {}
                if algorithm not in party_performance[party_count]:
                    party_performance[party_count][algorithm] = {"latency": [], "throughput": []}
                
                if "latency_seconds" in result.metrics:
                    party_performance[party_count][algorithm]["latency"].append(
                        result.metrics["latency_seconds"]
                    )
                if "throughput_ops_per_sec" in result.metrics:
                    party_performance[party_count][algorithm]["throughput"].append(
                        result.metrics["throughput_ops_per_sec"]
                    )
        
        # Print scalability trends
        logger.info("Scalability Trends:")
        for party_count in sorted(party_performance.keys()):
            logger.info(f"  Party Count: {party_count}")
            for algorithm in party_performance[party_count]:
                latencies = party_performance[party_count][algorithm]["latency"]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    logger.info(f"    {algorithm}: {avg_latency:.3f}s average latency")
    
    async def run_security_validation(self) -> None:
        """Run security validation study."""
        
        logger.info("Starting Security Validation Study")
        
        experiment = self.experiments["security_validation"]
        
        reports = await self.benchmark_framework.run_benchmark_suite(
            [experiment],
            parallel_jobs=4,
            save_results=True,
            results_dir=str(self.output_dir / "security_validation")
        )
        
        report = reports[0]
        
        logger.info(f"Security Validation Results:")
        logger.info(f"  Total runs: {report.total_experiments}")
        logger.info(f"  Successful runs: {report.successful_experiments}")
        
        # Analyze security vs performance trade-offs
        self._analyze_security_tradeoffs(report)
    
    def _analyze_security_tradeoffs(self, report) -> None:
        """Analyze security vs performance trade-offs."""
        
        security_performance = {}
        
        for result in report.results:
            if result.success:
                security_level = result.parameters.get("security_level", 128)
                algorithm = result.algorithm
                
                if algorithm not in security_performance:
                    security_performance[algorithm] = {}
                if security_level not in security_performance[algorithm]:
                    security_performance[algorithm][security_level] = {
                        "security_scores": [],
                        "latencies": []
                    }
                
                if "security_score" in result.metrics:
                    security_performance[algorithm][security_level]["security_scores"].append(
                        result.metrics["security_score"]
                    )
                if "latency_seconds" in result.metrics:
                    security_performance[algorithm][security_level]["latencies"].append(
                        result.metrics["latency_seconds"]
                    )
        
        logger.info("Security vs Performance Trade-offs:")
        for algorithm in security_performance:
            logger.info(f"  {algorithm}:")
            for sec_level in sorted(security_performance[algorithm].keys()):
                scores = security_performance[algorithm][sec_level]["security_scores"]
                latencies = security_performance[algorithm][sec_level]["latencies"]
                
                if scores and latencies:
                    avg_security = sum(scores) / len(scores)
                    avg_latency = sum(latencies) / len(latencies)
                    logger.info(f"    Security {sec_level}: {avg_security:.3f} score, {avg_latency:.3f}s latency")
    
    async def run_convergence_analysis(self) -> None:
        """Run algorithm convergence analysis."""
        
        logger.info("Starting Algorithm Convergence Analysis")
        
        experiment = self.experiments["algorithm_convergence"]
        
        reports = await self.benchmark_framework.run_benchmark_suite(
            [experiment],
            parallel_jobs=4,
            save_results=True,
            results_dir=str(self.output_dir / "convergence_analysis")
        )
        
        report = reports[0]
        
        logger.info(f"Convergence Analysis Results:")
        logger.info(f"  Total runs: {report.total_experiments}")
        logger.info(f"  Successful runs: {report.successful_experiments}")
        
        # Analyze convergence patterns
        self._analyze_convergence_patterns(report)
    
    def _analyze_convergence_patterns(self, report) -> None:
        """Analyze algorithm convergence patterns."""
        
        convergence_data = {}
        
        for result in report.results:
            if result.success:
                algorithm = result.algorithm
                variational_steps = result.parameters.get("variational_steps", 100)
                
                if algorithm not in convergence_data:
                    convergence_data[algorithm] = {}
                if variational_steps not in convergence_data[algorithm]:
                    convergence_data[algorithm][variational_steps] = {
                        "quantum_advantages": [],
                        "latencies": []
                    }
                
                if "quantum_speedup_factor" in result.metrics:
                    convergence_data[algorithm][variational_steps]["quantum_advantages"].append(
                        result.metrics["quantum_speedup_factor"]
                    )
                if "latency_seconds" in result.metrics:
                    convergence_data[algorithm][variational_steps]["latencies"].append(
                        result.metrics["latency_seconds"]
                    )
        
        logger.info("Convergence Patterns:")
        for algorithm in convergence_data:
            logger.info(f"  {algorithm}:")
            for steps in sorted(convergence_data[algorithm].keys()):
                advantages = convergence_data[algorithm][steps]["quantum_advantages"]
                latencies = convergence_data[algorithm][steps]["latencies"]
                
                if advantages:
                    avg_advantage = sum(advantages) / len(advantages)
                    avg_latency = sum(latencies) / len(latencies) if latencies else 0
                    logger.info(f"    {steps} steps: {avg_advantage:.3f}x advantage, {avg_latency:.3f}s")
    
    async def run_all_experiments(self) -> None:
        """Run all research validation experiments."""
        
        logger.info("Starting Comprehensive Research Validation Suite")
        start_time = time.time()
        
        # Run all experiments
        await self.run_performance_comparison()
        await self.run_scalability_analysis() 
        await self.run_security_validation()
        await self.run_convergence_analysis()
        
        total_time = time.time() - start_time
        
        logger.info(f"Research Validation Suite Completed")
        logger.info(f"Total execution time: {total_time:.1f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")
        
        # Generate comprehensive summary
        self._generate_research_summary()
    
    def _generate_research_summary(self) -> None:
        """Generate comprehensive research summary."""
        
        summary_file = self.output_dir / "research_validation_summary.md"
        
        with open(summary_file, "w") as f:
            f.write("# Quantum-Enhanced MPC Research Validation Summary\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This comprehensive validation study evaluates novel quantum-enhanced ")
            f.write("MPC algorithms against classical baselines using rigorous statistical analysis.\n\n")
            
            f.write("## Experiments Conducted\n\n")
            f.write("1. **Performance Comparison**: Comprehensive algorithm comparison across metrics\n")
            f.write("2. **Scalability Analysis**: Multi-party scalability evaluation\n")
            f.write("3. **Security Validation**: Security vs performance trade-off analysis\n")
            f.write("4. **Convergence Analysis**: Quantum optimization convergence study\n\n")
            
            f.write("## Key Contributions\n\n")
            f.write("- Novel Variational Quantum Eigenvalue Solver for MPC optimization\n")
            f.write("- Adaptive Quantum MPC Orchestrator with machine learning\n")
            f.write("- Comprehensive statistical validation framework\n")
            f.write("- Publication-ready experimental methodology\n\n")
            
            f.write("## Methodology\n\n")
            f.write("- Statistical significance testing with multiple comparison corrections\n")
            f.write("- Effect size analysis using Cohen's d\n")
            f.write("- Confidence interval computation\n")
            f.write("- Reproducible experimental design\n\n")
            
            f.write("## Results\n\n")
            f.write("Detailed results are available in the following directories:\n\n")
            f.write("- `performance_comparison/`: Algorithm performance benchmarks\n")
            f.write("- `scalability_analysis/`: Multi-party scalability results\n")
            f.write("- `security_validation/`: Security analysis results\n")
            f.write("- `convergence_analysis/`: Optimization convergence data\n\n")
            
            f.write("## Academic Quality\n\n")
            f.write("This validation follows academic standards for:\n")
            f.write("- Reproducible research methodology\n")
            f.write("- Statistical rigor and significance testing\n")
            f.write("- Comprehensive experimental design\n")
            f.write("- Publication-ready documentation\n\n")
        
        logger.info(f"Research summary saved to: {summary_file}")


async def main():
    """Main entry point for research validation demo."""
    
    parser = argparse.ArgumentParser(description="Quantum-Enhanced MPC Research Validation")
    parser.add_argument(
        "--experiments",
        choices=["all", "performance", "scalability", "security", "convergence"],
        default="all",
        help="Which experiments to run"
    )
    parser.add_argument(
        "--output",
        default="research_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize demo
    demo = ResearchValidationDemo(args.output)
    
    try:
        # Run selected experiments
        if args.experiments == "all":
            await demo.run_all_experiments()
        elif args.experiments == "performance":
            await demo.run_performance_comparison()
        elif args.experiments == "scalability":
            await demo.run_scalability_analysis()
        elif args.experiments == "security":
            await demo.run_security_validation()
        elif args.experiments == "convergence":
            await demo.run_convergence_analysis()
        
        logger.info("Research validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Research validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())