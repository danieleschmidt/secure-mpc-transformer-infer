#!/usr/bin/env python3
"""
Comprehensive test runner for secure MPC transformer.

This script provides various test running configurations for different scenarios.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any


class TestRunner:
    """Test runner with different configurations."""
    
    def __init__(self):
        self.base_cmd = ["python", "-m", "pytest"]
        self.test_dir = Path(__file__).parent
    
    def run_unit_tests(self, verbose: bool = True) -> int:
        """Run unit tests only."""
        cmd = self.base_cmd + [
            "tests/unit/",
            "-m", "not slow and not gpu",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("🧪 Running unit tests...")
        return subprocess.run(cmd).returncode
    
    def run_integration_tests(self, verbose: bool = True) -> int:
        """Run integration tests."""
        cmd = self.base_cmd + [
            "tests/integration/",
            "-m", "integration",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("🔗 Running integration tests...")
        return subprocess.run(cmd).returncode
    
    def run_security_tests(self, verbose: bool = True) -> int:
        """Run security-focused tests."""
        cmd = self.base_cmd + [
            "tests/security/",
            "-m", "security",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("🔒 Running security tests...")
        return subprocess.run(cmd).returncode
    
    def run_gpu_tests(self, verbose: bool = True) -> int:
        """Run GPU-enabled tests."""
        cmd = self.base_cmd + [
            "--gpu",
            "-m", "gpu",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("🖥️  Running GPU tests...")
        return subprocess.run(cmd).returncode
    
    def run_performance_tests(self, verbose: bool = True) -> int:
        """Run performance and benchmark tests."""
        cmd = self.base_cmd + [
            "tests/performance/",
            "--benchmark",
            "-m", "benchmark or performance",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("⚡ Running performance tests...")
        return subprocess.run(cmd).returncode
    
    def run_e2e_tests(self, verbose: bool = True) -> int:
        """Run end-to-end tests."""
        cmd = self.base_cmd + [
            "tests/e2e/",
            "-m", "e2e",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("🚀 Running end-to-end tests...")
        return subprocess.run(cmd).returncode
    
    def run_fast_tests(self, verbose: bool = True) -> int:
        """Run fast tests (excluding slow, gpu, benchmark)."""
        cmd = self.base_cmd + [
            "-m", "not slow and not gpu and not benchmark",
            "--tb=short",
            "--maxfail=3"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("⚡ Running fast tests...")
        return subprocess.run(cmd).returncode
    
    def run_all_tests(self, verbose: bool = True) -> int:
        """Run all tests."""
        cmd = self.base_cmd + [
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("🧪 Running all tests...")
        return subprocess.run(cmd).returncode
    
    def run_coverage_report(self) -> int:
        """Generate coverage report."""
        cmd = self.base_cmd + [
            "--cov=secure_mpc_transformer",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "-m", "not slow and not gpu and not benchmark"
        ]
        
        print("📊 Generating coverage report...")
        result = subprocess.run(cmd).returncode
        
        if result == 0:
            print("📊 Coverage report generated in htmlcov/")
        
        return result
    
    def run_specific_protocol(self, protocol: str, verbose: bool = True) -> int:
        """Run tests for specific MPC protocol."""
        cmd = self.base_cmd + [
            f"--protocol={protocol}",
            "-m", f"protocol or not protocol",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        print(f"🔐 Running tests for protocol: {protocol}")
        return subprocess.run(cmd).returncode
    
    def run_stress_tests(self, verbose: bool = True) -> int:
        """Run stress tests."""
        cmd = self.base_cmd + [
            "tests/performance/",
            "-m", "stress",
            "--tb=short",
            "--timeout=600"  # 10 minute timeout
        ]
        
        if verbose:
            cmd.append("-v")
        
        print("💪 Running stress tests...")
        return subprocess.run(cmd).returncode
    
    def run_ci_tests(self) -> int:
        """Run tests suitable for CI environment."""
        cmd = self.base_cmd + [
            "-m", "not slow and not gpu and not benchmark and not stress",
            "--tb=short",
            "--maxfail=5",
            "--cov=secure_mpc_transformer",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ]
        
        print("🤖 Running CI test suite...")
        return subprocess.run(cmd).returncode
    
    def run_smoke_tests(self) -> int:
        """Run basic smoke tests."""
        cmd = self.base_cmd + [
            "tests/unit/",
            "-k", "test_init or test_basic",
            "--tb=line",
            "--maxfail=1"
        ]
        
        print("💨 Running smoke tests...")
        return subprocess.run(cmd).returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test runner for secure MPC transformer")
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "security", "gpu", "performance", 
            "e2e", "fast", "all", "coverage", "stress", "ci", "smoke"
        ],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--protocol",
        default="semi_honest_3pc",
        help="MPC protocol to test"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (less verbose output)"
    )
    
    args = parser.parse_args()
    runner = TestRunner()
    verbose = not args.quiet
    
    # Test type dispatch
    test_functions = {
        "unit": runner.run_unit_tests,
        "integration": runner.run_integration_tests,
        "security": runner.run_security_tests,
        "gpu": runner.run_gpu_tests,
        "performance": runner.run_performance_tests,
        "e2e": runner.run_e2e_tests,
        "fast": runner.run_fast_tests,
        "all": runner.run_all_tests,
        "coverage": runner.run_coverage_report,
        "stress": runner.run_stress_tests,
        "ci": runner.run_ci_tests,
        "smoke": runner.run_smoke_tests,
    }
    
    if args.test_type == "protocol":
        result = runner.run_specific_protocol(args.protocol, verbose)
    else:
        test_func = test_functions[args.test_type]
        if args.test_type == "coverage":
            result = test_func()  # Coverage doesn't take verbose param
        else:
            result = test_func(verbose)
    
    if result == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    sys.exit(result)


if __name__ == "__main__":
    main()