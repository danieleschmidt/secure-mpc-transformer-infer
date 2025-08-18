#!/usr/bin/env python3
"""
Comprehensive test runner for Secure MPC Transformer.
Provides different test suites and reporting options.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


class TestRunner:
    """Comprehensive test runner with multiple test suites."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}

    def run_command(self, cmd: List[str], timeout: Optional[int] = None) -> Dict:
        """Run a command and capture results."""
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            return {
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out",
                "execution_time": timeout or 0,
                "success": False
            }

    def run_unit_tests(self) -> Dict:
        """Run unit tests."""
        print("🧪 Running unit tests...")
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/",
            "-v", "--tb=short",
            "--cov=secure_mpc_transformer",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit",
            "--cov-report=xml:coverage-unit.xml",
            "-m", "not slow and not gpu and not integration"
        ]
        return self.run_command(cmd, timeout=300)

    def run_integration_tests(self) -> Dict:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-v", "--tb=short",
            "--maxfail=3",
            "-m", "integration"
        ]
        return self.run_command(cmd, timeout=600)

    def run_security_tests(self) -> Dict:
        """Run security tests."""
        print("🔒 Running security tests...")
        cmd = [
            "python", "-m", "pytest",
            "tests/security/",
            "-v", "--tb=short",
            "-m", "security"
        ]
        return self.run_command(cmd, timeout=300)

    def run_performance_tests(self) -> Dict:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        cmd = [
            "python", "-m", "pytest",
            "tests/performance/",
            "-v", "--tb=short",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-json=benchmark-results.json"
        ]
        return self.run_command(cmd, timeout=900)

    def run_gpu_tests(self) -> Dict:
        """Run GPU tests."""
        print("🎮 Running GPU tests...")
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "-v", "--tb=short",
            "--gpu",
            "-m", "gpu"
        ]
        return self.run_command(cmd, timeout=600)

    def run_e2e_tests(self) -> Dict:
        """Run end-to-end tests."""
        print("🎯 Running end-to-end tests...")
        cmd = [
            "python", "-m", "pytest",
            "tests/e2e/",
            "-v", "--tb=short",
            "--maxfail=1",
            "-m", "e2e"
        ]
        return self.run_command(cmd, timeout=1200)

    def run_code_quality_checks(self) -> Dict:
        """Run code quality checks."""
        print("✨ Running code quality checks...")
        
        checks = {}
        
        # Black formatting check
        checks["black"] = self.run_command([
            "python", "-m", "black", "--check", "--diff", "src/", "tests/"
        ])
        
        # Import sorting check
        checks["isort"] = self.run_command([
            "python", "-m", "isort", "--check-only", "--diff", "src/", "tests/"
        ])
        
        # Linting
        checks["pylint"] = self.run_command([
            "python", "-m", "pylint", "src/secure_mpc_transformer"
        ])
        
        # Type checking
        checks["mypy"] = self.run_command([
            "python", "-m", "mypy", "src/"
        ])
        
        # Security scanning
        checks["bandit"] = self.run_command([
            "python", "-m", "bandit", "-r", "src/", "-ll"
        ])
        
        # Overall success
        all_passed = all(check["success"] for check in checks.values())
        
        return {
            "command": "code_quality_checks",
            "returncode": 0 if all_passed else 1,
            "stdout": json.dumps(checks, indent=2),
            "stderr": "",
            "execution_time": sum(check["execution_time"] for check in checks.values()),
            "success": all_passed,
            "details": checks
        }

    def run_dependency_checks(self) -> Dict:
        """Run dependency security checks."""
        print("📦 Running dependency checks...")
        
        checks = {}
        
        # Safety check for known vulnerabilities
        checks["safety"] = self.run_command([
            "python", "-m", "safety", "check", "--json"
        ])
        
        # License check
        checks["pip_licenses"] = self.run_command([
            "python", "-m", "pip_licenses", "--format=json"
        ])
        
        all_passed = all(check["success"] for check in checks.values())
        
        return {
            "command": "dependency_checks",
            "returncode": 0 if all_passed else 1,
            "stdout": json.dumps(checks, indent=2),
            "stderr": "",
            "execution_time": sum(check["execution_time"] for check in checks.values()),
            "success": all_passed,
            "details": checks
        }

    def generate_report(self, output_file: Optional[str] = None) -> None:
        """Generate a comprehensive test report."""
        print("📊 Generating test report...")
        
        total_time = sum(result["execution_time"] for result in self.results.values())
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        total_tests = len(self.results)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_test_suites": total_tests,
                "passed_test_suites": passed_tests,
                "failed_test_suites": total_tests - passed_tests,
                "total_execution_time": total_time,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            "results": self.results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Test report saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total test suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        
        if passed_tests < total_tests:
            print("\n❌ Some tests failed. Check the detailed output above.")
            return False
        else:
            print("\n✅ All test suites passed!")
            return True

    def run_test_suite(self, suite: str) -> bool:
        """Run a specific test suite."""
        suite_methods = {
            "unit": self.run_unit_tests,
            "integration": self.run_integration_tests,
            "security": self.run_security_tests,
            "performance": self.run_performance_tests,
            "gpu": self.run_gpu_tests,
            "e2e": self.run_e2e_tests,
            "quality": self.run_code_quality_checks,
            "dependencies": self.run_dependency_checks
        }
        
        if suite not in suite_methods:
            print(f"❌ Unknown test suite: {suite}")
            return False
        
        result = suite_methods[suite]()
        self.results[suite] = result
        
        if result["success"]:
            print(f"✅ {suite} tests passed")
        else:
            print(f"❌ {suite} tests failed")
            if self.verbose:
                print(f"STDOUT: {result['stdout']}")
                print(f"STDERR: {result['stderr']}")
        
        return result["success"]

    def run_all_tests(self, skip_slow: bool = False) -> bool:
        """Run all test suites."""
        print("🚀 Running comprehensive test suite...")
        
        # Define test order (fast to slow)
        test_suites = ["quality", "unit", "security", "integration"]
        
        if not skip_slow:
            test_suites.extend(["performance", "e2e"])
        
        # Add GPU tests if available
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            test_suites.append("gpu")
        
        test_suites.append("dependencies")
        
        success = True
        for suite in test_suites:
            suite_success = self.run_test_suite(suite)
            success = success and suite_success
            
            # Stop on first failure in critical tests
            if not suite_success and suite in ["quality", "unit", "security"]:
                print(f"💥 Critical test suite '{suite}' failed. Stopping execution.")
                break
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Secure MPC Transformer Test Runner")
    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "security", "performance", "gpu", "e2e", "quality", "dependencies", "all"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--output",
        help="Output file for test report (JSON format)"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow-running tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    if args.suite == "all":
        success = runner.run_all_tests(skip_slow=args.skip_slow)
    else:
        success = runner.run_test_suite(args.suite)
    
    runner.generate_report(args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
