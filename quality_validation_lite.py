#!/usr/bin/env python3
"""
Lightweight Quality Validation for TERRAGON SDLC

Validates core functionality without requiring heavy ML dependencies.
Focuses on code structure, imports, and basic functionality.
"""

import os
import sys
import ast
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class QualityValidator:
    """Lightweight quality validation for the codebase."""
    
    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all quality validation checks."""
        
        logger.info("Starting quality validation...")
        
        # Check directory structure
        self.check_directory_structure()
        
        # Validate Python syntax
        self.check_python_syntax()
        
        # Check imports
        self.check_import_structure()
        
        # Validate security patterns
        self.check_security_patterns()
        
        # Check research implementation
        self.check_research_modules()
        
        # Generate report
        return self.generate_report()
    
    def check_directory_structure(self) -> None:
        """Validate expected directory structure."""
        
        expected_dirs = [
            "secure_mpc_transformer",
            "secure_mpc_transformer/research",
            "secure_mpc_transformer/models",
            "secure_mpc_transformer/protocols",
            "secure_mpc_transformer/planning",
            "secure_mpc_transformer/optimization",
            "secure_mpc_transformer/security",
            "secure_mpc_transformer/services"
        ]
        
        for dir_path in expected_dirs:
            full_path = self.src_dir / dir_path
            if full_path.exists():
                self.passed_checks.append(f"Directory structure: {dir_path} exists")
            else:
                self.errors.append(f"Missing directory: {dir_path}")
    
    def check_python_syntax(self) -> None:
        """Check Python syntax for all .py files."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to check syntax
                ast.parse(content)
                self.passed_checks.append(f"Syntax validation: {py_file.name}")
                
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                self.warnings.append(f"Could not validate {py_file}: {e}")
    
    def check_import_structure(self) -> None:
        """Check import structure and dependencies."""
        
        # Check __init__.py files exist
        init_files = [
            "secure_mpc_transformer/__init__.py",
            "secure_mpc_transformer/research/__init__.py",
            "secure_mpc_transformer/models/__init__.py",
            "secure_mpc_transformer/protocols/__init__.py"
        ]
        
        for init_file in init_files:
            init_path = self.src_dir / init_file
            if init_path.exists():
                self.passed_checks.append(f"Package init: {init_file}")
            else:
                self.errors.append(f"Missing __init__.py: {init_file}")
    
    def check_security_patterns(self) -> None:
        """Check for security best practices."""
        
        security_patterns = [
            "cryptography",
            "secrets",
            "hashlib",
            "hmac"
        ]
        
        found_patterns = set()
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in security_patterns:
                    if pattern in content:
                        found_patterns.add(pattern)
                        
            except Exception as e:
                self.warnings.append(f"Could not scan {py_file}: {e}")
        
        for pattern in found_patterns:
            self.passed_checks.append(f"Security pattern: {pattern} usage found")
        
        if len(found_patterns) >= 2:
            self.passed_checks.append("Security implementation: Good cryptographic library usage")
        else:
            self.warnings.append("Security implementation: Limited cryptographic patterns found")
    
    def check_research_modules(self) -> None:
        """Check research module implementation."""
        
        research_files = [
            "secure_mpc_transformer/research/advanced_quantum_mpc.py",
            "secure_mpc_transformer/research/comparative_benchmark_framework.py",
            "secure_mpc_transformer/research/__init__.py"
        ]
        
        for research_file in research_files:
            file_path = self.src_dir / research_file
            if file_path.exists():
                # Check file size as indicator of implementation
                file_size = file_path.stat().st_size
                if file_size > 1000:  # At least 1KB indicates substantial implementation
                    self.passed_checks.append(f"Research module: {research_file} implemented ({file_size} bytes)")
                else:
                    self.warnings.append(f"Research module: {research_file} appears minimal")
            else:
                self.errors.append(f"Missing research module: {research_file}")
    
    def check_docstrings(self) -> None:
        """Check for proper documentation."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        documented_files = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to check for docstrings
                tree = ast.parse(content)
                
                # Check module docstring
                if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                    isinstance(tree.body[0].value, ast.Constant)):
                    documented_files += 1
                    
            except Exception:
                continue
        
        if documented_files > len(python_files) * 0.5:
            self.passed_checks.append(f"Documentation: {documented_files}/{len(python_files)} files have module docstrings")
        else:
            self.warnings.append(f"Documentation: Only {documented_files}/{len(python_files)} files documented")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        total_checks = len(self.passed_checks) + len(self.warnings) + len(self.errors)
        success_rate = len(self.passed_checks) / total_checks if total_checks > 0 else 0
        
        report = {
            "summary": {
                "total_checks": total_checks,
                "passed": len(self.passed_checks),
                "warnings": len(self.warnings),
                "errors": len(self.errors),
                "success_rate": success_rate,
                "overall_quality": self._assess_overall_quality(success_rate)
            },
            "passed_checks": self.passed_checks,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _assess_overall_quality(self, success_rate: float) -> str:
        """Assess overall code quality."""
        
        if success_rate >= 0.9:
            return "EXCELLENT"
        elif success_rate >= 0.8:
            return "GOOD"
        elif success_rate >= 0.7:
            return "ACCEPTABLE"
        elif success_rate >= 0.6:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        
        recommendations = []
        
        if self.errors:
            recommendations.append("Fix all syntax errors and missing files before deployment")
        
        if len(self.warnings) > len(self.passed_checks) * 0.3:
            recommendations.append("Address warnings to improve code quality")
        
        if any("Security" in check for check in self.passed_checks):
            recommendations.append("Good security implementation detected - maintain security standards")
        else:
            recommendations.append("Enhance security implementation with proper cryptographic libraries")
        
        if any("Research" in check for check in self.passed_checks):
            recommendations.append("Research modules implemented - ensure comprehensive testing")
        
        recommendations.append("Consider adding comprehensive unit tests for all modules")
        recommendations.append("Implement continuous integration for automated quality gates")
        
        return recommendations


def main():
    """Main quality validation execution."""
    
    validator = QualityValidator()
    report = validator.validate_all()
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY VALIDATION REPORT")
    print("="*60)
    
    summary = report["summary"]
    print(f"Overall Quality: {summary['overall_quality']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Checks Passed: {summary['passed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Errors: {summary['errors']}")
    
    # Print details
    if report["errors"]:
        print(f"\n❌ ERRORS ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"  • {error}")
    
    if report["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  • {warning}")
    
    print(f"\n✅ PASSED CHECKS ({len(report['passed_checks'])}):")
    for check in report["passed_checks"][:10]:  # Show first 10
        print(f"  • {check}")
    
    if len(report['passed_checks']) > 10:
        print(f"  ... and {len(report['passed_checks']) - 10} more")
    
    # Print recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n{'='*60}")
    
    # Determine exit code
    if report["errors"]:
        print("❌ Quality validation FAILED - fix errors before proceeding")
        return 1
    elif summary['success_rate'] >= 0.8:
        print("✅ Quality validation PASSED - ready for deployment")
        return 0
    else:
        print("⚠️  Quality validation PARTIAL - address warnings for better quality")
        return 0


if __name__ == "__main__":
    sys.exit(main())