#!/usr/bin/env python3
"""
Comprehensive Security Audit Report Generator
for Secure MPC Transformer System

This script analyzes the codebase and generates a detailed security audit report
covering all aspects of the secure MPC transformer implementation.
"""

import os
import sys
import ast
import json
import time
import hashlib
import subprocess
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
    """Represents a security finding."""
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SecurityMetric:
    """Represents a security metric."""
    name: str
    value: Any
    description: str
    status: str  # "good", "warning", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SecurityAuditor:
    """Comprehensive security audit system."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.src_path = self.repo_path / "src"
        self.findings: List[SecurityFinding] = []
        self.metrics: List[SecurityMetric] = []
        self.file_hashes: Dict[str, str] = {}
        
    def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        logger.info("Starting comprehensive security audit...")
        start_time = time.time()
        
        # Audit categories
        audit_categories = [
            ("Cryptographic Implementation", self.audit_cryptographic_implementation),
            ("Key Management", self.audit_key_management),
            ("Protocol Security", self.audit_protocol_security),
            ("Input Validation", self.audit_input_validation),
            ("Error Handling", self.audit_error_handling),
            ("Authentication & Authorization", self.audit_authentication),
            ("Network Security", self.audit_network_security),
            ("Data Protection", self.audit_data_protection),
            ("Logging & Monitoring", self.audit_logging_monitoring),
            ("Configuration Security", self.audit_configuration_security),
            ("Dependency Security", self.audit_dependency_security),
            ("Code Quality", self.audit_code_quality),
        ]
        
        audit_results = {}
        
        for category_name, audit_func in audit_categories:
            logger.info(f"Auditing {category_name}...")
            try:
                category_findings = audit_func()
                audit_results[category_name] = {
                    "findings_count": len(category_findings),
                    "findings": category_findings
                }
            except Exception as e:
                logger.error(f"Error auditing {category_name}: {e}")
                audit_results[category_name] = {
                    "error": str(e),
                    "findings_count": 0,
                    "findings": []
                }
        
        # Generate overall assessment
        total_duration = time.time() - start_time
        overall_assessment = self.generate_overall_assessment()
        
        return {
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "audit_duration_seconds": total_duration,
            "repository_path": str(self.repo_path),
            "overall_assessment": overall_assessment,
            "categories": audit_results,
            "all_findings": [f.to_dict() for f in self.findings],
            "security_metrics": [m.to_dict() for m in self.metrics],
            "file_integrity": self.file_hashes,
            "recommendations": self.generate_recommendations(),
            "compliance_status": self.assess_compliance()
        }
    
    def audit_cryptographic_implementation(self) -> List[Dict[str, Any]]:
        """Audit cryptographic implementations."""
        findings = []
        
        # Check key management implementation
        key_manager_file = self.src_path / "secure_mpc_transformer" / "security" / "key_manager.py"
        if key_manager_file.exists():
            findings.extend(self.analyze_cryptographic_file(key_manager_file))
        
        # Check protocol implementations
        protocols_dir = self.src_path / "secure_mpc_transformer" / "protocols"
        if protocols_dir.exists():
            for protocol_file in protocols_dir.glob("*.py"):
                findings.extend(self.analyze_cryptographic_file(protocol_file))
        
        # Analyze cryptographic constants and parameters
        findings.extend(self.check_cryptographic_constants())
        
        return [f.to_dict() for f in findings]
    
    def analyze_cryptographic_file(self, file_path: Path) -> List[SecurityFinding]:
        """Analyze a file for cryptographic security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            # Calculate file hash for integrity
            self.file_hashes[str(file_path.relative_to(self.repo_path))] = hashlib.sha256(content.encode()).hexdigest()
            
            # Check for weak cryptographic practices
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check for weak random number generation
                    if (hasattr(node.func, 'attr') and 
                        node.func.attr in ['random', 'rand'] and
                        hasattr(node.func, 'value') and
                        hasattr(node.func.value, 'id') and
                        node.func.value.id in ['random', 'np', 'numpy']):
                        
                        findings.append(SecurityFinding(
                            severity="medium",
                            category="Cryptographic Implementation",
                            title="Potentially weak random number generation",
                            description=f"Use of {node.func.value.id}.{node.func.attr} detected. Consider using cryptographically secure random number generation.",
                            file_path=str(file_path.relative_to(self.repo_path)),
                            line_number=node.lineno,
                            recommendation="Use secrets.token_bytes() or torch.randn() for cryptographic applications",
                            cwe_id="CWE-338"
                        ))
                
                # Check for hardcoded cryptographic keys or secrets
                if isinstance(node, ast.Str) and len(node.s) > 16:
                    # Simple heuristic for potential keys
                    if any(keyword in node.s.lower() for keyword in ['key', 'secret', 'password', 'token']):
                        findings.append(SecurityFinding(
                            severity="high",
                            category="Cryptographic Implementation", 
                            title="Potential hardcoded secret",
                            description="String literal contains keywords suggesting it might be a hardcoded secret",
                            file_path=str(file_path.relative_to(self.repo_path)),
                            line_number=node.lineno,
                            recommendation="Use environment variables or secure key management for secrets",
                            cwe_id="CWE-798"
                        ))
                
                # Check for weak hash algorithms
                if (isinstance(node, ast.Call) and
                    hasattr(node.func, 'attr') and
                    node.func.attr in ['md5', 'sha1'] and
                    hasattr(node.func, 'value')):
                    
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="Cryptographic Implementation",
                        title="Weak hash algorithm detected",
                        description=f"Use of {node.func.attr} hash algorithm detected",
                        file_path=str(file_path.relative_to(self.repo_path)),
                        line_number=node.lineno,
                        recommendation="Use SHA-256 or stronger hash algorithms",
                        cwe_id="CWE-327"
                    ))
                
            # Check for proper key size parameters
            if "key_size" in content or "security_level" in content:
                # Look for minimum security levels
                if "128" not in content and "256" not in content:
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="Cryptographic Implementation",
                        title="Security level verification needed",
                        description="Verify that cryptographic operations use adequate key sizes (>=128 bits)",
                        file_path=str(file_path.relative_to(self.repo_path)),
                        recommendation="Ensure minimum 128-bit security level for all cryptographic operations"
                    ))
            
        except Exception as e:
            findings.append(SecurityFinding(
                severity="low",
                category="Cryptographic Implementation",
                title="File analysis error",
                description=f"Could not analyze file {file_path}: {e}",
                file_path=str(file_path.relative_to(self.repo_path))
            ))
        
        return findings
    
    def check_cryptographic_constants(self) -> List[SecurityFinding]:
        """Check for appropriate cryptographic constants."""
        findings = []
        
        # Check for security level definitions
        security_files = [
            self.src_path / "secure_mpc_transformer" / "protocols" / "base.py",
            self.src_path / "secure_mpc_transformer" / "security" / "key_manager.py"
        ]
        
        min_security_level_found = False
        
        for file_path in security_files:
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Look for security level constants
                if "128" in content or "security_level" in content:
                    min_security_level_found = True
                    
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        if not min_security_level_found:
            findings.append(SecurityFinding(
                severity="medium",
                category="Cryptographic Implementation",
                title="Security level constants not found",
                description="Could not verify minimum security level constants in cryptographic implementations",
                recommendation="Ensure all cryptographic operations specify minimum 128-bit security level"
            ))
        
        # Add positive findings for good practices
        key_manager_file = self.src_path / "secure_mpc_transformer" / "security" / "key_manager.py"
        if key_manager_file.exists():
            try:
                with open(key_manager_file, 'r') as f:
                    content = f.read()
                    
                good_practices = []
                if "secrets.token_bytes" in content:
                    good_practices.append("Cryptographically secure random generation")
                if "Fernet" in content:
                    good_practices.append("Authenticated encryption")
                if "rotation" in content.lower():
                    good_practices.append("Key rotation support")
                if "expires_at" in content:
                    good_practices.append("Key expiration management")
                
                if good_practices:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Cryptographic Implementation",
                        title="Good cryptographic practices found",
                        description=f"Implementation includes: {', '.join(good_practices)}",
                        file_path=str(key_manager_file.relative_to(self.repo_path))
                    ))
                    
            except Exception as e:
                logger.warning(f"Could not analyze key manager: {e}")
        
        return findings
    
    def audit_key_management(self) -> List[Dict[str, Any]]:
        """Audit key management practices."""
        findings = []
        
        key_manager_file = self.src_path / "secure_mpc_transformer" / "security" / "key_manager.py"
        
        if not key_manager_file.exists():
            findings.append(SecurityFinding(
                severity="critical",
                category="Key Management",
                title="Key manager implementation not found",
                description="No key management implementation found in expected location"
            ))
        else:
            try:
                with open(key_manager_file, 'r') as f:
                    content = f.read()
                
                # Check for key management best practices
                key_practices = {
                    "Key Generation": "generate_" in content,
                    "Key Rotation": "rotation" in content.lower() or "rotate" in content.lower(),
                    "Key Derivation": "derive" in content.lower() or "hkdf" in content.upper(),
                    "Key Expiration": "expires" in content.lower() or "ttl" in content.lower(),
                    "Key Revocation": "revoke" in content.lower(),
                    "Secure Storage": "encrypt" in content.lower() and "storage" in content.lower(),
                    "Access Control": "permissions" in content.lower(),
                }
                
                implemented_practices = []
                missing_practices = []
                
                for practice, implemented in key_practices.items():
                    if implemented:
                        implemented_practices.append(practice)
                    else:
                        missing_practices.append(practice)
                
                if implemented_practices:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Key Management",
                        title="Key management practices implemented",
                        description=f"Implemented: {', '.join(implemented_practices)}",
                        file_path=str(key_manager_file.relative_to(self.repo_path))
                    ))
                
                if missing_practices:
                    severity = "high" if len(missing_practices) > 3 else "medium"
                    findings.append(SecurityFinding(
                        severity=severity,
                        category="Key Management",
                        title="Missing key management practices",
                        description=f"Consider implementing: {', '.join(missing_practices)}",
                        file_path=str(key_manager_file.relative_to(self.repo_path)),
                        recommendation="Implement comprehensive key lifecycle management"
                    ))
                
                # Check for secure key storage
                if "master_key" in content.lower():
                    if "environment" in content.lower() or "env" in content:
                        findings.append(SecurityFinding(
                            severity="info",
                            category="Key Management",
                            title="Environment-based key storage",
                            description="Master key retrieval from environment variables detected",
                            file_path=str(key_manager_file.relative_to(self.repo_path))
                        ))
                    else:
                        findings.append(SecurityFinding(
                            severity="medium",
                            category="Key Management",
                            title="Key storage mechanism review needed",
                            description="Review master key storage mechanism for security",
                            file_path=str(key_manager_file.relative_to(self.repo_path)),
                            recommendation="Ensure master keys are stored securely (HSM, environment, etc.)"
                        ))
                        
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="medium",
                    category="Key Management",
                    title="Key manager analysis error",
                    description=f"Could not analyze key manager: {e}"
                ))
        
        return [f.to_dict() for f in findings]
    
    def audit_protocol_security(self) -> List[Dict[str, Any]]:
        """Audit MPC protocol security implementations."""
        findings = []
        
        protocols_dir = self.src_path / "secure_mpc_transformer" / "protocols"
        
        if not protocols_dir.exists():
            findings.append(SecurityFinding(
                severity="critical",
                category="Protocol Security",
                title="Protocol implementations not found",
                description="MPC protocol implementations directory not found"
            ))
            return [f.to_dict() for f in findings]
        
        # Check protocol implementations
        protocol_files = list(protocols_dir.glob("*.py"))
        
        if not protocol_files:
            findings.append(SecurityFinding(
                severity="high",
                category="Protocol Security", 
                title="No protocol implementations found",
                description="No MPC protocol implementation files found"
            ))
        
        security_properties = {
            "Privacy": ["share", "secret", "privacy"],
            "Correctness": ["verify", "reconstruct", "validate"],
            "Integrity": ["integrity", "authentic", "mac"],
            "Malicious Security": ["malicious", "verify", "proof"],
            "Semi-honest Security": ["semi_honest", "passive"]
        }
        
        for protocol_file in protocol_files:
            if protocol_file.name == "__init__.py":
                continue
                
            try:
                with open(protocol_file, 'r') as f:
                    content = f.read()
                
                # Check for security properties
                found_properties = []
                for property_name, keywords in security_properties.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        found_properties.append(property_name)
                
                if found_properties:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Protocol Security",
                        title=f"Security properties in {protocol_file.name}",
                        description=f"Implements: {', '.join(found_properties)}",
                        file_path=str(protocol_file.relative_to(self.repo_path))
                    ))
                
                # Check for potential security issues
                if "TODO" in content or "FIXME" in content:
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="Protocol Security",
                        title="Incomplete implementation markers",
                        description="TODO/FIXME comments found - ensure implementation is complete",
                        file_path=str(protocol_file.relative_to(self.repo_path)),
                        recommendation="Complete all protocol implementations before production use"
                    ))
                
                # Check for error handling in protocols
                if "except" in content and "raise" in content:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Protocol Security",
                        title="Error handling present",
                        description="Protocol includes error handling mechanisms",
                        file_path=str(protocol_file.relative_to(self.repo_path))
                    ))
                elif "except" not in content:
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="Protocol Security", 
                        title="Limited error handling",
                        description="Protocol may lack comprehensive error handling",
                        file_path=str(protocol_file.relative_to(self.repo_path)),
                        recommendation="Implement robust error handling to prevent information leakage",
                        cwe_id="CWE-209"
                    ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="low",
                    category="Protocol Security",
                    title="Protocol analysis error",
                    description=f"Could not analyze {protocol_file.name}: {e}",
                    file_path=str(protocol_file.relative_to(self.repo_path))
                ))
        
        return [f.to_dict() for f in findings]
    
    def audit_input_validation(self) -> List[Dict[str, Any]]:
        """Audit input validation and sanitization."""
        findings = []
        
        # Check validation implementation
        validation_files = [
            self.src_path / "secure_mpc_transformer" / "utils" / "validators.py",
            self.src_path / "secure_mpc_transformer" / "validation" / "schema_validator.py",
            self.src_path / "secure_mpc_transformer" / "api" / "routes.py"
        ]
        
        validation_present = False
        
        for validation_file in validation_files:
            if not validation_file.exists():
                continue
                
            validation_present = True
            
            try:
                with open(validation_file, 'r') as f:
                    content = f.read()
                
                # Check for validation patterns
                validation_patterns = {
                    "Input Sanitization": ["sanitize", "clean", "escape"],
                    "Type Checking": ["isinstance", "type", "dtype"],
                    "Range Validation": ["min", "max", "range", "bounds"],
                    "Schema Validation": ["schema", "validate", "jsonschema"],
                    "SQL Injection Prevention": ["parameterized", "prepared", "escape"],
                    "XSS Prevention": ["escape", "sanitize", "html"]
                }
                
                found_validations = []
                for validation_type, keywords in validation_patterns.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        found_validations.append(validation_type)
                
                if found_validations:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Input Validation",
                        title=f"Validation mechanisms in {validation_file.name}",
                        description=f"Implements: {', '.join(found_validations)}",
                        file_path=str(validation_file.relative_to(self.repo_path))
                    ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="low",
                    category="Input Validation",
                    title="Validation file analysis error",
                    description=f"Could not analyze {validation_file.name}: {e}"
                ))
        
        if not validation_present:
            findings.append(SecurityFinding(
                severity="high",
                category="Input Validation",
                title="Input validation system not found",
                description="No comprehensive input validation system found",
                recommendation="Implement input validation and sanitization for all user inputs",
                cwe_id="CWE-20"
            ))
        
        # Check API endpoints for validation
        api_dir = self.src_path / "secure_mpc_transformer" / "api"
        if api_dir.exists():
            for api_file in api_dir.glob("*.py"):
                try:
                    with open(api_file, 'r') as f:
                        content = f.read()
                    
                    # Look for request validation in API endpoints
                    if "request" in content.lower():
                        if "validate" not in content.lower() and "schema" not in content.lower():
                            findings.append(SecurityFinding(
                                severity="medium",
                                category="Input Validation",
                                title="API endpoint validation needed",
                                description=f"API file {api_file.name} handles requests but may lack validation",
                                file_path=str(api_file.relative_to(self.repo_path)),
                                recommendation="Implement request validation for all API endpoints",
                                cwe_id="CWE-20"
                            ))
                        else:
                            findings.append(SecurityFinding(
                                severity="info",
                                category="Input Validation",
                                title="API validation present",
                                description=f"API file {api_file.name} includes validation mechanisms",
                                file_path=str(api_file.relative_to(self.repo_path))
                            ))
                
                except Exception as e:
                    logger.warning(f"Could not analyze API file {api_file}: {e}")
        
        return [f.to_dict() for f in findings]
    
    def audit_error_handling(self) -> List[Dict[str, Any]]:
        """Audit error handling and information disclosure."""
        findings = []
        
        error_handling_file = self.src_path / "secure_mpc_transformer" / "utils" / "error_handling.py"
        
        if not error_handling_file.exists():
            findings.append(SecurityFinding(
                severity="high",
                category="Error Handling",
                title="Centralized error handling not found",
                description="No centralized error handling system found"
            ))
        else:
            try:
                with open(error_handling_file, 'r') as f:
                    content = f.read()
                
                # Check for good error handling practices
                error_practices = {
                    "Custom Exceptions": "Exception" in content and "class" in content,
                    "Error Classification": "category" in content.lower() or "severity" in content.lower(),
                    "Secure Logging": "logging" in content and "logger" in content,
                    "Error Sanitization": "sanitize" in content.lower() or "redact" in content.lower(),
                    "Stack Trace Handling": "traceback" in content,
                    "User-Safe Messages": "user_message" in content
                }
                
                implemented = []
                missing = []
                
                for practice, present in error_practices.items():
                    if present:
                        implemented.append(practice)
                    else:
                        missing.append(practice)
                
                if implemented:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Error Handling",
                        title="Error handling practices implemented",
                        description=f"Implements: {', '.join(implemented)}",
                        file_path=str(error_handling_file.relative_to(self.repo_path))
                    ))
                
                if missing:
                    severity = "medium" if len(missing) < 3 else "high"
                    findings.append(SecurityFinding(
                        severity=severity,
                        category="Error Handling",
                        title="Missing error handling practices",
                        description=f"Consider implementing: {', '.join(missing)}",
                        file_path=str(error_handling_file.relative_to(self.repo_path))
                    ))
                
                # Check for information disclosure risks
                if "password" in content.lower() or "secret" in content.lower():
                    if "redact" in content.lower() or "sanitize" in content.lower():
                        findings.append(SecurityFinding(
                            severity="info",
                            category="Error Handling",
                            title="Sensitive data handling present",
                            description="Error handler includes sensitive data protection",
                            file_path=str(error_handling_file.relative_to(self.repo_path))
                        ))
                    else:
                        findings.append(SecurityFinding(
                            severity="medium",
                            category="Error Handling",
                            title="Potential sensitive data exposure",
                            description="Error handling may expose sensitive information",
                            file_path=str(error_handling_file.relative_to(self.repo_path)),
                            recommendation="Implement sensitive data redaction in error messages",
                            cwe_id="CWE-209"
                        ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="medium",
                    category="Error Handling",
                    title="Error handling analysis error",
                    description=f"Could not analyze error handling: {e}"
                ))
        
        return [f.to_dict() for f in findings]
    
    def audit_authentication(self) -> List[Dict[str, Any]]:
        """Audit authentication and authorization mechanisms."""
        findings = []
        
        # Check for authentication components
        auth_files = [
            self.src_path / "secure_mpc_transformer" / "security" / "session_manager.py",
            self.src_path / "secure_mpc_transformer" / "api" / "middleware.py"
        ]
        
        auth_present = False
        
        for auth_file in auth_files:
            if not auth_file.exists():
                continue
                
            auth_present = True
            
            try:
                with open(auth_file, 'r') as f:
                    content = f.read()
                
                # Check for authentication mechanisms
                auth_mechanisms = {
                    "Session Management": ["session", "cookie"],
                    "Token Authentication": ["token", "jwt", "bearer"],
                    "Multi-factor Authentication": ["mfa", "2fa", "totp"],
                    "Password Hashing": ["hash", "bcrypt", "pbkdf2", "scrypt"],
                    "Rate Limiting": ["rate", "limit", "throttle"],
                    "CSRF Protection": ["csrf", "xsrf"],
                    "Secure Headers": ["header", "security"]
                }
                
                found_mechanisms = []
                for mechanism, keywords in auth_mechanisms.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        found_mechanisms.append(mechanism)
                
                if found_mechanisms:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Authentication & Authorization",
                        title=f"Auth mechanisms in {auth_file.name}",
                        description=f"Implements: {', '.join(found_mechanisms)}",
                        file_path=str(auth_file.relative_to(self.repo_path))
                    ))
                
                # Check for security issues
                if "password" in content.lower():
                    if "plain" in content.lower() or "clear" in content.lower():
                        findings.append(SecurityFinding(
                            severity="critical",
                            category="Authentication & Authorization",
                            title="Potential plaintext password handling",
                            description="Code may handle passwords in plaintext",
                            file_path=str(auth_file.relative_to(self.repo_path)),
                            recommendation="Always hash passwords before storage/comparison",
                            cwe_id="CWE-256"
                        ))
                    elif "hash" in content.lower() or "crypt" in content.lower():
                        findings.append(SecurityFinding(
                            severity="info", 
                            category="Authentication & Authorization",
                            title="Password hashing detected",
                            description="Password hashing mechanisms found",
                            file_path=str(auth_file.relative_to(self.repo_path))
                        ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="low",
                    category="Authentication & Authorization",
                    title="Auth file analysis error",
                    description=f"Could not analyze {auth_file.name}: {e}"
                ))
        
        if not auth_present:
            findings.append(SecurityFinding(
                severity="medium",
                category="Authentication & Authorization",
                title="Authentication system not found",
                description="No comprehensive authentication system found",
                recommendation="Implement authentication and authorization if the system handles sensitive operations"
            ))
        
        return [f.to_dict() for f in findings]
    
    def audit_network_security(self) -> List[Dict[str, Any]]:
        """Audit network security measures."""
        findings = []
        
        # Check for network security implementations
        security_files = [
            self.src_path / "secure_mpc_transformer" / "security" / "ddos_protection.py",
            self.src_path / "secure_mpc_transformer" / "security" / "threat_detector.py",
            self.src_path / "secure_mpc_transformer" / "api" / "server.py"
        ]
        
        network_security_features = {
            "DDoS Protection": ["ddos", "rate_limit", "throttle"],
            "TLS/SSL": ["tls", "ssl", "https"],
            "IP Filtering": ["whitelist", "blacklist", "ip_filter"],
            "Request Validation": ["validate", "sanitize"],
            "Threat Detection": ["threat", "anomaly", "intrusion"],
            "Firewall": ["firewall", "iptables"],
            "Load Balancing": ["load_balance", "proxy"]
        }
        
        found_features = []
        
        for security_file in security_files:
            if not security_file.exists():
                continue
                
            try:
                with open(security_file, 'r') as f:
                    content = f.read()
                
                for feature, keywords in network_security_features.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        if feature not in found_features:
                            found_features.append(feature)
                            findings.append(SecurityFinding(
                                severity="info",
                                category="Network Security",
                                title=f"{feature} implementation found",
                                description=f"{feature} mechanisms detected in {security_file.name}",
                                file_path=str(security_file.relative_to(self.repo_path))
                            ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="low",
                    category="Network Security",
                    title="Network security analysis error",
                    description=f"Could not analyze {security_file.name}: {e}"
                ))
        
        # Check for missing critical network security features
        critical_features = ["DDoS Protection", "TLS/SSL", "Request Validation"]
        missing_critical = [f for f in critical_features if f not in found_features]
        
        if missing_critical:
            findings.append(SecurityFinding(
                severity="medium",
                category="Network Security",
                title="Missing critical network security features",
                description=f"Consider implementing: {', '.join(missing_critical)}",
                recommendation="Implement comprehensive network security measures"
            ))
        
        return [f.to_dict() for f in findings]
    
    def audit_data_protection(self) -> List[Dict[str, Any]]:
        """Audit data protection and privacy measures."""
        findings = []
        
        # Check for data protection implementations
        data_files = [
            self.src_path / "secure_mpc_transformer" / "models",
            self.src_path / "secure_mpc_transformer" / "protocols",
            self.src_path / "secure_mpc_transformer" / "caching"
        ]
        
        data_protection_features = {
            "Encryption at Rest": ["encrypt", "cipher", "aes"],
            "Encryption in Transit": ["tls", "ssl", "transport"],
            "Data Anonymization": ["anonymize", "pseudonym"],
            "Access Control": ["permission", "access", "authorize"],
            "Data Retention": ["retention", "expire", "cleanup"],
            "Secure Deletion": ["secure_delete", "wipe", "shred"],
            "Privacy Preservation": ["privacy", "differential", "k_anonymity"]
        }
        
        found_features = []
        
        for data_path in data_files:
            if not data_path.exists():
                continue
                
            for data_file in data_path.glob("*.py"):
                if data_file.name == "__init__.py":
                    continue
                    
                try:
                    with open(data_file, 'r') as f:
                        content = f.read()
                    
                    for feature, keywords in data_protection_features.items():
                        if any(keyword in content.lower() for keyword in keywords):
                            if feature not in found_features:
                                found_features.append(feature)
                                findings.append(SecurityFinding(
                                    severity="info",
                                    category="Data Protection",
                                    title=f"{feature} implementation found",
                                    description=f"{feature} mechanisms detected",
                                    file_path=str(data_file.relative_to(self.repo_path))
                                ))
                    
                except Exception as e:
                    continue  # Skip files that can't be read
        
        # Check for potential data leakage in logs
        log_files = list(self.src_path.rglob("*.py"))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Check for potential sensitive data in logging
                if "logger" in content and any(keyword in content.lower() for keyword in ["password", "secret", "key", "token"]):
                    # Check if there's redaction
                    if "redact" not in content.lower() and "sanitize" not in content.lower():
                        findings.append(SecurityFinding(
                            severity="medium",
                            category="Data Protection",
                            title="Potential sensitive data logging",
                            description="Logging code may expose sensitive data",
                            file_path=str(log_file.relative_to(self.repo_path)),
                            recommendation="Implement log sanitization for sensitive data",
                            cwe_id="CWE-532"
                        ))
                        break  # Only report once
                
            except Exception:
                continue
        
        return [f.to_dict() for f in findings]
    
    def audit_logging_monitoring(self) -> List[Dict[str, Any]]:
        """Audit logging and monitoring capabilities."""
        findings = []
        
        # Check monitoring implementations
        monitoring_files = [
            self.src_path / "secure_mpc_transformer" / "monitoring",
            self.src_path / "secure_mpc_transformer" / "resilience" / "health_checks.py"
        ]
        
        monitoring_features = {
            "Health Checks": ["health", "status", "alive"],
            "Metrics Collection": ["metrics", "prometheus", "grafana"],
            "Distributed Tracing": ["trace", "span", "jaeger"],
            "Circuit Breakers": ["circuit", "breaker", "fallback"],
            "Alerting": ["alert", "notification", "alarm"],
            "Audit Logging": ["audit", "security_log"],
            "Performance Monitoring": ["performance", "latency", "throughput"]
        }
        
        found_features = []
        
        for monitoring_path in monitoring_files:
            if isinstance(monitoring_path, Path) and monitoring_path.is_file():
                # Single file
                monitoring_files_to_check = [monitoring_path]
            elif isinstance(monitoring_path, Path) and monitoring_path.is_dir():
                # Directory
                monitoring_files_to_check = list(monitoring_path.glob("*.py"))
            else:
                continue
            
            for monitoring_file in monitoring_files_to_check:
                if monitoring_file.name == "__init__.py":
                    continue
                    
                try:
                    with open(monitoring_file, 'r') as f:
                        content = f.read()
                    
                    for feature, keywords in monitoring_features.items():
                        if any(keyword in content.lower() for keyword in keywords):
                            if feature not in found_features:
                                found_features.append(feature)
                                findings.append(SecurityFinding(
                                    severity="info",
                                    category="Logging & Monitoring",
                                    title=f"{feature} implementation found",
                                    description=f"{feature} capabilities detected",
                                    file_path=str(monitoring_file.relative_to(self.repo_path))
                                ))
                    
                except Exception as e:
                    continue
        
        # Check for comprehensive logging
        if "Audit Logging" not in found_features:
            findings.append(SecurityFinding(
                severity="medium",
                category="Logging & Monitoring",
                title="Audit logging not found",
                description="No dedicated security audit logging found",
                recommendation="Implement comprehensive audit logging for security events"
            ))
        
        if "Health Checks" not in found_features:
            findings.append(SecurityFinding(
                severity="low",
                category="Logging & Monitoring",
                title="Health monitoring not found",
                description="No health check system found",
                recommendation="Implement health checks for production monitoring"
            ))
        
        return [f.to_dict() for f in findings]
    
    def audit_configuration_security(self) -> List[Dict[str, Any]]:
        """Audit configuration security."""
        findings = []
        
        # Check configuration files
        config_files = [
            self.repo_path / "config" / "example_config.json",
            self.src_path / "secure_mpc_transformer" / "config.py",
            self.repo_path / ".env.example"
        ]
        
        for config_file in config_files:
            if not config_file.exists():
                continue
                
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Check for potential security issues in config
                security_issues = []
                
                # Check for hardcoded secrets
                if any(keyword in content.lower() for keyword in ["password", "secret", "key", "token"]):
                    if "example" in config_file.name.lower() or "template" in config_file.name.lower():
                        findings.append(SecurityFinding(
                            severity="info",
                            category="Configuration Security",
                            title="Example configuration found",
                            description=f"Example config file provides security parameter templates",
                            file_path=str(config_file.relative_to(self.repo_path))
                        ))
                    else:
                        # Check if values are placeholders
                        if "xxx" in content.lower() or "placeholder" in content.lower() or "${" in content:
                            findings.append(SecurityFinding(
                                severity="info",
                                category="Configuration Security",
                                title="Template configuration values",
                                description="Configuration uses placeholder values",
                                file_path=str(config_file.relative_to(self.repo_path))
                            ))
                        else:
                            findings.append(SecurityFinding(
                                severity="high",
                                category="Configuration Security",
                                title="Potential hardcoded secrets in config",
                                description="Configuration file may contain hardcoded secrets",
                                file_path=str(config_file.relative_to(self.repo_path)),
                                recommendation="Use environment variables for sensitive configuration",
                                cwe_id="CWE-798"
                            ))
                
                # Check for security-related configuration options
                security_configs = {
                    "TLS/SSL": ["ssl", "tls", "cert"],
                    "Authentication": ["auth", "login", "token"],
                    "Encryption": ["encrypt", "cipher", "key"],
                    "Security Level": ["security_level", "strength"],
                    "Timeout": ["timeout", "expire"],
                    "Rate Limiting": ["rate", "limit", "throttle"]
                }
                
                found_configs = []
                for config_type, keywords in security_configs.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        found_configs.append(config_type)
                
                if found_configs:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Configuration Security",
                        title=f"Security configurations in {config_file.name}",
                        description=f"Includes: {', '.join(found_configs)}",
                        file_path=str(config_file.relative_to(self.repo_path))
                    ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="low",
                    category="Configuration Security",
                    title="Config analysis error",
                    description=f"Could not analyze {config_file.name}: {e}"
                ))
        
        return [f.to_dict() for f in findings]
    
    def audit_dependency_security(self) -> List[Dict[str, Any]]:
        """Audit dependency security."""
        findings = []
        
        # Check requirements/dependencies
        dependency_files = [
            self.repo_path / "requirements.txt",
            self.repo_path / "pyproject.toml",
            self.repo_path / "setup.py"
        ]
        
        dependencies_found = False
        
        for dep_file in dependency_files:
            if not dep_file.exists():
                continue
                
            dependencies_found = True
            
            try:
                with open(dep_file, 'r') as f:
                    content = f.read()
                
                # Check for version pinning
                if "==" in content or "~=" in content:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Dependency Security",
                        title="Version pinning detected",
                        description=f"Dependencies are version-pinned in {dep_file.name}",
                        file_path=str(dep_file.relative_to(self.repo_path))
                    ))
                else:
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="Dependency Security",
                        title="Unpinned dependencies",
                        description=f"Dependencies may not be version-pinned in {dep_file.name}",
                        file_path=str(dep_file.relative_to(self.repo_path)),
                        recommendation="Pin dependency versions for reproducible builds",
                        cwe_id="CWE-1104"
                    ))
                
                # Check for known security-focused dependencies
                security_deps = ["cryptography", "pyjwt", "bcrypt", "passlib", "certifi"]
                found_security_deps = [dep for dep in security_deps if dep in content.lower()]
                
                if found_security_deps:
                    findings.append(SecurityFinding(
                        severity="info",
                        category="Dependency Security",
                        title="Security-focused dependencies",
                        description=f"Uses security libraries: {', '.join(found_security_deps)}",
                        file_path=str(dep_file.relative_to(self.repo_path))
                    ))
                
            except Exception as e:
                findings.append(SecurityFinding(
                    severity="low",
                    category="Dependency Security",
                    title="Dependency analysis error",
                    description=f"Could not analyze {dep_file.name}: {e}"
                ))
        
        if not dependencies_found:
            findings.append(SecurityFinding(
                severity="medium",
                category="Dependency Security",
                title="No dependency files found",
                description="No dependency management files found",
                recommendation="Use dependency management (requirements.txt, pyproject.toml, etc.)"
            ))
        
        # Check if there's a lock file
        lock_files = [
            self.repo_path / "poetry.lock",
            self.repo_path / "Pipfile.lock",
            self.repo_path / "requirements.lock"
        ]
        
        lock_file_found = any(lf.exists() for lf in lock_files)
        
        if lock_file_found:
            findings.append(SecurityFinding(
                severity="info",
                category="Dependency Security",
                title="Dependency lock file found",
                description="Project uses dependency locking for reproducible builds"
            ))
        else:
            findings.append(SecurityFinding(
                severity="low",
                category="Dependency Security",
                title="No dependency lock file",
                description="Consider using dependency locking (poetry.lock, Pipfile.lock, etc.)",
                recommendation="Use dependency locking for reproducible and secure builds"
            ))
        
        return [f.to_dict() for f in findings]
    
    def audit_code_quality(self) -> List[Dict[str, Any]]:
        """Audit overall code quality and security practices."""
        findings = []
        
        # Count Python files and lines of code
        python_files = list(self.src_path.rglob("*.py"))
        total_lines = 0
        total_files = len(python_files)
        
        security_patterns = {
            "TODO/FIXME": 0,
            "Hardcoded IPs": 0,
            "Print Statements": 0,
            "Exception Catching": 0,
            "Logging Usage": 0,
            "Type Annotations": 0
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    total_lines += len(lines)
                
                # Count security-relevant patterns
                content_lower = content.lower()
                
                if "todo" in content_lower or "fixme" in content_lower:
                    security_patterns["TODO/FIXME"] += 1
                
                # Simple IP address pattern
                import re
                ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
                if re.search(ip_pattern, content) and "127.0.0.1" not in content and "0.0.0.0" not in content:
                    security_patterns["Hardcoded IPs"] += 1
                
                if "print(" in content:
                    security_patterns["Print Statements"] += 1
                
                if "except" in content:
                    security_patterns["Exception Catching"] += 1
                
                if "logger" in content or "logging" in content:
                    security_patterns["Logging Usage"] += 1
                
                if "->" in content or ": " in content:  # Simple type annotation check
                    security_patterns["Type Annotations"] += 1
                
            except Exception:
                continue
        
        # Generate code quality findings
        findings.append(SecurityFinding(
            severity="info",
            category="Code Quality",
            title="Codebase statistics",
            description=f"Project contains {total_files} Python files with {total_lines:,} lines of code"
        ))
        
        # Analyze patterns
        if security_patterns["TODO/FIXME"] > total_files * 0.1:  # More than 10% of files
            findings.append(SecurityFinding(
                severity="medium",
                category="Code Quality",
                title="High number of TODO/FIXME comments",
                description=f"Found TODO/FIXME in {security_patterns['TODO/FIXME']} files",
                recommendation="Address outstanding TODO/FIXME items before production"
            ))
        
        if security_patterns["Hardcoded IPs"] > 0:
            findings.append(SecurityFinding(
                severity="medium",
                category="Code Quality", 
                title="Potential hardcoded IP addresses",
                description=f"Found potential hardcoded IPs in {security_patterns['Hardcoded IPs']} files",
                recommendation="Use configuration files for IP addresses",
                cwe_id="CWE-798"
            ))
        
        if security_patterns["Print Statements"] > 5:
            findings.append(SecurityFinding(
                severity="low",
                category="Code Quality",
                title="Print statements in code",
                description=f"Found print statements in {security_patterns['Print Statements']} files",
                recommendation="Use logging instead of print statements"
            ))
        
        if security_patterns["Exception Catching"] / total_files > 0.5:
            findings.append(SecurityFinding(
                severity="info",
                category="Code Quality",
                title="Good exception handling coverage",
                description=f"Exception handling found in {security_patterns['Exception Catching']} files"
            ))
        
        if security_patterns["Logging Usage"] / total_files > 0.3:
            findings.append(SecurityFinding(
                severity="info",
                category="Code Quality",
                title="Good logging coverage",
                description=f"Logging usage found in {security_patterns['Logging Usage']} files"
            ))
        
        return [f.to_dict() for f in findings]
    
    def generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall security assessment."""
        if not self.findings:
            return {
                "status": "unknown",
                "message": "No findings generated"
            }
        
        # Count findings by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for finding in self.findings:
            severity = finding.severity
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        total_issues = sum(severity_counts[s] for s in ["critical", "high", "medium", "low"])
        total_findings = len(self.findings)
        
        # Determine overall status
        if severity_counts["critical"] > 0:
            status = "critical"
            message = f"Critical security issues found ({severity_counts['critical']} critical)"
        elif severity_counts["high"] > 3:
            status = "high_risk"
            message = f"Multiple high-risk issues found ({severity_counts['high']} high)"
        elif severity_counts["high"] > 0:
            status = "medium_risk"
            message = f"Some high-risk issues found ({severity_counts['high']} high)"
        elif total_issues > 10:
            status = "needs_attention"
            message = f"Multiple issues need attention ({total_issues} total issues)"
        elif total_issues > 0:
            status = "good"
            message = f"Generally secure with minor issues ({total_issues} issues)"
        else:
            status = "excellent"
            message = "No security issues identified"
        
        # Calculate security score (0-100)
        security_score = max(0, 100 - (
            severity_counts["critical"] * 25 +
            severity_counts["high"] * 10 +
            severity_counts["medium"] * 3 +
            severity_counts["low"] * 1
        ))
        
        return {
            "status": status,
            "message": message,
            "security_score": security_score,
            "total_findings": total_findings,
            "total_issues": total_issues,
            "severity_distribution": severity_counts,
            "info_findings": severity_counts["info"]
        }
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate security recommendations."""
        recommendations = []
        
        if not self.findings:
            return recommendations
        
        # Group findings by category and severity
        critical_findings = [f for f in self.findings if f.severity == "critical"]
        high_findings = [f for f in self.findings if f.severity == "high"]
        medium_findings = [f for f in self.findings if f.severity == "medium"]
        
        # Priority recommendations for critical issues
        if critical_findings:
            recommendations.append({
                "priority": "critical",
                "title": "Address Critical Security Issues Immediately",
                "description": f"Found {len(critical_findings)} critical security issues that must be fixed before production deployment",
                "actions": [f.title for f in critical_findings[:5]]
            })
        
        # High priority recommendations
        if high_findings:
            recommendations.append({
                "priority": "high",
                "title": "Fix High-Risk Security Issues",
                "description": f"Address {len(high_findings)} high-risk security issues",
                "actions": [f.title for f in high_findings[:5]]
            })
        
        # Medium priority recommendations
        if len(medium_findings) > 5:
            recommendations.append({
                "priority": "medium",
                "title": "Improve Security Posture",
                "description": f"Address {len(medium_findings)} medium-priority security improvements",
                "actions": [f.title for f in medium_findings[:5]]
            })
        
        # Category-specific recommendations
        category_counts = {}
        for finding in self.findings:
            if finding.severity in ["critical", "high", "medium"]:
                category_counts[finding.category] = category_counts.get(finding.category, 0) + 1
        
        # Top problematic categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for category, count in top_categories:
            if count >= 3:
                recommendations.append({
                    "priority": "medium",
                    "title": f"Strengthen {category}",
                    "description": f"Multiple issues found in {category} ({count} issues)",
                    "actions": [f"Review and improve {category.lower()} implementations"]
                })
        
        # General security recommendations
        recommendations.append({
            "priority": "low",
            "title": "Security Best Practices",
            "description": "Continue following security best practices",
            "actions": [
                "Regular security code reviews",
                "Automated security testing in CI/CD",
                "Keep dependencies updated",
                "Monitor security advisories",
                "Conduct periodic penetration testing"
            ]
        })
        
        return recommendations
    
    def assess_compliance(self) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        compliance = {
            "OWASP_Top_10": {"score": 0, "max_score": 10, "issues": []},
            "NIST_Cybersecurity": {"score": 0, "max_score": 5, "issues": []},
            "SOC2": {"score": 0, "max_score": 5, "issues": []},
            "ISO_27001": {"score": 0, "max_score": 8, "issues": []}
        }
        
        # OWASP Top 10 assessment
        owasp_checks = {
            "Injection": ["Input Validation", "SQL injection"],
            "Broken_Authentication": ["Authentication & Authorization"],  
            "Sensitive_Data_Exposure": ["Data Protection", "Cryptographic Implementation"],
            "XML_External_Entities": ["Input Validation"],
            "Broken_Access_Control": ["Authentication & Authorization"],
            "Security_Misconfiguration": ["Configuration Security"],
            "XSS": ["Input Validation"],
            "Insecure_Deserialization": ["Input Validation"],
            "Known_Vulnerabilities": ["Dependency Security"],
            "Logging_Monitoring": ["Logging & Monitoring"]
        }
        
        for owasp_item, categories in owasp_checks.items():
            # Check if we have issues in these categories
            has_issues = any(
                f for f in self.findings 
                if f.category in categories and f.severity in ["critical", "high"]
            )
            if not has_issues:
                compliance["OWASP_Top_10"]["score"] += 1
            else:
                compliance["OWASP_Top_10"]["issues"].append(owasp_item)
        
        # NIST Cybersecurity Framework
        nist_functions = ["Identify", "Protect", "Detect", "Respond", "Recover"]
        for function in nist_functions:
            # Simplified assessment based on presence of security controls
            if function == "Protect" and any("Cryptographic" in f.category for f in self.findings if f.severity == "info"):
                compliance["NIST_Cybersecurity"]["score"] += 1
            elif function == "Detect" and any("Monitoring" in f.category for f in self.findings if f.severity == "info"):
                compliance["NIST_Cybersecurity"]["score"] += 1
            elif function == "Respond" and any("Error Handling" in f.category for f in self.findings if f.severity == "info"):
                compliance["NIST_Cybersecurity"]["score"] += 1
            else:
                # Basic score for having security implementations
                if any(f.severity == "info" for f in self.findings):
                    compliance["NIST_Cybersecurity"]["score"] += 0.5
        
        compliance["NIST_Cybersecurity"]["score"] = min(5, int(compliance["NIST_Cybersecurity"]["score"]))
        
        return compliance

def main():
    """Run the security audit."""
    print("=" * 80)
    print("Secure MPC Transformer - Comprehensive Security Audit")
    print("=" * 80)
    
    auditor = SecurityAuditor()
    
    try:
        results = auditor.run_full_audit()
        
        # Save results
        output_file = "security_audit_report.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Display summary
        print(f"\nAudit completed in {results['audit_duration_seconds']:.2f} seconds")
        print(f"Report saved to: {output_file}")
        
        assessment = results["overall_assessment"]
        print(f"\n" + "=" * 50)
        print("SECURITY ASSESSMENT SUMMARY")
        print("=" * 50)
        print(f"Overall Status: {assessment['status'].upper()}")
        print(f"Security Score: {assessment['security_score']}/100")
        print(f"Total Findings: {assessment['total_findings']}")
        print(f"Issues to Address: {assessment['total_issues']}")
        
        severity_dist = assessment["severity_distribution"]
        print(f"\nSeverity Distribution:")
        for severity in ["critical", "high", "medium", "low", "info"]:
            count = severity_dist.get(severity, 0)
            if count > 0:
                print(f"  {severity.capitalize()}: {count}")
        
        # Display top recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\n" + "=" * 50)
            print("TOP SECURITY RECOMMENDATIONS")
            print("=" * 50)
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. [{rec['priority'].upper()}] {rec['title']}")
                print(f"   {rec['description']}")
                if rec.get('actions'):
                    for action in rec['actions'][:3]:
                        print(f"   • {action}")
                print()
        
        # Display compliance status
        compliance = results.get("compliance_status", {})
        if compliance:
            print("=" * 50)
            print("COMPLIANCE STATUS")
            print("=" * 50)
            for standard, data in compliance.items():
                score = data.get("score", 0)
                max_score = data.get("max_score", 1)
                percentage = (score / max_score * 100) if max_score > 0 else 0
                print(f"{standard}: {score}/{max_score} ({percentage:.1f}%)")
        
        # Exit code based on security status
        if assessment["status"] in ["critical", "high_risk"]:
            print(f"\n❌ CRITICAL: Security issues must be addressed before production")
            return 1
        elif assessment["status"] == "medium_risk":
            print(f"\n⚠️  WARNING: Address security issues for production deployment")
            return 0
        else:
            print(f"\n✅ GOOD: Security posture is acceptable for production")
            return 0
            
    except Exception as e:
        logger.error(f"Security audit failed: {e}")
        print(f"\n❌ Security audit failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)