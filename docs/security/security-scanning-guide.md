# Security Scanning Guide

This document outlines the security scanning tools and processes for the Secure MPC Transformer project.

## Overview

Security is paramount for MPC systems. This guide covers automated and manual security scanning procedures to ensure defensive security practices.

## Automated Security Tools

### 1. Static Application Security Testing (SAST)

#### Bandit - Python Security Linter

**Purpose**: Detect common security issues in Python code

**Configuration**: `.bandit`
```yaml
# .bandit configuration
skips: []
tests: []
exclude_dirs:
  - tests/
  - build/
  - dist/

# Custom rules for MPC security
assert_used:
  skips: ['B101']  # Allow asserts in MPC protocols for correctness

hardcoded_password_string:
  word_list: 'secrets.txt'  # Custom password wordlist
```

**Usage**:
```bash
# Run Bandit scan
bandit -r src/ -f json -o bandit-report.json

# Run with high confidence only
bandit -r src/ -i -ll

# Exclude test files
bandit -r src/ --exclude tests/
```

#### Semgrep - Multi-language Static Analysis

**Purpose**: Advanced pattern-based security scanning

**Configuration**: `.semgrep.yml`
```yaml
rules:
  - id: hardcoded-crypto-key
    pattern: |
      key = "..."
    message: "Hardcoded cryptographic key detected"
    languages: [python]
    severity: ERROR
    
  - id: insecure-random
    pattern: |
      random.random()
    message: "Use cryptographically secure random for security operations"
    languages: [python]
    severity: WARNING
    
  - id: timing-attack
    pattern: |
      if $SECRET == $INPUT:
        ...
    message: "Potential timing attack vulnerability"
    languages: [python]
    severity: ERROR
```

**Usage**:
```bash
# Run Semgrep with custom rules
semgrep --config=.semgrep.yml src/

# Use community rules
semgrep --config=p/security-audit src/

# Output to SARIF format
semgrep --sarif --output=semgrep.sarif src/
```

### 2. Dependency Vulnerability Scanning

#### Safety - Python Package Vulnerability Scanner

**Purpose**: Check for known vulnerabilities in dependencies

**Configuration**: `pyproject.toml`
```toml
[tool.safety]
# Ignore specific vulnerabilities (with justification)
ignore = [
    # Example: 12345  # torch vulnerability fixed in next release
]

# Check only production dependencies
exclude = ["dev", "test"]
```

**Usage**:
```bash
# Basic vulnerability check
safety check

# Check with JSON output
safety check --json --output safety-report.json

# Check specific requirements file
safety check -r requirements.txt

# Check and continue on failure
safety check --continue-on-error
```

#### pip-audit - Advanced Dependency Auditing

**Purpose**: Comprehensive dependency vulnerability scanning

**Usage**:
```bash
# Audit installed packages
pip-audit

# Audit requirements file
pip-audit -r requirements.txt

# Output to JSON
pip-audit --format=json --output=pip-audit.json

# Fix vulnerabilities automatically
pip-audit --fix
```

### 3. Secret Detection

#### GitLeaks - Git Secret Scanner

**Purpose**: Detect secrets in Git history and files

**Configuration**: `.gitleaks.toml`
```toml
title = "Secure MPC Transformer GitLeaks Config"

[[rules]]
id = "private-key"
description = "Private key detected"
regex = '''-----BEGIN [A-Z]+ PRIVATE KEY-----'''
tags = ["key", "private"]

[[rules]]
id = "api-key"
description = "API key detected"
regex = '''[aA][pP][iI]_?[kK][eE][yY].*['|\"](0x)?[a-fA-F0-9]+['|\"]'''
tags = ["key", "api"]

[[rules]]
id = "mpc-secret"
description = "MPC secret share detected"
regex = '''secret_share.*['|\"](0x)?[a-fA-F0-9]{64,}['|\"]'''
tags = ["mpc", "secret"]

[allowlist]
description = "Allowlist for test files"
files = [
    '''tests/.*test.*\.py$''',
    '''examples/.*\.py$'''
]
```

**Usage**:
```bash
# Scan current repository
gitleaks detect

# Scan with custom config
gitleaks detect --config .gitleaks.toml

# Scan specific directory
gitleaks detect --source src/

# Output to SARIF
gitleaks detect --report-format sarif --report-path gitleaks.sarif
```

#### TruffleHog - Deep Secret Scanning

**Purpose**: Entropy-based secret detection

**Usage**:
```bash
# Scan filesystem
trufflehog filesystem .

# Scan Git repository
trufflehog git https://github.com/user/repo

# Scan with custom patterns
trufflehog --regex-file custom-patterns.txt .
```

### 4. Container Security Scanning

#### Trivy - Container Vulnerability Scanner

**Purpose**: Scan container images for vulnerabilities

**Usage**:
```bash
# Scan Docker image
trivy image mpc-transformer:latest

# Scan with SARIF output
trivy image --format sarif --output trivy.sarif mpc-transformer:latest

# Scan filesystem
trivy fs .

# Scan with severity filtering
trivy image --severity HIGH,CRITICAL mpc-transformer:latest
```

#### Grype - Container and Filesystem Scanner

**Purpose**: Alternative container vulnerability scanning

**Usage**:
```bash
# Scan container image
grype mpc-transformer:latest

# Scan local directory
grype dir:.

# Output to JSON
grype -o json mpc-transformer:latest
```

## Manual Security Assessment

### 1. Code Review Checklist

#### Cryptographic Implementation
- [ ] Use of cryptographically secure random number generators
- [ ] Proper key management and storage
- [ ] Constant-time operations for sensitive comparisons
- [ ] Secure memory clearing after use
- [ ] Proper entropy sources for key generation

#### MPC Protocol Security
- [ ] Correct secret sharing implementation
- [ ] Protection against timing attacks
- [ ] Input validation for all network data
- [ ] Proper error handling without information leakage
- [ ] Secure communication channels (TLS 1.3+)

#### Input Validation
- [ ] All external inputs validated
- [ ] Buffer overflow protection
- [ ] SQL injection prevention (if applicable)
- [ ] Cross-site scripting prevention (if applicable)
- [ ] Path traversal protection

### 2. Security Testing

#### Penetration Testing

**Network Security Testing**:
```bash
# Port scanning
nmap -sV -sC localhost

# SSL/TLS testing
testssl.sh --parallel https://localhost:8443

# Network traffic analysis
tcpdump -i lo port 50051
```

**Application Security Testing**:
```bash
# Fuzzing MPC inputs
python security/fuzz_mpc_inputs.py

# Protocol conformance testing
python security/test_protocol_conformance.py

# Timing attack testing
python security/test_timing_attacks.py
```

### 3. Compliance Scanning

#### License Compliance

**Purpose**: Ensure all dependencies have compatible licenses

**Usage**:
```bash
# Check licenses with pip-licenses
pip-licenses --format=json --output-file licenses.json

# Generate license report
pip-licenses --format=html --output-file licenses.html

# Check for GPL licenses
pip-licenses | grep GPL
```

#### SBOM Generation

**Purpose**: Generate Software Bill of Materials

**Usage**:
```bash
# Generate SPDX SBOM
syft packages . -o spdx-json > sbom.spdx.json

# Generate CycloneDX SBOM
syft packages . -o cyclonedx-json > sbom.cyclonedx.json

# Generate from container image
syft packages docker:mpc-transformer:latest -o table
```

## CI/CD Integration

### GitHub Actions Security Workflows

The security scanning is integrated into CI/CD via GitHub Actions:

**Security Workflow** (`.github/workflows/security.yml`):
```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: Run Safety
      run: safety check --json --output safety-report.json
    
    - name: Run GitLeaks
      uses: gitleaks/gitleaks-action@v2
    
    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: security-results.sarif
```

### Pre-commit Hooks

Security scanning in pre-commit hooks:

**`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', '.bandit']
  
  - repo: local
    hooks:
      - id: safety-check
        name: Safety Check
        entry: safety check
        language: system
        pass_filenames: false
```

## Security Metrics and Monitoring

### Security Dashboards

**Grafana Dashboard Panels**:
- Vulnerability count trends
- Security scan frequency
- Failed security checks
- Dependency update status

**Prometheus Metrics**:
```python
# Security metrics collection
from prometheus_client import Counter, Histogram

security_scans_total = Counter('security_scans_total', 'Total security scans')
vulnerabilities_found = Counter('vulnerabilities_found_total', 'Vulnerabilities found')
scan_duration = Histogram('security_scan_duration_seconds', 'Security scan duration')
```

### Alert Configuration

**Alert Rules**:
```yaml
groups:
  - name: security_alerts
    rules:
      - alert: HighSeverityVulnerability
        expr: vulnerabilities_found{severity="high"} > 0
        for: 0m
        annotations:
          summary: "High severity vulnerability detected"
      
      - alert: SecurityScanFailed
        expr: security_scans_total{status="failed"} > 0
        for: 5m
        annotations:
          summary: "Security scan failed"
```

## Incident Response

### Security Incident Workflow

1. **Detection**: Automated alerts from security tools
2. **Assessment**: Evaluate severity and impact
3. **Containment**: Isolate affected systems
4. **Eradication**: Remove vulnerabilities
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Update security measures

### Emergency Procedures

**Critical Vulnerability Response**:
1. Immediate deployment halt
2. Security team notification
3. Vulnerability assessment
4. Patch development and testing
5. Emergency deployment approval
6. Post-incident review

## Security Tool Configuration

### Tool Integration Scripts

**Security scan aggregator** (`scripts/security_scan.py`):
```python
#!/usr/bin/env python3
"""Aggregate security scan results."""

import json
import subprocess
from pathlib import Path

def run_security_scans():
    """Run all security scanning tools."""
    results = {}
    
    # Run Bandit
    cmd = ["bandit", "-r", "src/", "-f", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    results['bandit'] = json.loads(result.stdout) if result.stdout else {}
    
    # Run Safety
    cmd = ["safety", "check", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    results['safety'] = json.loads(result.stdout) if result.stdout else {}
    
    # Run pip-audit
    cmd = ["pip-audit", "--format=json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    results['pip_audit'] = json.loads(result.stdout) if result.stdout else {}
    
    return results

if __name__ == "__main__":
    results = run_security_scans()
    with open("security-report.json", "w") as f:
        json.dump(results, f, indent=2)
```

### Configuration Templates

**Bandit configuration** (`.bandit`):
```yaml
exclude_dirs:
  - tests
  - venv
  - .venv

tests:
  - B101  # assert_used
  - B102  # exec_used
  - B103  # set_bad_file_permissions
  - B104  # hardcoded_bind_all_interfaces
  - B105  # hardcoded_password_string
  - B106  # hardcoded_password_funcarg
  - B107  # hardcoded_password_default

skips:
  - B101  # Allow asserts in tests
```

## Best Practices

### Security Development Lifecycle

1. **Secure Design**: Threat modeling and security requirements
2. **Secure Coding**: Secure coding standards and guidelines
3. **Security Testing**: Regular security assessments
4. **Security Deployment**: Secure configuration management
5. **Security Monitoring**: Continuous security monitoring

### Tool Selection Criteria

- **Coverage**: Broad vulnerability detection
- **Accuracy**: Low false positive rate
- **Integration**: CI/CD and IDE integration
- **Performance**: Reasonable scan times
- **Reporting**: Clear, actionable reports

### Continuous Improvement

- Regular tool updates and configuration reviews
- Security metrics analysis and trend monitoring
- Team training on new security tools and techniques
- Integration of lessons learned from incidents

This security scanning guide ensures comprehensive protection for the MPC transformer system while maintaining development velocity and operational efficiency.