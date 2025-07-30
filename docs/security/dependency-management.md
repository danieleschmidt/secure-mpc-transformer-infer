# Dependency Management and Security

## Overview

This document outlines secure dependency management practices for the Secure MPC Transformer project, focusing on cryptographic library security, vulnerability management, and supply chain protection.

## Dependency Categories

### Core Cryptographic Dependencies

```toml
# pyproject.toml - Core crypto dependencies
[project.dependencies]
cryptography = ">=41.0.0"  # Well-audited cryptographic primitives
pycryptodome = ">=3.19.0"  # Additional crypto algorithms
tenseal = ">=0.3.14"       # Homomorphic encryption for tensors
seal-python = ">=4.1.0"   # Microsoft SEAL Python bindings
```

### MPC Protocol Libraries

```toml
# Specialized MPC frameworks
mp-spdz = ">=0.3.8"        # Generic MPC protocols
aby3 = ">=1.0.0"           # 3-party computation framework
fantastic-four = ">=0.2.0"  # 4-party GPU-accelerated protocols
```

### Machine Learning Stack

```toml
# PyTorch ecosystem
torch = ">=2.3.0"
transformers = ">=4.40.0"
onnx = ">=1.16.0"
onnxruntime-gpu = ">=1.18.0"

# CUDA acceleration
cupy-cuda12x = ">=12.0.0"
triton = ">=2.3.0"
```

## Security Practices

### 1. Dependency Pinning Strategy

#### Production Dependencies
```toml
# Pin major and minor versions for stability
cryptography = "~=41.0.0"
torch = "~=2.3.0"
transformers = "~=4.40.0"
```

#### Development Dependencies
```toml
# More flexible versioning for dev tools
pytest = ">=7.4.0"
black = ">=23.7.0"
ruff = ">=0.0.287"
```

#### Critical Security Libraries
```toml
# Pin exact versions for crypto libraries
seal-python = "==4.1.0"
tenseal = "==0.3.14"
pycryptodome = "==3.19.1"
```

### 2. Vulnerability Scanning

#### Safety Configuration
```toml
# pyproject.toml
[tool.safety]
ignore = [
    # Document any intentionally ignored vulnerabilities
    # "12345",  # Low-severity issue in dev dependency
]
audit-and-monitor = true
```

#### Automated Vulnerability Checks
```bash
# Daily vulnerability scanning
safety check --json --output safety-report.json

# Check for known security issues
bandit -r src/ -f json -o bandit-report.json

# Audit Python packages
pip-audit --format=json --output=audit-report.json
```

### 3. Supply Chain Security

#### Package Verification
```bash
# Verify package signatures and checksums
pip install --require-hashes -r requirements-hashes.txt

# Use pip-tools for reproducible builds
pip-compile --generate-hashes requirements.in
```

#### SBOM Generation
```yaml
# Generate Software Bill of Materials
name: Generate SBOM
run: |
  pip install cyclonedx-bom
  cyclonedx-py -o sbom.json
  
  # Validate SBOM format
  cyclonedx validate --input-file sbom.json
```

### 4. License Compliance

#### License Scanning
```bash
# Scan for license compatibility
pip-licenses --format=json --output-file=licenses.json

# Check for restrictive licenses
pip-licenses --fail-on GPL-3.0+ --fail-on AGPL-3.0+
```

#### Approved Licenses
```yaml
# .github/workflows/license-check.yml
approved_licenses:
  - MIT
  - Apache-2.0
  - BSD-3-Clause
  - ISC
  - Python-2.0
```

## Cryptographic Library Security

### 1. SEAL (Microsoft Homomorphic Encryption)

#### Security Considerations
- **Version Pinning**: Use exact version 4.1.0 for production
- **Parameter Validation**: Validate all encryption parameters
- **Memory Security**: Use secure memory allocation
- **Performance vs Security**: Balance polynomial modulus with security needs

```python
# Secure SEAL configuration
seal_params = seal.EncryptionParameters(seal.scheme_type.bfv)
seal_params.set_poly_modulus_degree(16384)  # 128-bit security
seal_params.set_coeff_modulus(seal.CoeffModulus.BFVDefault(16384))
seal_params.set_plain_modulus(seal.PlainModulus.Batching(16384, 20))
```

### 2. TenSEAL Integration

#### Secure Practices
- **Context Serialization**: Secure context sharing between parties
- **Precision Management**: Handle floating-point precision securely
- **Noise Budget**: Monitor and manage noise accumulation

```python
# Secure TenSEAL context
import tenseal as ts

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = pow(2, 40)
context.generate_galois_keys()
```

### 3. MP-SPDZ Integration

#### Protocol Security
- **Network Security**: Use TLS for all inter-party communication
- **Input Validation**: Validate all secret shares
- **Randomness**: Use cryptographically secure random number generation

```python
# Secure MP-SPDZ configuration
mpspdz_config = {
    'protocol': 'malicious-shamir',
    'security_level': 128,
    'network_security': 'tls1.3',
    'random_seed': secrets.randbits(256)
}
```

## Development Environment Security

### 1. Virtual Environment Isolation

```bash
# Create isolated environment
python -m venv venv-secure-mpc
source venv-secure-mpc/bin/activate

# Install dependencies with hash verification
pip install --require-hashes -r requirements-locked.txt
```

### 2. Pre-commit Security Hooks

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/PyCQA/safety
  rev: 2.3.5
  hooks:
    - id: safety
      args: [--short-report]

- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
    - id: detect-secrets
      args: ['--baseline', '.secrets.baseline']
```

### 3. Container Security

```dockerfile
# Secure multi-stage Docker build
FROM python:3.10-slim as crypto-builder

# Install cryptographic libraries with verification
RUN apt-get update && apt-get install -y \
    libseal-dev=4.1.0 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Verify library integrity
RUN sha256sum /usr/lib/libseal.so | grep "expected_hash"

FROM python:3.10-slim

# Copy verified libraries
COPY --from=crypto-builder /usr/lib/libseal.so /usr/lib/

# Install Python dependencies with hash verification
COPY requirements-hashes.txt .
RUN pip install --require-hashes -r requirements-hashes.txt

# Run as non-root user
RUN useradd --create-home --shell /bin/bash mpc
USER mpc
```

## Dependency Update Process

### 1. Automated Updates

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "security-team"
    open-pull-requests-limit: 5
    
    # Security-only updates for crypto libraries
    ignore:
      - dependency-name: "seal-python"
        update-types: ["version-update:semver-minor"]
      - dependency-name: "tenseal"
        update-types: ["version-update:semver-minor"]
```

### 2. Manual Review Process

#### Security-Critical Updates
1. **Assessment**: Review security implications
2. **Testing**: Run full test suite with new versions
3. **Validation**: Verify cryptographic correctness
4. **Rollback Plan**: Prepare rollback procedures

#### Update Workflow
```bash
# Update dependency versions  
pip-compile --upgrade requirements.in

# Generate new hashes
pip-compile --generate-hashes requirements.in

# Test with new dependencies
pytest tests/

# Security validation
python scripts/validate_crypto_setup.py

# Performance benchmarking
python benchmarks/run_all.py --compare-baseline
```

### 3. Emergency Security Updates

#### Critical Vulnerability Response
1. **Assessment**: Evaluate vulnerability impact (< 4 hours)
2. **Patching**: Apply security patches (< 24 hours)
3. **Testing**: Validate fixes don't break functionality
4. **Deployment**: Roll out updates to production
5. **Monitoring**: Monitor for issues post-deployment

```bash
# Emergency update script
#!/bin/bash
set -e

# Update vulnerable dependency
pip install --upgrade vulnerable-package==secure-version

# Run security validation
python scripts/security_validation.py

# Deploy if validation passes
if [ $? -eq 0 ]; then
    docker build -t secure-mpc:emergency-patch .
    kubectl rollout restart deployment/mpc-nodes
fi
```

## Monitoring and Alerting

### 1. Vulnerability Monitoring

```python
# vulnerability_monitor.py
import safety
import requests
import json

def check_vulnerabilities():
    """Daily vulnerability checking"""
    # Run safety check
    result = safety.check(packages=get_installed_packages())
    
    # Check for new CVEs
    cve_data = requests.get('https://cve.mitre.org/api/v1/vulnerabilities')
    
    # Alert on critical issues
    if critical_vulnerabilities_found(result, cve_data):
        send_security_alert()
```

### 2. License Compliance Monitoring

```python
# license_monitor.py
import pkg_resources
import requests

def monitor_license_changes():
    """Monitor for license changes in dependencies"""
    for package in pkg_resources.working_set:
        license_info = get_package_license(package)
        if license_changed(package, license_info):
            alert_legal_team(package, license_info)
```

### 3. Supply Chain Monitoring

```yaml
# Supply chain monitoring configuration
monitoring:
  package_sources:
    - pypi.org
    - conda-forge
  
  alerts:
    - new_maintainer_added
    - package_deleted
    - suspicious_download_patterns
    - typosquatting_detected
  
  validation:
    - signature_verification
    - checksum_validation
    - build_reproducibility
```

## Best Practices Summary

### Development
- ✅ Use virtual environments for isolation
- ✅ Pin cryptographic libraries to exact versions
- ✅ Verify package integrity with hashes
- ✅ Run security scans on every commit
- ✅ Monitor for license compliance

### Production
- ✅ Use locked requirements with hashes
- ✅ Implement vulnerability scanning in CI/CD
- ✅ Generate and validate SBOMs
- ✅ Monitor for dependency changes
- ✅ Maintain emergency update procedures

### Security
- ✅ Regular security audits of dependencies
- ✅ Cryptographic library version control
- ✅ Supply chain attack prevention
- ✅ Incident response procedures
- ✅ Continuous monitoring and alerting

## Resources and References

### Security Tools
- [Safety](https://github.com/pyupio/safety) - Python dependency vulnerability scanner
- [Bandit](https://github.com/PyCQA/bandit) - Security linter for Python
- [pip-audit](https://github.com/pypa/pip-audit) - Vulnerability scanner
- [CycloneDX](https://cyclonedx.org/) - SBOM generation

### Standards and Guidelines
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [SLSA Supply Chain Security](https://slsa.dev/)
- [OpenSSF Scorecard](https://github.com/ossf/scorecard)

### Cryptographic Resources
- [Microsoft SEAL Documentation](https://github.com/microsoft/SEAL)
- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)
- [MP-SPDZ Documentation](https://github.com/data61/MP-SPDZ)