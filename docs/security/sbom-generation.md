# Software Bill of Materials (SBOM) Generation

This document describes the SBOM generation process for the secure MPC transformer project, ensuring supply chain transparency and security.

## Overview

SBOMs provide visibility into software components and dependencies, enabling:
- Vulnerability tracking and management
- License compliance verification
- Supply chain security analysis
- Incident response and remediation

## SBOM Standards

We generate SBOMs in multiple formats to ensure compatibility:

### SPDX Format
- **Primary format**: SPDX 2.3 JSON
- **Use case**: Industry standard, tool compatibility
- **Generated for**: All releases, container images

### CycloneDX Format  
- **Secondary format**: CycloneDX 1.5 JSON
- **Use case**: Vulnerability management integration
- **Generated for**: Security scanning, dependency analysis

## Generation Methods

### 1. Python Package SBOM

Using `syft` for Python package analysis:

```bash
# Install syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate Python SBOM
syft packages dir:. \
  --output spdx-json=sbom-python.spdx.json \
  --output cyclonedx-json=sbom-python.cyclonedx.json
```

### 2. Container Image SBOM

For Docker containers:

```bash
# Build container
docker build -t secure-mpc-transformer:latest .

# Generate container SBOM
syft packages secure-mpc-transformer:latest \
  --output spdx-json=sbom-container.spdx.json \
  --exclude '/tmp,/var/cache,/var/log'
```

### 3. Source Code SBOM

Including development dependencies:

```bash
# Generate comprehensive source SBOM
syft packages dir:. \
  --include-dev-deps \
  --output spdx-json=sbom-source.spdx.json
```

## SBOM Content Standards

### Required Components
1. **Direct Dependencies**: All packages listed in pyproject.toml
2. **Transitive Dependencies**: All sub-dependencies
3. **System Libraries**: OS-level packages in containers
4. **GPU Libraries**: CUDA, cuDNN, and related libraries
5. **Cryptographic Libraries**: SEAL, TenSEAL, and crypto dependencies

### Metadata Requirements
```json
{
  "creationInfo": {
    "created": "2025-01-15T10:30:00Z",
    "creators": ["Tool: syft", "Organization: Terragon Labs"],
    "licenseListVersion": "3.21"
  },
  "name": "secure-mpc-transformer",
  "documentNamespace": "https://github.com/terragon/secure-mpc-transformer/sbom-{uuid}",
  "packages": [
    {
      "name": "torch",
      "downloadLocation": "https://pypi.org/project/torch/",
      "filesAnalyzed": false,
      "licenseConcluded": "BSD-3-Clause",
      "copyrightText": "Copyright PyTorch contributors"
    }
  ]
}
```

### Security-Critical Components
Special attention for cryptographic and MPC components:

```json
{
  "name": "seal-python",
  "versionInfo": "4.1.0",
  "supplier": "Organization: Microsoft Research",
  "downloadLocation": "https://github.com/microsoft/SEAL-Python/",
  "packageVerificationCode": {
    "packageVerificationCodeValue": "d6a770ba38583ed4bb4525bd96e50461655d2758"
  },
  "licenseConcluded": "MIT",
  "externalRefs": [
    {
      "referenceCategory": "SECURITY",
      "referenceType": "cpe23Type",
      "referenceLocator": "cpe:2.3:a:microsoft:seal:4.1.0:*:*:*:*:python:*:*"
    }
  ]
}
```

## Automation Integration

### GitHub Actions Integration

```yaml
name: Generate SBOM
on:
  push:
    tags: ['v*']
  release:
    types: [published]

jobs:
  sbom:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Syft
        uses: anchore/sbom-action@v0
        
      - name: Generate Python SBOM
        run: |
          syft packages dir:. \
            --output spdx-json=sbom-python.spdx.json \
            --output cyclonedx-json=sbom-python.cyclonedx.json
            
      - name: Generate Container SBOM
        run: |
          docker build -t temp-image .
          syft packages temp-image \
            --output spdx-json=sbom-container.spdx.json
            
      - name: Upload SBOMs
        uses: actions/upload-artifact@v4
        with:
          name: sboms
          path: sbom-*.json
          
      - name: Attach to Release
        if: github.event_name == 'release'
        uses: softprops/action-gh-release@v1
        with:
          files: sbom-*.json
```

### Container Registry Integration

```yaml
- name: Push SBOM to Registry
  run: |
    cosign attach sbom \
      ${{ env.IMAGE_NAME }}:${{ github.sha }} \
      --sbom sbom-container.spdx.json
```

## Vulnerability Mapping

### CVE Integration
Map SBOM components to known vulnerabilities:

```bash
# Generate vulnerability report from SBOM
grype sbom:sbom-container.spdx.json \
  --output json > vulnerability-report.json
```

### Continuous Monitoring
```yaml
- name: Monitor Dependencies
  run: |
    # Check for new CVEs against SBOM
    grype sbom:sbom-python.spdx.json \
      --fail-on high \
      --only-fixed
```

## Compliance Standards

### NTIA Minimum Elements
Our SBOMs include all NTIA-required elements:
- [x] Supplier name
- [x] Component name  
- [x] Version of component
- [x] Other unique identifiers
- [x] Dependency relationships
- [x] Author of SBOM data
- [x] Timestamp of SBOM data

### Industry Standards
- **SPDX 2.3**: Full compliance
- **CycloneDX 1.5**: Full compliance  
- **SWID Tags**: Generated for major components
- **SLSA Provenance**: Linked to build process

## Storage and Distribution

### Repository Storage
```
sboms/
├── releases/
│   ├── v0.1.0/
│   │   ├── sbom-python.spdx.json
│   │   ├── sbom-container.spdx.json
│   │   └── sbom-source.spdx.json
│   └── v0.2.0/
└── latest/
    ├── sbom-python.spdx.json
    └── sbom-container.spdx.json
```

### Container Registry
SBOMs are attached to container images using cosign:
```bash
# Attach SBOM
cosign attach sbom image:tag --sbom sbom.spdx.json

# Verify SBOM
cosign verify-attestation image:tag --type spdxjson
```

## SBOM Validation

### Schema Validation
```bash
# Validate SPDX format
spdx-tools-python validate sbom-container.spdx.json

# Validate CycloneDX format
cyclonedx-cli validate --input-file sbom-container.cyclonedx.json
```

### Content Validation
```python
#!/usr/bin/env python3
"""SBOM validation script"""

import json
import sys

def validate_sbom(sbom_path):
    with open(sbom_path) as f:
        sbom = json.load(f)
    
    # Check required fields
    required_fields = ['name', 'creationInfo', 'packages']
    for field in required_fields:
        if field not in sbom:
            print(f"Missing required field: {field}")
            return False
    
    # Validate packages
    for pkg in sbom.get('packages', []):
        if not pkg.get('name'):
            print(f"Package missing name: {pkg}")
            return False
            
    return True

if __name__ == '__main__':
    if validate_sbom(sys.argv[1]):
        print("SBOM validation passed")
    else:
        print("SBOM validation failed")
        sys.exit(1)
```

## Security Considerations

### SBOM Protection
- SBOMs may contain sensitive information about internal dependencies
- Sign SBOMs with cosign to ensure integrity
- Limit access to detailed SBOMs in private repositories

### Vulnerability Response
1. **Detection**: Automated scanning of SBOMs for new CVEs
2. **Assessment**: Impact analysis on MPC protocols and security
3. **Remediation**: Coordinated updates and testing
4. **Communication**: Security advisories and updated SBOMs

### Privacy Concerns
- Exclude internal/proprietary components from public SBOMs
- Use generic identifiers for custom MPC implementations
- Sanitize file paths and internal URLs

## Tooling and Integration

### Recommended Tools
- **Syft**: Primary SBOM generation
- **Grype**: Vulnerability scanning
- **Cosign**: SBOM signing and attestation
- **SPDX Tools**: Validation and conversion
- **CycloneDX CLI**: Alternative format support

### Integration Points
- GitHub Actions workflows
- Container registries (GHCR, Docker Hub)
- Vulnerability management systems
- Dependency tracking tools
- Compliance reporting systems