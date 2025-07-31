# SLSA Compliance Framework

This document outlines the Supply Chain Levels for Software Artifacts (SLSA) compliance implementation for the secure MPC transformer project.

## Overview

SLSA is a security framework for ensuring the integrity of software artifacts throughout the software supply chain. We implement SLSA Level 3 compliance for production releases.

## SLSA Levels Implementation

### SLSA Level 1: Source + Build
**Status**: ✅ Implemented

Requirements:
- [x] Source code version controlled (Git)
- [x] Build process generates provenance
- [x] Provenance available to consumers

### SLSA Level 2: Hosted + Build  
**Status**: ✅ Implemented

Additional requirements:
- [x] Hosted build service (GitHub Actions)
- [x] Service-generated provenance
- [x] Build service authenticity

### SLSA Level 3: Hardened Builds
**Status**: 🔄 In Progress

Additional requirements:
- [x] Non-falsifiable provenance
- [x] Isolated build environment
- [ ] Ephemeral build environment (planned)

### SLSA Level 4: Reproducible + Two-party
**Status**: 📋 Planned

Future requirements:
- [ ] Reproducible builds
- [ ] Two independent build attestations

## Provenance Generation

### Build Provenance Schema
We use the SLSA Provenance v1.0 format:

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "subject": [
    {
      "name": "ghcr.io/terragon/secure-mpc-transformer",
      "digest": {
        "sha256": "abc123..."
      }
    }
  ],
  "predicateType": "https://slsa.dev/provenance/v1",
  "predicate": {
    "buildDefinition": {
      "buildType": "https://github.com/actions/runner",
      "externalParameters": {
        "workflow": {
          "ref": "refs/heads/main",
          "repository": "https://github.com/terragon/secure-mpc-transformer"
        }
      },
      "internalParameters": {
        "github": {
          "event_name": "push",
          "ref": "refs/heads/main",
          "sha": "def456..."
        }
      }
    },
    "runDetails": {
      "builder": {
        "id": "https://github.com/actions/runner/github-hosted"
      },
      "metadata": {
        "invocationId": "github-actions-run-123",
        "completedOn": "2025-01-15T10:30:00Z"
      }
    }
  }
}
```

### GitHub Actions Integration

#### Build and Provenance Workflow
```yaml
name: Build with SLSA Provenance
on:
  push:
    tags: ['v*']

permissions:
  contents: read
  packages: write
  id-token: write  # Required for SLSA

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.image.outputs.digest }}
      
    steps:
      - uses: actions/checkout@v4
      
      - name: Build container
        id: image
        run: |
          docker build -t temp-image .
          digest=$(docker inspect temp-image --format='{{index .RepoDigests 0}}' | cut -d'@' -f2)
          echo "digest=$digest" >> $GITHUB_OUTPUT
          
      - name: Push to registry
        run: |
          echo "${{ secrets.GHCR_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker tag temp-image ghcr.io/${{ github.repository }}:${{ github.sha }}
          docker push ghcr.io/${{ github.repository }}:${{ github.sha }}

  provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.10.0
    with:
      image: ghcr.io/${{ github.repository }}
      digest: ${{ needs.build.outputs.digest }}
    secrets:
      registry-username: ${{ github.actor }}
      registry-password: ${{ secrets.GHCR_TOKEN }}
```

### Container Provenance
For container images, we generate provenance including:

```json
{
  "buildDefinition": {
    "buildType": "https://github.com/docker/build-push-action",
    "externalParameters": {
      "source": {
        "uri": "https://github.com/terragon/secure-mpc-transformer",
        "ref": "refs/tags/v0.1.0"
      },
      "dockerfile": "docker/Dockerfile.gpu"
    }
  },
  "materials": [
    {
      "uri": "git+https://github.com/terragon/secure-mpc-transformer@refs/tags/v0.1.0",
      "digest": {
        "sha1": "abc123..."
      }
    }
  ]
}
```

## Verification Process

### Consumer Verification
Users can verify SLSA provenance:

```bash
# Install slsa-verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Verify container provenance
slsa-verifier verify-image \
  ghcr.io/terragon/secure-mpc-transformer:v0.1.0 \
  --source-uri github.com/terragon/secure-mpc-transformer \
  --source-tag v0.1.0
```

### Automated Verification in CI
```yaml
- name: Verify SLSA Provenance
  run: |
    slsa-verifier verify-artifact \
      ./secure-mpc-transformer-v0.1.0.tar.gz \
      --provenance-path ./provenance.json \
      --source-uri github.com/terragon/secure-mpc-transformer \
      --source-tag v0.1.0
```

## Build Environment Hardening

### Isolated Build Environment
GitHub Actions runners provide isolation:
- Fresh VM for each build
- No network access during sensitive operations
- Ephemeral build environment
- Signed attestations from GitHub

### Security Measures
```yaml
jobs:
  secure-build:
    runs-on: ubuntu-latest
    container:
      image: ubuntu:22.04
      options: --security-opt no-new-privileges
      
    steps:
      - name: Harden environment
        run: |
          # Remove package managers to prevent tampering
          rm -f /usr/bin/apt /usr/bin/apt-get /usr/bin/dpkg
          
          # Set restrictive umask
          umask 077
          
      - name: Verify source integrity
        run: |
          # Verify Git commit signature
          git verify-commit HEAD
          
          # Check for unexpected files
          find . -name "*.py" -exec python -m py_compile {} \;
```

### Build Reproducibility
For reproducible builds:

```dockerfile
# Use specific base image digest
FROM ubuntu:22.04@sha256:abc123...

# Set build timestamp for reproducibility
ARG BUILD_DATE=2025-01-15T10:30:00Z
ENV BUILD_DATE=$BUILD_DATE

# Use deterministic package versions
RUN apt-get update && apt-get install -y \
    python3=3.10.12-1~22.04 \
    python3-pip=22.0.2+dfsg-1ubuntu0.4
```

## Attestation Management

### Multiple Attestation Types
We generate multiple attestations:

1. **Build Provenance**: How the artifact was built
2. **SBOM**: What components are included  
3. **Vulnerability Scan**: Security assessment
4. **Test Results**: Quality assurance

```yaml
- name: Generate Multiple Attestations
  run: |
    # Build provenance (automatic via SLSA)
    
    # SBOM attestation
    cosign attest --predicate sbom.spdx.json \
      --type spdxjson \
      ${{ env.IMAGE }}@${{ env.DIGEST }}
      
    # Vulnerability scan attestation
    cosign attest --predicate vuln-scan.json \
      --type vuln \
      ${{ env.IMAGE }}@${{ env.DIGEST }}
      
    # Test results attestation  
    cosign attest --predicate test-results.json \
      --type custom \
      ${{ env.IMAGE }}@${{ env.DIGEST }}
```

### Attestation Storage
Attestations are stored in:
- OCI registry (attached to artifacts)
- Transparency log (Rekor)
- Release artifacts (GitHub Releases)

## Policy Enforcement

### Admission Control
Kubernetes admission controller example:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-slsa-provenance
spec:
  validationFailureAction: enforce
  background: false
  rules:
    - name: check-slsa-provenance
      match:
        any:
        - resources:
            kinds:
            - Pod
      verifyImages:
      - imageReferences:
        - "ghcr.io/terragon/secure-mpc-transformer:*"
        attestors:
        - entries:
          - keyless:
              subject: "https://github.com/terragon/secure-mpc-transformer/.github/workflows/build.yml@refs/heads/main"
              issuer: "https://token.actions.githubusercontent.com"
        attestations:
        - predicateType: https://slsa.dev/provenance/v1
```

### Supply Chain Policies
Example policy for dependency verification:

```yaml
# policy.yaml
apiVersion: tekton.dev/v1beta1
kind: Policy
metadata:
  name: slsa-requirements
spec:
  resources:
    - name: secure-mpc-transformer
      policy:
        - type: slsa-provenance-available
          with:
            minLevel: 3
        - type: source-code-signed
          with:
            trustedKeys: 
              - keyless: true
                issuer: https://github.com/login/oauth
        - type: sbom-required
          with:
            format: spdx
```

## Monitoring and Compliance

### Compliance Dashboard
Track SLSA compliance metrics:

```python
# compliance_monitor.py
import requests
import json
from datetime import datetime, timedelta

class SLSAMonitor:
    def __init__(self):
        self.metrics = {}
        
    def check_provenance_coverage(self):
        """Check what percentage of releases have provenance"""
        # Query GitHub API for releases
        releases = self.get_releases()
        
        coverage = 0
        for release in releases:
            if self.has_slsa_provenance(release):
                coverage += 1
                
        return coverage / len(releases) * 100
        
    def verify_attestations(self, image_digest):
        """Verify all attestations for an image"""
        attestations = self.get_attestations(image_digest)
        
        required_types = [
            'https://slsa.dev/provenance/v1',
            'https://spdx.dev/Document',
            'https://cyclonedx.org/schema'
        ]
        
        return all(t in attestations for t in required_types)
```

### Alerting
Set up alerts for compliance violations:

```yaml
# prometheus-alerts.yml
groups:
  - name: slsa-compliance
    rules:
      - alert: SLSAProvenanceMissing
        expr: slsa_provenance_coverage_percent < 95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "SLSA provenance coverage below threshold"
          
      - alert: UnverifiedArtifact
        expr: slsa_verification_failures_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Artifact failed SLSA verification"
```

## Integration with Security Tools

### CI/CD Integration
SLSA verification in deployment pipelines:

```yaml
- name: Verify Deployment Artifacts
  run: |
    # Verify all container images have SLSA provenance
    for image in $(kubectl get deploy -o jsonpath='{.items[*].spec.template.spec.containers[*].image}'); do
      echo "Verifying $image"
      slsa-verifier verify-image "$image" \
        --source-uri github.com/terragon/secure-mpc-transformer
    done
```

### Security Scanning Integration
Include SLSA verification in security scans:

```yaml
- name: Comprehensive Security Scan
  run: |
    # Traditional vulnerability scan
    trivy image ${{ env.IMAGE }}
    
    # SLSA provenance verification
    slsa-verifier verify-image ${{ env.IMAGE }}
    
    # Attestation verification
    cosign verify-attestation ${{ env.IMAGE }} \
      --type spdxjson \
      --certificate-identity-regexp="^https://github.com/terragon/"
```

## Future Enhancements

### SLSA Level 4 Roadmap
1. **Reproducible Builds**: Implement hermetic build process
2. **Two-Party Verification**: Add independent build verification
3. **Enhanced Provenance**: Include more detailed build information
4. **Policy as Code**: Formalize supply chain policies

### Advanced Features
- Cross-compilation provenance for multiple architectures
- Hardware security module (HSM) integration for signing
- Zero-trust supply chain verification
- Continuous compliance monitoring and reporting