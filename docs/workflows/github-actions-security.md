# GitHub Actions Security Configuration

This document outlines the security configuration for GitHub Actions workflows in the secure MPC transformer repository.

## Security Principles

1. **Minimal Permissions**: Each workflow uses the minimum required permissions
2. **Secret Management**: All secrets are managed through GitHub Secrets or external vaults
3. **Supply Chain Security**: Dependencies are pinned and verified
4. **Artifact Security**: Build artifacts are signed and verified
5. **Environment Isolation**: Different security levels for different environments

## Required GitHub Secrets

### Core Secrets
```yaml
# Container Registry
DOCKER_USERNAME: Username for Docker Hub
DOCKER_PASSWORD: Token for Docker Hub (not password)
GHCR_TOKEN: GitHub Container Registry token

# Security Scanning
SNYK_TOKEN: Snyk vulnerability scanning token
SEMGREP_TOKEN: Semgrep security analysis token

# Deployment
KUBECONFIG_STAGING: Base64-encoded kubeconfig for staging
KUBECONFIG_PROD: Base64-encoded kubeconfig for production

# Code Signing
COSIGN_PRIVATE_KEY: Private key for container signing
COSIGN_PASSWORD: Password for private key
```

### Optional Secrets
```yaml
# External Services
PROMETHEUS_URL: Monitoring endpoint
GRAFANA_API_KEY: Dashboard automation
SLACK_WEBHOOK: Notification webhook

# Performance Testing
BENCHMARK_API_KEY: Performance tracking service
```

## Workflow Security Templates

### Basic Security Workflow
```yaml
name: Security Checks
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read
  security-events: write

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Semgrep
        uses: semgrep/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/python
```

### Container Security Template
```yaml
name: Container Security
on:
  push:
    paths: ['docker/**', 'Dockerfile*']

permissions:
  contents: read
  packages: write
  security-events: write

jobs:
  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build container
        run: docker build -t test-image .
        
      - name: Run Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: test-image
          format: sarif
          output: trivy-results.sarif
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
```

## Security Scanning Configuration

### SARIF Integration
All security tools should output SARIF format for GitHub Security tab integration:

```yaml
- name: Upload SARIF results
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
    category: security-scan
```

### Required Security Checks
1. **Static Analysis**: Semgrep, CodeQL, Bandit
2. **Secret Scanning**: GitLeaks, detect-secrets
3. **Dependency Scanning**: Snyk, Safety, Dependabot
4. **Container Scanning**: Trivy, Grype
5. **Infrastructure Scanning**: Checkov, TFSec

## Branch Protection Rules

### Main Branch
```yaml
required_status_checks:
  - security-scan
  - dependency-check
  - container-scan
  - lint-and-test
enforce_admins: true
required_pull_request_reviews:
  required_approving_review_count: 2
  dismiss_stale_reviews: true
  require_code_owner_reviews: true
restrict_pushes: true
```

### Development Branch
```yaml
required_status_checks:
  - security-scan
  - lint-and-test
required_pull_request_reviews:
  required_approving_review_count: 1
  require_code_owner_reviews: true
```

## Environment Protection

### Production Environment
```yaml
protection_rules:
  - type: required_reviewers
    reviewers: ["@security-team", "@crypto-team"]
  - type: wait_timer
    wait_timer: 5  # 5 minute delay
deployment_branch_policy:
  protected_branches: true
  custom_branch_policies: false
```

### Staging Environment
```yaml
protection_rules:
  - type: required_reviewers
    reviewers: ["@danieleschmidt"]
deployment_branch_policy:
  protected_branches: false
  custom_branch_policies: true
  custom_branches: ["develop", "staging/*"]
```

## Artifact Security

### Container Signing
```yaml
- name: Sign container
  run: |
    echo "$COSIGN_PRIVATE_KEY" | cosign sign --key - \
      ${{ env.IMAGE_NAME }}:${{ github.sha }}
  env:
    COSIGN_PRIVATE_KEY: ${{ secrets.COSIGN_PRIVATE_KEY }}
    COSIGN_PASSWORD: ${{ secrets.COSIGN_PASSWORD }}
```

### SBOM Generation
```yaml
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    image: ${{ env.IMAGE_NAME }}:${{ github.sha }}
    format: spdx-json
    output-file: sbom.spdx.json
    
- name: Upload SBOM
  uses: actions/upload-artifact@v4
  with:
    name: sbom
    path: sbom.spdx.json
```

## Monitoring and Alerting

### Security Event Notification
```yaml
- name: Notify security team
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SECURITY_SLACK_WEBHOOK }}
    text: "Security scan failed in ${{ github.repository }}"
```

### Metrics Collection
```yaml
- name: Record security metrics
  run: |
    curl -X POST "$PROMETHEUS_PUSHGATEWAY/metrics/job/github-actions" \
      --data "security_scan_duration_seconds $(date +%s)"
```

## Compliance Requirements

### Required for MATURING Level
1. All workflows must have security scanning
2. Container images must be signed
3. SBOMs must be generated for releases
4. Security events must be logged
5. Failed security checks must block deployment

### Audit Trail
All security-related workflow runs are logged and retained for compliance:
- Workflow execution logs: 90 days
- Security scan results: 1 year
- Container signatures: Permanent
- SBOMs: Permanent

## Emergency Procedures

### Security Incident Response
1. Disable affected workflows immediately
2. Revoke compromised secrets
3. Audit recent workflow runs
4. Update security configurations
5. Re-enable workflows after verification

### Secret Rotation
1. Generate new secrets in external systems
2. Update GitHub Secrets
3. Test workflows in staging
4. Monitor for failures
5. Update documentation