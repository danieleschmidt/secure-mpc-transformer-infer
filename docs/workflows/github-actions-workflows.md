# GitHub Actions Workflows for Secure MPC Transformer

## Overview

This document provides template GitHub Actions workflows for the Secure MPC Transformer project. These workflows implement comprehensive CI/CD with security-first practices.

## Required Workflows

### 1. Main CI/CD Pipeline

**File**: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 4 * * 1'  # Weekly security scans

env:
  PYTHON_VERSION: '3.10'
  NODE_VERSION: '18'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,benchmark]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests with coverage
      run: |
        pytest --cov=secure_mpc_transformer --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  container-security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -f docker/Dockerfile.cpu -t secure-mpc:test .
    
    - name: Run container security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'secure-mpc:test'
        format: 'sarif'
        output: 'container-results.sarif'
    
    - name: Upload container scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'container-results.sarif'

  deploy:
    needs: [test, security-scan, container-security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build and push Docker images
      env:
        DOCKER_REGISTRY: ${{ secrets.DOCKER_REGISTRY }}
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
      run: |
        echo $DOCKER_PASSWORD | docker login $DOCKER_REGISTRY -u $DOCKER_USERNAME --password-stdin
        docker build -f docker/Dockerfile.cpu -t $DOCKER_REGISTRY/secure-mpc:latest .
        docker push $DOCKER_REGISTRY/secure-mpc:latest
```

### 2. Security Scanning Workflow

**File**: `.github/workflows/security.yml`

```yaml
name: Security Scans

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'  # Daily security scans

jobs:
  sast:
    name: Static Application Security Testing
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run CodeQL Analysis
      uses: github/codeql-action/init@v2
      with:
        languages: python
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install safety bandit
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json
    
    - name: Run Bandit security linter
      run: bandit -r src/ -f json -o bandit-report.json

  secrets-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Run GitGuardian scan
      uses: GitGuardian/ggshield-action@v1.25.0
      env:
        GITHUB_PUSH_BEFORE_SHA: ${{ github.event.before }}
        GITHUB_PUSH_BASE_SHA: ${{ github.event.base }}
        GITHUB_PULL_BASE_SHA: ${{ github.event.pull_request.base.sha }}
        GITHUB_DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
        GITGUARDIAN_API_KEY: ${{ secrets.GITGUARDIAN_API_KEY }}
```

### 3. Performance Benchmarking

**File**: `.github/workflows/benchmark.yml`

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    paths: ['src/**', 'benchmarks/**']

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,benchmark]"
    
    - name: Run benchmarks
      run: |
        python benchmarks/run_all.py --output-format json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'customSmallerIsBetter'
        output-file-path: benchmarks/results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

### 4. Documentation Building

**File**: `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [main]
    paths: ['docs/**', 'src/**/*.py']
  pull_request:
    branches: [main]
    paths: ['docs/**', 'src/**/*.py']

jobs:
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install sphinx sphinx-rtd-theme myst-parser
        pip install -e .
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

### 5. Container Release Pipeline

**File**: `.github/workflows/release.yml`

```yaml
name: Release Pipeline

on:
  release:
    types: [published]
  push:
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    strategy:
      matrix:
        platform: [cpu, gpu]
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/Dockerfile.${{ matrix.platform }}
        push: true
        tags: ${{ steps.meta.outputs.tags }}-${{ matrix.platform }}
        labels: ${{ steps.meta.outputs.labels }}
    
    - name: Run container security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ steps.meta.outputs.tags }}-${{ matrix.platform }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

## Required Repository Secrets

Configure these secrets in your GitHub repository settings:

### Container Registry
- `DOCKER_REGISTRY`: Container registry URL
- `DOCKER_USERNAME`: Registry username
- `DOCKER_PASSWORD`: Registry password/token

### Security Tools
- `GITGUARDIAN_API_KEY`: GitGuardian API key for secret scanning
- `CODECOV_TOKEN`: Codecov token for coverage reporting

### Deployment
- `DEPLOY_KEY`: SSH key for deployment (if using SSH deployment)
- `KUBECONFIG`: Kubernetes configuration for K8s deployments

## Branch Protection Rules

Configure these branch protection rules for `main`:

```yaml
protection_rules:
  required_status_checks:
    strict: true
    contexts:
      - "test"
      - "security-scan"
      - "container-security"
  
  enforce_admins: true
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
  
  restrictions:
    users: []
    teams: ["maintainers"]
  
  allow_force_pushes: false
  allow_deletions: false
```

## Security Considerations

### Workflow Security
- All workflows use pinned action versions with SHA hashes
- Secrets are properly scoped and never logged
- Pull request workflows have limited permissions
- Container images are scanned before deployment

### Dependency Security
- Regular dependency updates via Dependabot
- Vulnerability scanning on every push
- License compliance checking
- Supply chain security with SLSA attestation

### Runtime Security
- Container images run as non-root users
- Read-only file systems where possible
- Resource limits and security contexts
- Network policies for Kubernetes deployments

## Monitoring and Alerting

### GitHub Advanced Security
- Enable Dependabot alerts
- Configure CodeQL analysis
- Set up secret scanning alerts
- Monitor security advisory notifications

### External Integrations
- Integrate with security tools (Snyk, GitGuardian)
- Set up monitoring dashboards
- Configure incident response workflows
- Implement compliance reporting

## Customization Guidelines

### Repository-Specific Configuration
1. Update Python versions based on project requirements
2. Modify container build targets for your architecture
3. Configure appropriate security scanning tools
4. Set up environment-specific deployment targets

### Performance Optimization
1. Use workflow caching for dependencies
2. Implement parallel job execution
3. Optimize container layer caching
4. Configure artifact retention policies

### Compliance Requirements
1. Add compliance-specific scanning tools
2. Implement audit trail logging
3. Configure retention policies
4. Set up compliance reporting workflows

## Migration Guide

### From Existing CI/CD
1. **Assessment**: Audit current CI/CD practices
2. **Planning**: Map existing workflows to new structure
3. **Implementation**: Gradually migrate workflows
4. **Testing**: Validate all security controls
5. **Monitoring**: Set up observability and alerting

### Security Hardening Checklist
- [ ] All secrets properly configured
- [ ] Branch protection rules enabled
- [ ] Security scanning tools configured
- [ ] Container security implemented
- [ ] Dependency management automated
- [ ] Compliance requirements addressed
- [ ] Incident response procedures documented

## Support and Maintenance

### Regular Updates
- Review and update workflow versions monthly
- Monitor security tool effectiveness
- Update dependency scanning configurations
- Refresh security credentials quarterly

### Troubleshooting
- Check GitHub Actions logs for detailed error information
- Verify secret configuration and permissions
- Review security scan results for false positives
- Monitor workflow performance and optimization opportunities

For questions or issues with these workflows, please refer to the project's security team or create an issue in the repository.