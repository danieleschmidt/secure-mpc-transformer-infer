# GitHub Actions Workflow Templates

This document provides complete, ready-to-use GitHub Actions workflow templates for the Secure MPC Transformer project.

## Implementation Guide

Since workflow files cannot be created automatically, these templates must be manually implemented in `.github/workflows/` directory.

## 1. Core CI/CD Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.10"
  CUDA_VERSION: "12.0"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Cache pre-commit hooks
      uses: actions/cache@v3
      with:
        path: ~/.cache/pre-commit
        key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
    
    - name: Install pre-commit
      run: pip install pre-commit
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files

  test-cpu:
    runs-on: ubuntu-latest
    needs: lint-and-format
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
        cache-dependency-path: 'pyproject.toml'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ \
          --cov=secure_mpc_transformer \
          --cov-report=xml \
          --cov-report=term-missing \
          --junitxml=junit.xml
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-cpu-${{ matrix.python-version }}
        fail_ci_if_error: false

  test-gpu:
    runs-on: self-hosted
    needs: lint-and-format
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up CUDA environment
      run: |
        export PATH=/usr/local/cuda-${{ env.CUDA_VERSION }}/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-${{ env.CUDA_VERSION }}/lib64:$LD_LIBRARY_PATH
        nvidia-smi
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install GPU dependencies
      run: |
        pip install -e ".[dev,gpu]"
    
    - name: Build CUDA kernels
      run: |
        cd kernels/cuda
        make clean && make all
        cd ../..
    
    - name: Run GPU integration tests
      run: |
        pytest tests/integration/ \
          --gpu \
          --timeout=600 \
          --tb=short \
          -v
    
    - name: Cleanup GPU memory
      if: always()
      run: |
        python -c "import torch; torch.cuda.empty_cache()" || true
        nvidia-smi

  build-package:
    runs-on: ubuntu-latest
    needs: [test-cpu]
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install build dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: |
        python -m build
    
    - name: Check package
      run: |
        twine check dist/*
    
    - name: Upload package artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package
        path: dist/
        retention-days: 7
```

## 2. Security Scanning Workflow

**File**: `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM
  workflow_dispatch:

jobs:
  secret-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Run GitLeaks
      uses: gitleaks/gitleaks-action@v2
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Upload GitLeaks results
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: gitleaks-report
        path: results.sarif

  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        pip install safety bandit
        pip install -e ".[dev]"
    
    - name: Run Bandit security scan
      run: |
        bandit -r src/ -f json -o bandit-report.json
      continue-on-error: true
    
    - name: Run Safety vulnerability check
      run: |
        safety check --json --output safety-report.json
      continue-on-error: true
    
    - name: Run pip-audit
      run: |
        pip install pip-audit
        pip-audit --format=json --output=pip-audit-report.json
      continue-on-error: true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          pip-audit-report.json

  codeql-analysis:
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        queries: security-and-quality
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"

  container-scan:
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -f docker/Dockerfile.cpu -t mpc-transformer:scan .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'mpc-transformer:scan'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
```

## 3. Performance Benchmarking Workflow

**File**: `.github/workflows/benchmark.yml`

```yaml
name: Performance Benchmarking

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
    types: [ labeled ]
  schedule:
    - cron: '0 3 * * 0'  # Weekly Sunday 3 AM
  workflow_dispatch:
    inputs:
      benchmark_type:
        description: 'Benchmark type to run'
        required: true
        default: 'quick'
        type: choice
        options:
        - quick
        - full
        - comparison

jobs:
  benchmark:
    runs-on: self-hosted
    if: |
      github.event_name == 'schedule' ||
      github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'pull_request' && contains(github.event.label.name, 'benchmark')) ||
      (github.event_name == 'push' && github.ref == 'refs/heads/main')
    
    timeout-minutes: 120
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up environment
      run: |
        export PATH=/usr/local/cuda-12.0/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
        nvidia-smi
    
    - name: Install dependencies
      run: |
        pip install -e ".[benchmark,gpu]"
    
    - name: Prepare benchmark environment
      run: |
        mkdir -p benchmarks/results
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    - name: Run quick benchmarks
      if: github.event.inputs.benchmark_type == 'quick' || github.event_name != 'workflow_dispatch'
      run: |
        python benchmarks/benchmark_bert.py \
          --model bert-base-uncased \
          --batch-size 1 \
          --iterations 10 \
          --output benchmarks/results/bert-quick-$(date +%Y%m%d-%H%M%S).json
    
    - name: Run full benchmark suite
      if: github.event.inputs.benchmark_type == 'full' || github.event_name == 'schedule'
      run: |
        python benchmarks/run_all.py \
          --gpu \
          --models bert-base,distilbert-base \
          --output benchmarks/results/full-$(date +%Y%m%d-%H%M%S).json
    
    - name: Run protocol comparison
      if: github.event.inputs.benchmark_type == 'comparison'
      run: |
        python benchmarks/compare_protocols.py \
          --protocols semi_honest_3pc,malicious_3pc \
          --output benchmarks/results/protocols-$(date +%Y%m%d-%H%M%S).json
    
    - name: Generate performance report
      run: |
        python benchmarks/generate_report.py \
          --input benchmarks/results/ \
          --output benchmarks/benchmark-report.html \
          --format html
    
    - name: Store baseline for comparison
      if: github.ref == 'refs/heads/main'
      run: |
        cp benchmarks/results/*.json benchmarks/baselines/
        git config --global user.name 'GitHub Actions'
        git config --global user.email 'actions@github.com'
        git add benchmarks/baselines/
        git commit -m "chore: update performance baselines [skip ci]" || true
        git push || true
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ github.run_id }}
        path: |
          benchmarks/results/
          benchmarks/benchmark-report.html
        retention-days: 30
    
    - name: Comment benchmark results on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const path = 'benchmarks/benchmark-report.html';
          if (fs.existsSync(path)) {
            const report = fs.readFileSync(path, 'utf8');
            const summary = report.match(/<summary[^>]*>(.*?)<\/summary>/s)?.[1] || 'Performance benchmark completed';
            
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🚀 Performance Benchmark Results\n\n${summary}\n\n[View detailed report](https://github.com/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId})`
            });
          }
    
    - name: Check for performance regressions
      if: github.event_name == 'pull_request'
      run: |
        python benchmarks/check_regression.py \
          --current benchmarks/results/ \
          --baseline benchmarks/baselines/ \
          --threshold 0.1 || echo "Performance regression detected!"
```

## 4. Container Build Workflow

**File**: `.github/workflows/docker.yml`

```yaml
name: Build and Push Container Images

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]
    paths: [ 'docker/**', 'src/**', 'pyproject.toml' ]
  workflow_dispatch:

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
        variant: [cpu, gpu]
        include:
          - variant: cpu
            dockerfile: docker/Dockerfile.cpu
            platforms: linux/amd64,linux/arm64
          - variant: gpu
            dockerfile: docker/Dockerfile.gpu
            platforms: linux/amd64
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        flavor: |
          suffix=-${{ matrix.variant }},onlatest=true
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ${{ matrix.dockerfile }}
        platforms: ${{ matrix.platforms }}
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha,scope=${{ matrix.variant }}
        cache-to: type=gha,mode=max,scope=${{ matrix.variant }}
        build-args: |
          BUILDKIT_INLINE_CACHE=1
    
    - name: Test container image
      if: matrix.variant == 'cpu'
      run: |
        docker run --rm ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}-${{ matrix.variant }} \
          python -c "import secure_mpc_transformer; print('Package import successful')"
```

## 5. Dependency Management Workflow

**File**: `.github/workflows/dependencies.yml`

```yaml
name: Dependency Management

on:
  schedule:
    - cron: '0 1 * * 1'  # Weekly Monday 1 AM
  workflow_dispatch:
    inputs:
      update_type:
        description: 'Type of update to perform'
        required: true
        default: 'patch'
        type: choice
        options:
        - patch
        - minor
        - major

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    if: github.repository_owner == 'your-username'  # Prevent forks from running
    
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependency management tools
      run: |
        pip install pip-tools safety bandit
    
    - name: Generate updated requirements
      run: |
        pip-compile --upgrade --resolver=backtracking pyproject.toml
        pip-compile --upgrade --resolver=backtracking --extra dev pyproject.toml
        pip-compile --upgrade --resolver=backtracking --extra gpu pyproject.toml
    
    - name: Check for security vulnerabilities
      run: |
        pip install -r requirements.txt
        safety check --json --output safety-check.json || true
        
        # Create summary
        echo "## Security Scan Results" > security-summary.md
        if [ -f safety-check.json ]; then
          python -c "
import json
with open('safety-check.json') as f:
    data = json.load(f)
    if data.get('vulnerabilities'):
        print('⚠️ Security vulnerabilities found!')
        for vuln in data['vulnerabilities']:
            print(f'- {vuln[\"package_name\"]} {vuln[\"installed_version\"]}: {vuln[\"advisory\"]}')
    else:
        print('✅ No security vulnerabilities found')
" >> security-summary.md
        fi
    
    - name: Run dependency checks
      run: |
        # Check for deprecated packages
        pip list --outdated --format=json > outdated.json
        
        # Generate dependency report
        echo "## Dependency Update Summary" > update-summary.md
        echo "" >> update-summary.md
        echo "### Updated Packages" >> update-summary.md
        
        python -c "
import json
try:
    with open('outdated.json') as f:
        outdated = json.load(f)
    if outdated:
        for pkg in outdated[:10]:  # Limit to top 10
            print(f'- {pkg[\"name\"]}: {pkg[\"version\"]} → {pkg[\"latest_version\"]}')
    else:
        print('No packages to update')
except:
    print('Unable to generate update summary')
" >> update-summary.md
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: "chore: update dependencies"
        title: "🔒 Automated Dependency Updates"
        body: |
          ## Dependency Updates
          
          This PR contains automated dependency updates generated on $(date).
          
          ### Security Status
          $(cat security-summary.md)
          
          ### Package Updates
          $(cat update-summary.md)
          
          ### Checklist
          - [ ] All tests pass
          - [ ] No new security vulnerabilities introduced
          - [ ] Performance benchmarks acceptable
          - [ ] Breaking changes documented
          
          ### Automation Info
          - Generated by: GitHub Actions
          - Trigger: ${{ github.event_name }}
          - Workflow: ${{ github.workflow }}
        branch: automated/dependency-updates
        delete-branch: true
        labels: |
          dependencies
          automated
          security
```

## 6. Release Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., v1.0.0)'
        required: true
        type: string

env:
  PYTHON_VERSION: "3.10"

jobs:
  validate-release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Validate version format
      run: |
        if [[ ! "${{ github.ref_name }}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
          echo "Invalid version format: ${{ github.ref_name }}"
          exit 1
        fi
    
    - name: Check changelog
      run: |
        if ! grep -q "${{ github.ref_name }}" CHANGELOG.md 2>/dev/null; then
          echo "Warning: Version ${{ github.ref_name }} not found in CHANGELOG.md"
        fi

  build-and-test:
    runs-on: ubuntu-latest
    needs: validate-release
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run comprehensive tests
      run: |
        pytest tests/ --cov=secure_mpc_transformer --cov-fail-under=80
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package
        path: dist/

  build-containers:
    runs-on: ubuntu-latest
    needs: validate-release
    permissions:
      contents: read
      packages: write
    
    strategy:
      matrix:
        variant: [cpu, gpu]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push release image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./docker/Dockerfile.${{ matrix.variant }}
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:${{ github.ref_name }}-${{ matrix.variant }}
          ghcr.io/${{ github.repository }}:latest-${{ matrix.variant }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  publish-pypi:
    runs-on: ubuntu-latest
    needs: [build-and-test]
    environment: release
    permissions:
      id-token: write
    
    steps:
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: python-package
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://upload.pypi.org/legacy/

  create-github-release:
    runs-on: ubuntu-latest
    needs: [build-and-test, build-containers, publish-pypi]
    permissions:
      contents: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Generate release notes
      run: |
        # Extract changelog section for this version
        if [ -f CHANGELOG.md ]; then
          awk '/^## \[${{ github.ref_name }}\]/{flag=1; next} /^## \[/{flag=0} flag' CHANGELOG.md > release-notes.md
        else
          echo "Release ${{ github.ref_name }}" > release-notes.md
          echo "" >> release-notes.md
          echo "Auto-generated release notes:" >> release-notes.md
          git log --pretty=format:"- %s" $(git describe --tags --abbrev=0 HEAD^)..HEAD >> release-notes.md
        fi
    
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: python-package
        path: dist/
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        body_path: release-notes.md
        files: |
          dist/*
        generate_release_notes: true
        make_latest: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  notify-release:
    runs-on: ubuntu-latest
    needs: [create-github-release]
    if: always()
    
    steps:
    - name: Notify release status
      run: |
        if [ "${{ needs.create-github-release.result }}" == "success" ]; then
          echo "✅ Release ${{ github.ref_name }} completed successfully"
        else
          echo "❌ Release ${{ github.ref_name }} failed"
          exit 1
        fi
```

## Implementation Checklist

### 1. Repository Setup
- [ ] Create `.github/workflows/` directory
- [ ] Enable GitHub Actions in repository settings
- [ ] Configure branch protection rules for main branch

### 2. Secrets Configuration
- [ ] Add `GITHUB_TOKEN` (automatically provided)
- [ ] Configure `PYPI_API_TOKEN` for releases
- [ ] Set up container registry credentials

### 3. Self-Hosted Runners (for GPU tests)
- [ ] Set up GPU-enabled self-hosted runner
- [ ] Configure CUDA environment
- [ ] Install required dependencies

### 4. External Integrations
- [ ] Enable Dependabot in repository settings
- [ ] Configure Codecov integration
- [ ] Set up CodeQL security scanning

### 5. Workflow Implementation Priority
1. **Critical**: `ci.yml` (basic testing and quality)
2. **High**: `security.yml` (security scanning)
3. **Medium**: `docker.yml` (container builds)
4. **Medium**: `dependencies.yml` (dependency management)
5. **Low**: `benchmark.yml` (performance tracking)
6. **Low**: `release.yml` (release automation)

### 6. Testing Workflow Implementation
1. Copy template content to appropriate `.github/workflows/` files
2. Customize environment variables and settings
3. Test with a small change on a feature branch
4. Monitor workflow execution and adjust as needed
5. Document any custom modifications

## Maintenance Notes

- Review and update workflow versions quarterly
- Monitor workflow performance and optimize caching
- Update security scanning tools regularly
- Adjust benchmark thresholds based on performance trends
- Keep documentation synchronized with workflow changes