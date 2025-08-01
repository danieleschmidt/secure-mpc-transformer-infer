# Required Manual Setup for GitHub Workflows

Due to GitHub App permission limitations, the workflow files in `docs/workflows/examples/` must be manually created in the `.github/workflows/` directory.

## Quick Setup

1. **Create the workflows directory:**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy workflow templates:**
   ```bash
   cp docs/workflows/examples/*.yml .github/workflows/
   ```

3. **Configure required secrets and settings** (see sections below)

## Required GitHub Repository Settings

### Branch Protection Rules

Configure branch protection for `main` branch:

1. Go to Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable the following:
   - [x] Require a pull request before merging
   - [x] Require approvals (minimum 2)
   - [x] Dismiss stale PR approvals when new commits are pushed
   - [x] Require review from code owners
   - [x] Require status checks to pass before merging
   - [x] Require branches to be up to date before merging
   - [x] Require conversation resolution before merging
   - [x] Include administrators

### Required Status Checks

Add these status checks to branch protection:
- `quality / Code Quality`
- `test-unit / Unit Tests`
- `test-integration / Integration Tests`
- `security / Static Security Analysis`
- `build-docker / Build Docker Images`

## Required Repository Secrets

### Essential Secrets

Configure these secrets in Settings → Secrets and variables → Actions:

```bash
# Docker Registry Access
GHCR_TOKEN                    # GitHub Container Registry token

# Deployment Access
STAGING_KUBECONFIG           # Base64 encoded kubeconfig for staging
PRODUCTION_KUBECONFIG        # Base64 encoded kubeconfig for production

# Dependency Updates
DEPENDENCY_UPDATE_TOKEN      # Personal access token for dependency PRs

# Security Scanning
SECURITY_SCAN_TOKEN          # Token for security scanning services

# Notifications
SLACK_WEBHOOK_URL           # Slack webhook for critical alerts
TEAMS_WEBHOOK_URL           # Microsoft Teams webhook (optional)
```

### Optional Secrets

```bash
# External Services
CODECOV_TOKEN               # Codecov integration token
SONAR_TOKEN                # SonarCloud token
SNYK_TOKEN                 # Snyk security scanning token

# Monitoring
DATADOG_API_KEY            # Datadog monitoring (if used)
NEWRELIC_API_KEY           # New Relic monitoring (if used)
```

## Repository Variables

Configure these variables in Settings → Secrets and variables → Actions → Variables:

```bash
# Container Registry
REGISTRY_URL=ghcr.io
IMAGE_NAME=secure-mpc-transformer

# Deployment
STAGING_URL=https://staging.mpc-transformer.example.com
PRODUCTION_URL=https://mpc-transformer.example.com

# Security
SECURITY_EMAIL=security@mpc-transformer.example.com
```

## Self-Hosted Runners (for GPU Tests)

### Setup GPU Runner

1. **Prepare GPU machine:**
   ```bash
   # Install NVIDIA drivers and Docker
   sudo apt update
   sudo apt install -y nvidia-driver-520 docker.io
   
   # Install NVIDIA Container Toolkit
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Install GitHub Actions Runner:**
   ```bash
   # Create runner user
   sudo useradd -m -s /bin/bash github-runner
   sudo usermod -aG docker github-runner
   
   # Download and configure runner
   sudo -u github-runner bash -c "
   cd /home/github-runner
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.308.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.308.0/actions-runner-linux-x64-2.308.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.308.0.tar.gz
   "
   ```

3. **Configure runner:**
   ```bash
   # Get registration token from GitHub repo settings
   sudo -u github-runner bash -c "
   cd /home/github-runner/actions-runner
   ./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN --labels self-hosted,gpu --work _work
   "
   ```

4. **Install as service:**
   ```bash
   sudo ./svc.sh install github-runner
   sudo ./svc.sh start
   ```

## Environment Configuration

### Staging Environment

1. **Kubernetes cluster setup:**
   ```bash
   # Create namespace
   kubectl create namespace mpc-staging
   
   # Create service account for GitHub Actions
   kubectl create serviceaccount github-actions -n mpc-staging
   kubectl create clusterrolebinding github-actions-binding \
     --clusterrole=cluster-admin \
     --serviceaccount=mpc-staging:github-actions
   ```

2. **Generate kubeconfig:**
   ```bash
   # Get service account token
   SECRET_NAME=$(kubectl get serviceaccount github-actions -n mpc-staging -o jsonpath='{.secrets[0].name}')
   TOKEN=$(kubectl get secret $SECRET_NAME -n mpc-staging -o jsonpath='{.data.token}' | base64 -d)
   
   # Create kubeconfig
   kubectl config set-cluster staging --server=https://your-k8s-api-server
   kubectl config set-credentials github-actions --token=$TOKEN
   kubectl config set-context staging --cluster=staging --user=github-actions --namespace=mpc-staging
   kubectl config use-context staging
   
   # Encode for GitHub secret
   kubectl config view --raw | base64 -w 0
   ```

### Production Environment

Follow similar steps as staging but with production cluster credentials.

## Workflow Customization

### Modify Workflow Files

Each workflow file may need customization for your environment:

1. **Update Docker registry settings:**
   ```yaml
   env:
     REGISTRY: your-registry.com  # Change from ghcr.io if needed
     IMAGE_NAME: your-org/secure-mpc-transformer
   ```

2. **Update deployment URLs:**
   ```yaml
   environment:
     name: staging
     url: https://your-staging-url.com  # Update your URLs
   ```

3. **Configure notification channels:**
   ```yaml
   - name: Notify on failure
     if: failure()
     run: |
       curl -X POST -H 'Content-type: application/json' \
         --data '{"text":"Build failed: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}' \
         ${{ secrets.SLACK_WEBHOOK_URL }}
   ```

### Enable/Disable Workflows

Control which workflows run by modifying the `on:` triggers:

```yaml
# Disable workflow temporarily
on:
  # push:
  #   branches: [ main ]
  workflow_dispatch:  # Manual trigger only
```

## Required GitHub Apps and Integrations

### Recommended Apps

1. **Codecov** - Code coverage reporting
2. **Dependabot** - Automated dependency updates (alternative to custom workflow)
3. **CodeQL** - Advanced security scanning
4. **Renovate** - Dependency management (alternative to Dependabot)

### Security Integrations

1. **Snyk** - Vulnerability scanning
2. **FOSSA** - License compliance
3. **GitGuardian** - Secret scanning
4. **Checkmarx** - SAST scanning

## Troubleshooting

### Common Issues

1. **Workflow permissions:**
   ```bash
   # Ensure GITHUB_TOKEN has sufficient permissions
   permissions:
     contents: read
     packages: write
     security-events: write
   ```

2. **Self-hosted runner connectivity:**
   ```bash
   # Check runner status
   sudo ./svc.sh status
   
   # View runner logs
   sudo journalctl -u actions.runner.* -f
   ```

3. **Kubernetes deployment failures:**
   ```bash
   # Verify kubeconfig
   kubectl cluster-info
   
   # Check deployment status
   kubectl get pods -n mpc-staging
   kubectl describe deployment mpc-transformer-staging -n mpc-staging
   ```

### Getting Help

1. **GitHub Discussions:** Use repository discussions for workflow questions
2. **Issues:** Create issues for workflow bugs or enhancement requests
3. **Documentation:** Refer to GitHub Actions documentation
4. **Security:** Contact security team for security-related workflow issues

## Verification Checklist

After setting up workflows, verify:

- [ ] All required secrets are configured
- [ ] Branch protection rules are active
- [ ] Self-hosted runners are connected and healthy
- [ ] Staging environment is accessible
- [ ] Production environment is properly secured
- [ ] Notification channels are working
- [ ] Docker registry access is configured
- [ ] Security scanning is enabled
- [ ] Dependency updates are automated

## Maintenance

### Regular Tasks

1. **Monthly:**
   - Review and rotate access tokens
   - Update self-hosted runner software
   - Review security scan results

2. **Quarterly:**
   - Review workflow performance and optimization
   - Update workflow dependencies
   - Audit permissions and access

3. **Annually:**
   - Complete security audit of CI/CD pipeline
   - Review and update workflow documentation
   - Benchmark workflow performance