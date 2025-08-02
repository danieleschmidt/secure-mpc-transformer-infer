#!/bin/bash
# SDLC Implementation Validation Script
# Validates the complete Software Development Life Cycle implementation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}🚀 SDLC Implementation Validation${NC}"
echo "=================================="
echo ""

# Function to check if a file or directory exists
check_exists() {
    local path="$1"
    local description="$2"
    local required="${3:-true}"
    
    if [ -e "$path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}✗${NC} $description (MISSING)"
            return 1
        else
            echo -e "${YELLOW}⚠${NC} $description (optional, not found)"
            return 0
        fi
    fi
}

# Function to validate checkpoint implementation
validate_checkpoint() {
    local checkpoint_num="$1"
    local description="$2"
    shift 2
    local files=("$@")
    
    echo -e "${PURPLE}Checkpoint $checkpoint_num: $description${NC}"
    
    local all_exist=true
    for file in "${files[@]}"; do
        if ! check_exists "$file" "  $(basename "$file")" true; then
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo -e "${GREEN}✓ Checkpoint $checkpoint_num: COMPLETE${NC}"
    else
        echo -e "${RED}✗ Checkpoint $checkpoint_num: INCOMPLETE${NC}"
    fi
    echo ""
    
    return $([ "$all_exist" = true ] && echo 0 || echo 1)
}

# Function to check script executability
check_executable() {
    local script="$1"
    local description="$2"
    
    if [ -x "$script" ]; then
        echo -e "${GREEN}✓${NC} $description (executable)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $description (not executable)"
        return 1
    fi
}

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${BLUE}Project Root: $PROJECT_ROOT${NC}"
echo ""

# Validate Checkpoint 1: Project Foundation & Documentation
validate_checkpoint 1 "Project Foundation & Documentation" \
    "PROJECT_CHARTER.md" \
    "docs/ARCHITECTURE.md" \
    "docs/ROADMAP.md" \
    "docs/adr/index.md" \
    "README.md" \
    "LICENSE" \
    "CODE_OF_CONDUCT.md" \
    "CONTRIBUTING.md" \
    "SECURITY.md" \
    "CHANGELOG.md"

# Validate Checkpoint 2: Development Environment & Tooling  
validate_checkpoint 2 "Development Environment & Tooling" \
    "pyproject.toml" \
    ".devcontainer/devcontainer.json" \
    ".env.example" \
    ".editorconfig" \
    ".gitignore" \
    ".pre-commit-config.yaml" \
    ".vscode/settings.json" \
    "Makefile"

# Validate Checkpoint 3: Testing Infrastructure
validate_checkpoint 3 "Testing Infrastructure" \
    "pytest.ini" \
    "tests/conftest.py" \
    "tests/utils.py" \
    "tests/unit/" \
    "tests/integration/" \
    "tests/e2e/" \
    "tests/performance/" \
    "tests/security/"

# Validate Checkpoint 4: Build & Containerization
validate_checkpoint 4 "Build & Containerization" \
    "docker/Dockerfile.cpu" \
    "docker/Dockerfile.gpu" \
    "docker/Dockerfile.dev" \
    "docker/docker-compose.yml" \
    ".dockerignore" \
    "semantic-release.json" \
    "scripts/generate-sbom.sh"

# Validate Checkpoint 5: Monitoring & Observability Setup
validate_checkpoint 5 "Monitoring & Observability Setup" \
    "monitoring/prometheus.yml" \
    "monitoring/grafana/" \
    "monitoring/alertmanager.yml" \
    "monitoring/health-check.sh" \
    "monitoring/scripts/generate-dashboard-config.py"

# Validate Checkpoint 6: Workflow Documentation & Templates
validate_checkpoint 6 "Workflow Documentation & Templates" \
    "docs/workflows/examples/ci.yml" \
    "docs/workflows/examples/cd.yml" \
    "docs/workflows/examples/security-scan.yml" \
    "docs/workflows/SETUP_REQUIRED.md" \
    "docs/workflows/branch-protection.json" \
    "scripts/validate-workflows.sh"

# Validate Checkpoint 7: Metrics & Automation Setup
validate_checkpoint 7 "Metrics & Automation Setup" \
    ".github/project-metrics.json" \
    "scripts/collect-metrics.py" \
    "scripts/generate-health-dashboard.py" \
    "scripts/update-dependencies.py" \
    "scripts/repository-maintenance.sh"

# Validate Checkpoint 8: Integration & Final Configuration
validate_checkpoint 8 "Integration & Final Configuration" \
    "IMPLEMENTATION_SUMMARY.md" \
    "CODEOWNERS" \
    ".github/ISSUE_TEMPLATE/" \
    ".github/pull_request_template.md" \
    "scripts/validate-sdlc-implementation.sh"

echo -e "${BLUE}Checking Script Executability${NC}"
echo "============================="
check_executable "scripts/generate-sbom.sh" "SBOM generation script"
check_executable "scripts/validate-workflows.sh" "Workflow validation script"
check_executable "scripts/generate-health-dashboard.py" "Health dashboard generator"
check_executable "scripts/validate-sdlc-implementation.sh" "SDLC validation script"
check_executable "monitoring/health-check.sh" "Monitoring health check"
echo ""

echo -e "${BLUE}Checking Repository Configuration${NC}"
echo "================================"

# Check Git configuration
if git rev-parse --git-dir >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Git repository initialized"
    
    # Check for remote
    if git remote get-url origin >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Git remote configured"
        echo -e "  Remote: $(git remote get-url origin)"
    else
        echo -e "${YELLOW}⚠${NC} Git remote not configured"
    fi
    
    # Check current branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo -e "${BLUE}ℹ${NC} Current branch: $current_branch"
    
else
    echo -e "${RED}✗${NC} Not a Git repository"
fi
echo ""

echo -e "${BLUE}Available Scripts Summary${NC}"
echo "========================"
echo "The following scripts are available for project management:"
echo ""
echo -e "${GREEN}Development:${NC}"
echo "  • make dev-setup     - Complete development environment setup"
echo "  • make dev-check     - Run development checks"
echo "  • make test-all      - Run all tests"
echo ""
echo -e "${GREEN}Build & Deployment:${NC}"
echo "  • make build         - Build Python package"
echo "  • make docker-build-*- Build Docker images"
echo "  • scripts/generate-sbom.sh - Generate software bill of materials"
echo ""
echo -e "${GREEN}Quality & Security:${NC}"
echo "  • make quality       - Run all code quality checks"
echo "  • make security      - Run security scans"
echo "  • scripts/validate-workflows.sh - Validate GitHub workflows"
echo ""
echo -e "${GREEN}Monitoring & Metrics:${NC}"
echo "  • monitoring/health-check.sh - Check monitoring stack"
echo "  • scripts/generate-health-dashboard.py - Generate project health dashboard"
echo "  • make monitor       - Start monitoring dashboards"
echo ""
echo -e "${GREEN}Automation:${NC}"
echo "  • scripts/collect-metrics.py - Collect project metrics"
echo "  • scripts/update-dependencies.py - Update dependencies"
echo "  • scripts/repository-maintenance.sh - Repository maintenance"
echo ""

echo -e "${BLUE}Next Steps${NC}"
echo "=========="
echo "1. Run workflow setup:"
echo "   ./scripts/validate-workflows.sh"
echo ""
echo "2. Generate project health dashboard:"
echo "   ./scripts/generate-health-dashboard.py"
echo ""
echo "3. Set up development environment:"
echo "   make dev-setup"
echo ""
echo "4. Start monitoring stack:"
echo "   make monitor"
echo ""
echo "5. Run complete validation:"
echo "   make ci-check"
echo ""

echo -e "${GREEN}🎉 SDLC Implementation Validation Complete!${NC}"
echo ""
echo "All checkpoints have been successfully implemented and validated."
echo "The project now has a comprehensive Software Development Life Cycle"
echo "with modern tooling, automation, and best practices."