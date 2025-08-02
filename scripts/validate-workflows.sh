#!/bin/bash
# Workflow validation script for Secure MPC Transformer
# Validates GitHub Actions workflow configurations and repository setup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKFLOWS_DIR=".github/workflows"
EXAMPLES_DIR="docs/workflows/examples"
REQUIRED_WORKFLOWS=("ci.yml" "cd.yml" "security-scan.yml" "dependency-update.yml")

echo -e "${BLUE}GitHub Workflows Validation${NC}"
echo "============================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check workflow file syntax
validate_workflow_syntax() {
    local workflow_file="$1"
    local workflow_name=$(basename "$workflow_file" .yml)
    
    echo -e "${YELLOW}Validating $workflow_name workflow syntax...${NC}"
    
    # Check if file exists
    if [ ! -f "$workflow_file" ]; then
        echo -e "${RED}✗ Workflow file not found: $workflow_file${NC}"
        return 1
    fi
    
    # Validate YAML syntax
    if command_exists yq; then
        if yq eval '.' "$workflow_file" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Valid YAML syntax${NC}"
        else
            echo -e "${RED}✗ Invalid YAML syntax in $workflow_file${NC}"
            return 1
        fi
    elif command_exists python3; then
        if python3 -c "
import yaml
import sys
try:
    with open('$workflow_file', 'r') as f:
        yaml.safe_load(f)
    print('✓ Valid YAML syntax')
except yaml.YAMLError as e:
    print(f'✗ Invalid YAML syntax: {e}')
    sys.exit(1)
"; then
            echo -e "${GREEN}✓ Valid YAML syntax${NC}"
        else
            echo -e "${RED}✗ Invalid YAML syntax in $workflow_file${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ Cannot validate YAML syntax (yq or python3 with PyYAML required)${NC}"
    fi
    
    # Check required workflow structure
    if grep -q "name:" "$workflow_file" && \
       grep -q "on:" "$workflow_file" && \
       grep -q "jobs:" "$workflow_file"; then
        echo -e "${GREEN}✓ Valid workflow structure${NC}"
    else
        echo -e "${RED}✗ Missing required workflow sections (name, on, jobs)${NC}"
        return 1
    fi
    
    return 0
}

# Function to check if workflows are properly set up
check_workflows_setup() {
    echo -e "${YELLOW}Checking workflows setup...${NC}"
    
    # Check if workflows directory exists
    if [ ! -d "$WORKFLOWS_DIR" ]; then
        echo -e "${RED}✗ Workflows directory not found: $WORKFLOWS_DIR${NC}"
        echo -e "${BLUE}  Run: mkdir -p $WORKFLOWS_DIR${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Workflows directory exists${NC}"
    
    # Check each required workflow
    local missing_workflows=()
    for workflow in "${REQUIRED_WORKFLOWS[@]}"; do
        if [ -f "$WORKFLOWS_DIR/$workflow" ]; then
            echo -e "${GREEN}✓ Found $workflow${NC}"
            if ! validate_workflow_syntax "$WORKFLOWS_DIR/$workflow"; then
                echo -e "${RED}✗ $workflow has validation errors${NC}"
            fi
        else
            echo -e "${RED}✗ Missing $workflow${NC}"
            missing_workflows+=("$workflow")
        fi
    done
    
    # Suggest copying from examples if workflows are missing
    if [ ${#missing_workflows[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Missing workflows detected. To copy from examples:${NC}"
        for workflow in "${missing_workflows[@]}"; do
            if [ -f "$EXAMPLES_DIR/$workflow" ]; then
                echo -e "${BLUE}  cp $EXAMPLES_DIR/$workflow $WORKFLOWS_DIR/$workflow${NC}"
            else
                echo -e "${RED}  Example not found: $EXAMPLES_DIR/$workflow${NC}"
            fi
        done
        return 1
    fi
    
    return 0
}

# Function to check GitHub repository configuration
check_github_config() {
    echo -e "${YELLOW}Checking GitHub repository configuration...${NC}"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}✗ Not a git repository${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Git repository detected${NC}"
    
    # Check for GitHub remote
    if git remote get-url origin | grep -q github.com; then
        echo -e "${GREEN}✓ GitHub remote detected${NC}"
        local repo_url=$(git remote get-url origin)
        echo -e "${BLUE}  Repository: $repo_url${NC}"
    else
        echo -e "${YELLOW}⚠ GitHub remote not detected${NC}"
    fi
    
    # Check if gh CLI is available for advanced checks
    if command_exists gh; then
        echo -e "${YELLOW}Checking repository settings with GitHub CLI...${NC}"
        
        # Check if authenticated
        if gh auth status > /dev/null 2>&1; then
            echo -e "${GREEN}✓ GitHub CLI authenticated${NC}"
            
            # Check repository settings
            local repo_info=$(gh repo view --json name,isPrivate,hasIssuesEnabled,hasWikiEnabled,visibility 2>/dev/null || echo "{}")
            if [ "$repo_info" != "{}" ]; then
                echo -e "${GREEN}✓ Repository accessible via GitHub API${NC}"
                
                # Extract specific settings
                local has_issues=$(echo "$repo_info" | grep -o '"hasIssuesEnabled":[^,]*' | cut -d: -f2)
                if [ "$has_issues" = "true" ]; then
                    echo -e "${GREEN}✓ Issues enabled${NC}"
                else
                    echo -e "${YELLOW}⚠ Issues not enabled${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ Cannot access repository via GitHub API${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ GitHub CLI not authenticated${NC}"
            echo -e "${BLUE}  Run: gh auth login${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ GitHub CLI not available for advanced checks${NC}"
        echo -e "${BLUE}  Install: https://cli.github.com/${NC}"
    fi
}

# Function to check required secrets and environment variables
check_secrets_config() {
    echo -e "${YELLOW}Checking secrets configuration...${NC}"
    
    # List of required secrets
    local required_secrets=(
        "CODECOV_TOKEN"
        "DOCKER_HUB_USERNAME"
        "DOCKER_HUB_TOKEN"
        "SNYK_TOKEN"
    )
    
    echo -e "${BLUE}Required repository secrets:${NC}"
    for secret in "${required_secrets[@]}"; do
        echo -e "${BLUE}  - $secret${NC}"
    done
    
    echo ""
    echo -e "${YELLOW}To configure secrets:${NC}"
    echo -e "${BLUE}1. Go to Settings → Secrets and variables → Actions${NC}"
    echo -e "${BLUE}2. Click 'New repository secret'${NC}"
    echo -e "${BLUE}3. Add each required secret${NC}"
    
    # Check if GitHub CLI can verify secrets
    if command_exists gh && gh auth status > /dev/null 2>&1; then
        echo ""
        echo -e "${YELLOW}Checking configured secrets...${NC}"
        
        if gh secret list > /dev/null 2>&1; then
            local configured_secrets=$(gh secret list --json name -q '.[].name' 2>/dev/null || echo "")
            
            for secret in "${required_secrets[@]}"; do
                if echo "$configured_secrets" | grep -q "^$secret$"; then
                    echo -e "${GREEN}✓ $secret configured${NC}"
                else
                    echo -e "${RED}✗ $secret not configured${NC}"
                fi
            done
        else
            echo -e "${YELLOW}⚠ Cannot access repository secrets${NC}"
        fi
    fi
}

# Function to generate workflow setup commands
generate_setup_commands() {
    echo -e "${YELLOW}Setup Commands Summary${NC}"
    echo "====================="
    
    echo ""
    echo -e "${BLUE}1. Copy workflow files:${NC}"
    echo "   mkdir -p .github/workflows"
    echo "   cp docs/workflows/examples/*.yml .github/workflows/"
    
    echo ""
    echo -e "${BLUE}2. Install validation tools:${NC}"
    echo "   # For YAML validation"
    echo "   pip install PyYAML"
    echo "   # OR"
    echo "   brew install yq"
    
    echo ""
    echo -e "${BLUE}3. Install GitHub CLI:${NC}"
    echo "   # macOS"
    echo "   brew install gh"
    echo "   # Ubuntu/Debian"
    echo "   sudo apt install gh"
    echo "   # Then authenticate"
    echo "   gh auth login"
    
    echo ""
    echo -e "${BLUE}4. Configure repository settings:${NC}"
    echo "   gh repo edit --enable-issues --enable-wiki"
    echo "   gh api repos/:owner/:repo --method PATCH -f allow_merge_commit=false"
    
    echo ""
    echo -e "${BLUE}5. Set up branch protection:${NC}"
    echo "   gh api repos/:owner/:repo/branches/main/protection \\"
    echo "     --method PUT \\"
    echo "     --input docs/workflows/branch-protection.json"
}

# Main execution
main() {
    local exit_code=0
    
    echo -e "${BLUE}Starting workflow validation...${NC}"
    echo ""
    
    # Run all checks
    if ! check_workflows_setup; then
        exit_code=1
    fi
    
    echo ""
    if ! check_github_config; then
        exit_code=1
    fi
    
    echo ""
    check_secrets_config
    
    echo ""
    generate_setup_commands
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ All validations passed!${NC}"
    else
        echo -e "${RED}✗ Some validations failed. Please review the output above.${NC}"
    fi
    
    return $exit_code
}

# Execute main function
main "$@"