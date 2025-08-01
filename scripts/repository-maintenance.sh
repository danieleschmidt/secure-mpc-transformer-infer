#!/bin/bash
# Repository maintenance automation script
# Performs regular maintenance tasks for the Secure MPC Transformer project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT=$(git rev-parse --show-toplevel)
BACKUP_DIR="$REPO_ROOT/.maintenance/backups"
LOG_FILE="$REPO_ROOT/.maintenance/maintenance.log"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

setup_maintenance_dir() {
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Initialize log file
    echo "=== Repository Maintenance Log ===" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

cleanup_temporary_files() {
    log_info "Cleaning up temporary files..."
    
    # Clean Python cache files
    find "$REPO_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$REPO_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$REPO_ROOT" -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Clean test artifacts
    rm -rf "$REPO_ROOT"/.pytest_cache 2>/dev/null || true
    rm -rf "$REPO_ROOT"/.coverage 2>/dev/null || true
    rm -rf "$REPO_ROOT"/htmlcov 2>/dev/null || true
    rm -rf "$REPO_ROOT"/.mypy_cache 2>/dev/null || true
    rm -rf "$REPO_ROOT"/.ruff_cache 2>/dev/null || true
    
    # Clean build artifacts
    rm -rf "$REPO_ROOT"/build 2>/dev/null || true
    rm -rf "$REPO_ROOT"/dist 2>/dev/null || true
    find "$REPO_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    
    # Clean temporary files
    find "$REPO_ROOT" -type f -name "*.tmp" -delete 2>/dev/null || true
    find "$REPO_ROOT" -type f -name "*.temp" -delete 2>/dev/null || true
    find "$REPO_ROOT" -type f -name "*~" -delete 2>/dev/null || true
    
    # Clean log files older than 30 days
    find "$REPO_ROOT" -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true
    
    log_success "Temporary files cleaned up"
}

update_dependencies() {
    log_info "Checking for dependency updates..."
    
    # Check Python dependencies
    if command -v pip-audit &> /dev/null; then
        log_info "Running pip-audit for security vulnerabilities..."
        pip-audit --desc --output audit-results.json --format=json 2>/dev/null || log_warning "pip-audit failed"
    fi
    
    # Check for outdated packages
    if command -v pip &> /dev/null; then
        log_info "Checking for outdated Python packages..."
        pip list --outdated --format=json > outdated-packages.json 2>/dev/null || log_warning "Could not check outdated packages"
        
        if [ -s outdated-packages.json ]; then
            OUTDATED_COUNT=$(jq '. | length' outdated-packages.json 2>/dev/null || echo "0")
            if [ "$OUTDATED_COUNT" -gt 0 ]; then
                log_warning "$OUTDATED_COUNT outdated packages found"
            else
                log_success "All packages are up to date"
            fi
        fi
        
        rm -f outdated-packages.json
    fi
    
    # Update pre-commit hooks
    if [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
        log_info "Updating pre-commit hooks..."
        pre-commit autoupdate 2>/dev/null || log_warning "pre-commit autoupdate failed"
    fi
}

check_security() {
    log_info "Running security checks..."
    
    # Check for secrets
    if command -v detect-secrets &> /dev/null; then
        log_info "Scanning for secrets..."
        detect-secrets scan --all-files --baseline .secrets.baseline \
            --exclude-files='\.git/.*' \
            --exclude-files='.*\.pyc' \
            --exclude-files='.*\.log' \
            > secrets-scan.json 2>/dev/null || log_warning "Secret scan failed"
        
        if [ -f secrets-scan.json ]; then
            NEW_SECRETS=$(jq '.results | length' secrets-scan.json 2>/dev/null || echo "0")
            if [ "$NEW_SECRETS" -gt 0 ]; then
                log_warning "$NEW_SECRETS potential secrets found"
            else
                log_success "No new secrets detected"
            fi
            rm -f secrets-scan.json
        fi
    fi
    
    # Check file permissions
    log_info "Checking file permissions..."
    
    # Find files with overly permissive permissions
    PERMISSIVE_FILES=$(find "$REPO_ROOT" -type f -perm /o+w ! -path "*/.git/*" | wc -l)
    if [ "$PERMISSIVE_FILES" -gt 0 ]; then
        log_warning "$PERMISSIVE_FILES files are world-writable"
    else
        log_success "File permissions are secure"
    fi
    
    # Check for executable files that shouldn't be
    SUSPICIOUS_EXECUTABLES=$(find "$REPO_ROOT" -name "*.py" -perm +111 ! -path "*/.git/*" ! -path "*/scripts/*" | wc -l)
    if [ "$SUSPICIOUS_EXECUTABLES" -gt 0 ]; then
        log_warning "$SUSPICIOUS_EXECUTABLES Python files are executable (may be intentional)"
    fi
}

optimize_git_repository() {
    log_info "Optimizing Git repository..."
    
    # Git garbage collection
    git gc --auto --prune=now 2>/dev/null || log_warning "Git garbage collection failed"
    
    # Clean up remote tracking branches
    git remote prune origin 2>/dev/null || log_warning "Remote prune failed"
    
    # Count objects before and after
    OBJECTS_BEFORE=$(git count-objects -v | grep "count " | awk '{print $2}')
    git repack -ad 2>/dev/null || log_warning "Git repack failed"
    OBJECTS_AFTER=$(git count-objects -v | grep "count " | awk '{print $2}')
    
    if [ "$OBJECTS_BEFORE" -gt "$OBJECTS_AFTER" ]; then
        SAVED=$((OBJECTS_BEFORE - OBJECTS_AFTER))
        log_success "Git optimization complete. Saved $SAVED objects"
    else
        log_success "Git repository is already optimized"
    fi
    
    # Check repository size
    REPO_SIZE=$(du -sh "$REPO_ROOT/.git" | cut -f1)
    log_info "Repository size: $REPO_SIZE"
}

validate_configuration() {
    log_info "Validating configuration files..."
    
    # Validate JSON files
    JSON_FILES=$(find "$REPO_ROOT" -name "*.json" ! -path "*/.git/*" ! -path "*/node_modules/*")
    for json_file in $JSON_FILES; do
        if ! jq empty < "$json_file" 2>/dev/null; then
            log_error "Invalid JSON: $json_file"
        fi
    done
    
    # Validate YAML files
    YAML_FILES=$(find "$REPO_ROOT" -name "*.yml" -o -name "*.yaml" ! -path "*/.git/*")
    for yaml_file in $YAML_FILES; do
        if command -v yamllint &> /dev/null; then
            if ! yamllint -d relaxed "$yaml_file" 2>/dev/null; then
                log_warning "YAML issues in: $yaml_file"
            fi
        fi
    done
    
    # Validate Python syntax
    PYTHON_FILES=$(find "$REPO_ROOT/src" -name "*.py" 2>/dev/null || true)
    for py_file in $PYTHON_FILES; do
        if ! python -m py_compile "$py_file" 2>/dev/null; then
            log_error "Python syntax error: $py_file"
        fi
    done
    
    # Check required files exist
    REQUIRED_FILES=(
        "pyproject.toml"
        "README.md"
        "LICENSE"
        ".gitignore"
        ".pre-commit-config.yaml"
    )
    
    for required_file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$REPO_ROOT/$required_file" ]; then
            log_warning "Missing required file: $required_file"
        fi
    done
    
    log_success "Configuration validation complete"
}

backup_important_files() {
    log_info "Creating backup of important files..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup configuration files
    CONFIG_FILES=(
        "pyproject.toml"
        ".pre-commit-config.yaml"
        ".github/project-metrics.json"
        "monitoring/prometheus.yml"
        "monitoring/alertmanager.yml"
        "docker/docker-compose.yml"
    )
    
    for config_file in "${CONFIG_FILES[@]}"; do
        if [ -f "$REPO_ROOT/$config_file" ]; then
            mkdir -p "$BACKUP_PATH/$(dirname "$config_file")"
            cp "$REPO_ROOT/$config_file" "$BACKUP_PATH/$config_file"
        fi
    done
    
    # Compress backup
    tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "backup_$BACKUP_TIMESTAMP"
    rm -rf "$BACKUP_PATH"
    
    # Clean old backups (keep last 10)
    ls -t "$BACKUP_DIR"/backup_*.tar.gz | tail -n +11 | xargs rm -f 2>/dev/null || true
    
    log_success "Backup created: backup_$BACKUP_TIMESTAMP.tar.gz"
}

generate_health_report() {
    log_info "Generating repository health report..."
    
    HEALTH_REPORT="$REPO_ROOT/.maintenance/health-report.md"
    
    cat << EOF > "$HEALTH_REPORT"
# Repository Health Report

**Generated:** $(date)
**Repository:** $(git remote get-url origin 2>/dev/null || echo "Unknown")
**Branch:** $(git branch --show-current)
**Last Commit:** $(git log -1 --format="%h %s" 2>/dev/null || echo "Unknown")

## Repository Statistics

- **Total Files:** $(find "$REPO_ROOT" -type f ! -path "*/.git/*" | wc -l)
- **Python Files:** $(find "$REPO_ROOT" -name "*.py" ! -path "*/.git/*" | wc -l)
- **Documentation Files:** $(find "$REPO_ROOT" -name "*.md" ! -path "*/.git/*" | wc -l)
- **Test Files:** $(find "$REPO_ROOT" -path "*/tests/*" -name "*.py" | wc -l)
- **Repository Size:** $(du -sh "$REPO_ROOT" | cut -f1)
- **Git Objects:** $(git count-objects -v | grep "count " | awk '{print $2}')

## Recent Activity

### Last 10 Commits
EOF
    
    git log --oneline -10 >> "$HEALTH_REPORT" 2>/dev/null || echo "No commit history available" >> "$HEALTH_REPORT"
    
    cat << EOF >> "$HEALTH_REPORT"

### Branch Information
- **Current Branch:** $(git branch --show-current)
- **Total Branches:** $(git branch -a | wc -l)
- **Tracking Branches:** $(git branch -r | wc -l)

### Configuration Status
- **Pre-commit Hooks:** $([ -f .pre-commit-config.yaml ] && echo "✅ Configured" || echo "❌ Missing")
- **CI/CD Workflows:** $([ -d .github/workflows ] && echo "✅ Present" || echo "❌ Missing")
- **Docker Configuration:** $([ -f docker/docker-compose.yml ] && echo "✅ Present" || echo "❌ Missing")
- **Monitoring Setup:** $([ -f monitoring/prometheus.yml ] && echo "✅ Present" || echo "❌ Missing")

## Maintenance Tasks Completed

- [x] Temporary files cleaned
- [x] Dependencies checked
- [x] Security scan performed
- [x] Git repository optimized
- [x] Configuration validated
- [x] Backup created
- [x] Health report generated

## Recommendations

EOF
    
    # Add recommendations based on findings
    if [ -f outdated-packages.json ] && [ -s outdated-packages.json ]; then
        echo "- 🔄 Update outdated Python packages" >> "$HEALTH_REPORT"
    fi
    
    if [ ! -f "$REPO_ROOT/.github/workflows/ci.yml" ]; then
        echo "- ⚙️ Setup CI/CD workflows (see docs/workflows/SETUP_REQUIRED.md)" >> "$HEALTH_REPORT"
    fi
    
    if [ ! -d "tests" ]; then
        echo "- 🧪 Add comprehensive test suite" >> "$HEALTH_REPORT"
    fi
    
    echo "- 📊 Regular metrics collection recommended (weekly)" >> "$HEALTH_REPORT"
    echo "- 🚨 Review security scan results regularly" >> "$HEALTH_REPORT"
    echo "- 📋 Keep documentation up to date" >> "$HEALTH_REPORT"
    
    log_success "Health report generated: $HEALTH_REPORT"
}

update_metrics() {
    log_info "Updating project metrics..."
    
    if [ -f "$REPO_ROOT/scripts/collect-metrics.py" ]; then
        python "$REPO_ROOT/scripts/collect-metrics.py" --all --report --output "$REPO_ROOT/.maintenance/metrics-report.md" 2>/dev/null || log_warning "Metrics collection failed"
    else
        log_warning "Metrics collection script not found"
    fi
}

show_summary() {
    echo ""
    log_info "=== MAINTENANCE SUMMARY ==="
    
    # Count warnings and errors from log
    WARNINGS=$(grep -c "\[WARNING\]" "$LOG_FILE" 2>/dev/null || echo "0")
    ERRORS=$(grep -c "\[ERROR\]" "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo "Warnings: $WARNINGS"
    echo "Errors: $ERRORS"
    echo ""
    
    if [ "$ERRORS" -gt 0 ]; then
        echo "❌ Maintenance completed with errors. Review the log file:"
        echo "   $LOG_FILE"
        return 1
    elif [ "$WARNINGS" -gt 0 ]; then
        echo "⚠️  Maintenance completed with warnings. Review the log file:"
        echo "   $LOG_FILE"
        return 0
    else
        echo "✅ Maintenance completed successfully!"
        return 0
    fi
}

main() {
    echo "🔧 Starting repository maintenance..."
    echo "Repository: $REPO_ROOT"
    echo ""
    
    # Setup
    setup_maintenance_dir
    
    # Run maintenance tasks
    cleanup_temporary_files
    update_dependencies
    check_security
    optimize_git_repository
    validate_configuration
    backup_important_files
    update_metrics
    generate_health_report
    
    # Show summary
    show_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean-only)
            cleanup_temporary_files
            exit 0
            ;;
        --security-only)
            setup_maintenance_dir
            check_security
            show_summary
            exit $?
            ;;
        --backup-only)
            setup_maintenance_dir
            backup_important_files
            exit 0
            ;;
        --help)
            cat << EOF
Repository Maintenance Script

Usage: $0 [OPTIONS]

Options:
    --clean-only      Only clean temporary files
    --security-only   Only run security checks
    --backup-only     Only create backup
    --help           Show this help

Default: Run all maintenance tasks
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Run maintenance if no specific option provided
main