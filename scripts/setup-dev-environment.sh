#\!/bin/bash
# Development environment setup script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

main() {
    log_info "Setting up development environment..."
    
    # Install dependencies
    if [[ -f "pyproject.toml" ]]; then
        pip install -e ".[dev,benchmark]"
        log_success "Python dependencies installed"
    fi
    
    # Setup pre-commit
    if [[ -f ".pre-commit-config.yaml" ]]; then
        pre-commit install
        log_success "Pre-commit hooks installed"
    fi
    
    log_success "Development environment setup completed\!"
}

main "$@"
EOF < /dev/null
