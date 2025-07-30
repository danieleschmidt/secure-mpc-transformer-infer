#!/bin/bash
set -euo pipefail

# Secure MPC Transformer - Development Environment Setup
# This script sets up a complete development environment with security best practices

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.10"
VENV_NAME="secure-mpc-dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check if running with appropriate privileges
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        error_exit "This script should not be run as root for security reasons"
    fi
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get >/dev/null 2>&1; then
            DISTRO="debian"
        elif command -v yum >/dev/null 2>&1; then
            DISTRO="redhat"
        else
            error_exit "Unsupported Linux distribution"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error_exit "Unsupported operating system: $OSTYPE"
    fi
    log_info "Detected OS: $OS"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available memory (minimum 8GB recommended)
    if [[ "$OS" == "linux" ]]; then
        MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
    elif [[ "$OS" == "macos" ]]; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    fi
    
    if [[ $MEMORY_GB -lt 8 ]]; then
        log_warning "System has ${MEMORY_GB}GB RAM. Minimum 8GB recommended for MPC operations"
    fi
    
    # Check disk space (minimum 10GB)
    DISK_SPACE_GB=$(df "$PROJECT_ROOT" | awk 'NR==2{print int($4/1024/1024)}')
    if [[ $DISK_SPACE_GB -lt 10 ]]; then
        error_exit "Insufficient disk space. Need at least 10GB available, found ${DISK_SPACE_GB}GB"
    fi
    
    log_success "System requirements check passed"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        if [[ "$DISTRO" == "debian" ]]; then
            sudo apt-get update
            sudo apt-get install -y \
                python3.10 \
                python3.10-dev \
                python3.10-venv \
                python3-pip \
                build-essential \
                libssl-dev \
                libffi-dev \
                libgmp-dev \
                libmpfr-dev \
                libmpc-dev \
                libnuma-dev \
                git \
                curl \
                wget \
                gnupg \
                software-properties-common
        elif [[ "$DISTRO" == "redhat" ]]; then
            sudo yum update -y
            sudo yum install -y \
                python310 \
                python310-devel \
                gcc \
                gcc-c++ \
                make \
                openssl-devel \
                libffi-devel \
                gmp-devel \
                mpfr-devel \
                libmpc-devel \
                numactl-devel \
                git \
                curl \
                wget \
                gnupg
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew >/dev/null 2>&1; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew update
        brew install \
            python@3.10 \
            gmp \
            mpfr \
            libmpc \
            openssl \
            libffi \
            git \
            curl \
            wget \
            gnupg
    fi
    
    log_success "System dependencies installed"
}

# Setup Python virtual environment
setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment
    python3.10 -m venv "$VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip and setuptools
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [[ -f "pyproject.toml" ]]; then
        pip install -e ".[dev,benchmark]"
    else
        error_exit "pyproject.toml not found. Are you in the correct directory?"
    fi
    
    log_success "Python environment setup complete"
}

# Install cryptographic libraries
install_crypto_libraries() {
    log_info "Installing cryptographic libraries..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    
    # Install Microsoft SEAL (if available)
    if [[ "$OS" == "linux" ]]; then
        # Try to install from system packages first
        if [[ "$DISTRO" == "debian" ]]; then
            if apt-cache search libseal-dev | grep -q libseal-dev; then
                sudo apt-get install -y libseal-dev
                log_success "Microsoft SEAL installed from system packages"
            else
                log_warning "Microsoft SEAL not available in system packages. Will use Python bindings only."
            fi
        fi
    fi
    
    # Install TenSEAL and other crypto libraries
    pip install tenseal pycryptodome cryptography
    
    # Verify installation
    python -c "import tenseal; print(f'TenSEAL version: {tenseal.__version__}')" || \
        log_warning "TenSEAL installation may have issues"
    
    log_success "Cryptographic libraries installed"
}

# Setup GPU support (optional)
setup_gpu_support() {
    log_info "Setting up GPU support (optional)..."
    
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "NVIDIA GPU detected, installing CUDA support..."
        
        # Activate virtual environment
        source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
        
        # Install CUDA-enabled packages
        pip install cupy-cuda12x torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        # Verify GPU installation
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        python -c "import cupy; print(f'CuPy version: {cupy.__version__}')" 2>/dev/null || \
            log_warning "CuPy installation may have issues"
        
        log_success "GPU support configured"
    else
        log_info "No NVIDIA GPU detected, skipping GPU setup"
        
        # Install CPU-only versions
        source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
}

# Setup development tools
setup_development_tools() {
    log_info "Setting up development tools..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    
    # Install pre-commit hooks
    pre-commit install
    pre-commit install --hook-type commit-msg
    
    # Setup secrets detection baseline
    if [[ ! -f "$PROJECT_ROOT/.secrets.baseline" ]]; then
        detect-secrets scan --baseline .secrets.baseline
    fi
    
    # Create development configuration
    mkdir -p "$PROJECT_ROOT/config/dev"
    
    if [[ ! -f "$PROJECT_ROOT/config/dev/local.env" ]]; then
        cat > "$PROJECT_ROOT/config/dev/local.env" << EOF
# Development environment configuration
MPC_PROTOCOL=semi_honest_3pc
SECURITY_LEVEL=128
GPU_ENABLED=auto
LOG_LEVEL=DEBUG
REDIS_URL=redis://localhost:6379
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF
        log_info "Created development configuration at config/dev/local.env"
    fi
    
    log_success "Development tools configured"
}

# Setup Docker environment
setup_docker_environment() {
    log_info "Setting up Docker environment..."
    
    if ! command -v docker >/dev/null 2>&1; then
        log_warning "Docker not found. Please install Docker manually."
        return
    fi
    
    # Build development Docker image
    cd "$PROJECT_ROOT"
    docker build -f docker/Dockerfile.dev -t secure-mpc:dev .
    
    # Setup docker-compose for development
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f docker/docker-compose.yml build
        log_success "Docker development environment ready"
    else
        log_warning "docker-compose not found. Please install docker-compose for full development setup."
    fi
}

# Setup monitoring stack
setup_monitoring() {
    log_info "Setting up monitoring stack..."
    
    if command -v docker-compose >/dev/null 2>&1; then
        cd "$PROJECT_ROOT"
        docker-compose -f monitoring/docker-compose.monitoring.yml up -d
        
        log_info "Monitoring stack started:"
        log_info "  - Prometheus: http://localhost:9090"
        log_info "  - Grafana: http://localhost:3000 (admin/admin)"
        
        log_success "Monitoring stack configured"
    else
        log_warning "docker-compose not available, skipping monitoring setup"
    fi
}

# Run tests to verify installation
run_verification_tests() {
    log_info "Running verification tests..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    
    cd "$PROJECT_ROOT"
    
    # Run basic tests
    python -c "
import sys
import pkg_resources
import torch
import transformers
import cryptography

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('Transformers version:', transformers.__version__)
print('Cryptography version:', cryptography.__version__)
print('CUDA available:', torch.cuda.is_available())

# Test basic MPC imports
try:
    from src.secure_mpc_transformer import SecurityConfig
    print('MPC modules import: OK')
except ImportError as e:
    print('MPC modules import: FAILED -', e)
"
    
    # Run unit tests if available
    if [[ -d "tests" ]]; then
        python -m pytest tests/unit/ -v --tb=short || \
            log_warning "Some unit tests failed. This may be expected in a fresh setup."
    fi
    
    log_success "Verification tests completed"
}

# Create development scripts
create_dev_scripts() {
    log_info "Creating development scripts..."
    
    mkdir -p "$PROJECT_ROOT/scripts/dev"
    
    # Create activation script
    cat > "$PROJECT_ROOT/scripts/dev/activate.sh" << EOF
#!/bin/bash
# Activate development environment
cd "$PROJECT_ROOT"
source "$VENV_NAME/bin/activate"
export PYTHONPATH="\$PWD/src:\$PYTHONPATH"
source config/dev/local.env
echo "Development environment activated"
echo "Python: \$(which python)"
echo "Project root: \$PWD"
EOF
    chmod +x "$PROJECT_ROOT/scripts/dev/activate.sh"
    
    # Create test runner script
    cat > "$PROJECT_ROOT/scripts/dev/run-tests.sh" << EOF
#!/bin/bash
cd "$PROJECT_ROOT"
source "$VENV_NAME/bin/activate"
python -m pytest tests/ -v --cov=secure_mpc_transformer --cov-report=html
EOF
    chmod +x "$PROJECT_ROOT/scripts/dev/run-tests.sh"
    
    # Create benchmark runner script
    cat > "$PROJECT_ROOT/scripts/dev/run-benchmarks.sh" << EOF
#!/bin/bash
cd "$PROJECT_ROOT"
source "$VENV_NAME/bin/activate"
python benchmarks/run_all.py --output-format json
EOF
    chmod +x "$PROJECT_ROOT/scripts/dev/run-benchmarks.sh"
    
    log_success "Development scripts created"
}

# Generate security report
generate_security_report() {
    log_info "Generating security report..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    
    cd "$PROJECT_ROOT"
    
    # Create security report
    cat > security-setup-report.md << EOF
# Security Setup Report

## System Information
- OS: $OS
- Python: $(python --version)
- Date: $(date)

## Installed Security Tools
- Pre-commit hooks: $(pre-commit --version)
- Secrets detection: $(detect-secrets --version)
- Bandit: $(bandit --version 2>/dev/null || echo "Not installed")
- Safety: $(safety --version 2>/dev/null || echo "Not installed")

## Security Configuration
- Virtual environment: $VENV_NAME
- Secrets baseline: .secrets.baseline
- Pre-commit hooks: Installed and configured

## Recommendations
1. Regularly update dependencies with 'pip install --upgrade -e .[dev]'
2. Run 'pre-commit run --all-files' before committing
3. Periodically run 'safety check' for vulnerability scanning
4. Keep secrets out of version control
5. Use environment variables for configuration

## Next Steps
1. Run 'source scripts/dev/activate.sh' to activate the environment
2. Create a '.env' file with your local configuration
3. Run tests with 'scripts/dev/run-tests.sh'
4. Start coding securely!
EOF
    
    log_success "Security report generated: security-setup-report.md"
}

# Main setup function
main() {
    log_info "Starting Secure MPC Transformer development environment setup..."
    
    check_privileges
    detect_os
    check_system_requirements
    install_system_dependencies
    setup_python_environment
    install_crypto_libraries
    setup_gpu_support
    setup_development_tools
    setup_docker_environment
    setup_monitoring
    run_verification_tests
    create_dev_scripts
    generate_security_report
    
    log_success "Development environment setup complete!"
    echo ""
    echo "To get started:"
    echo "  1. source scripts/dev/activate.sh"
    echo "  2. python -c 'from src.secure_mpc_transformer import SecurityConfig; print(\"Ready to go!\")'"
    echo "  3. Read security-setup-report.md for important security information"
    echo ""
    echo "Happy secure coding! 🔒"
}

# Run setup if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi