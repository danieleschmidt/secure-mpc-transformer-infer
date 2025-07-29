# Comprehensive Development Guide

This guide provides everything needed for secure MPC transformer development, from environment setup to advanced debugging techniques.

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.10+ (3.11 recommended for better performance)
- **CUDA**: 12.0+ with compatible GPU drivers
- **System Memory**: 64GB+ RAM (128GB for large model testing)
- **GPU Memory**: 24GB+ VRAM (RTX 4090, A5000, or better)
- **Network**: 1Gbps+ for multi-party testing
- **Storage**: 100GB+ free space for models and datasets

### Required Tools
```bash
# Core development tools
pip install pre-commit black ruff mypy pytest

# Security tools
pip install bandit safety detect-secrets

# Performance tools
pip install memory-profiler py-spy

# Container tools (optional)
docker --version  # Docker 24.0+
docker compose --version  # Compose v2
```

## 🚀 Environment Setup

### Option 1: Conda Environment (Recommended)

```bash
# Create isolated environment
conda create -n mpc-transformer python=3.10
conda activate mpc-transformer

# Install system-level dependencies
conda install -c conda-forge cmake ninja

# Install project with all development dependencies
pip install -e ".[dev,gpu,benchmark]"

# Setup development tools
pre-commit install
pre-commit install --hook-type commit-msg

# Initialize secrets baseline
detect-secrets scan --baseline .secrets.baseline

# Verify installation
python -c "import secure_mpc_transformer; print('✅ Installation successful')"
```

### Option 2: Docker Development Environment

```bash
# Build development image with all tools
docker build -f docker/Dockerfile.dev -t mpc-transformer:dev .

# Run development container with GPU support
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -v ~/.ssh:/root/.ssh:ro \
  -p 8080:8080 \
  -p 6006:6006 \
  --name mpc-dev \
  mpc-transformer:dev

# Alternative: Use docker-compose for development
docker-compose -f docker/docker-compose.dev.yml up -d
```

### Option 3: VS Code DevContainer

```bash
# Open in VS Code with DevContainer extension
code .
# Command palette: "Dev Containers: Reopen in Container"
```

## 💻 Development Workflow

### 1. Enhanced Code Quality Pipeline

```bash
# Comprehensive formatting and linting
make format  # or: black src/ tests/ && isort src/ tests/
make lint    # or: ruff check src/ tests/ --fix
make typecheck  # or: mypy src/ --strict

# Security scanning
make security  # or: bandit -r src/ && safety check

# Run complete pre-commit suite
pre-commit run --all-files

# Advanced code quality checks
ruff check --select=ALL src/  # All available rules
mypy src/ --strict --show-error-codes
black --check --diff src/  # Show what would change
```

### 2. Comprehensive Testing Strategy

```bash
# Unit tests with parallel execution
pytest tests/unit/ -n auto --dist=loadgroup

# Integration tests with GPU validation
pytest tests/integration/ --gpu --timeout=300

# End-to-end protocol tests
pytest tests/e2e/ --slow --protocol=aby3

# Performance regression tests
pytest tests/benchmark/ --benchmark-only --benchmark-compare

# Security-focused tests
pytest tests/security/ --runslow

# Coverage with detailed reporting
pytest --cov=secure_mpc_transformer \
       --cov-report=html \
       --cov-report=term-missing \
       --cov-fail-under=85

# Memory leak detection
pytest tests/ --memray
```

### 3. Advanced Building and Packaging

```bash
# Clean build environment
make clean

# Build CUDA kernels with debugging
cd kernels/cuda
make clean && make debug
cd ../..

# Build Python package with all variants
python -m build --wheel --sdist

# Build and test container images
docker build -f docker/Dockerfile.cpu -t mpc-transformer:cpu .
docker build -f docker/Dockerfile.gpu -t mpc-transformer:gpu .

# Validate package integrity
twine check dist/*
pip install dist/*.whl && python -c "import secure_mpc_transformer"

# Multi-platform builds
docker buildx build --platform linux/amd64,linux/arm64 -t mpc-transformer:multi .
```

### 4. Performance Optimization Workflow

```bash
# Profile CPU performance
python -m cProfile -o profile.stats benchmarks/benchmark_bert.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# GPU profiling with detailed metrics
nsys profile --stats=true -o gpu_profile.qdrep python benchmarks/benchmark_bert.py
ncu --set full -o gpu_metrics python benchmarks/gpu_kernel_test.py

# Memory usage analysis
python -m memory_profiler benchmarks/memory_test.py
python -m pympler.asizeof benchmarks/memory_objects.py

# Real-time performance monitoring
py-spy top --pid $(pgrep -f python) --duration 60
```

## Architecture Overview

```
src/
├── secure_mpc_transformer/
│   ├── protocols/          # MPC protocol implementations
│   ├── models/            # Transformer model wrappers
│   ├── gpu/               # CUDA kernel interfaces
│   ├── network/           # Communication layer
│   └── utils/             # Utility functions
├── kernels/
│   └── cuda/              # CUDA kernel implementations
└── tests/
    ├── unit/              # Unit tests
    ├── integration/       # Integration tests
    └── benchmark/         # Performance tests
```

## Adding New Features

### New MPC Protocol

1. Create protocol class in `src/secure_mpc_transformer/protocols/`
2. Implement required interface methods
3. Add GPU kernel support if needed
4. Write comprehensive tests
5. Update documentation

### GPU Kernel Optimization

1. Implement CUDA kernel in `kernels/cuda/`
2. Add Python wrapper in `src/secure_mpc_transformer/gpu/`
3. Benchmark against existing implementation
4. Update performance documentation

## Security Guidelines

- Never commit private keys or certificates
- Use constant-time algorithms for cryptographic operations
- Validate all inputs from network and files
- Use secure random number generation
- Clear sensitive data from memory after use

## Performance Profiling

```bash
# CPU profiling
python -m cProfile -o profile.stats benchmark_script.py

# GPU profiling
nsys profile python benchmark_script.py

# Memory profiling
python -m memory_profiler benchmark_script.py
```

## Debugging

### CUDA Debugging

```bash
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# Use compute-sanitizer
compute-sanitizer python test_script.py
```

### MPC Protocol Debugging

```bash
# Enable protocol tracing
export MPC_DEBUG=1
export MPC_TRACE_PROTOCOL=1

python debug_protocol.py
```

## Contributing Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Performance impact assessed
- [ ] Backward compatibility maintained