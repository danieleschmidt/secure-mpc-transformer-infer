# Development Guide

## Prerequisites

- Python 3.10+
- CUDA 12.0+ (for GPU acceleration)
- Docker (optional)
- 64GB+ RAM recommended

## Environment Setup

### Option 1: Conda Environment

```bash
# Create environment
conda create -n mpc-transformer python=3.10
conda activate mpc-transformer

# Install development dependencies
pip install -e ".[dev,gpu]"

# Install pre-commit hooks
pre-commit install
```

### Option 2: Docker Development

```bash
# Build development image
docker build -f docker/Dockerfile.dev -t mpc-transformer:dev .

# Run development container
docker run -it --gpus all -v $(pwd):/workspace mpc-transformer:dev
```

## Development Workflow

### 1. Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### 2. Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires GPU)
pytest tests/integration/ --gpu

# Benchmark tests
pytest tests/benchmark/ --benchmark

# Coverage report
pytest --cov=secure_mpc_transformer --cov-report=html
```

### 3. Building

```bash
# Build Python package
python -m build

# Build CUDA kernels
cd kernels/cuda
make clean && make all
cd ../..

# Run installation test
pip install dist/*.whl
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