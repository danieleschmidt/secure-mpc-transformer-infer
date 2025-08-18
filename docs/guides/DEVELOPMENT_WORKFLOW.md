# Development Workflow Guide

This guide outlines the development process, code standards, and best practices for contributing to the Secure MPC Transformer project.

## Development Environment Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.0+
- Docker and Docker Compose
- Git with LFS support

### Initial Setup

```bash
# Clone repository with submodules
git clone --recursive https://github.com/danieleschmidt/secure-mpc-transformer-infer.git
cd secure-mpc-transformer-infer

# Create development environment
conda create -n mpc-transformer-dev python=3.10
conda activate mpc-transformer-dev

# Install development dependencies
pip install -e ".[dev,gpu,quantum-planning,test]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -m secure_mpc_transformer.test_installation
```

### Development Tools

```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
pylint src/
flake8 src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
safety check

# Documentation
sphinx-build -b html docs/ docs/_build/
```

## Code Organization

### Project Structure

```
src/secure_mpc_transformer/
├── core/                 # Core MPC protocols and algorithms
├── models/              # Transformer model implementations
├── gpu/                 # CUDA kernels and GPU acceleration
├── protocols/           # MPC protocol implementations
├── security/            # Security and cryptographic utilities
├── api/                 # REST API and service interfaces
├── monitoring/          # Metrics and observability
├── utils/               # Shared utilities and helpers
└── cli/                 # Command-line interface
```

### Module Guidelines

#### Core Modules (`core/`)
- Fundamental MPC algorithms and data structures
- Protocol-agnostic implementations
- Abstract base classes for extensibility

#### Model Modules (`models/`)
- Transformer architecture implementations
- Model loading and preprocessing
- Secure inference pipelines

#### GPU Modules (`gpu/`)
- CUDA kernel implementations
- Memory management utilities
- Performance optimization tools

#### Protocol Modules (`protocols/`)
- Specific MPC protocol implementations
- Network communication handlers
- Cryptographic primitive implementations

## Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific modifications:

```python
# File header template
"""
Secure MPC Transformer - [Module Description]

This module implements [brief description of functionality].

Security considerations:
- [Security consideration 1]
- [Security consideration 2]

Performance notes:
- [Performance note 1]
- [Performance note 2]
"""

# Import organization
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes

from secure_mpc_transformer.core import BaseProtocol
from secure_mpc_transformer.utils import logger
```

### Code Quality Standards

#### Type Hints
All public functions must include type hints:

```python
from typing import Dict, List, Optional, Union

def secure_inference(
    model: SecureTransformer,
    inputs: List[str],
    security_config: SecurityConfig,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Perform secure inference with the given model and inputs."""
    pass
```

#### Docstrings
Use Google-style docstrings:

```python
def encrypt_tensor(
    tensor: torch.Tensor,
    public_key: PublicKey,
    noise_level: float = 1e-6
) -> EncryptedTensor:
    """Encrypt a tensor using homomorphic encryption.
    
    Args:
        tensor: Input tensor to encrypt
        public_key: Public key for encryption
        noise_level: Noise level for differential privacy
        
    Returns:
        Encrypted tensor with security metadata
        
    Raises:
        SecurityError: If encryption fails due to security constraints
        
    Security:
        This function adds differential privacy noise before encryption
        to prevent inference attacks on the encrypted values.
        
    Performance:
        GPU acceleration is used when available. Memory usage scales
        linearly with tensor size.
    """
```

#### Error Handling
Use specific exception types and provide actionable error messages:

```python
from secure_mpc_transformer.utils.exceptions import (
    SecurityError,
    ProtocolError,
    GPUError
)

def validate_security_config(config: SecurityConfig) -> None:
    """Validate security configuration parameters."""
    if config.security_level < 128:
        raise SecurityError(
            f"Security level {config.security_level} is insufficient. "
            f"Minimum required: 128 bits. "
            f"Recommendation: Use 128 or 256 bits."
        )
```

### Security Coding Guidelines

#### Input Validation
Always validate inputs, especially from external sources:

```python
def process_user_input(text: str) -> str:
    """Process and sanitize user input."""
    # Length validation
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input too long: {len(text)} > {MAX_INPUT_LENGTH}")
    
    # Content validation
    if not text.strip():
        raise ValueError("Empty input not allowed")
    
    # Sanitization
    sanitized = sanitize_text(text)
    
    # Additional security checks
    if contains_injection_patterns(sanitized):
        raise SecurityError("Potential injection attack detected")
    
    return sanitized
```

#### Cryptographic Operations
Follow cryptographic best practices:

```python
def generate_secure_random(size: int) -> bytes:
    """Generate cryptographically secure random bytes."""
    # Use cryptographically secure random generator
    return secrets.token_bytes(size)

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Compare bytes in constant time to prevent timing attacks."""
    return secrets.compare_digest(a, b)
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/               # Unit tests for individual functions
├── integration/        # Integration tests for component interaction
├── e2e/               # End-to-end tests for complete workflows
├── performance/       # Performance and benchmark tests
├── security/          # Security-specific tests
├── fixtures/          # Test data and fixtures
└── utils/             # Testing utilities
```

### Unit Testing

```python
import pytest
import torch
from unittest.mock import Mock, patch

from secure_mpc_transformer.models import SecureTransformer
from secure_mpc_transformer.security import SecurityConfig

class TestSecureTransformer:
    """Test suite for SecureTransformer class."""
    
    @pytest.fixture
    def security_config(self):
        """Create test security configuration."""
        return SecurityConfig(
            protocol="3pc_semi_honest",
            security_level=128,
            gpu_acceleration=True
        )
    
    @pytest.fixture
    def model(self, security_config):
        """Create test model instance."""
        return SecureTransformer.from_pretrained(
            "bert-base-uncased",
            security_config=security_config
        )
    
    def test_inference_basic(self, model):
        """Test basic inference functionality."""
        text = "The capital of France is [MASK]."
        result = model.inference(text)
        
        assert result is not None
        assert "prediction" in result
        assert result["confidence"] > 0.0
    
    @pytest.mark.gpu
    def test_gpu_acceleration(self, model):
        """Test GPU acceleration is working."""
        assert model.config.gpu_acceleration
        assert torch.cuda.is_available()
        
        # Verify GPU memory usage
        before = torch.cuda.memory_allocated()
        result = model.inference("Test input")
        after = torch.cuda.memory_allocated()
        
        assert after > before  # GPU memory was used
    
    @pytest.mark.security
    def test_security_guarantees(self, model):
        """Test security properties are maintained."""
        # Test differential privacy
        result1 = model.inference("Sensitive data 1")
        result2 = model.inference("Sensitive data 2")
        
        # Results should be different due to noise
        assert result1["prediction"] != result2["prediction"]
        
        # Test timing attack resistance
        import time
        times = []
        for _ in range(10):
            start = time.time()
            model.inference("Test input")
            times.append(time.time() - start)
        
        # Timing should be relatively consistent
        assert max(times) / min(times) < 1.1  # Within 10% variance
```

### Integration Testing

```python
@pytest.mark.integration
class TestMPCProtocols:
    """Test MPC protocol integration."""
    
    def test_three_party_computation(self):
        """Test 3-party MPC protocol end-to-end."""
        # Setup three parties
        parties = [
            MPCParty(party_id=0, role="data_owner"),
            MPCParty(party_id=1, role="compute"),
            MPCParty(party_id=2, role="compute")
        ]
        
        # Start coordination
        coordinator = MPCCoordinator(parties)
        coordinator.start()
        
        try:
            # Run secure computation
            result = coordinator.run_inference(
                "Test input for secure computation",
                model="bert-base-uncased"
            )
            
            assert result is not None
            assert "prediction" in result
            assert result["security_level"] == 128
            
        finally:
            coordinator.stop()
```

### Performance Testing

```python
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_inference_latency(self, benchmark):
        """Benchmark inference latency."""
        model = SecureTransformer.from_pretrained("bert-base-uncased")
        text = "The capital of France is [MASK]."
        
        result = benchmark(model.inference, text)
        
        # Assert performance requirements
        assert benchmark.stats['mean'] < 60.0  # Under 60 seconds
    
    @pytest.mark.gpu
    def test_gpu_utilization(self):
        """Test GPU utilization efficiency."""
        model = SecureTransformer.from_pretrained("bert-base-uncased")
        
        gpu_monitor = GPUMonitor()
        gpu_monitor.start()
        
        # Run batch inference
        texts = ["Sample text"] * 32
        results = model.inference_batch(texts)
        
        metrics = gpu_monitor.stop()
        
        # GPU should be well utilized
        assert metrics.utilization > 0.8  # 80%+ utilization
        assert metrics.memory_efficiency > 0.7  # 70%+ memory efficiency
```

## Code Review Process

### Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names
   - `feature/quantum-planning-optimization`
   - `fix/gpu-memory-leak`
   - `security/timing-attack-mitigation`

2. **Commit Messages**: Follow conventional commits
   ```
   feat(protocols): add quantum-enhanced MPC protocol
   
   - Implement quantum superposition for task scheduling
   - Add quantum annealing optimization
   - Include comprehensive security analysis
   
   Closes #123
   ```

3. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Security Impact
   - Security implications of changes
   - Any new attack vectors introduced
   - Mitigation strategies implemented
   
   ## Performance Impact
   - Performance improvements/regressions
   - Benchmark results before/after
   - Memory usage changes
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Security tests pass
   - [ ] Performance benchmarks run
   
   ## Documentation
   - [ ] Code comments updated
   - [ ] API documentation updated
   - [ ] User documentation updated
   ```

### Review Checklist

#### Security Review
- [ ] Input validation is comprehensive
- [ ] Cryptographic operations follow best practices
- [ ] No hardcoded secrets or keys
- [ ] Timing attack resistance verified
- [ ] Side-channel attack mitigation in place

#### Performance Review
- [ ] GPU kernels are optimized
- [ ] Memory usage is efficient
- [ ] Network communication is minimized
- [ ] Benchmark results meet requirements

#### Code Quality Review
- [ ] Type hints are complete and accurate
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate and secure
- [ ] Documentation is complete and accurate

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

#### Pre-Release
- [ ] All tests pass (unit, integration, security, performance)
- [ ] Security audit completed
- [ ] Performance benchmarks meet requirements
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

#### Release
- [ ] Version bumped in all relevant files
- [ ] Git tag created and signed
- [ ] Docker images built and pushed
- [ ] PyPI package published
- [ ] GitHub release created with notes

#### Post-Release
- [ ] Deployment documentation updated
- [ ] Community notified (Discord, mailing list)
- [ ] Monitoring alerts configured
- [ ] Hotfix process documented

## Debugging and Profiling

### Debug Mode Setup

```python
# Enable comprehensive debugging
import os
os.environ['MPC_DEBUG'] = 'true'
os.environ['MPC_LOG_LEVEL'] = 'DEBUG'
os.environ['MPC_PROFILE'] = 'true'

# Enable specific component debugging
os.environ['MPC_DEBUG_PROTOCOLS'] = 'true'
os.environ['MPC_DEBUG_GPU'] = 'true'
os.environ['MPC_DEBUG_NETWORK'] = 'true'
```

### Performance Profiling

```python
# CPU profiling
import cProfile
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = model.inference(text)

profiler.disable()
profiler.dump_stats('profile.prof')

# GPU profiling
from secure_mpc_transformer.profiling import GPUProfiler
gpu_profiler = GPUProfiler()
with gpu_profiler:
    result = model.inference(text)

gpu_report = gpu_profiler.get_report()
```

### Memory Debugging

```python
# Memory leak detection
import tracemalloc
tracemalloc.start()

# Your code here
result = model.inference(text)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## Documentation Standards

### API Documentation

Use Sphinx with autodoc for API documentation:

```python
def secure_inference(
    model: SecureTransformer,
    inputs: List[str],
    **kwargs
) -> InferenceResult:
    """Perform secure multi-party inference.
    
    This function coordinates secure computation across multiple parties
    to perform transformer inference while preserving input privacy.
    
    Args:
        model: Pre-trained secure transformer model
        inputs: List of input texts to process
        **kwargs: Additional configuration options
        
    Returns:
        InferenceResult: Results with predictions and metadata
        
    Raises:
        SecurityError: If security constraints are violated
        ProtocolError: If MPC protocol fails
        
    Example:
        >>> config = SecurityConfig(protocol="3pc", security_level=128)
        >>> model = SecureTransformer.from_pretrained("bert-base", config=config)
        >>> result = secure_inference(model, ["Sample text"])
        >>> print(result.predictions)
        
    Note:
        This function requires at least 3 parties for secure computation.
        GPU acceleration is recommended for production workloads.
        
    Security:
        - Input data is never exposed in plaintext to individual parties
        - Differential privacy noise is added to prevent inference attacks
        - All network communication is encrypted with TLS 1.3
        
    Performance:
        - GPU acceleration provides 10-20x speedup
        - Batch processing improves throughput significantly
        - Memory usage scales linearly with input size
    """
```

## Contributing Guidelines

### Getting Started

1. Read the [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
2. Review the [CONTRIBUTING.md](../CONTRIBUTING.md)
3. Join our [Discord community](https://discord.gg/secure-mpc)
4. Look for "good first issue" labels on GitHub

### Making Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes following this guide
4. Add tests and documentation
5. Submit a pull request

### Community Support

- **Technical Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Issues with enhancement label
- **Security Issues**: security@secure-mpc-transformer.org

## Tools and Resources

### Development Tools

- **IDE**: VSCode with Python and CUDA extensions
- **Debugging**: PyCharm Professional with GPU debugger
- **Profiling**: NVIDIA Nsight Systems and Compute
- **Testing**: pytest with coverage reporting
- **Documentation**: Sphinx with autodoc

### External Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Cryptography Library](https://cryptography.io/)
- [MPC Protocols Survey](https://eprint.iacr.org/2020/300)

This development workflow ensures high code quality, security, and performance while maintaining a collaborative and welcoming development environment.