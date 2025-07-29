# Contributing to Secure MPC Transformer Inference

Thank you for your interest in contributing to our secure multi-party computation (MPC) transformer inference project!

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Set up the development environment (see Development Setup below)
4. Make your changes with appropriate tests
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/secure-mpc-transformer-infer.git
cd secure-mpc-transformer-infer

# Create conda environment
conda create -n mpc-transformer python=3.10
conda activate mpc-transformer

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Areas for Contribution

### High Priority
- **New MPC Protocols**: Implement additional secure computation protocols
- **GPU Optimization**: CUDA kernel improvements for homomorphic encryption
- **Model Support**: Add support for more transformer architectures
- **Documentation**: Improve protocol specifications and tutorials

### Medium Priority
- **Testing**: Expand test coverage for edge cases
- **Benchmarking**: Add new performance evaluation scenarios
- **Communication**: Optimize network protocols between parties

## Security Guidelines

- **Never commit cryptographic keys or certificates**
- **Use secure coding practices** for all cryptographic operations
- **Validate all inputs** to prevent injection attacks
- **Follow responsible disclosure** for security vulnerabilities

## Code Standards

- Follow PEP 8 for Python code
- Include type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new functionality
- Update documentation for API changes

## Pull Request Process

1. Ensure all tests pass: `pytest tests/`
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address any feedback promptly

## Reporting Issues

- Use the GitHub issue tracker
- Include minimal reproducible examples
- For security issues, email security@secure-mpc-transformer.org privately

## Questions?

- Open a discussion on GitHub
- Join our research collaboration channel
- Review existing documentation and tutorials