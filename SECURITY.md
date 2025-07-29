# Security Policy

## Supported Versions

This project implements state-of-the-art secure multi-party computation (MPC) protocols. Security support is provided for the following versions:

| Version | Supported          | Security Level |
| ------- | ------------------ | -------------- |
| 1.0.x   | :white_check_mark: | 128-bit        |
| 0.9.x   | :x:                | -              |
| < 0.9   | :x:                | -              |

## Reporting Vulnerabilities

**Please do not report security vulnerabilities through public GitHub issues.**

### Reporting Process

1. **Email**: Send details to security@secure-mpc-transformer.org
2. **PGP**: Use our public key for sensitive communications (key ID: [TO_BE_ADDED])
3. **Response Time**: We aim to acknowledge reports within 48 hours
4. **Disclosure**: Coordinated disclosure after fix is available

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fixes (if any)

## Security Considerations

### Cryptographic Security

- **Protocol Security**: Implements malicious-secure 3PC and semi-honest protocols
- **Key Management**: Uses secure key generation and distribution
- **Communication**: All inter-party communication is encrypted

### Implementation Security

- **Input Validation**: All inputs are validated and sanitized
- **Memory Safety**: CUDA kernels are bounds-checked
- **Side-Channel Resistance**: Constant-time implementations where applicable

### Operational Security

- **Secrets Management**: Never log or persist cryptographic secrets
- **Network Security**: Use TLS 1.3 for all network communications
- **Container Security**: Docker images are regularly scanned for vulnerabilities

## Known Security Limitations

1. **Timing Attacks**: Some operations may leak timing information
2. **Research Code**: This is research software - not production-ready
3. **GPU Memory**: Secrets may persist in GPU memory between operations

## Security Audit Status

- **Last Audit**: [TO_BE_SCHEDULED]
- **Audit Firm**: [TO_BE_DETERMINED]
- **Report**: Will be published after completion

## Security Updates

Security updates are distributed through:
- GitHub Security Advisories
- Email notifications to registered users
- Docker image updates with security tags

## Best Practices for Users

1. **Environment**: Use isolated environments for MPC computation
2. **Key Storage**: Store cryptographic keys in secure hardware when possible
3. **Network**: Use private networks or VPNs for party communication
4. **Monitoring**: Monitor for unusual computational patterns or network activity

## Contact Information

- **Security Team**: security@secure-mpc-transformer.org
- **General Questions**: GitHub Discussions
- **Research Collaboration**: research@secure-mpc-transformer.org