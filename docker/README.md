# Docker Configuration for Secure MPC Transformer

This directory contains Docker configurations for the Secure MPC Transformer project, supporting development, testing, and production deployments.

## Available Images

### 1. CPU-Only Image (`Dockerfile.cpu`)
- **Purpose**: Production deployment without GPU requirements
- **Base**: `python:3.10-slim`
- **Features**: Lightweight, secure, optimized for CPU-only inference
- **Use Cases**: Testing, lightweight deployments, CI/CD

### 2. GPU-Enabled Image (`Dockerfile.gpu`)
- **Purpose**: Production deployment with CUDA acceleration
- **Base**: `nvidia/cuda:12.0-devel-ubuntu22.04`
- **Features**: CUDA 12.0+, cuDNN, GPU-optimized MPC kernels
- **Use Cases**: High-performance inference, benchmarking

### 3. Development Image (`Dockerfile.dev`)
- **Purpose**: Development environment with all tools
- **Base**: `python:3.10-slim`
- **Features**: Jupyter Lab, development tools, debugging utilities
- **Use Cases**: Local development, experimentation, debugging

## Quick Start

### Development Environment

```bash
# Start development environment
docker-compose up dev

# Access Jupyter Lab
open http://localhost:8888

# Execute commands in container
docker-compose exec dev bash
```

### Multi-Party Computation Demo

```bash
# Start 3-party MPC setup
docker-compose up mpc-party-0 mpc-party-1 mpc-party-2 redis

# Check party status
docker-compose logs mpc-party-0
docker-compose logs mpc-party-1 
docker-compose logs mpc-party-2
```

### Production Deployment

```bash
# CPU-only deployment
docker-compose up app-cpu

# GPU-enabled deployment (requires NVIDIA Docker)
docker-compose up app-gpu
```

## Environment Variables

### Common Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python module search path | `/app/src` |
| `MPC_DEBUG` | Enable debug logging | `0` |
| `MPC_PROTOCOL` | MPC protocol to use | `semi_honest_3pc` |
| `MPC_PARTY_ID` | Party identifier (0, 1, 2) | `0` |

### GPU-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | Available GPU devices | `all` |
| `NVIDIA_VISIBLE_DEVICES` | NVIDIA GPU visibility | `all` |
| `NVIDIA_DRIVER_CAPABILITIES` | Driver capabilities | `compute,utility` |

### Multi-Party Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MPC_NUM_PARTIES` | Number of parties | `3` |
| `MPC_PORT` | Communication port | `50051` |
| `MPC_PEERS` | Peer addresses | `party1:50051,party2:50051` |

## Build Instructions

### Building Individual Images

```bash
# Build CPU image
docker build -f docker/Dockerfile.cpu -t mpc-transformer:cpu .

# Build GPU image
docker build -f docker/Dockerfile.gpu -t mpc-transformer:gpu .

# Build development image
docker build -f docker/Dockerfile.dev -t mpc-transformer:dev .
```

### Multi-Architecture Builds

```bash
# Build for multiple architectures (CPU only)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.cpu \
  -t mpc-transformer:cpu-multi \
  --push .
```

## Usage Examples

### Development Workflow

```bash
# Start development environment
docker-compose up -d dev

# Install additional packages
docker-compose exec dev pip install --user new-package

# Run tests
docker-compose exec dev pytest tests/

# Format code
docker-compose exec dev black src/

# Run linting
docker-compose exec dev ruff check src/
```

### Running Benchmarks

```bash
# GPU benchmark
docker-compose run --rm app-gpu python benchmarks/benchmark_bert.py

# CPU benchmark comparison
docker-compose run --rm app-cpu python benchmarks/compare_protocols.py
```

### Security Scanning

```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image mpc-transformer:cpu

# Check for secrets
docker run --rm -v "$(pwd)":/workspace \
  trufflesecurity/trufflehog:latest filesystem /workspace
```

## Configuration Files

### Docker Compose Profiles

The `docker-compose.yml` supports multiple profiles:

```bash
# Default services
docker-compose up

# Include monitoring stack
docker-compose --profile monitoring up

# Run tests
docker-compose --profile test up
```

### Monitoring Setup

Enable monitoring with Prometheus and Grafana:

```bash
# Create monitoring configuration
mkdir -p docker/monitoring
cp examples/prometheus.yml docker/monitoring/
cp examples/grafana-dashboard.json docker/monitoring/grafana/

# Start with monitoring
docker-compose --profile monitoring up
```

## Security Considerations

### Image Security

- **Non-root user**: All containers run as non-root user (UID 1000)
- **Minimal attack surface**: Production images exclude development tools
- **Regular updates**: Base images updated monthly
- **Vulnerability scanning**: Integrated with CI/CD pipeline

### Network Security

- **Isolated network**: Containers communicate via dedicated bridge network
- **TLS encryption**: Inter-party communication uses TLS 1.3
- **Port restriction**: Only necessary ports exposed
- **Secret management**: Sensitive data via Docker secrets or environment

### Runtime Security

```bash
# Run with security options
docker run --rm \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  --cap-add=CHOWN,SETGID,SETUID \
  --read-only \
  --tmpfs /tmp \
  mpc-transformer:cpu
```

## Troubleshooting

### Common Issues

#### GPU Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Verify GPU access in container
docker-compose exec app-gpu nvidia-smi
```

#### Build Failures

```bash
# Clear build cache
docker builder prune -f

# Rebuild without cache
docker-compose build --no-cache

# Check build logs
docker-compose build 2>&1 | tee build.log
```

#### Memory Issues

```bash
# Increase Docker memory limit
docker run --rm -m 8g mpc-transformer:gpu

# Monitor memory usage
docker stats
```

### Debugging

#### Container Debugging

```bash
# Access running container
docker-compose exec dev bash

# Debug startup issues
docker-compose up --debug dev

# Check container logs
docker-compose logs -f app-cpu
```

#### Network Debugging

```bash
# Test inter-container connectivity
docker-compose exec mpc-party-0 ping mpc-party-1

# Check network configuration
docker network inspect docker_mpc-network
```

## Performance Optimization

### Build Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile.gpu .

# Multi-stage build optimization
docker build --target development -f docker/Dockerfile.gpu .
```

### Runtime Optimization

```bash
# Optimize for CPU performance
docker run --rm \
  --cpus="4.0" \
  --memory="8g" \
  --shm-size="2g" \
  mpc-transformer:cpu

# Optimize for GPU performance
docker run --rm \
  --gpus all \
  --memory="16g" \
  --shm-size="4g" \
  mpc-transformer:gpu
```

## Maintenance

### Regular Tasks

```bash
# Update base images
docker pull python:3.10-slim
docker pull nvidia/cuda:12.0-devel-ubuntu22.04

# Clean up unused resources
docker system prune -f

# Update dependencies
docker-compose build --pull
```

### Health Monitoring

```bash
# Check container health
docker-compose ps

# Monitor resource usage
docker stats

# View health check logs
docker inspect --format='{{.State.Health}}' mpc-transformer-cpu
```

## Integration with CI/CD

The Docker configuration integrates with GitHub Actions workflows:

- **ci.yml**: Uses development image for testing
- **docker.yml**: Builds and pushes production images
- **benchmark.yml**: Uses GPU image for performance testing

See `docs/workflows/` for complete workflow templates.

## Contributing

When modifying Docker configurations:

1. Test all build stages
2. Verify security settings
3. Update documentation
4. Test with both CPU and GPU
5. Check multi-architecture compatibility

## Support

For Docker-related issues:

- Check existing [GitHub Issues](https://github.com/yourusername/secure-mpc-transformer/issues)
- Review [troubleshooting section](#troubleshooting)
- Consult [Docker documentation](https://docs.docker.com/)
- Contact maintainers for complex issues