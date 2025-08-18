# Getting Started with Secure MPC Transformer

This guide will help you quickly set up and run your first secure multi-party computation (MPC) transformer inference.

## Quick Start (5 minutes)

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with 16GB+ VRAM (RTX 3080 or better)
- Python 3.10+

### Run Your First Secure Inference

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/secure-mpc-transformer-infer.git
cd secure-mpc-transformer-infer

# Start with Docker Compose
docker compose up -d

# Wait for initialization (check logs)
docker compose logs -f

# Run a test inference
curl -X POST http://localhost:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is [MASK].",
    "model": "bert-base-uncased",
    "security_level": 128
  }'
```

Expected output:
```json
{
  "prediction": "Paris",
  "confidence": 0.94,
  "computation_time_ms": 42000,
  "security_level": 128,
  "protocol": "3pc_malicious"
}
```

## Installation Options

### Option 1: Docker (Recommended)

```bash
# Pull pre-built image
docker pull securempc/transformer-inference:latest

# Run with GPU support
docker run --gpus all -p 8080:8080 \
  securempc/transformer-inference:latest
```

### Option 2: Build from Source

```bash
# Install system dependencies
sudo apt-get install -y libseal-dev libprotobuf-dev

# Create Python environment
conda create -n mpc-transformer python=3.10
conda activate mpc-transformer

# Install package
pip install -e ".[gpu]"

# Build GPU kernels
cd kernels/cuda && make all && cd ../..

# Run tests
pytest tests/ -v
```

### Option 3: Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/danieleschmidt/secure-mpc-transformer-infer.git

# Install development dependencies
pip install -e ".[dev,gpu,quantum-planning]"

# Set up pre-commit hooks
pre-commit install

# Run development server
python main.py --dev
```

## Core Concepts

### Multi-Party Computation (MPC)

MPC allows multiple parties to jointly compute a function over their inputs while keeping those inputs private.

```python
from secure_mpc_transformer import SecureTransformer, SecurityConfig

# Configure security level
config = SecurityConfig(
    protocol="3pc",           # 3-party computation
    security_level=128,       # 128-bit security
    gpu_acceleration=True     # Use GPU kernels
)

# Load model
model = SecureTransformer.from_pretrained(
    "bert-base-uncased", 
    security_config=config
)
```

### Security Protocols

| Protocol | Security Model | Performance | Use Case |
|----------|----------------|-------------|----------|
| `3pc_semi_honest` | Semi-honest | Fastest | Development/testing |
| `3pc_malicious` | Malicious | Moderate | Production |
| `aby3` | Malicious | Slower | High security |

### GPU Acceleration

GPU acceleration provides 10-20x speedup for cryptographic operations:

```python
# Enable GPU acceleration
config = SecurityConfig(
    gpu_acceleration=True,
    gpu_memory_limit="20GB",
    batch_size=32
)
```

## Basic Usage Examples

### Text Classification

```python
from secure_mpc_transformer import SecureTransformer

model = SecureTransformer.from_pretrained("bert-base-uncased")

# Single prediction
result = model.classify(
    "This movie is absolutely fantastic!",
    labels=["positive", "negative"]
)
print(f"Sentiment: {result.label}, Confidence: {result.confidence:.2f}")

# Batch prediction
texts = [
    "Great product, highly recommended!",
    "Terrible experience, avoid at all costs.",
    "Average quality, nothing special."
]
results = model.classify_batch(texts, labels=["positive", "negative"])
```

### Question Answering

```python
context = """
The company reported strong quarterly earnings with revenue 
growing 15% year-over-year to $2.3 billion.
"""

question = "What was the revenue growth rate?"

answer = model.answer_question(
    question=question,
    context=context
)
print(f"Answer: {answer.text}")
print(f"Confidence: {answer.confidence:.2f}")
```

### Named Entity Recognition

```python
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

entities = model.extract_entities(text)
for entity in entities:
    print(f"{entity.text}: {entity.label} ({entity.confidence:.2f})")
```

## Multi-Party Setup

### Local Development (Single Machine)

```python
from secure_mpc_transformer import MPCCoordinator

# Start 3-party local setup
coordinator = MPCCoordinator(
    num_parties=3,
    local=True,
    ports=[8001, 8002, 8003]
)

# Initialize all parties
coordinator.start_all_parties()

# Run computation
result = coordinator.run_inference(
    "The capital of [MASK] is Berlin.",
    model="bert-base-uncased"
)
```

### Distributed Setup (Multiple Machines)

```bash
# Party 0 (Data Owner)
python -m secure_mpc_transformer.party \
  --party-id 0 \
  --role data-owner \
  --peers party1.example.com:8001,party2.example.com:8001 \
  --port 8001

# Party 1 (Compute)
python -m secure_mpc_transformer.party \
  --party-id 1 \
  --role compute \
  --peers party0.example.com:8001,party2.example.com:8001 \
  --port 8001

# Party 2 (Compute)
python -m secure_mpc_transformer.party \
  --party-id 2 \
  --role compute \
  --peers party0.example.com:8001,party1.example.com:8001 \
  --port 8001
```

## Configuration

### Environment Variables

```bash
# Security settings
export MPC_SECURITY_LEVEL=128
export MPC_PROTOCOL=3pc_malicious
export MPC_GPU_ACCELERATION=true

# Network settings
export MPC_BIND_ADDRESS=0.0.0.0
export MPC_PORT=8001
export MPC_TLS_ENABLED=true

# Performance settings
export MPC_BATCH_SIZE=32
export MPC_GPU_MEMORY_LIMIT=20GB
export MPC_WORKER_THREADS=8
```

### Configuration File

```yaml
# config/mpc-config.yaml
security:
  protocol: "3pc_malicious"
  security_level: 128
  tls_enabled: true
  
performance:
  gpu_acceleration: true
  batch_size: 32
  worker_threads: 8
  memory_limit: "20GB"
  
networking:
  bind_address: "0.0.0.0"
  port: 8001
  timeout: 30
  
models:
  cache_dir: "/app/models"
  default_model: "bert-base-uncased"
  preload_models: ["bert-base-uncased", "roberta-base"]
```

## Monitoring and Debugging

### Health Checks

```bash
# Check system health
curl http://localhost:8080/health

# Check party connectivity
curl http://localhost:8080/health/parties

# Check GPU status
curl http://localhost:8080/health/gpu
```

### Metrics and Monitoring

```python
from secure_mpc_transformer.monitoring import MetricsCollector

collector = MetricsCollector()

# Start monitoring
collector.start()

# Run computation
result = model.inference("Sample text")

# View metrics
metrics = collector.get_metrics()
print(f"Computation time: {metrics.computation_time:.2f}s")
print(f"GPU utilization: {metrics.gpu_utilization:.1%}")
print(f"Network bytes: {metrics.network_bytes}")
```

### Debugging

```python
import logging
from secure_mpc_transformer import set_log_level

# Enable debug logging
set_log_level(logging.DEBUG)

# Enable protocol tracing
model = SecureTransformer.from_pretrained(
    "bert-base-uncased",
    debug_mode=True,
    trace_protocol=True
)
```

## Performance Tuning

### GPU Optimization

```python
config = SecurityConfig(
    gpu_acceleration=True,
    gpu_optimization_level=3,      # 0=none, 1=basic, 2=aggressive, 3=maximum
    gpu_memory_pool=True,          # Pre-allocate GPU memory
    mixed_precision=True,          # Use FP16 where possible
    kernel_fusion=True             # Fuse GPU kernels
)
```

### Network Optimization

```python
network_config = NetworkConfig(
    compression=True,              # Compress network traffic
    batch_messages=True,           # Batch small messages
    tcp_nodelay=True,             # Disable Nagle's algorithm
    socket_buffer_size="64KB"      # Increase buffer size
)
```

### Model Optimization

```python
# Pre-compile model for target hardware
model = SecureTransformer.from_pretrained(
    "bert-base-uncased",
    compile_for_gpu=True,          # Pre-compile GPU kernels
    quantization="int8",           # Use 8-bit quantization
    model_parallelism=True         # Split model across GPUs
)
```

## Troubleshooting

### Common Issues

#### GPU Out of Memory
```bash
# Reduce batch size
export MPC_BATCH_SIZE=16

# Enable memory pooling
export MPC_GPU_MEMORY_POOL=true

# Use gradient checkpointing
export MPC_GRADIENT_CHECKPOINTING=true
```

#### Network Connection Issues
```bash
# Check connectivity
telnet party1.example.com 8001

# Enable debug logging
export MPC_LOG_LEVEL=DEBUG

# Check firewall settings
sudo ufw status
```

#### Performance Issues
```bash
# Enable profiling
export MPC_ENABLE_PROFILING=true

# Check GPU utilization
nvidia-smi -l 1

# Monitor network usage
iftop -i eth0
```

### Debug Commands

```bash
# Test GPU kernels
python -m secure_mpc_transformer.test_gpu

# Benchmark protocols
python -m secure_mpc_transformer.benchmark \
  --protocols 3pc_semi_honest,3pc_malicious \
  --models bert-base,roberta-base

# Validate installation
python -m secure_mpc_transformer.validate
```

## Next Steps

1. **Read the Architecture Guide**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
2. **Explore Examples**: Check the `examples/` directory
3. **Join the Community**: [Discord](https://discord.gg/secure-mpc)
4. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Support

- **Documentation**: [https://docs.secure-mpc-transformer.org](https://docs.secure-mpc-transformer.org)
- **Issues**: [GitHub Issues](https://github.com/danieleschmidt/secure-mpc-transformer-infer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danieleschmidt/secure-mpc-transformer-infer/discussions)
- **Email**: support@secure-mpc-transformer.org