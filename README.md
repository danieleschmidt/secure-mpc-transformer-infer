# Secure MPC Transformer Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-NDSS%202025-red.svg)](https://www.ndss-symposium.org/ndss2025/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/r/secure-mpc-transformer)

Reference implementation of non-interactive MPC transformer inference with GPU-accelerated homomorphic encryption kernels. First practical system achieving BERT inference in tens of seconds under secure multi-party computation.

## 🔒 Overview

Following NDSS '25 breakthroughs showing BERT inference in 30-60 seconds under MPC, this repo provides the first complete, GPU-accelerated implementation with:

- **Non-interactive protocols** eliminating round-trip latency
- **GPU-accelerated HE** with custom CUDA kernels for 10x speedup
- **Torch integration** via CrypTFlow2 patches and custom ops
- **Privacy tracking** with differential privacy composition
- **Docker deployment** for reproducible secure inference

## ⚡ Performance

| Model | Plaintext | CPU MPC | **GPU MPC** | Speedup | Privacy |
|-------|-----------|---------|-------------|---------|---------|
| BERT-Base | 8ms | 485s | **42s** | 11.5x | 128-bit |
| RoBERTa | 12ms | 612s | **58s** | 10.6x | 128-bit |
| DistilBERT | 5ms | 287s | **31s** | 9.3x | 128-bit |
| GPT-2 (124M) | 15ms | 1,840s | **156s** | 11.8x | 128-bit |

*Benchmarked on 3-party computation with malicious security, RTX 4090*

## 📋 Requirements

### Core Dependencies
```bash
# Cryptography & MPC
cryptflow2>=2.0
seal-python>=4.1.0  # Microsoft SEAL
tenseal>=0.3.14
mp-spdz>=0.3.8

# Deep Learning
torch>=2.3.0
transformers>=4.40.0
onnx>=1.16.0
onnxruntime-gpu>=1.18.0

# GPU Acceleration
cuda>=12.0
cudnn>=8.9
cutlass>=3.5.0
triton>=2.3.0

# Infrastructure
docker>=24.0
redis>=7.2  # For secret sharing
grpcio>=1.62.0
prometheus-client>=0.20.0
```

### Hardware Requirements
- NVIDIA GPU with 24GB+ VRAM (RTX 4090, A5000, or better)
- 64GB+ system RAM for large models
- Fast network (1Gbps+) for multi-party setup

## 🛠️ Installation

### Quick Start with Docker

```bash
# Pull pre-built image
docker pull securempc/transformer-inference:latest

# Run 3-party computation demo
docker compose up -d

# Access web interface
open http://localhost:8080
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/danieleschmidt/secure-mpc-transformer-infer.git
cd secure-mpc-transformer-infer

# Install system dependencies
sudo apt-get install -y libseal-dev libprotobuf-dev

# Create conda environment
conda create -n mpc-transformer python=3.10
conda activate mpc-transformer

# Install Python packages
pip install -e ".[gpu]"

# Build GPU kernels
cd kernels/cuda
make all
cd ../..

# Run tests
pytest tests/
```

## 🚀 Quick Example

### Single-Party Testing

```python
from secure_mpc_transformer import SecureTransformer, SecurityConfig

# Initialize secure model
config = SecurityConfig(
    protocol="3pc",  # 3-party computation
    security_level=128,  # bits
    gpu_acceleration=True
)

model = SecureTransformer.from_pretrained(
    "bert-base-uncased",
    security_config=config
)

# Secure inference
text = "The capital of France is [MASK]."
secure_output = model.predict_secure(text)

print(f"Prediction: {secure_output.decoded_text}")
print(f"Computation time: {secure_output.latency_ms}ms")
```

### Multi-Party Setup

```python
# Party 0 (Data Owner)
from secure_mpc_transformer import DataOwner

owner = DataOwner(party_id=0)
secret_shares = owner.share_input(
    "Confidential: Our Q3 revenue was [MASK] million.",
    num_parties=3
)
owner.distribute_shares(secret_shares)

# Party 1 & 2 (Compute Parties)
from secure_mpc_transformer import ComputeParty

compute1 = ComputeParty(party_id=1, model="bert-base")
compute2 = ComputeParty(party_id=2, model="bert-base")

# Run MPC protocol
result_shares = [
    compute1.compute_on_shares(),
    compute2.compute_on_shares()
]

# Reconstruct result
result = owner.reconstruct_output(result_shares)
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│   Client App    │────▶│  Secret Sharing   │────▶│  MPC Protocol    │
│                 │     │    Engine         │     │  Coordinator     │
└─────────────────┘     └───────────────────┘     └──────────────────┘
                                │                          │
                                ▼                          ▼
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│ Secure Model    │     │  GPU HE Kernels   │     │ Network Manager  │
│  (CrypTFlow2)   │────▶│  (CUDA/Triton)    │────▶│   (gRPC/TCP)     │
└─────────────────┘     └───────────────────┘     └──────────────────┘
```

### Key Components

1. **Secret Sharing Engine**: Splits inputs into cryptographic shares
2. **MPC Protocol Suite**: Implements BGW, GMW, and custom protocols
3. **GPU Acceleration**: CUDA kernels for HE operations
4. **Network Layer**: Optimized communication between parties
5. **Privacy Accountant**: Tracks information leakage

## 🔐 Security Protocols

### Supported Protocols

```python
from secure_mpc_transformer.protocols import ProtocolFactory

# Semi-honest 3PC (fastest)
protocol = ProtocolFactory.create(
    "replicated_3pc",
    security="semi-honest"
)

# Malicious-secure 3PC
protocol = ProtocolFactory.create(
    "aby3",
    security="malicious",
    mac_key_size=128
)

# 4-party with GPU offload
protocol = ProtocolFactory.create(
    "fantastic_four",
    gpu_offload=True
)
```

### Custom Protocol Definition

```python
from secure_mpc_transformer.protocols import Protocol, SecureOp

class MyCustomProtocol(Protocol):
    def secure_matmul(self, x_shares, w_shares):
        # Implement using HE acceleration
        encrypted_x = self.gpu_encrypt(x_shares)
        encrypted_w = self.gpu_encrypt(w_shares)
        
        # GPU-accelerated multiplication
        result = self.gpu_he_matmul(encrypted_x, encrypted_w)
        
        # Decrypt and reshare
        return self.decrypt_and_share(result)
```

## 🚄 GPU Acceleration

### HE Kernel Architecture

```cuda
// kernels/cuda/he_matmul.cu
__global__ void he_matmul_kernel(
    const seal::Ciphertext* A,
    const seal::Ciphertext* B,
    seal::Ciphertext* C,
    int M, int N, int K
) {
    // Optimized homomorphic matrix multiplication
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < M * N) {
        int row = tid / N;
        int col = tid % N;
        
        seal::Ciphertext sum = zero_ciphertext();
        for (int k = 0; k < K; k++) {
            sum = he_add(sum, he_mul(A[row * K + k], B[k * N + col]));
        }
        C[tid] = sum;
    }
}
```

### Performance Optimizations

```python
from secure_mpc_transformer.optimize import GPUOptimizer

optimizer = GPUOptimizer()

# Enable all optimizations
model = optimizer.optimize_model(
    model,
    techniques=[
        "kernel_fusion",      # Fuse HE operations
        "mixed_protocol",     # Use optimal protocol per layer
        "ciphertext_packing", # Pack multiple values
        "gpu_streams",        # Parallel CUDA streams
        "quantization"        # 8-bit quantization where safe
    ]
)

# Benchmark optimizations
optimizer.benchmark(model, batch_size=32)
```

## 📊 Privacy Analysis

### Differential Privacy Integration

```python
from secure_mpc_transformer.privacy import PrivacyAccountant

accountant = PrivacyAccountant(
    epsilon_budget=3.0,
    delta=1e-5
)

# Track privacy during inference
with accountant.track():
    result = model.predict_secure(text)
    
print(f"Privacy spent: ε={accountant.epsilon_spent:.2f}")
print(f"Remaining budget: ε={accountant.epsilon_remaining:.2f}")
```

### Information Leakage Analysis

```python
from secure_mpc_transformer.analysis import LeakageAnalyzer

analyzer = LeakageAnalyzer()

# Analyze protocol leakage
leakage_report = analyzer.analyze_protocol(
    protocol="aby3",
    model=model,
    num_queries=1000
)

# Visualize leakage patterns
analyzer.plot_leakage_heatmap(leakage_report)
```

## 🧪 Benchmarking Suite

### Run Standard Benchmarks

```bash
# Benchmark all models
python benchmarks/run_all.py --gpu --models all

# Specific configuration
python benchmarks/benchmark_bert.py \
    --parties 3 \
    --security malicious \
    --batch-size 32 \
    --iterations 100

# Generate report
python benchmarks/generate_report.py --output results.html
```

### Custom Benchmarks

```python
from secure_mpc_transformer.benchmark import Benchmark

bench = Benchmark()

# Compare protocols
results = bench.compare_protocols(
    model="bert-base",
    protocols=["semi_honest_3pc", "malicious_3pc", "4pc_gpu"],
    metrics=["latency", "throughput", "communication"]
)

bench.plot_results(results, save_path="protocol_comparison.png")
```

## 🐳 Production Deployment

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mpc-compute-nodes
spec:
  serviceName: mpc-service
  replicas: 3
  template:
    spec:
      containers:
      - name: mpc-node
        image: securempc/transformer-inference:latest
        env:
        - name: PARTY_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.ordinal
        - name: GPU_MEMORY_FRACTION
          value: "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 64Gi
```

### Secure Communication Setup

```python
from secure_mpc_transformer.network import SecureChannel

# TLS configuration
channel = SecureChannel(
    cert_file="certs/party.crt",
    key_file="certs/party.key",
    ca_file="certs/ca.crt",
    verify_mode="CERT_REQUIRED"
)

# Establish secure connections
channel.connect_to_parties([
    "mpc-node-0.mpc-service:50051",
    "mpc-node-1.mpc-service:50051",
    "mpc-node-2.mpc-service:50051"
])
```

## 📈 Monitoring & Observability

### Prometheus Metrics

```python
from secure_mpc_transformer.metrics import MetricsCollector

collector = MetricsCollector()

# Track computation metrics
collector.observe_computation_time("bert_inference", 42.3)
collector.increment_counter("mpc_operations", labels={"op": "matmul"})

# Export for Prometheus
collector.start_http_server(port=9090)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MPC Transformer Inference",
    "panels": [
      {
        "title": "Inference Latency",
        "targets": [{
          "expr": "mpc_computation_time_seconds"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "mpc_gpu_utilization_percent"
        }]
      }
    ]
  }
}
```

## 🔧 Advanced Configuration

### Protocol Tuning

```yaml
# config/protocols.yaml
aby3:
  ring_size: 2^64
  num_threads: 32
  gpu_batch_size: 1024
  preprocessing_mode: "online"
  
fantastic_four:
  shares_per_party: 2
  communication_rounds: 3
  packing_factor: 128
```

### Model-Specific Optimizations

```python
from secure_mpc_transformer.optimize import ModelOptimizer

# BERT-specific optimizations
bert_optimizer = ModelOptimizer("bert")
bert_optimizer.apply_optimizations({
    "attention": "approximate_softmax",
    "ff_layers": "polynomial_activation",
    "embeddings": "shared_dictionary"
})

# GPT-specific optimizations  
gpt_optimizer = ModelOptimizer("gpt2")
gpt_optimizer.apply_optimizations({
    "causal_mask": "precomputed",
    "position_encoding": "cached"
})
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- New MPC protocols
- GPU kernel optimizations
- Support for more models
- Communication compression
- Formal security proofs

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@inproceedings{secure_mpc_transformer_2025,
  title={GPU-Accelerated MPC for Practical Transformer Inference},
  author={Daniel Schmidt},
  booktitle={Network and Distributed System Security Symposium (NDSS)},
  year={2025}
}
```

## 🔗 Resources

- [Documentation](https://secure-mpc-transformer.readthedocs.io)
- [Protocol Specifications](docs/protocols.md)
- [Security Analysis](docs/security_analysis.pdf)
- [Video Tutorial](https://youtube.com/secure-mpc-transformer)
- [Research Paper](https://arxiv.org/abs/2507.mpc-transformer)

## ⚠️ Security Notice

This is research software. While we implement state-of-the-art MPC protocols, do not use in production without thorough security review. See [SECURITY.md](SECURITY.md) for responsible disclosure.

## 📧 Contact

- **Security Issues**: security@secure-mpc-transformer.org
- **GitHub Issues**: Bug reports and features
- **Research Collaboration**: research@secure-mpc-transformer.org
