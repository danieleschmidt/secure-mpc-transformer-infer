# User Guide: Secure MPC Transformer

Complete guide for using the Secure MPC Transformer system for privacy-preserving AI inference.

## Overview

The Secure MPC Transformer enables privacy-preserving inference on transformer models using multi-party computation (MPC). Your data remains encrypted and private while still enabling powerful AI predictions.

## Key Benefits

- **Privacy-Preserving**: Your data never leaves your control in plaintext
- **Production-Ready**: GPU-accelerated for real-world performance
- **Secure**: 128-bit cryptographic security against malicious adversaries
- **Compatible**: Works with popular transformer models (BERT, RoBERTa, GPT-2)

## Quick Start

### 1. Installation

Choose your preferred installation method:

#### Option A: Docker (Recommended)
```bash
# Pull and run the pre-built image
docker run --gpus all -p 8080:8080 \
  securempc/transformer-inference:latest
```

#### Option B: Python Package
```bash
pip install secure-mpc-transformer[gpu]
```

#### Option C: Build from Source
```bash
git clone https://github.com/danieleschmidt/secure-mpc-transformer-infer.git
cd secure-mpc-transformer-infer
pip install -e ".[gpu]"
```

### 2. Your First Secure Inference

```python
from secure_mpc_transformer import SecureTransformer, SecurityConfig

# Configure security settings
config = SecurityConfig(
    protocol="3pc_malicious",     # 3-party malicious-secure protocol
    security_level=128,           # 128-bit security
    gpu_acceleration=True         # Enable GPU acceleration
)

# Load model
model = SecureTransformer.from_pretrained(
    "bert-base-uncased",
    security_config=config
)

# Perform secure inference
result = model.predict(
    "The capital of France is [MASK].",
    top_k=5
)

print(f"Prediction: {result.predictions[0].token}")
print(f"Confidence: {result.predictions[0].score:.2f}")
```

### 3. REST API Usage

Start the API server:
```bash
python -m secure_mpc_transformer.server --port 8080
```

Make inference requests:
```bash
curl -X POST http://localhost:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is [MASK].",
    "model": "bert-base-uncased",
    "security_level": 128,
    "top_k": 5
  }'
```

## Core Concepts

### Multi-Party Computation (MPC)

MPC allows multiple parties to jointly compute a function while keeping their inputs private:

```
Party 1 (You)     Party 2 (Cloud)    Party 3 (Cloud)
    |                   |                   |
  Input A            Input B            Input C
    |                   |                   |
    └────────── Secure Computation ────────┘
                         |
                   Final Result
```

**Key Properties:**
- **Privacy**: No party sees others' raw inputs
- **Correctness**: Result is mathematically correct
- **Security**: Protects against malicious behavior

### Security Protocols

| Protocol | Security Level | Performance | Best For |
|----------|---------------|-------------|----------|
| `3pc_semi_honest` | Semi-honest adversaries | Fastest | Development/testing |
| `3pc_malicious` | Malicious adversaries | Moderate | Production use |
| `aby3` | Malicious with abort | Slower | High-security applications |

### GPU Acceleration

GPU acceleration provides significant speedup:

```python
# Performance comparison
config_cpu = SecurityConfig(gpu_acceleration=False)
config_gpu = SecurityConfig(gpu_acceleration=True)

# CPU: ~8 minutes for BERT inference
model_cpu = SecureTransformer.from_pretrained("bert-base", config=config_cpu)

# GPU: ~45 seconds for BERT inference  
model_gpu = SecureTransformer.from_pretrained("bert-base", config=config_gpu)
```

## Usage Patterns

### Text Classification

```python
from secure_mpc_transformer import SecureTransformer

model = SecureTransformer.from_pretrained("bert-base-uncased")

# Sentiment analysis
text = "This product exceeded my expectations!"
result = model.classify(text, labels=["positive", "negative", "neutral"])

print(f"Sentiment: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Multi-class classification
text = "Breaking: Stock market reaches new highs"
result = model.classify(text, labels=["politics", "sports", "business", "technology"])
```

### Question Answering

```python
# Load QA model
qa_model = SecureTransformer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
"""

question = "Who designed the Eiffel Tower?"

answer = qa_model.answer_question(
    question=question,
    context=context
)

print(f"Answer: {answer.text}")
print(f"Confidence: {answer.confidence:.2f}")
print(f"Start position: {answer.start}")
print(f"End position: {answer.end}")
```

### Named Entity Recognition

```python
# Load NER model
ner_model = SecureTransformer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."

entities = ner_model.extract_entities(text)

for entity in entities:
    print(f"{entity.text}: {entity.label} (confidence: {entity.confidence:.2f})")

# Output:
# Apple Inc.: ORG (confidence: 0.99)
# Steve Jobs: PER (confidence: 0.98)
# Cupertino: LOC (confidence: 0.95)
# California: LOC (confidence: 0.97)
```

### Text Generation

```python
# Load GPT-2 model
gpt_model = SecureTransformer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence will"

generated = gpt_model.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    top_p=0.9
)

print(f"Generated text: {generated.text}")
```

### Batch Processing

```python
# Process multiple inputs efficiently
texts = [
    "This movie is fantastic!",
    "The weather is terrible today.",
    "I love this new restaurant.",
    "The service was disappointing."
]

# Batch inference for better performance
results = model.classify_batch(
    texts,
    labels=["positive", "negative"],
    batch_size=8
)

for text, result in zip(texts, results):
    print(f"'{text}' -> {result.label} ({result.confidence:.2f})")
```

## Multi-Party Setup

### Single Machine (Development)

For development and testing, you can run all parties on one machine:

```python
from secure_mpc_transformer import MPCCoordinator

# Start local 3-party setup
coordinator = MPCCoordinator(
    num_parties=3,
    local=True,
    ports=[8001, 8002, 8003]
)

coordinator.start_all_parties()

try:
    result = coordinator.run_inference(
        "The capital of [MASK] is Berlin.",
        model="bert-base-uncased"
    )
    print(f"Result: {result}")
finally:
    coordinator.stop_all_parties()
```

### Distributed Setup (Production)

For production, deploy parties across different machines:

#### Data Owner (Your Machine)
```python
from secure_mpc_transformer import DataOwnerParty

party = DataOwnerParty(
    party_id=0,
    peers=["compute1.example.com:8001", "compute2.example.com:8001"],
    tls_cert="/path/to/cert.pem",
    tls_key="/path/to/key.pem"
)

party.start()

# Share your data securely
secret_shares = party.share_input("Confidential business data for analysis")
party.distribute_shares(secret_shares)

# Get result
result = party.get_computation_result()
print(f"Secure result: {result}")
```

#### Compute Parties (Cloud Providers)
```bash
# Compute Party 1
python -m secure_mpc_transformer.party \
  --party-id 1 \
  --role compute \
  --peers data-owner.example.com:8001,compute2.example.com:8001 \
  --port 8001 \
  --tls-cert /path/to/cert.pem \
  --tls-key /path/to/key.pem

# Compute Party 2  
python -m secure_mpc_transformer.party \
  --party-id 2 \
  --role compute \
  --peers data-owner.example.com:8001,compute1.example.com:8001 \
  --port 8001 \
  --tls-cert /path/to/cert.pem \
  --tls-key /path/to/key.pem
```

## Configuration

### Security Configuration

```python
from secure_mpc_transformer import SecurityConfig

# Basic configuration
config = SecurityConfig(
    protocol="3pc_malicious",
    security_level=128,
    gpu_acceleration=True
)

# Advanced configuration
config = SecurityConfig(
    # Protocol settings
    protocol="3pc_malicious",
    security_level=256,                    # Higher security
    
    # Performance settings
    gpu_acceleration=True,
    batch_size=32,
    worker_threads=8,
    
    # Privacy settings
    differential_privacy=True,
    dp_epsilon=1.0,                        # Privacy budget
    dp_delta=1e-5,                         # Delta parameter
    
    # Network settings
    tls_enabled=True,
    network_timeout=30,
    compression_enabled=True,
    
    # Debugging
    debug_mode=False,
    log_level="INFO"
)
```

### Environment Variables

```bash
# Security settings
export MPC_SECURITY_LEVEL=128
export MPC_PROTOCOL=3pc_malicious
export MPC_GPU_ACCELERATION=true

# Performance settings
export MPC_BATCH_SIZE=32
export MPC_WORKER_THREADS=8
export MPC_GPU_MEMORY_LIMIT=20GB

# Network settings
export MPC_PORT=8001
export MPC_TLS_ENABLED=true
export MPC_NETWORK_TIMEOUT=30

# Privacy settings
export MPC_DIFFERENTIAL_PRIVACY=true
export MPC_DP_EPSILON=1.0
export MPC_DP_DELTA=1e-5
```

### Configuration File

```yaml
# config/mpc-config.yaml
security:
  protocol: "3pc_malicious"
  security_level: 128
  tls_enabled: true
  certificate_path: "/path/to/cert.pem"
  private_key_path: "/path/to/key.pem"

performance:
  gpu_acceleration: true
  batch_size: 32
  worker_threads: 8
  memory_limit: "20GB"
  optimization_level: 2

privacy:
  differential_privacy: true
  epsilon: 1.0
  delta: 1e-5
  noise_mechanism: "gaussian"

networking:
  port: 8001
  timeout: 30
  compression: true
  buffer_size: "64KB"

models:
  cache_directory: "/app/models"
  preload_models:
    - "bert-base-uncased"
    - "roberta-base"
  default_model: "bert-base-uncased"
```

## Performance Optimization

### GPU Optimization

```python
# Optimize for your specific GPU
config = SecurityConfig(
    gpu_acceleration=True,
    gpu_optimization_level=3,        # Maximum optimization
    gpu_memory_pool=True,            # Pre-allocate memory
    mixed_precision=True,            # Use FP16 where safe
    kernel_fusion=True               # Fuse operations
)

# Multi-GPU support
config = SecurityConfig(
    gpu_acceleration=True,
    gpu_devices=[0, 1, 2, 3],       # Use multiple GPUs
    model_parallelism=True           # Split model across GPUs
)
```

### Batch Processing

```python
# Process multiple inputs together for better throughput
texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

# Better: Process as batch
results = model.classify_batch(texts, batch_size=4)

# Worse: Process individually
results = [model.classify(text) for text in texts]
```

### Model Optimization

```python
# Optimize model for inference
optimized_model = model.optimize_for_inference(
    techniques=[
        "quantization",              # Reduce precision
        "kernel_fusion",             # Fuse operations
        "memory_optimization",       # Optimize memory layout
        "protocol_selection"         # Choose optimal protocol per layer
    ]
)

# Pre-compile for target hardware
compiled_model = optimized_model.compile(
    target_gpu="RTX4090",
    optimization_level=3
)
```

## Monitoring and Observability

### Built-in Metrics

```python
from secure_mpc_transformer.monitoring import MetricsCollector

collector = MetricsCollector()
collector.start()

# Your inference code
result = model.predict("Sample text")

# Get performance metrics
metrics = collector.get_metrics()
print(f"Inference time: {metrics.inference_time:.2f}s")
print(f"GPU utilization: {metrics.gpu_utilization:.1%}")
print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
print(f"Network latency: {metrics.network_latency:.2f}ms")
```

### Health Monitoring

```python
# Check system health
health = model.get_health_status()
print(f"System status: {health.status}")
print(f"GPU available: {health.gpu_available}")
print(f"Network connectivity: {health.network_status}")
print(f"Security status: {health.security_status}")

# Party connectivity
party_status = model.check_party_connectivity()
for party_id, status in party_status.items():
    print(f"Party {party_id}: {status}")
```

### Prometheus Integration

```python
# Export metrics to Prometheus
from secure_mpc_transformer.monitoring import PrometheusExporter

exporter = PrometheusExporter(port=9090)
exporter.start()

# Metrics are automatically exported:
# - mpc_inference_duration_seconds
# - mpc_gpu_utilization_percent  
# - mpc_memory_usage_bytes
# - mpc_network_latency_seconds
# - mpc_security_violations_total
```

## Error Handling and Troubleshooting

### Common Issues

#### GPU Out of Memory
```python
try:
    result = model.predict(text)
except GPUOutOfMemoryError:
    # Reduce batch size and retry
    model.config.batch_size = model.config.batch_size // 2
    result = model.predict(text)
```

#### Network Connectivity Issues
```python
try:
    result = model.predict(text)
except NetworkTimeoutError as e:
    print(f"Network timeout: {e}")
    # Check party connectivity
    status = model.check_party_connectivity()
    print(f"Party status: {status}")
```

#### Security Violations
```python
try:
    result = model.predict(text)
except SecurityViolationError as e:
    print(f"Security violation detected: {e}")
    # Check security logs
    logs = model.get_security_logs()
    print(f"Recent security events: {logs}")
```

### Debug Mode

```python
# Enable debug mode for detailed logging
config = SecurityConfig(
    debug_mode=True,
    log_level="DEBUG",
    trace_protocols=True
)

model = SecureTransformer.from_pretrained(
    "bert-base-uncased",
    security_config=config
)

# Debug information will be logged
result = model.predict("Debug this inference")
```

## Security Best Practices

### Input Validation

```python
def validate_input(text: str) -> str:
    """Validate and sanitize user input."""
    # Check length
    if len(text) > 10000:
        raise ValueError("Input too long")
    
    # Remove potentially harmful content
    sanitized = sanitize_text(text)
    
    return sanitized

# Use validation in your code
user_input = validate_input(user_provided_text)
result = model.predict(user_input)
```

### Key Management

```python
# Rotate keys regularly
from secure_mpc_transformer.security import KeyManager

key_manager = KeyManager()
key_manager.rotate_encryption_keys()
key_manager.rotate_signing_keys()

# Backup keys securely
key_manager.backup_keys("/secure/backup/location")
```

### Audit Logging

```python
# Enable comprehensive audit logging
from secure_mpc_transformer.security import AuditLogger

audit_logger = AuditLogger()
audit_logger.enable()

# All security-relevant events are logged
result = model.predict("Sensitive data")

# Review audit logs
logs = audit_logger.get_logs(since="2024-01-01")
for log in logs:
    print(f"{log.timestamp}: {log.event_type} - {log.message}")
```

## Integration Examples

### Web Application Integration

```python
from flask import Flask, request, jsonify
from secure_mpc_transformer import SecureTransformer

app = Flask(__name__)
model = SecureTransformer.from_pretrained("bert-base-uncased")

@app.route("/api/classify", methods=["POST"])
def classify_text():
    try:
        data = request.get_json()
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        result = model.classify(text, labels=["positive", "negative"])
        
        return jsonify({
            "prediction": result.label,
            "confidence": result.confidence,
            "processing_time": result.processing_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### Microservices Architecture

```python
# Async microservice using FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from secure_mpc_transformer import SecureTransformer

app = FastAPI()
model = SecureTransformer.from_pretrained("bert-base-uncased")

class InferenceRequest(BaseModel):
    text: str
    model_name: str = "bert-base-uncased"
    security_level: int = 128

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float

@app.post("/inference", response_model=InferenceResponse)
async def perform_inference(request: InferenceRequest):
    try:
        result = await model.predict_async(request.text)
        return InferenceResponse(
            prediction=result.prediction,
            confidence=result.confidence,
            processing_time=result.processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Jupyter Notebook Usage

```python
# Install in Jupyter environment
!pip install secure-mpc-transformer[gpu]

# Import and setup
from secure_mpc_transformer import SecureTransformer, SecurityConfig
import matplotlib.pyplot as plt

# Configure for notebook use
config = SecurityConfig(
    protocol="3pc_semi_honest",  # Faster for development
    gpu_acceleration=True,
    debug_mode=True
)

model = SecureTransformer.from_pretrained("bert-base-uncased", security_config=config)

# Interactive analysis
texts = ["Great product!", "Terrible service.", "Average quality."]
results = model.classify_batch(texts, labels=["positive", "negative"])

# Visualize results
confidences = [r.confidence for r in results]
labels = [r.label for r in results]

plt.bar(range(len(texts)), confidences, color=['green' if l == 'positive' else 'red' for l in labels])
plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))])
plt.ylabel("Confidence")
plt.title("Sentiment Analysis Results")
plt.show()
```

## Advanced Features

### Differential Privacy

```python
# Enable differential privacy for additional protection
config = SecurityConfig(
    differential_privacy=True,
    dp_epsilon=1.0,           # Privacy budget
    dp_delta=1e-5,            # Delta parameter
    dp_mechanism="gaussian"    # Noise mechanism
)

model = SecureTransformer.from_pretrained("bert-base-uncased", security_config=config)

# Privacy budget tracking
privacy_budget = model.get_privacy_budget()
print(f"Remaining privacy budget: {privacy_budget.remaining_epsilon}")
```

### Custom Models

```python
# Use your own fine-tuned models
model = SecureTransformer.from_pretrained("/path/to/your/custom/model")

# Or load from Hugging Face Hub
model = SecureTransformer.from_pretrained("your-username/your-model")
```

### Federated Learning Integration

```python
# Participate in federated learning
from secure_mpc_transformer.federated import FederatedClient

client = FederatedClient(
    client_id="your_client_id",
    server_url="https://federated-server.example.com"
)

# Contribute to model training while preserving privacy
client.participate_in_training(
    local_data=your_private_data,
    model_name="bert-base-uncased"
)
```

## Support and Community

### Getting Help

- **Documentation**: [https://docs.secure-mpc-transformer.org](https://docs.secure-mpc-transformer.org)
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time community support
- **Email**: support@secure-mpc-transformer.org

### Contributing

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for details.

### Professional Support

For enterprise customers:
- **Professional Services**: Custom integration support
- **Security Audits**: Third-party security assessments  
- **Training**: On-site training and workshops
- **SLA Support**: 24/7 support with SLA guarantees

Contact: enterprise@secure-mpc-transformer.org

## What's Next?

1. **Explore Examples**: Check out the `examples/` directory
2. **Read Architecture Guide**: Understand the system internals
3. **Join Community**: Connect with other users on Discord
4. **Contribute**: Help improve the project

The Secure MPC Transformer enables a new era of privacy-preserving AI. Start building secure applications today!