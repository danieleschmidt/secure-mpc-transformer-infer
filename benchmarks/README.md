# Performance Benchmarking Suite

This directory contains comprehensive performance benchmarking tools for the Secure MPC Transformer project.

## Overview

The benchmarking suite evaluates:
- **Inference Latency**: Time to process individual inputs
- **Throughput**: Requests processed per second
- **Memory Usage**: RAM and GPU memory consumption
- **Protocol Efficiency**: MPC protocol overhead
- **Scalability**: Performance across different input sizes

## Benchmark Scripts

### Core Benchmarks

| Script | Purpose | Hardware | Duration |
|--------|---------|----------|----------|
| `benchmark_bert.py` | BERT model inference | CPU/GPU | 5-10 min |
| `benchmark_protocols.py` | MPC protocol comparison | CPU/GPU | 15-30 min |
| `benchmark_scalability.py` | Scaling behavior analysis | GPU | 30-60 min |
| `benchmark_memory.py` | Memory usage profiling | CPU/GPU | 10-20 min |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `run_all.py` | Execute full benchmark suite |
| `compare_results.py` | Compare benchmark runs |
| `generate_report.py` | Create performance reports |
| `check_regression.py` | Detect performance regressions |

## Quick Start

### Run Basic Benchmarks

```bash
# Single model benchmark
python benchmarks/benchmark_bert.py --model bert-base-uncased --iterations 100

# Protocol comparison
python benchmarks/benchmark_protocols.py --protocols semi_honest_3pc,malicious_3pc

# Full benchmark suite
python benchmarks/run_all.py --quick
```

### GPU Benchmarks

```bash
# Ensure GPU is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run GPU-optimized benchmarks
python benchmarks/benchmark_bert.py --gpu --batch-size 32

# Memory profiling with GPU
python benchmarks/benchmark_memory.py --gpu --profile-memory
```

## Benchmark Configuration

### Environment Variables

```bash
export BENCHMARK_GPU=1                    # Enable GPU benchmarks
export BENCHMARK_ITERATIONS=100           # Number of iterations
export BENCHMARK_WARMUP=10               # Warmup iterations
export BENCHMARK_TIMEOUT=3600            # Timeout in seconds
export BENCHMARK_RESULTS_DIR=results/    # Results directory
```

### Configuration File

Create `benchmarks/config.yaml`:

```yaml
# Benchmark configuration
default:
  iterations: 100
  warmup: 10
  timeout: 3600
  
models:
  bert-base:
    batch_sizes: [1, 8, 16, 32]
    sequence_lengths: [128, 256, 512]
  
protocols:
  semi_honest_3pc:
    security_level: 128
    num_parties: 3
  malicious_3pc:
    security_level: 128
    num_parties: 3
    
hardware:
  cpu:
    threads: [1, 2, 4, 8]
  gpu:
    memory_fraction: 0.9
    mixed_precision: true
```

## Results Analysis

### Output Formats

Benchmarks support multiple output formats:

```bash
# JSON format (default)
python benchmark_bert.py --output results.json

# CSV format
python benchmark_bert.py --output results.csv --format csv

# Human-readable format
python benchmark_bert.py --format table
```

### Results Structure

```json
{
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "git_commit": "abc123",
    "python_version": "3.10.12",
    "torch_version": "2.3.0",
    "cuda_version": "12.0"
  },
  "system_info": {
    "cpu": "Intel Xeon E5-2686 v4",
    "memory_gb": 64,
    "gpu": "NVIDIA RTX 4090",
    "gpu_memory_gb": 24
  },
  "benchmarks": [
    {
      "name": "bert_inference",
      "config": {
        "model": "bert-base-uncased", 
        "batch_size": 1,
        "sequence_length": 128
      },
      "metrics": {
        "latency_ms": 42.3,
        "throughput_qps": 23.6,
        "memory_mb": 1024,
        "gpu_memory_mb": 2048
      },
      "statistics": {
        "mean": 42.3,
        "median": 41.8,
        "std": 2.1,
        "min": 38.7,
        "max": 48.9,
        "p95": 46.2,
        "p99": 47.8
      }
    }
  ]
}
```

## Continuous Benchmarking

### GitHub Actions Integration

Benchmarks integrate with CI/CD via `.github/workflows/benchmark.yml`:

```yaml
# Trigger benchmarks on PR with [benchmark] label
- name: Run Performance Benchmarks
  if: contains(github.event.label.name, 'benchmark')
  run: python benchmarks/run_all.py --quick --output pr-results.json
```

### Regression Detection

```bash
# Compare current results with baseline
python benchmarks/check_regression.py \
  --current results/current.json \
  --baseline results/baseline.json \
  --threshold 0.1

# Generate regression report
python benchmarks/generate_report.py \
  --type regression \
  --current results/current.json \
  --baseline results/baseline.json
```

## Custom Benchmarks

### Adding New Benchmarks

1. Create benchmark script in `benchmarks/`
2. Follow the benchmark interface:

```python
from benchmarks.base import BaseBenchmark

class MyCustomBenchmark(BaseBenchmark):
    def setup(self):
        # Initialize resources
        pass
    
    def run_iteration(self):
        # Single benchmark iteration
        # Return metrics dict
        return {"latency_ms": 42.0}
    
    def teardown(self):
        # Clean up resources
        pass
```

3. Register in `benchmarks/registry.py`
4. Add configuration to `benchmarks/config.yaml`

### Benchmark Best Practices

- **Warmup**: Always include warmup iterations
- **Statistics**: Collect multiple samples for statistical significance
- **Isolation**: Minimize external factors affecting results
- **Reproducibility**: Document hardware and software configuration
- **Baseline**: Maintain baseline results for comparison

## Performance Monitoring

### Grafana Dashboard

Performance metrics can be visualized in Grafana:

1. Configure Prometheus metrics collection
2. Import dashboard from `monitoring/grafana/performance-dashboard.json`
3. View trends and alerts

### Alerts

Set up performance regression alerts:

```yaml
# prometheus/alerts.yml
groups:
  - name: performance
    rules:
      - alert: PerformanceRegression
        expr: benchmark_latency_ms > benchmark_baseline_ms * 1.2
        for: 5m
        annotations:
          summary: "Performance regression detected"
```

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4 cores, 8GB RAM
- **Storage**: 10GB free space
- **Network**: 100Mbps for multi-party benchmarks

### Recommended Requirements

- **CPU**: 8+ cores, 32GB+ RAM
- **GPU**: NVIDIA RTX 4090 or better, 24GB+ VRAM
- **Storage**: SSD with 50GB+ free space
- **Network**: 1Gbps+ for realistic MPC scenarios

## Troubleshooting

### Common Issues

#### GPU Out of Memory

```bash
# Reduce batch size
python benchmark_bert.py --batch-size 1

# Use CPU fallback
python benchmark_bert.py --no-gpu

# Monitor GPU memory
nvidia-smi -l 1
```

#### Inconsistent Results

```bash
# Increase warmup iterations
python benchmark_bert.py --warmup 20

# Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Check thermal throttling
cat /proc/cpuinfo | grep MHz
```

#### Network Issues in MPC Benchmarks

```bash
# Test network connectivity
ping -c 5 mpc-party-1

# Check port availability
netstat -an | grep 50051

# Use local loopback
python benchmark_protocols.py --localhost
```

## Contributing

When adding benchmarks:

1. Follow naming convention: `benchmark_*.py`
2. Include comprehensive documentation
3. Add unit tests in `tests/benchmarks/`
4. Update this README
5. Ensure reproducible results

## Support

For benchmark-related issues:

- Check [troubleshooting section](#troubleshooting)
- Review existing benchmark implementations
- Consult performance optimization guides
- Open GitHub issue with benchmark logs