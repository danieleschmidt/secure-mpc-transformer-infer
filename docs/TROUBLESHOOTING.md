# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Secure MPC Transformer system.

## Quick Diagnosis

### System Health Check

```bash
# Run comprehensive health check
python -m secure_mpc_transformer.diagnostics --full

# Check individual components
python -m secure_mpc_transformer.diagnostics --gpu
python -m secure_mpc_transformer.diagnostics --network
python -m secure_mpc_transformer.diagnostics --protocols
```

### Common Issues Quick Reference

| Issue | Quick Fix | Details |
|-------|-----------|---------|
| GPU OOM | Reduce batch size | [GPU Issues](#gpu-issues) |
| Network timeout | Check firewall | [Network Issues](#network-issues) |
| Slow inference | Enable GPU acceleration | [Performance Issues](#performance-issues) |
| Protocol failure | Verify party coordination | [Protocol Issues](#protocol-issues) |

## Installation Issues

### Docker Issues

#### Container Won't Start
```bash
# Check logs
docker logs secure-mpc-transformer

# Common causes and fixes
docker system prune -f              # Clean up disk space
docker pull securempc/transformer-inference:latest  # Update image
docker compose down && docker compose up -d         # Restart services
```

#### GPU Not Detected in Container
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Fix: Install nvidia-docker2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Build Issues

#### Missing Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
  libseal-dev \
  libprotobuf-dev \
  build-essential \
  cmake \
  cuda-toolkit-12-0

# macOS
brew install protobuf cmake
```

#### CUDA Compilation Errors
```bash
# Check CUDA version compatibility
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Rebuild with specific CUDA version
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
pip install --force-reinstall torch torchvision torchaudio
```

## Runtime Issues

### GPU Issues

#### Out of Memory Errors
```python
# Symptoms
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB

# Solutions
import os
os.environ['MPC_BATCH_SIZE'] = '16'  # Reduce batch size
os.environ['MPC_GPU_MEMORY_LIMIT'] = '16GB'  # Set memory limit
os.environ['MPC_ENABLE_MEMORY_POOL'] = 'true'  # Enable memory pooling

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

#### GPU Kernel Launch Failures
```bash
# Check GPU status
nvidia-smi

# Common fixes
sudo nvidia-smi -r                    # Reset GPU
sudo modprobe -r nvidia && sudo modprobe nvidia  # Reload driver

# Check for overheating
nvidia-smi -q -d temperature
```

#### Performance Degradation
```python
# Enable GPU profiling
from secure_mpc_transformer.profiling import GPUProfiler

profiler = GPUProfiler()
profiler.start()

# Your inference code here
result = model.inference(text)

report = profiler.stop()
print(f"GPU utilization: {report.gpu_utilization:.1%}")
print(f"Memory usage: {report.memory_usage}")
```

### Network Issues

#### Connection Timeouts
```bash
# Test connectivity between parties
telnet party1.example.com 8001
nc -zv party1.example.com 8001

# Check firewall rules
sudo ufw status
iptables -L

# Enable debug logging
export MPC_LOG_LEVEL=DEBUG
export MPC_NETWORK_DEBUG=true
```

#### TLS Certificate Issues
```bash
# Generate new certificates
python -m secure_mpc_transformer.generate_certs \
  --hostnames party0.example.com,party1.example.com,party2.example.com

# Test TLS connection
openssl s_client -connect party1.example.com:8001 -servername party1.example.com
```

#### Message Corruption
```python
# Enable message verification
config = SecurityConfig(
    enable_message_verification=True,
    network_checksum=True,
    retry_on_corruption=True
)

# Check network statistics
from secure_mpc_transformer.monitoring import NetworkMonitor
monitor = NetworkMonitor()
stats = monitor.get_network_stats()
print(f"Corrupted messages: {stats.corrupted_messages}")
print(f"Retransmissions: {stats.retransmissions}")
```

### Protocol Issues

#### Party Synchronization Failures
```python
# Check party status
from secure_mpc_transformer import MPCCoordinator

coordinator = MPCCoordinator()
status = coordinator.check_party_status()
for party_id, party_status in status.items():
    print(f"Party {party_id}: {party_status}")

# Resynchronize parties
coordinator.resynchronize_parties()
```

#### Share Reconstruction Errors
```python
# Symptoms
ValueError: Invalid share reconstruction: shares do not match

# Debugging
config = SecurityConfig(
    enable_share_verification=True,
    redundant_shares=True,
    corruption_detection=True
)

# Check for timing attacks
from secure_mpc_transformer.security import TimingAnalyzer
analyzer = TimingAnalyzer()
analyzer.detect_timing_anomalies(operation_logs)
```

### Performance Issues

#### Slow Inference Times
```python
# Enable detailed profiling
import cProfile
profiler = cProfile.Profile()
profiler.enable()

result = model.inference(text)

profiler.disable()
profiler.dump_stats('inference_profile.prof')

# Analyze with snakeviz
# pip install snakeviz
# snakeviz inference_profile.prof
```

#### High Memory Usage
```python
# Memory profiling
from memory_profiler import profile

@profile
def run_inference():
    return model.inference("Sample text")

result = run_inference()

# Check for memory leaks
import gc
import psutil

before = psutil.Process().memory_info().rss
result = model.inference(text)
gc.collect()
after = psutil.Process().memory_info().rss
print(f"Memory increase: {(after - before) / 1024 / 1024:.1f} MB")
```

#### CPU Bottlenecks
```bash
# CPU profiling
perf record -g python main.py
perf report

# Check CPU affinity
taskset -c 0-7 python main.py  # Use specific CPU cores

# Enable NUMA optimization
numactl --cpubind=0 --membind=0 python main.py
```

## Security Issues

### Certificate Problems
```bash
# Check certificate validity
openssl x509 -in cert.pem -text -noout

# Verify certificate chain
openssl verify -CAfile ca.pem cert.pem

# Generate new certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### Key Management Issues
```python
# Rotate keys
from secure_mpc_transformer.security import KeyManager

key_manager = KeyManager()
key_manager.rotate_all_keys()

# Verify key integrity
key_manager.verify_key_integrity()

# Backup keys securely
key_manager.backup_keys('/secure/backup/location')
```

### Security Audit Failures
```bash
# Run security audit
python -m secure_mpc_transformer.security.audit --full

# Fix common vulnerabilities
python -m secure_mpc_transformer.security.fix --auto

# Generate security report
python -m secure_mpc_transformer.security.report --output security_report.pdf
```

## Monitoring and Debugging

### Enable Debug Logging
```python
import logging
from secure_mpc_transformer import set_log_level

# Set debug level
set_log_level(logging.DEBUG)

# Enable specific component logging
logging.getLogger('secure_mpc_transformer.protocols').setLevel(logging.DEBUG)
logging.getLogger('secure_mpc_transformer.gpu').setLevel(logging.DEBUG)
logging.getLogger('secure_mpc_transformer.network').setLevel(logging.DEBUG)
```

### Performance Monitoring
```python
from secure_mpc_transformer.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Your code here
result = model.inference(text)

metrics = monitor.stop()
print(f"Total time: {metrics.total_time:.2f}s")
print(f"GPU time: {metrics.gpu_time:.2f}s")
print(f"Network time: {metrics.network_time:.2f}s")
print(f"CPU utilization: {metrics.cpu_utilization:.1%}")
```

### Memory Profiling
```python
from secure_mpc_transformer.profiling import MemoryProfiler

profiler = MemoryProfiler()
with profiler:
    result = model.inference(text)

report = profiler.get_report()
print(f"Peak memory: {report.peak_memory_mb:.1f} MB")
print(f"GPU memory: {report.gpu_memory_mb:.1f} MB")
```

## Advanced Debugging

### Protocol Trace Analysis
```python
# Enable protocol tracing
config = SecurityConfig(
    enable_protocol_trace=True,
    trace_detail_level=3
)

model = SecureTransformer.from_pretrained(
    "bert-base-uncased",
    security_config=config
)

# Analyze trace
from secure_mpc_transformer.debugging import ProtocolTracer
tracer = ProtocolTracer()
tracer.load_trace('protocol_trace.log')
tracer.analyze_timing()
tracer.detect_anomalies()
```

### GPU Kernel Debugging
```bash
# Use CUDA debugger
cuda-gdb python
(gdb) set environment CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
(gdb) run main.py

# Profile GPU kernels
nvprof python main.py
nsight-sys python main.py
```

### Network Analysis
```bash
# Capture network traffic
sudo tcpdump -i any -w mpc_traffic.pcap host party1.example.com

# Analyze with Wireshark
wireshark mpc_traffic.pcap

# Monitor bandwidth usage
iftop -i eth0
nethogs
```

## Getting Help

### Information to Collect

When reporting issues, please include:

```bash
# System information
python -m secure_mpc_transformer.diagnostics --system-info

# Error logs
tail -n 100 /var/log/secure-mpc-transformer/error.log

# Configuration
cat config/mpc-config.yaml

# Performance metrics
python -m secure_mpc_transformer.benchmark --quick
```

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time community support
- **Email**: security@secure-mpc-transformer.org for security issues
- **Documentation**: https://docs.secure-mpc-transformer.org

### Professional Support

For enterprise support and consulting:
- **Professional Services**: support@secure-mpc-transformer.org
- **Security Audits**: security-audit@secure-mpc-transformer.org
- **Custom Development**: consulting@secure-mpc-transformer.org

## Contributing Fixes

Found a bug? Please contribute back:

1. Fork the repository
2. Create a feature branch: `git checkout -b fix/issue-description`
3. Add tests for your fix
4. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.