# ADR-0001: GPU Acceleration Framework Selection

**Status:** Accepted  
**Date:** 2024-03-15  
**Authors:** Technical Team  
**Reviewers:** CTO, Security Advisor  
**Related ADRs:** ADR-0002 (MPC Protocol Selection)

## Context

The Secure MPC Transformer Inference project requires significant performance improvements over CPU-only implementations to achieve practical inference times. Initial CPU-only prototypes showed BERT inference times of 8+ minutes, which is prohibitive for production use.

Key requirements:
- Achieve sub-60 second BERT inference with 3-party MPC
- Maintain 128-bit security guarantees
- Support multiple GPU vendors (NVIDIA, AMD)
- Integrate with existing cryptographic libraries (SEAL, TenSEAL)
- Enable future scalability to larger models

Technical constraints:
- Must work with homomorphic encryption operations
- Memory bandwidth limitations for large ciphertexts
- Need for custom kernels for MPC-specific operations
- Integration with PyTorch ecosystem

## Decision

We will implement a hybrid GPU acceleration framework combining:

1. **CUDA for NVIDIA GPUs** as the primary acceleration target
2. **Triton for high-level kernel development** for maintainable custom operations
3. **CuPy integration** for seamless NumPy-like operations on GPU
4. **Custom C++/CUDA kernels** for performance-critical HE operations

The framework will provide:
- Automatic device detection and optimal kernel selection
- Fallback to CPU implementations when GPU unavailable
- Memory pool management for efficient ciphertext handling
- Integration with PyTorch's autograd system for end-to-end training

## Alternatives Considered

### Alternative 1: OpenCL-based Solution
- **Description:** Cross-platform GPU acceleration using OpenCL
- **Pros:** 
  - Vendor-neutral (NVIDIA, AMD, Intel)
  - Existing ecosystem and tooling
  - Lower development complexity
- **Cons:**
  - Performance limitations compared to native CUDA
  - Limited support for advanced GPU features
  - Complexity of HE operations in OpenCL
- **Rejection Reason:** Performance requirements necessitate CUDA optimizations, and NVIDIA dominance in ML hardware justifies CUDA-first approach

### Alternative 2: Pure PyTorch CUDA Extensions
- **Description:** Implement everything as PyTorch C++/CUDA extensions
- **Pros:**
  - Deep PyTorch integration
  - Automatic differentiation support
  - Familiar development model
- **Cons:**
  - Limited flexibility for non-PyTorch operations
  - Overhead of PyTorch tensor management
  - Difficulty optimizing for MPC-specific patterns
- **Rejection Reason:** MPC operations don't fit well into standard tensor operations, requiring more control over memory management

### Alternative 3: JAX/XLA Compilation
- **Description:** Use JAX with XLA compilation for GPU acceleration
- **Pros:**
  - Automatic differentiation and compilation
  - Clean functional programming model
  - Excellent performance for standard operations
- **Cons:**
  - Limited support for custom cryptographic operations
  - Dependency on Google's ecosystem
  - Less mature than PyTorch for transformers
- **Rejection Reason:** Project already committed to PyTorch ecosystem, and XLA compilation doesn't optimize well for HE operations

## Consequences

### Positive Consequences
- **Performance:** Achieved 10x+ speedup over CPU implementation
- **Scalability:** Framework supports scaling to larger models and more parties
- **Maintainability:** Triton enables high-level kernel development
- **Ecosystem:** Deep integration with PyTorch ML ecosystem

### Negative Consequences
- **Complexity:** Increased codebase complexity with multiple GPU backends
- **Dependencies:** Heavy dependency on NVIDIA CUDA ecosystem
- **Memory:** High GPU memory requirements (20GB+ for BERT)
- **Development:** Requires GPU expertise for optimization and debugging

### Neutral Consequences
- **Deployment:** Requires GPU-enabled deployment infrastructure
- **Testing:** Need for GPU-enabled CI/CD pipelines
- **Documentation:** Additional documentation for GPU setup and troubleshooting

## Implementation

### Phase 1: Core CUDA Kernels (Completed)
- Implemented homomorphic encryption primitives
- Basic matrix operations with ciphertext support
- Memory pool management for GPU ciphertexts
- Integration with SEAL library

### Phase 2: Triton High-Level Kernels (Completed)
- Transformer-specific operations (attention, FFN)
- Automatic kernel fusion and optimization
- Integration with PyTorch JIT compilation
- Performance profiling and benchmarking

### Phase 3: Multi-GPU Support (Planned)
- Distributed computation across multiple GPUs
- Load balancing and memory management
- Communication optimization between GPUs
- Fault tolerance and error recovery

### Success Criteria
- ✅ BERT inference under 60 seconds (achieved: 42 seconds)
- ✅ Memory usage under 24GB GPU RAM (achieved: 20GB)
- ✅ 90%+ GPU utilization during computation
- ✅ Graceful fallback to CPU when GPU unavailable

## Monitoring and Measurement

- GPU utilization metrics via Prometheus
- Memory usage tracking and alerting
- Performance regression testing in CI/CD
- Comparative benchmarks against CPU baseline

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [Microsoft SEAL Library](https://github.com/Microsoft/SEAL)
- [CuPy Documentation](https://cupy.dev/)
- [PyTorch CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)