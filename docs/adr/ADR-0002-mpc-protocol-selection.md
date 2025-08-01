# ADR-0002: MPC Protocol Selection Criteria

**Status:** Accepted  
**Date:** 2024-04-02  
**Authors:** Security Team, Cryptography Lead  
**Reviewers:** Security Advisor, Technical Lead  
**Related ADRs:** ADR-0001 (GPU Acceleration Framework)

## Context

The project requires selecting appropriate Multi-Party Computation (MPC) protocols that balance security, performance, and practical deployment constraints. The choice of protocol significantly impacts:

- Security guarantees (semi-honest vs malicious)
- Performance characteristics (round complexity, communication)
- GPU acceleration potential
- Number of parties supported
- Implementation complexity

Key requirements:
- Support for transformer neural network operations
- Sub-60 second inference for BERT-base model
- 128-bit computational security level
- Practical deployment in cloud environments
- Compatibility with GPU acceleration framework

Security requirements:
- Protection against semi-honest adversaries (minimum)
- Option for malicious security in sensitive deployments
- Information-theoretic or computational security guarantees
- Resistance to timing and side-channel attacks

## Decision

We will implement a multi-protocol framework supporting:

### Primary Protocol: ABY3 (3-Party Replicated Secret Sharing)
- **Security Model:** Malicious security with honest majority
- **Party Count:** 3 parties
- **Communication Rounds:** 1 round for most operations
- **GPU Acceleration:** Full support via custom kernels

### Secondary Protocol: Semi-Honest 3PC  
- **Security Model:** Semi-honest security
- **Party Count:** 3 parties  
- **Communication Rounds:** 0-1 rounds (non-interactive where possible)
- **GPU Acceleration:** Optimal performance with GPU offload

### Experimental Protocol: 4-Party GPU-Offload
- **Security Model:** Semi-honest with GPU as trusted party
- **Party Count:** 4 parties (3 compute + 1 GPU server)
- **Communication Rounds:** Reduced via GPU preprocessing
- **GPU Acceleration:** Specialized for transformer operations

## Alternatives Considered

### Alternative 1: BGW (Shamir Secret Sharing)
- **Description:** Classical BGW protocol with polynomial secret sharing
- **Pros:**
  - Strong theoretical foundation
  - Supports arbitrary number of parties
  - Well-studied security properties
- **Cons:**
  - High communication complexity (O(n²))
  - Limited GPU acceleration potential
  - Polynomial degree increases with adversaries
- **Rejection Reason:** Communication overhead prohibitive for large models, difficulty with GPU optimization

### Alternative 2: GMW (Garbled Circuits)
- **Description:** Garbled circuit approach for secure computation
- **Pros:**
  - Supports malicious security
  - Constant round complexity
  - Well-established in practice
- **Cons:**
  - Circuit size explosion for neural networks
  - Limited reusability of garbled circuits
  - Poor GPU acceleration characteristics
- **Rejection Reason:** Circuit complexity for transformers makes this approach impractical

### Alternative 3: SPDZ Protocol Family
- **Description:** SPDZ-style protocols with preprocessing and MAC verification
- **Pros:**
  - Strong malicious security
  - Efficient online phase
  - Good theoretical properties
- **Cons:**
  - Complex preprocessing phase
  - Higher memory requirements
  - Implementation complexity
- **Rejection Reason:** Preprocessing overhead and complexity outweigh benefits for inference-only use case

### Alternative 4: Homomorphic Encryption Only
- **Description:** Pure HE approach without secret sharing
- **Pros:**
  - No communication during computation
  - Strong security guarantees
  - Simpler threat model
- **Cons:**
  - Extremely high computational overhead
  - Limited operation support
  - Ciphertext size explosion
- **Rejection Reason:** Performance requirements cannot be met with current HE technology

## Consequences

### Positive Consequences
- **Performance:** ABY3 achieves optimal round complexity for most operations
- **Security:** Malicious security available when required
- **Flexibility:** Multiple protocols allow optimization for different scenarios
- **GPU Integration:** Protocols selected specifically for GPU acceleration potential

### Negative Consequences
- **Complexity:** Multiple protocol implementations increase codebase complexity
- **Party Requirements:** 3-party minimum may limit some deployment scenarios
- **Trust Assumptions:** Honest majority assumption required for strongest security
- **Implementation Effort:** Custom GPU kernels required for each protocol

### Neutral Consequences
- **Deployment:** Requires coordination between 3 parties minimum
- **Communication:** Network requirements scale with model size
- **Storage:** Secret shares require additional storage overhead

## Implementation

### Phase 1: Semi-Honest 3PC (Completed)
- Basic secret sharing arithmetic
- GPU-accelerated matrix operations
- Integration with transformer layers
- Performance optimization and tuning

### Phase 2: ABY3 Malicious Security (Completed)
- MAC-based authentication
- Zero-knowledge proofs for verification
- Robust error detection and recovery
- Security audit and formal analysis

### Phase 3: 4-Party GPU Offload (In Progress)
- GPU server trust model design
- Specialized kernels for GPU party
- Communication protocol optimization
- Performance evaluation and comparison

### Success Criteria
- ✅ Sub-60 second BERT inference with ABY3
- ✅ 128-bit security level verification
- ✅ Successful malicious security audit
- ✅ GPU utilization >80% during computation

## Security Analysis

### Threat Model
- **Semi-Honest Adversaries:** Honest-but-curious parties follow protocol
- **Malicious Adversaries:** Arbitrary deviations from protocol
- **Network Security:** TLS encryption for all communication
- **Side Channels:** Constant-time implementations where feasible

### Security Guarantees
- **Confidentiality:** Input privacy preserved under honest majority
- **Integrity:** Output correctness guaranteed with malicious security
- **Availability:** Graceful degradation with party failures
- **Verifiability:** Zero-knowledge proofs for critical operations

### Assumptions
- Honest majority (2 out of 3 parties honest)
- Secure communication channels
- Trusted setup for preprocessing (where required)
- Secure random number generation

## Performance Benchmarks

| Protocol | BERT Inference | Communication | GPU Memory |
|----------|---------------|---------------|-------------|
| Semi-Honest 3PC | 35s | 2.1 GB | 18 GB |
| ABY3 (Malicious) | 42s | 2.8 GB | 20 GB |
| 4-Party GPU | 28s | 1.9 GB | 22 GB |

## References

- [ABY3: A Mixed Protocol Framework](https://eprint.iacr.org/2018/403.pdf)
- [Secure Multiparty Computation and Secret Sharing](https://www.cambridge.org/core/books/secure-multiparty-computation-and-secret-sharing/8EED3137A3F90DBAEC8B92D5F8C09A81)
- [GPU-Accelerated MPC Protocols](https://research.nvidia.com/publication/gpu-accelerated-secure-multiparty-computation)
- [SPDZ Protocol Specification](https://eprint.iacr.org/2011/535.pdf)
- [BGW Protocol Analysis](https://dl.acm.org/doi/10.1145/62212.62213)