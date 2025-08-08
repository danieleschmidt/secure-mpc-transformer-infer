"""
Secure MPC Transformer Inference

GPU-accelerated secure multi-party computation for transformer inference
with Generation 3 scaling features for enterprise-grade performance.
"""

__version__ = "0.3.0"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"

from .models.secure_transformer import SecureTransformer, TransformerConfig
from .protocols.factory import ProtocolFactory
from .config import SecurityConfig
from .services import InferenceService, SecurityService, ModelService
from .planning import QuantumTaskPlanner, QuantumScheduler, TaskPriority

# Generation 3 Scaling Features
from .optimization import (
    ModelOptimizer, OptimizationConfig,
    GPUMemoryManager, CUDAStreamManager,
    TensorFusionEngine, FusionConfig,
    SIMDProcessor,
    AdaptiveBatchManager,
    MemoryMappedModelLoader
)

from .caching import (
    CacheManager, CacheConfig,
    L1TensorCache, L2ComponentCache, DistributedCache,
    CacheWarmer, CacheCoherenceManager
)

from .concurrency import (
    WorkerPool, DynamicWorkerPool, WorkerConfig,
    AsyncOptimizer, AsyncContext
)

from .profiling import (
    PerformanceProfiler, ProfilingConfig
)

__all__ = [
    # Core components
    "SecureTransformer", 
    "TransformerConfig",
    "ProtocolFactory", 
    "SecurityConfig",
    "InferenceService",
    "SecurityService", 
    "ModelService",
    "QuantumTaskPlanner",
    "QuantumScheduler",
    "TaskPriority",
    
    # Performance optimization
    "ModelOptimizer",
    "OptimizationConfig",
    "GPUMemoryManager",
    "CUDAStreamManager", 
    "TensorFusionEngine",
    "FusionConfig",
    "SIMDProcessor",
    "AdaptiveBatchManager",
    "MemoryMappedModelLoader",
    
    # Caching system
    "CacheManager",
    "CacheConfig",
    "L1TensorCache",
    "L2ComponentCache", 
    "DistributedCache",
    "CacheWarmer",
    "CacheCoherenceManager",
    
    # Concurrent processing
    "WorkerPool",
    "DynamicWorkerPool",
    "WorkerConfig", 
    "AsyncOptimizer",
    "AsyncContext",
    
    # Profiling and optimization
    "PerformanceProfiler",
    "ProfilingConfig"
]