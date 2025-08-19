"""
Secure MPC Transformer Inference

GPU-accelerated secure multi-party computation for transformer inference
with Generation 3 scaling features for enterprise-grade performance.
"""

__version__ = "0.3.0"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"

from .caching import (
    CacheCoherenceManager,
    CacheConfig,
    CacheManager,
    CacheWarmer,
    DistributedCache,
    L1TensorCache,
    L2ComponentCache,
)
from .concurrency import (
    AsyncContext,
    AsyncOptimizer,
    DynamicWorkerPool,
    WorkerConfig,
    WorkerPool,
)
from .config import SecurityConfig

# Autonomous SDLC Core
from .core import AutonomousExecutor, ExecutionMetrics, ExecutionPhase, ExecutionTask
from .models.secure_transformer import SecureTransformer, TransformerConfig

# Generation 3 Scaling Features
from .optimization import (
    AdaptiveBatchManager,
    CUDAStreamManager,
    FusionConfig,
    GPUMemoryManager,
    MemoryMappedModelLoader,
    ModelOptimizer,
    OptimizationConfig,
    SIMDProcessor,
    TensorFusionEngine,
)
from .planning import QuantumScheduler, QuantumTaskPlanner, TaskPriority
from .profiling import PerformanceProfiler, ProfilingConfig
from .protocols.factory import ProtocolFactory
from .services import InferenceService, ModelService, SecurityService

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
    "ProfilingConfig",

    # Autonomous SDLC
    "AutonomousExecutor",
    "ExecutionTask",
    "ExecutionPhase",
    "TaskPriority",
    "ExecutionMetrics"
]
