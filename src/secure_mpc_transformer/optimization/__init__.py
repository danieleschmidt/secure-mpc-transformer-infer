"""
Performance optimization module for secure MPC transformer system.

This module provides advanced optimization techniques including:
- Model quantization and pruning
- GPU memory management
- Tensor fusion and kernel optimization
- SIMD vectorization
- Adaptive batch sizing
- Memory-mapped file support
"""

from .model_optimizer import ModelOptimizer, OptimizationConfig
from .gpu_manager import GPUMemoryManager, CUDAStreamManager
from .tensor_fusion import TensorFusionEngine, FusionConfig
from .simd_operations import SIMDProcessor
from .adaptive_batching import AdaptiveBatchManager
from .memory_mapper import MemoryMappedModelLoader

__all__ = [
    'ModelOptimizer',
    'OptimizationConfig', 
    'GPUMemoryManager',
    'CUDAStreamManager',
    'TensorFusionEngine',
    'FusionConfig',
    'SIMDProcessor',
    'AdaptiveBatchManager',
    'MemoryMappedModelLoader'
]