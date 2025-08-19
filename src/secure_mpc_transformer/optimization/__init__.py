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

from .adaptive_batching import AdaptiveBatchManager
from .gpu_manager import CUDAStreamManager, GPUMemoryManager
from .memory_mapper import MemoryMappedModelLoader
from .model_optimizer import ModelOptimizer, OptimizationConfig
from .simd_operations import SIMDProcessor
from .tensor_fusion import FusionConfig, TensorFusionEngine

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
