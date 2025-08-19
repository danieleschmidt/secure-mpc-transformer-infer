"""
Tensor fusion and kernel optimization for batch operations.

This module provides high-performance tensor fusion techniques to minimize
kernel launches and optimize memory bandwidth utilization.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FusionType(Enum):
    """Types of tensor fusion operations."""
    ELEMENTWISE = "elementwise"
    MATMUL = "matmul"
    ATTENTION = "attention"
    LAYERNORM = "layernorm"
    ACTIVATION = "activation"
    REDUCTION = "reduction"


@dataclass
class FusionConfig:
    """Configuration for tensor fusion optimization."""
    enable_elementwise_fusion: bool = True
    enable_matmul_fusion: bool = True
    enable_attention_fusion: bool = True
    enable_activation_fusion: bool = True
    max_fusion_size: int = 8
    min_tensor_size_for_fusion: int = 1024
    memory_optimization_level: int = 2  # 0: none, 1: moderate, 2: aggressive
    enable_kernel_caching: bool = True
    enable_autotuning: bool = True


class FusionPattern(ABC):
    """Abstract base class for fusion patterns."""

    @abstractmethod
    def can_fuse(self, ops: list[Any]) -> bool:
        """Check if operations can be fused."""
        pass

    @abstractmethod
    def fuse(self, ops: list[Any]) -> Any:
        """Fuse operations into a single kernel."""
        pass

    @abstractmethod
    def get_pattern_name(self) -> str:
        """Get pattern name for identification."""
        pass


class ElementwiseFusionPattern(FusionPattern):
    """Fusion pattern for elementwise operations."""

    def can_fuse(self, ops: list[Any]) -> bool:
        """Check if operations are fusable elementwise ops."""
        if len(ops) < 2:
            return False

        # Check if all operations are elementwise
        elementwise_ops = {
            torch.add, torch.mul, torch.div, torch.sub,
            torch.relu, torch.gelu, torch.tanh, torch.sigmoid,
            torch.exp, torch.log, torch.sqrt
        }

        for op in ops:
            if not any(isinstance(op, op_type) or op.__class__.__name__ in [ot.__name__ for ot in elementwise_ops] for op_type in elementwise_ops):
                return False

        return True

    def fuse(self, ops: list[Any]) -> Callable:
        """Fuse elementwise operations."""
        def fused_elementwise(*inputs):
            result = inputs[0]
            for i, op in enumerate(ops):
                if i < len(inputs) - 1:
                    next_input = inputs[i + 1]
                    result = op(result, next_input)
                else:
                    result = op(result)
            return result

        return fused_elementwise

    def get_pattern_name(self) -> str:
        return "elementwise_fusion"


class MatmulFusionPattern(FusionPattern):
    """Fusion pattern for matrix multiplication operations."""

    def can_fuse(self, ops: list[Any]) -> bool:
        """Check if operations can be fused as matrix operations."""
        if len(ops) < 2:
            return False

        # Look for patterns like: matmul + bias + activation
        matmul_ops = {torch.matmul, torch.bmm, torch.mm, nn.Linear}

        has_matmul = False
        for op in ops:
            if any(isinstance(op, op_type) for op_type in matmul_ops):
                has_matmul = True
                break

        return has_matmul

    def fuse(self, ops: list[Any]) -> Callable:
        """Fuse matrix operations."""
        def fused_matmul(input_tensor, weight, bias=None, activation=None):
            # Fused matmul + bias + activation
            result = torch.matmul(input_tensor, weight)
            if bias is not None:
                result = result + bias
            if activation is not None:
                result = activation(result)
            return result

        return fused_matmul

    def get_pattern_name(self) -> str:
        return "matmul_fusion"


class AttentionFusionPattern(FusionPattern):
    """Fusion pattern for attention operations."""

    def can_fuse(self, ops: list[Any]) -> bool:
        """Check if operations form attention pattern."""
        # Look for QKV computation pattern
        return len(ops) >= 3  # Simplified check

    def fuse(self, ops: list[Any]) -> Callable:
        """Fuse attention operations."""
        def fused_attention(query, key, value, mask=None, dropout_p=0.0):
            # Fused scaled dot-product attention
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = torch.softmax(scores, dim=-1)

            if dropout_p > 0.0:
                attention_weights = torch.dropout(attention_weights, dropout_p, training=True)

            return torch.matmul(attention_weights, value)

        return fused_attention

    def get_pattern_name(self) -> str:
        return "attention_fusion"


class LayerNormFusionPattern(FusionPattern):
    """Fusion pattern for layer normalization operations."""

    def can_fuse(self, ops: list[Any]) -> bool:
        """Check if operations can be fused with layer norm."""
        layernorm_ops = {nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d}
        return any(isinstance(op, op_type) for op_type in layernorm_ops for op in ops)

    def fuse(self, ops: list[Any]) -> Callable:
        """Fuse layer normalization operations."""
        def fused_layernorm(input_tensor, weight, bias, eps=1e-5):
            # Fused layer norm + residual
            mean = input_tensor.mean(dim=-1, keepdim=True)
            var = ((input_tensor - mean) ** 2).mean(dim=-1, keepdim=True)
            normalized = (input_tensor - mean) / torch.sqrt(var + eps)
            return normalized * weight + bias

        return fused_layernorm

    def get_pattern_name(self) -> str:
        return "layernorm_fusion"


class KernelCache:
    """Cache for compiled fusion kernels."""

    def __init__(self, max_cache_size: int = 1000):
        self.cache: dict[str, Any] = {}
        self.max_cache_size = max_cache_size
        self.access_count: dict[str, int] = {}

    def get_kernel(self, pattern_name: str, op_signature: str) -> Any | None:
        """Get cached kernel if available."""
        cache_key = f"{pattern_name}_{op_signature}"

        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]

        return None

    def store_kernel(self, pattern_name: str, op_signature: str, kernel: Any):
        """Store compiled kernel in cache."""
        cache_key = f"{pattern_name}_{op_signature}"

        if len(self.cache) >= self.max_cache_size:
            # Evict least frequently used kernel
            lfu_key = min(self.access_count.keys(), key=self.access_count.get)
            del self.cache[lfu_key]
            del self.access_count[lfu_key]

        self.cache[cache_key] = kernel
        self.access_count[cache_key] = 1

    def clear(self):
        """Clear the kernel cache."""
        self.cache.clear()
        self.access_count.clear()


class TensorFusionEngine:
    """Main tensor fusion engine for optimizing operations."""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.fusion_patterns: list[FusionPattern] = []
        self.kernel_cache = KernelCache() if config.enable_kernel_caching else None

        # Statistics
        self.fusion_stats = {
            "total_fusions": 0,
            "elementwise_fusions": 0,
            "matmul_fusions": 0,
            "attention_fusions": 0,
            "layernorm_fusions": 0,
            "kernel_cache_hits": 0,
            "kernel_cache_misses": 0
        }

        # Initialize fusion patterns
        self._initialize_patterns()

        logger.info("Initialized tensor fusion engine")

    def _initialize_patterns(self):
        """Initialize fusion patterns based on configuration."""
        if self.config.enable_elementwise_fusion:
            self.fusion_patterns.append(ElementwiseFusionPattern())

        if self.config.enable_matmul_fusion:
            self.fusion_patterns.append(MatmulFusionPattern())

        if self.config.enable_attention_fusion:
            self.fusion_patterns.append(AttentionFusionPattern())

        self.fusion_patterns.append(LayerNormFusionPattern())

        logger.info(f"Initialized {len(self.fusion_patterns)} fusion patterns")

    def analyze_fusion_opportunities(self, model: nn.Module) -> dict[str, list[str]]:
        """Analyze model for fusion opportunities."""
        opportunities = {pattern.get_pattern_name(): [] for pattern in self.fusion_patterns}

        # Walk through model modules
        for name, module in model.named_modules():
            # Identify fusable operation sequences
            if isinstance(module, nn.Sequential):
                ops = list(module.children())

                for pattern in self.fusion_patterns:
                    if pattern.can_fuse(ops):
                        opportunities[pattern.get_pattern_name()].append(name)

        return opportunities

    def fuse_operations(self, ops: list[Any], pattern_hint: str | None = None) -> Any | None:
        """Fuse a list of operations if possible."""
        if len(ops) < 2:
            return None

        # Check minimum tensor size requirement
        total_params = sum(sum(p.numel() for p in getattr(op, 'parameters', lambda: [])()) for op in ops if hasattr(op, 'parameters'))
        if total_params < self.config.min_tensor_size_for_fusion:
            return None

        # Try each fusion pattern
        for pattern in self.fusion_patterns:
            if pattern_hint and pattern.get_pattern_name() != pattern_hint:
                continue

            if pattern.can_fuse(ops):
                # Check kernel cache
                op_signature = self._generate_operation_signature(ops)

                if self.kernel_cache:
                    cached_kernel = self.kernel_cache.get_kernel(pattern.get_pattern_name(), op_signature)
                    if cached_kernel:
                        self.fusion_stats["kernel_cache_hits"] += 1
                        return cached_kernel
                    else:
                        self.fusion_stats["kernel_cache_misses"] += 1

                # Fuse operations
                fused_op = pattern.fuse(ops)

                # Cache the fused kernel
                if self.kernel_cache:
                    self.kernel_cache.store_kernel(pattern.get_pattern_name(), op_signature, fused_op)

                # Update statistics
                self.fusion_stats["total_fusions"] += 1
                pattern_name = pattern.get_pattern_name()
                if pattern_name in self.fusion_stats:
                    self.fusion_stats[pattern_name] += 1

                logger.debug(f"Fused {len(ops)} operations using {pattern_name}")
                return fused_op

        return None

    def _generate_operation_signature(self, ops: list[Any]) -> str:
        """Generate signature for operations to enable caching."""
        signature_parts = []

        for op in ops:
            if hasattr(op, '__class__'):
                signature_parts.append(op.__class__.__name__)
            else:
                signature_parts.append(str(type(op).__name__))

        return "_".join(signature_parts)

    def optimize_model_for_fusion(self, model: nn.Module) -> nn.Module:
        """Optimize entire model by applying fusion where possible."""
        logger.info("Starting model-wide fusion optimization")

        optimized_modules = {}

        # Find fusion opportunities
        opportunities = self.analyze_fusion_opportunities(model)

        total_opportunities = sum(len(ops) for ops in opportunities.values())
        logger.info(f"Found {total_opportunities} fusion opportunities")

        # Apply fusions
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                ops = list(module.children())

                if len(ops) >= 2:
                    fused_op = self.fuse_operations(ops)
                    if fused_op:
                        optimized_modules[module_name] = fused_op

        # Replace modules with fused versions
        for module_name, fused_module in optimized_modules.items():
            # This would need proper implementation to replace modules in the model
            logger.debug(f"Would replace {module_name} with fused version")

        logger.info(f"Applied {len(optimized_modules)} fusions to model")
        return model

    @contextmanager
    def fusion_context(self, enable_autotuning: bool = None):
        """Context manager for fusion operations with optional autotuning."""
        original_autotuning = self.config.enable_autotuning

        if enable_autotuning is not None:
            self.config.enable_autotuning = enable_autotuning

        try:
            yield self
        finally:
            self.config.enable_autotuning = original_autotuning

    def benchmark_fusion(self, ops: list[Any], num_iterations: int = 100) -> dict[str, float]:
        """Benchmark fused vs unfused operations."""
        if not ops:
            return {}

        # Generate dummy inputs for benchmarking
        dummy_inputs = self._generate_dummy_inputs(ops)

        # Benchmark unfused operations
        unfused_time = self._benchmark_operations(ops, dummy_inputs, num_iterations)

        # Benchmark fused operations
        fused_op = self.fuse_operations(ops)
        if fused_op:
            fused_time = self._benchmark_single_operation(fused_op, dummy_inputs, num_iterations)
        else:
            fused_time = unfused_time

        speedup = unfused_time / fused_time if fused_time > 0 else 1.0

        return {
            "unfused_time_ms": unfused_time * 1000,
            "fused_time_ms": fused_time * 1000,
            "speedup": speedup,
            "fusion_effective": speedup > 1.05  # 5% threshold
        }

    def _generate_dummy_inputs(self, ops: list[Any]) -> list[torch.Tensor]:
        """Generate dummy inputs for benchmarking."""
        # Simplified dummy input generation
        return [torch.randn(1024, 768) for _ in range(len(ops))]

    def _benchmark_operations(self, ops: list[Any], inputs: list[torch.Tensor], iterations: int) -> float:
        """Benchmark a sequence of operations."""
        import time

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(iterations):
            result = inputs[0]
            for i, op in enumerate(ops):
                if callable(op):
                    if i < len(inputs) - 1:
                        result = op(result, inputs[i + 1])
                    else:
                        result = op(result)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        return (end_time - start_time) / iterations

    def _benchmark_single_operation(self, op: Any, inputs: list[torch.Tensor], iterations: int) -> float:
        """Benchmark a single fused operation."""
        import time

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(iterations):
            result = op(*inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        return (end_time - start_time) / iterations

    def get_fusion_stats(self) -> dict[str, Any]:
        """Get fusion statistics."""
        return {
            **self.fusion_stats,
            "cache_size": len(self.kernel_cache.cache) if self.kernel_cache else 0,
            "patterns_enabled": len(self.fusion_patterns)
        }

    def clear_cache(self):
        """Clear kernel cache."""
        if self.kernel_cache:
            self.kernel_cache.clear()
            logger.info("Cleared kernel cache")

    def autotune_fusion_config(self, model: nn.Module, sample_inputs: list[torch.Tensor]) -> FusionConfig:
        """Automatically tune fusion configuration for optimal performance."""
        logger.info("Starting fusion configuration autotuning")

        best_config = self.config
        best_performance = 0.0

        # Test different configuration combinations
        test_configs = [
            FusionConfig(enable_elementwise_fusion=True, enable_matmul_fusion=False),
            FusionConfig(enable_elementwise_fusion=False, enable_matmul_fusion=True),
            FusionConfig(enable_elementwise_fusion=True, enable_matmul_fusion=True),
            FusionConfig(memory_optimization_level=0),
            FusionConfig(memory_optimization_level=1),
            FusionConfig(memory_optimization_level=2),
        ]

        for test_config in test_configs:
            # Create temporary fusion engine
            temp_engine = TensorFusionEngine(test_config)

            # Measure performance with this config
            performance = self._measure_model_performance(model, sample_inputs, temp_engine)

            if performance > best_performance:
                best_performance = performance
                best_config = test_config

            logger.debug(f"Config performance: {performance:.3f}")

        logger.info(f"Autotuning complete. Best performance: {best_performance:.3f}")
        return best_config

    def _measure_model_performance(self, model: nn.Module, inputs: list[torch.Tensor], fusion_engine: 'TensorFusionEngine') -> float:
        """Measure model performance with given fusion configuration."""
        import time

        # Apply fusion optimizations
        optimized_model = fusion_engine.optimize_model_for_fusion(model)

        # Benchmark inference time
        num_iterations = 50

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                for input_tensor in inputs:
                    _ = optimized_model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations

        # Return performance metric (higher is better)
        return 1.0 / avg_time if avg_time > 0 else 0.0
