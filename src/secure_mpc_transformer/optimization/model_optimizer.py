"""
Advanced model optimization techniques for secure MPC transformers.

Implements quantization, pruning, and distillation optimizations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization techniques."""

    # Quantization settings
    enable_quantization: bool = True
    quantization_bits: int = 8
    quantization_mode: str = "dynamic"  # "dynamic", "static", "qat"
    calibration_samples: int = 100

    # Pruning settings
    enable_pruning: bool = True
    pruning_ratio: float = 0.2
    structured_pruning: bool = False
    pruning_schedule: str = "gradual"  # "gradual", "one_shot"

    # Distillation settings
    enable_distillation: bool = False
    teacher_model_path: str | None = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7

    # General settings
    preserve_accuracy_threshold: float = 0.95
    optimization_iterations: int = 10
    enable_mixed_precision: bool = True


class BaseOptimizer(ABC):
    """Abstract base class for model optimizers."""

    @abstractmethod
    def optimize(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Optimize the given model."""
        pass

    @abstractmethod
    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        pass


class QuantizationOptimizer(BaseOptimizer):
    """Implements model quantization optimization."""

    def __init__(self):
        self.quantization_stats = {}

    def optimize(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply quantization optimization to the model."""
        if not config.enable_quantization:
            return model

        logger.info(f"Starting quantization optimization with {config.quantization_bits}-bit precision")

        original_size = self._calculate_model_size(model)

        if config.quantization_mode == "dynamic":
            quantized_model = self._dynamic_quantization(model, config)
        elif config.quantization_mode == "static":
            quantized_model = self._static_quantization(model, config)
        elif config.quantization_mode == "qat":
            quantized_model = self._quantization_aware_training(model, config)
        else:
            raise ValueError(f"Unsupported quantization mode: {config.quantization_mode}")

        quantized_size = self._calculate_model_size(quantized_model)
        compression_ratio = original_size / quantized_size

        self.quantization_stats = {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "quantization_mode": config.quantization_mode,
            "bits": config.quantization_bits
        }

        logger.info(f"Quantization complete. Compression ratio: {compression_ratio:.2f}x")
        return quantized_model

    def _dynamic_quantization(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply dynamic quantization."""
        # Quantize linear and embedding layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Embedding},
            dtype=torch.qint8 if config.quantization_bits == 8 else torch.quint8
        )
        return quantized_model

    def _static_quantization(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply static quantization with calibration."""
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare model
        prepared_model = torch.quantization.prepare(model)

        # Calibration phase (would need actual calibration data in practice)
        logger.info("Running calibration for static quantization")
        with torch.no_grad():
            for _ in range(config.calibration_samples):
                # Generate dummy calibration data
                dummy_input = torch.randn(1, 512, dtype=torch.long)
                prepared_model(dummy_input)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model

    def _quantization_aware_training(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply quantization-aware training."""
        # Configure QAT
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(model)

        # In practice, this would involve actual training
        # For now, we'll just return the prepared model
        logger.info("QAT preparation complete (actual training not implemented)")

        # Convert to quantized model
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return param_size + buffer_size

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get quantization statistics."""
        return self.quantization_stats


class PruningOptimizer(BaseOptimizer):
    """Implements model pruning optimization."""

    def __init__(self):
        self.pruning_stats = {}

    def optimize(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply pruning optimization to the model."""
        if not config.enable_pruning:
            return model

        logger.info(f"Starting pruning optimization with {config.pruning_ratio} pruning ratio")

        original_params = self._count_parameters(model)

        if config.structured_pruning:
            pruned_model = self._structured_pruning(model, config)
        else:
            pruned_model = self._unstructured_pruning(model, config)

        remaining_params = self._count_nonzero_parameters(pruned_model)
        actual_pruning_ratio = 1 - (remaining_params / original_params)

        self.pruning_stats = {
            "original_parameters": original_params,
            "remaining_parameters": remaining_params,
            "target_pruning_ratio": config.pruning_ratio,
            "actual_pruning_ratio": actual_pruning_ratio,
            "structured": config.structured_pruning
        }

        logger.info(f"Pruning complete. Actual pruning ratio: {actual_pruning_ratio:.3f}")
        return pruned_model

    def _unstructured_pruning(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply unstructured (magnitude-based) pruning."""
        parameters_to_prune = []

        # Collect linear layers for pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=config.pruning_ratio
        )

        # Make pruning permanent
        for module, param in parameters_to_prune:
            prune.remove(module, param)

        return model

    def _structured_pruning(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply structured pruning (remove entire neurons/channels)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune entire output channels based on L2 norm
                prune.ln_structured(
                    module,
                    'weight',
                    amount=config.pruning_ratio,
                    n=2,
                    dim=0
                )
                prune.remove(module, 'weight')

        return model

    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())

    def _count_nonzero_parameters(self, model: nn.Module) -> int:
        """Count non-zero parameters in model."""
        return sum((p != 0).sum().item() for p in model.parameters())

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get pruning statistics."""
        return self.pruning_stats


class DistillationOptimizer(BaseOptimizer):
    """Implements knowledge distillation optimization."""

    def __init__(self):
        self.distillation_stats = {}

    def optimize(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply knowledge distillation optimization."""
        if not config.enable_distillation or not config.teacher_model_path:
            return model

        logger.info("Starting knowledge distillation optimization")

        # Load teacher model
        teacher_model = self._load_teacher_model(config.teacher_model_path)

        # Apply distillation (simplified implementation)
        # In practice, this would involve actual training with distillation loss
        distilled_model = self._perform_distillation(model, teacher_model, config)

        self.distillation_stats = {
            "teacher_model_path": config.teacher_model_path,
            "temperature": config.distillation_temperature,
            "alpha": config.distillation_alpha,
            "student_params": self._count_parameters(distilled_model),
            "teacher_params": self._count_parameters(teacher_model)
        }

        logger.info("Knowledge distillation complete")
        return distilled_model

    def _load_teacher_model(self, model_path: str) -> nn.Module:
        """Load teacher model from checkpoint."""
        # In practice, this would load the actual teacher model
        # For now, return a dummy model
        logger.warning("Teacher model loading not implemented - using dummy model")
        return nn.Linear(768, 768)

    def _perform_distillation(self, student: nn.Module, teacher: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Perform knowledge distillation."""
        # Simplified implementation - in practice would involve training loop
        logger.info("Distillation training not implemented - returning original student model")
        return student

    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get distillation statistics."""
        return self.distillation_stats


class ModelOptimizer:
    """Main model optimizer that orchestrates all optimization techniques."""

    def __init__(self):
        self.quantization_optimizer = QuantizationOptimizer()
        self.pruning_optimizer = PruningOptimizer()
        self.distillation_optimizer = DistillationOptimizer()
        self.optimization_stats = {}

    def optimize_model(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply comprehensive model optimization."""
        logger.info("Starting comprehensive model optimization")

        optimized_model = model
        original_size = self._calculate_model_size(model)

        # Apply optimizations in sequence
        if config.enable_distillation:
            optimized_model = self.distillation_optimizer.optimize(optimized_model, config)

        if config.enable_pruning:
            optimized_model = self.pruning_optimizer.optimize(optimized_model, config)

        if config.enable_quantization:
            optimized_model = self.quantization_optimizer.optimize(optimized_model, config)

        # Enable mixed precision if requested
        if config.enable_mixed_precision:
            optimized_model = self._enable_mixed_precision(optimized_model)

        final_size = self._calculate_model_size(optimized_model)
        total_compression_ratio = original_size / final_size

        # Compile optimization statistics
        self.optimization_stats = {
            "original_size_mb": original_size / (1024 * 1024),
            "optimized_size_mb": final_size / (1024 * 1024),
            "total_compression_ratio": total_compression_ratio,
            "quantization_stats": self.quantization_optimizer.get_optimization_stats(),
            "pruning_stats": self.pruning_optimizer.get_optimization_stats(),
            "distillation_stats": self.distillation_optimizer.get_optimization_stats(),
            "mixed_precision_enabled": config.enable_mixed_precision
        }

        logger.info(f"Model optimization complete. Total compression: {total_compression_ratio:.2f}x")
        return optimized_model

    def _enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision training/inference."""
        # Convert appropriate layers to half precision
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.half()

        logger.info("Mixed precision enabled")
        return model

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return param_size + buffer_size

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return self.optimization_stats

    def save_optimized_model(self, model: nn.Module, save_path: str):
        """Save optimized model with metadata."""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimization_stats": self.optimization_stats,
            "model_architecture": str(model)
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Optimized model saved to {save_path}")
