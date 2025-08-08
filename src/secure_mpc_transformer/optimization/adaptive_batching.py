"""
Adaptive batch sizing based on resource availability and workload characteristics.

This module dynamically adjusts batch sizes to optimize throughput while
respecting memory and compute constraints.
"""

import torch
import psutil
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import numpy as np
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    GPU_UTILIZATION = "gpu_utilization"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ResourceConstraints:
    """Resource constraints for batch sizing."""
    max_memory_mb: float = 8192.0  # 8GB default
    max_gpu_memory_mb: float = 12288.0  # 12GB default
    max_cpu_utilization: float = 0.8  # 80%
    max_gpu_utilization: float = 0.9  # 90%
    safety_margin: float = 0.1  # 10% safety margin


@dataclass
class BatchingConfig:
    """Configuration for adaptive batching."""
    min_batch_size: int = 1
    max_batch_size: int = 128
    initial_batch_size: int = 8
    adaptation_rate: float = 0.1
    memory_check_interval: float = 1.0  # seconds
    performance_window_size: int = 50
    enable_predictive_scaling: bool = True
    enable_auto_tuning: bool = True
    warmup_batches: int = 10
    batch_size_step: int = 2


@dataclass
class BatchMetrics:
    """Metrics for a single batch execution."""
    batch_size: int
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    cpu_utilization: float
    gpu_utilization: float
    timestamp: float = field(default_factory=time.time)
    
    def efficiency_score(self) -> float:
        """Calculate efficiency score for this batch."""
        # Combine throughput with resource utilization efficiency
        memory_efficiency = 1.0 - (self.memory_usage_mb / 16384.0)  # Assume 16GB max
        gpu_efficiency = self.gpu_utilization / 100.0
        
        return (self.throughput_ops_per_sec * memory_efficiency * gpu_efficiency) / 1000.0


class ResourceMonitor:
    """Monitor system resources in real-time."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Current resource state
        self.current_resources = {
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0.0,
            ResourceType.GPU_MEMORY: 0.0,
            ResourceType.GPU_UTILIZATION: 0.0
        }
        
        # Resource history for trend analysis
        self.resource_history: Dict[ResourceType, deque] = {
            res_type: deque(maxlen=100) for res_type in ResourceType
        }
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._update_resources()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _update_resources(self):
        """Update current resource measurements."""
        with self._lock:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.current_resources[ResourceType.CPU] = cpu_percent
            self.resource_history[ResourceType.CPU].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            self.current_resources[ResourceType.MEMORY] = memory_percent
            self.resource_history[ResourceType.MEMORY].append(memory_percent)
            
            # GPU monitoring (if available)
            if torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_percent = (gpu_memory_mb / (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))) * 100
                    
                    self.current_resources[ResourceType.GPU_MEMORY] = gpu_memory_percent
                    self.resource_history[ResourceType.GPU_MEMORY].append(gpu_memory_percent)
                    
                    # GPU utilization (simplified - would need nvidia-ml-py for accurate measurement)
                    gpu_util = min(gpu_memory_percent * 1.2, 100.0)  # Rough approximation
                    self.current_resources[ResourceType.GPU_UTILIZATION] = gpu_util
                    self.resource_history[ResourceType.GPU_UTILIZATION].append(gpu_util)
                    
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
    
    def get_current_resources(self) -> Dict[ResourceType, float]:
        """Get current resource utilization."""
        with self._lock:
            return self.current_resources.copy()
    
    def get_resource_trend(self, resource_type: ResourceType, window_size: int = 10) -> float:
        """Get resource trend (positive = increasing, negative = decreasing)."""
        with self._lock:
            history = list(self.resource_history[resource_type])
            if len(history) < window_size:
                return 0.0
            
            recent = history[-window_size:]
            older = history[-2*window_size:-window_size] if len(history) >= 2*window_size else history[:-window_size]
            
            if not older:
                return 0.0
            
            recent_avg = np.mean(recent)
            older_avg = np.mean(older)
            
            return recent_avg - older_avg
    
    def check_constraints(self, constraints: ResourceConstraints) -> Dict[ResourceType, bool]:
        """Check if current resources violate constraints."""
        current = self.get_current_resources()
        violations = {}
        
        violations[ResourceType.CPU] = current[ResourceType.CPU] > constraints.max_cpu_utilization * 100
        violations[ResourceType.MEMORY] = current[ResourceType.MEMORY] > 90.0  # Use percentage
        violations[ResourceType.GPU_MEMORY] = current[ResourceType.GPU_MEMORY] > 90.0
        violations[ResourceType.GPU_UTILIZATION] = current[ResourceType.GPU_UTILIZATION] > constraints.max_gpu_utilization * 100
        
        return violations


class BatchSizePredictor:
    """Predict optimal batch size based on historical performance."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_history: deque = deque(maxlen=window_size)
        self.batch_size_performance: Dict[int, List[float]] = defaultdict(list)
    
    def record_batch_performance(self, metrics: BatchMetrics):
        """Record performance metrics for a batch."""
        self.performance_history.append(metrics)
        
        # Track performance by batch size
        efficiency = metrics.efficiency_score()
        self.batch_size_performance[metrics.batch_size].append(efficiency)
        
        # Keep only recent performance data per batch size
        if len(self.batch_size_performance[metrics.batch_size]) > 10:
            self.batch_size_performance[metrics.batch_size].pop(0)
    
    def predict_optimal_batch_size(self, constraints: ResourceConstraints) -> int:
        """Predict optimal batch size based on historical data."""
        if not self.batch_size_performance:
            return 8  # Default
        
        # Calculate average efficiency for each batch size
        batch_size_scores = {}
        for batch_size, performances in self.batch_size_performance.items():
            if performances:
                avg_performance = np.mean(performances)
                # Penalize very large batch sizes to account for resource constraints
                size_penalty = 1.0 - (batch_size / 128.0) * 0.1
                batch_size_scores[batch_size] = avg_performance * size_penalty
        
        if not batch_size_scores:
            return 8
        
        # Return batch size with highest score
        optimal_batch_size = max(batch_size_scores, key=batch_size_scores.get)
        return optimal_batch_size
    
    def get_performance_trend(self) -> float:
        """Get recent performance trend."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_metrics = list(self.performance_history)[-10:]
        older_metrics = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else []
        
        if not older_metrics:
            return 0.0
        
        recent_avg = np.mean([m.efficiency_score() for m in recent_metrics])
        older_avg = np.mean([m.efficiency_score() for m in older_metrics])
        
        return recent_avg - older_avg


class AdaptiveBatchManager:
    """Main adaptive batch manager that coordinates all components."""
    
    def __init__(self, config: Optional[BatchingConfig] = None, constraints: Optional[ResourceConstraints] = None):
        self.config = config or BatchingConfig()
        self.constraints = constraints or ResourceConstraints()
        
        # Components
        self.resource_monitor = ResourceMonitor()
        self.predictor = BatchSizePredictor()
        
        # Current state
        self.current_batch_size = self.config.initial_batch_size
        self.warmup_counter = 0
        self.last_adaptation = time.time()
        
        # Performance tracking
        self.recent_metrics: deque = deque(maxlen=self.config.performance_window_size)
        self.adaptation_history: List[Tuple[float, int, str]] = []  # (timestamp, batch_size, reason)
        
        # Auto-tuning state
        self.auto_tuning_active = False
        self.tuning_phase = "exploration"  # "exploration", "exploitation"
        self.exploration_batch_sizes = []
        
        logger.info(f"Initialized adaptive batch manager with initial batch size: {self.current_batch_size}")
    
    def start(self):
        """Start the adaptive batch manager."""
        self.resource_monitor.start_monitoring()
        
        if self.config.enable_auto_tuning:
            self._start_auto_tuning()
        
        logger.info("Adaptive batch manager started")
    
    def stop(self):
        """Stop the adaptive batch manager."""
        self.resource_monitor.stop_monitoring()
        self.auto_tuning_active = False
        logger.info("Adaptive batch manager stopped")
    
    def get_next_batch_size(self, current_workload_size: Optional[int] = None) -> int:
        """Get the next recommended batch size."""
        # Check resource constraints
        resource_violations = self.resource_monitor.check_constraints(self.constraints)
        
        if any(resource_violations.values()):
            # Reduce batch size if resources are constrained
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            self._record_adaptation("resource_constraint")
            
        elif self.config.enable_predictive_scaling:
            # Use predictor for batch size optimization
            predicted_size = self.predictor.predict_optimal_batch_size(self.constraints)
            
            # Gradually move towards predicted size
            if predicted_size > self.current_batch_size:
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    self.current_batch_size + self.config.batch_size_step
                )
                self._record_adaptation("predictive_increase")
            elif predicted_size < self.current_batch_size:
                self.current_batch_size = max(
                    self.config.min_batch_size,
                    self.current_batch_size - self.config.batch_size_step
                )
                self._record_adaptation("predictive_decrease")
        
        # Respect workload size limits
        if current_workload_size is not None:
            self.current_batch_size = min(self.current_batch_size, current_workload_size)
        
        # Ensure bounds
        self.current_batch_size = max(
            self.config.min_batch_size,
            min(self.config.max_batch_size, self.current_batch_size)
        )
        
        return self.current_batch_size
    
    def record_batch_execution(self, batch_size: int, execution_time: float, 
                              memory_usage_mb: float, **kwargs):
        """Record the execution of a batch."""
        current_resources = self.resource_monitor.get_current_resources()
        
        metrics = BatchMetrics(
            batch_size=batch_size,
            latency_ms=execution_time * 1000,
            throughput_ops_per_sec=batch_size / execution_time if execution_time > 0 else 0,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_usage_mb=kwargs.get('gpu_memory_mb', 0),
            cpu_utilization=current_resources.get(ResourceType.CPU, 0),
            gpu_utilization=current_resources.get(ResourceType.GPU_UTILIZATION, 0)
        )
        
        self.recent_metrics.append(metrics)
        self.predictor.record_batch_performance(metrics)
        
        # Update warmup counter
        if self.warmup_counter < self.config.warmup_batches:
            self.warmup_counter += 1
        
        logger.debug(f"Recorded batch execution: size={batch_size}, latency={execution_time:.3f}s")
    
    def _record_adaptation(self, reason: str):
        """Record a batch size adaptation."""
        self.adaptation_history.append((time.time(), self.current_batch_size, reason))
        self.last_adaptation = time.time()
        
        # Keep history bounded
        if len(self.adaptation_history) > 100:
            self.adaptation_history.pop(0)
        
        logger.debug(f"Adapted batch size to {self.current_batch_size}: {reason}")
    
    def _start_auto_tuning(self):
        """Start auto-tuning process."""
        self.auto_tuning_active = True
        
        # Generate exploration batch sizes
        self.exploration_batch_sizes = list(range(
            self.config.min_batch_size,
            min(self.config.max_batch_size + 1, 65),
            self.config.batch_size_step
        ))
        
        logger.info(f"Started auto-tuning with {len(self.exploration_batch_sizes)} exploration sizes")
    
    async def auto_tune_async(self, evaluation_function: Callable[[int], Tuple[float, Dict]]):
        """Asynchronous auto-tuning of batch size."""
        if not self.auto_tuning_active:
            return
        
        logger.info("Starting asynchronous auto-tuning")
        
        best_batch_size = self.config.initial_batch_size
        best_performance = 0.0
        
        for batch_size in self.exploration_batch_sizes:
            try:
                # Evaluate performance with this batch size
                performance, metrics = await asyncio.get_event_loop().run_in_executor(
                    None, evaluation_function, batch_size
                )
                
                if performance > best_performance:
                    best_performance = performance
                    best_batch_size = batch_size
                
                logger.debug(f"Auto-tune: batch_size={batch_size}, performance={performance:.3f}")
                
                # Allow other coroutines to run
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Auto-tuning error for batch size {batch_size}: {e}")
        
        # Update to best configuration
        self.current_batch_size = best_batch_size
        self._record_adaptation("auto_tuning_complete")
        
        logger.info(f"Auto-tuning complete. Optimal batch size: {best_batch_size}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.recent_metrics:
            return {"status": "no_data"}
        
        metrics_list = list(self.recent_metrics)
        
        return {
            "current_batch_size": self.current_batch_size,
            "total_batches": len(metrics_list),
            "average_latency_ms": np.mean([m.latency_ms for m in metrics_list]),
            "average_throughput": np.mean([m.throughput_ops_per_sec for m in metrics_list]),
            "average_efficiency": np.mean([m.efficiency_score() for m in metrics_list]),
            "resource_utilization": self.resource_monitor.get_current_resources(),
            "adaptation_count": len(self.adaptation_history),
            "warmup_progress": min(1.0, self.warmup_counter / self.config.warmup_batches),
            "recent_adaptations": self.adaptation_history[-5:] if self.adaptation_history else []
        }
    
    def optimize_for_latency(self):
        """Optimize batch sizing for minimum latency."""
        self.config.adaptation_rate = 0.2  # More aggressive adaptation
        self.current_batch_size = max(1, self.current_batch_size // 2)  # Smaller batches
        self._record_adaptation("latency_optimization")
        logger.info("Optimized for latency")
    
    def optimize_for_throughput(self):
        """Optimize batch sizing for maximum throughput."""
        self.config.adaptation_rate = 0.05  # Conservative adaptation
        # Allow larger batches within resource constraints
        current_resources = self.resource_monitor.get_current_resources()
        if current_resources[ResourceType.GPU_MEMORY] < 70:  # If GPU memory usage is low
            self.current_batch_size = min(self.config.max_batch_size, self.current_batch_size * 2)
            self._record_adaptation("throughput_optimization")
        logger.info("Optimized for throughput")
    
    def get_batch_size_recommendations(self, workload_characteristics: Dict[str, Any]) -> Dict[str, int]:
        """Get batch size recommendations for different scenarios."""
        base_size = self.current_batch_size
        
        recommendations = {
            "current": base_size,
            "latency_optimized": max(1, base_size // 2),
            "throughput_optimized": min(self.config.max_batch_size, base_size * 2),
            "memory_constrained": max(1, base_size // 4),
            "balanced": base_size
        }
        
        # Adjust based on workload characteristics
        if workload_characteristics.get("input_size", "medium") == "large":
            for key in recommendations:
                recommendations[key] = max(1, recommendations[key] // 2)
        
        if workload_characteristics.get("priority", "normal") == "high":
            recommendations["recommended"] = recommendations["latency_optimized"]
        else:
            recommendations["recommended"] = recommendations["throughput_optimized"]
        
        return recommendations
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()