"""
Comprehensive performance profiling and optimization hints system.

This module provides detailed profiling capabilities for the secure MPC
transformer system with actionable optimization recommendations.
"""

from .performance_profiler import PerformanceProfiler, ProfilingConfig, ProfileReport
from .optimization_hints import OptimizationHintEngine, OptimizationHint, HintType
from .metrics_collector import MetricsCollector, Metric, MetricType
from .bottleneck_detector import BottleneckDetector, BottleneckAnalysis
from .profiling_decorators import profile_function, profile_class, async_profile

__all__ = [
    'PerformanceProfiler',
    'ProfilingConfig',
    'ProfileReport',
    'OptimizationHintEngine',
    'OptimizationHint',
    'HintType',
    'MetricsCollector',
    'Metric',
    'MetricType',
    'BottleneckDetector',
    'BottleneckAnalysis',
    'profile_function',
    'profile_class',
    'async_profile'
]