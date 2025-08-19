"""
Comprehensive performance profiling and optimization hints system.

This module provides detailed profiling capabilities for the secure MPC
transformer system with actionable optimization recommendations.
"""

from .bottleneck_detector import BottleneckAnalysis, BottleneckDetector
from .metrics_collector import Metric, MetricsCollector, MetricType
from .optimization_hints import HintType, OptimizationHint, OptimizationHintEngine
from .performance_profiler import PerformanceProfiler, ProfileReport, ProfilingConfig
from .profiling_decorators import async_profile, profile_class, profile_function

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
