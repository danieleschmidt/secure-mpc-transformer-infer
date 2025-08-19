"""
Comprehensive performance profiler for the secure MPC transformer system.

This module provides detailed performance analysis including CPU, memory,
GPU usage, and system-level metrics.
"""

import contextlib
import cProfile
import functools
import io
import json
import logging
import pstats
import threading
import time
import tracemalloc
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import psutil

logger = logging.getLogger(__name__)

# Try to import GPU monitoring
try:
    import pynvml
    GPU_MONITORING_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logger.warning("NVIDIA GPU monitoring not available")


class ProfilingLevel(Enum):
    """Profiling detail levels."""
    BASIC = "basic"           # Basic timing and resource usage
    DETAILED = "detailed"     # Detailed function-level profiling
    COMPREHENSIVE = "comprehensive"  # Full system profiling with traces


@dataclass
class ProfilingConfig:
    """Configuration for performance profiling."""
    level: ProfilingLevel = ProfilingLevel.DETAILED
    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_gpu_profiling: bool = GPU_MONITORING_AVAILABLE
    enable_system_monitoring: bool = True

    # Sampling and collection
    sampling_interval_ms: float = 100.0  # 100ms
    max_samples: int = 10000
    enable_call_stack_traces: bool = True
    max_stack_depth: int = 50

    # Memory tracking
    trace_malloc: bool = True
    memory_growth_threshold_mb: float = 100.0  # Alert on 100MB+ growth

    # Output and reporting
    enable_realtime_monitoring: bool = True
    report_interval_seconds: int = 60
    export_detailed_reports: bool = True
    report_directory: str = "profiling_reports"

    # Performance thresholds
    slow_function_threshold_ms: float = 100.0
    memory_leak_threshold_mb: float = 50.0
    high_cpu_threshold_percent: float = 80.0


class ProfileSample(NamedTuple):
    """Single profile sample."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_utilization: float
    gpu_memory_mb: float
    thread_count: int
    function_name: str | None = None
    call_stack: list[str] | None = None


@dataclass
class FunctionProfile:
    """Profile data for a single function."""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    memory_usage: list[float] = field(default_factory=list)
    stack_traces: list[list[str]] = field(default_factory=list)

    def update(self, execution_time: float, memory_mb: float = 0.0, stack_trace: list[str] | None = None):
        """Update function profile with new execution data."""
        self.call_count += 1
        self.total_time += execution_time
        self.average_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)

        if memory_mb > 0:
            self.memory_usage.append(memory_mb)

        if stack_trace:
            self.stack_traces.append(stack_trace)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'name': self.name,
            'call_count': self.call_count,
            'total_time_ms': self.total_time * 1000,
            'average_time_ms': self.average_time * 1000,
            'min_time_ms': self.min_time * 1000 if self.min_time != float('inf') else 0,
            'max_time_ms': self.max_time * 1000,
            'average_memory_mb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'max_memory_mb': max(self.memory_usage) if self.memory_usage else 0
        }


@dataclass
class ProfileReport:
    """Comprehensive profile report."""
    profiling_duration: float
    total_samples: int
    function_profiles: dict[str, FunctionProfile]
    system_metrics: dict[str, Any]
    performance_issues: list[dict[str, Any]]
    optimization_hints: list[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'profiling_duration_seconds': self.profiling_duration,
            'total_samples': self.total_samples,
            'function_profiles': {
                name: profile.to_dict()
                for name, profile in self.function_profiles.items()
            },
            'system_metrics': self.system_metrics,
            'performance_issues': self.performance_issues,
            'optimization_hints': self.optimization_hints
        }

    def save_report(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class PerformanceProfiler:
    """Main performance profiler class."""

    def __init__(self, config: ProfilingConfig):
        self.config = config

        # Profile data storage
        self.samples: deque = deque(maxlen=config.max_samples)
        self.function_profiles: dict[str, FunctionProfile] = {}
        self.active_functions: dict[str, float] = {}  # function -> start_time

        # System monitoring
        self.process = psutil.Process()
        self.start_time = time.time()

        # GPU monitoring setup
        self.gpu_handle = None
        if config.enable_gpu_profiling and GPU_MONITORING_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                logger.warning(f"GPU monitoring setup failed: {e}")

        # Memory tracking
        if config.trace_malloc:
            tracemalloc.start()

        # cProfile integration
        self.cprofile = None
        if config.level == ProfilingLevel.COMPREHENSIVE:
            self.cprofile = cProfile.Profile()

        # Thread safety
        self._lock = threading.RLock()

        # Background monitoring
        self._monitoring_thread: threading.Thread | None = None
        self._monitoring_active = False

        # Reporting
        self.report_directory = Path(config.report_directory)
        self.report_directory.mkdir(exist_ok=True)

        logger.info(f"Performance profiler initialized with {config.level.value} level")

    def start_profiling(self):
        """Start comprehensive profiling."""
        logger.info("Starting performance profiling")

        self.start_time = time.time()

        # Start cProfile if enabled
        if self.cprofile:
            self.cprofile.enable()

        # Start background monitoring
        if self.config.enable_realtime_monitoring:
            self._start_background_monitoring()

        logger.info("Performance profiling started")

    def stop_profiling(self) -> ProfileReport:
        """Stop profiling and generate report."""
        logger.info("Stopping performance profiling")

        profiling_duration = time.time() - self.start_time

        # Stop cProfile if enabled
        if self.cprofile:
            self.cprofile.disable()

        # Stop background monitoring
        self._stop_background_monitoring()

        # Generate comprehensive report
        report = self._generate_report(profiling_duration)

        # Export detailed report if enabled
        if self.config.export_detailed_reports:
            timestamp = int(time.time())
            report_file = self.report_directory / f"profile_report_{timestamp}.json"
            report.save_report(str(report_file))
            logger.info(f"Detailed report saved to {report_file}")

        logger.info("Performance profiling stopped")
        return report

    @contextlib.contextmanager
    def profile_context(self, context_name: str = "profiling_context"):
        """Context manager for profiling a code block."""
        self.start_profiling()
        start_time = time.time()

        try:
            yield self
        finally:
            execution_time = time.time() - start_time
            report = self.stop_profiling()

            logger.info(f"Profiling context '{context_name}' completed in {execution_time:.3f}s")

    def profile_function(self, func_name: str):
        """Decorator for profiling individual functions."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_profiling(func_name, func, args, kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_async_with_profiling(func_name, func, args, kwargs)

            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper

        return decorator

    def _execute_with_profiling(self, func_name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with profiling."""
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()

        # Record function start
        with self._lock:
            self.active_functions[func_name] = start_time

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Record completion
            execution_time = time.perf_counter() - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before

            # Get stack trace if enabled
            stack_trace = None
            if self.config.enable_call_stack_traces:
                stack_trace = self._get_call_stack()

            # Update function profile
            self._update_function_profile(func_name, execution_time, memory_delta, stack_trace)

            return result

        finally:
            with self._lock:
                self.active_functions.pop(func_name, None)

    async def _execute_async_with_profiling(self, func_name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute async function with profiling."""
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()

        # Record function start
        with self._lock:
            self.active_functions[func_name] = start_time

        try:
            # Execute async function
            result = await func(*args, **kwargs)

            # Record completion
            execution_time = time.perf_counter() - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before

            # Get stack trace if enabled
            stack_trace = None
            if self.config.enable_call_stack_traces:
                stack_trace = self._get_call_stack()

            # Update function profile
            self._update_function_profile(func_name, execution_time, memory_delta, stack_trace)

            return result

        finally:
            with self._lock:
                self.active_functions.pop(func_name, None)

    def _update_function_profile(self, func_name: str, execution_time: float,
                                memory_delta: float, stack_trace: list[str] | None):
        """Update function profile with execution data."""
        with self._lock:
            if func_name not in self.function_profiles:
                self.function_profiles[func_name] = FunctionProfile(func_name)

            self.function_profiles[func_name].update(execution_time, memory_delta, stack_trace)

    def _start_background_monitoring(self):
        """Start background system monitoring thread."""
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="performance_monitor",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Background monitoring started")

    def _stop_background_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Background monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                sample = self._collect_system_sample()

                with self._lock:
                    self.samples.append(sample)

                # Sleep until next sample
                time.sleep(self.config.sampling_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Longer sleep on error

    def _collect_system_sample(self) -> ProfileSample:
        """Collect a single system performance sample."""
        timestamp = time.time()

        # CPU and memory
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Thread count
        thread_count = self.process.num_threads()

        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_mb = 0.0

        if self.gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_utilization = gpu_util.gpu

                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_mb = gpu_mem_info.used / (1024 * 1024)
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")

        return ProfileSample(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_utilization=gpu_utilization,
            gpu_memory_mb=gpu_memory_mb,
            thread_count=thread_count
        )

    def _generate_report(self, profiling_duration: float) -> ProfileReport:
        """Generate comprehensive profiling report."""
        with self._lock:
            # System metrics summary
            system_metrics = self._calculate_system_metrics()

            # Identify performance issues
            performance_issues = self._identify_performance_issues()

            # Generate optimization hints
            optimization_hints = self._generate_optimization_hints()

            return ProfileReport(
                profiling_duration=profiling_duration,
                total_samples=len(self.samples),
                function_profiles=self.function_profiles.copy(),
                system_metrics=system_metrics,
                performance_issues=performance_issues,
                optimization_hints=optimization_hints
            )

    def _calculate_system_metrics(self) -> dict[str, Any]:
        """Calculate summary system metrics."""
        if not self.samples:
            return {}

        cpu_values = [s.cpu_percent for s in self.samples]
        memory_values = [s.memory_mb for s in self.samples]
        gpu_util_values = [s.gpu_utilization for s in self.samples]
        gpu_mem_values = [s.gpu_memory_mb for s in self.samples]

        return {
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'average_mb': sum(memory_values) / len(memory_values),
                'max_mb': max(memory_values),
                'min_mb': min(memory_values),
                'peak_growth_mb': max(memory_values) - min(memory_values)
            },
            'gpu': {
                'average_utilization': sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0,
                'max_utilization': max(gpu_util_values) if gpu_util_values else 0,
                'average_memory_mb': sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else 0,
                'max_memory_mb': max(gpu_mem_values) if gpu_mem_values else 0
            }
        }

    def _identify_performance_issues(self) -> list[dict[str, Any]]:
        """Identify performance issues from profiling data."""
        issues = []

        # Check for slow functions
        for func_name, profile in self.function_profiles.items():
            if profile.average_time * 1000 > self.config.slow_function_threshold_ms:
                issues.append({
                    'type': 'slow_function',
                    'function': func_name,
                    'average_time_ms': profile.average_time * 1000,
                    'call_count': profile.call_count,
                    'severity': 'high' if profile.average_time * 1000 > self.config.slow_function_threshold_ms * 2 else 'medium'
                })

        # Check for high CPU usage
        if self.samples:
            avg_cpu = sum(s.cpu_percent for s in self.samples) / len(self.samples)
            if avg_cpu > self.config.high_cpu_threshold_percent:
                issues.append({
                    'type': 'high_cpu_usage',
                    'average_cpu_percent': avg_cpu,
                    'severity': 'high' if avg_cpu > 90 else 'medium'
                })

        # Check for memory leaks
        if len(self.samples) > 100:  # Need sufficient samples
            memory_values = [s.memory_mb for s in self.samples]
            memory_growth = memory_values[-1] - memory_values[0]

            if memory_growth > self.config.memory_leak_threshold_mb:
                issues.append({
                    'type': 'memory_growth',
                    'growth_mb': memory_growth,
                    'severity': 'high' if memory_growth > self.config.memory_leak_threshold_mb * 2 else 'medium'
                })

        return issues

    def _generate_optimization_hints(self) -> list[str]:
        """Generate optimization hints based on profiling data."""
        hints = []

        # Function-level hints
        for func_name, profile in self.function_profiles.items():
            if profile.average_time * 1000 > self.config.slow_function_threshold_ms:
                hints.append(f"Consider optimizing function '{func_name}' "
                           f"(avg time: {profile.average_time * 1000:.1f}ms)")

                # Memory usage hints
                if profile.memory_usage:
                    avg_memory = sum(profile.memory_usage) / len(profile.memory_usage)
                    if avg_memory > 100:  # > 100MB
                        hints.append(f"Function '{func_name}' uses significant memory "
                                   f"(avg: {avg_memory:.1f}MB) - consider optimization")

        # System-level hints
        if self.samples:
            avg_cpu = sum(s.cpu_percent for s in self.samples) / len(self.samples)
            max_memory = max(s.memory_mb for s in self.samples)

            if avg_cpu > 80:
                hints.append("High CPU usage detected - consider parallel processing or optimization")

            if max_memory > 8192:  # > 8GB
                hints.append("High memory usage detected - consider memory optimization techniques")

            # GPU hints
            if self.gpu_handle:
                avg_gpu = sum(s.gpu_utilization for s in self.samples) / len(self.samples)
                if avg_gpu < 30:
                    hints.append("Low GPU utilization - consider GPU acceleration opportunities")
                elif avg_gpu > 90:
                    hints.append("High GPU utilization - consider GPU memory optimization")

        return hints

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def _get_call_stack(self) -> list[str]:
        """Get current call stack."""
        try:
            import traceback
            stack = traceback.extract_stack(limit=self.config.max_stack_depth)
            return [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack]
        except:
            return []

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        sample = self._collect_system_sample()

        return {
            'timestamp': sample.timestamp,
            'cpu_percent': sample.cpu_percent,
            'memory_mb': sample.memory_mb,
            'gpu_utilization': sample.gpu_utilization,
            'gpu_memory_mb': sample.gpu_memory_mb,
            'thread_count': sample.thread_count,
            'active_functions': len(self.active_functions),
            'total_function_profiles': len(self.function_profiles)
        }

    def export_cprofile_stats(self, output_file: str):
        """Export cProfile statistics to file."""
        if not self.cprofile:
            logger.warning("cProfile not enabled")
            return

        # Create stats object
        s = io.StringIO()
        ps = pstats.Stats(self.cprofile, stream=s)
        ps.sort_stats('tottime')
        ps.print_stats()

        # Write to file
        with open(output_file, 'w') as f:
            f.write(s.getvalue())

        logger.info(f"cProfile stats exported to {output_file}")

    def reset_profiling_data(self):
        """Reset all profiling data."""
        with self._lock:
            self.samples.clear()
            self.function_profiles.clear()
            self.active_functions.clear()

        if self.cprofile:
            self.cprofile.clear()

        logger.info("Profiling data reset")
