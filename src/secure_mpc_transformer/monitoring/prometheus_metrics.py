"""Advanced Prometheus metrics collection and export system."""

import asyncio
import gc
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PrometheusMetric:
    """Prometheus metric definition."""

    name: str
    metric_type: MetricType
    help_text: str
    labels: list[str]
    value: float | dict[str, float]
    timestamp: float | None = None

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        lines = []

        # Add HELP and TYPE comments
        lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} {self.metric_type.value}")

        if isinstance(self.value, dict):
            # Multiple label combinations
            for label_combo, value in self.value.items():
                if label_combo:
                    lines.append(f"{self.name}{{{label_combo}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        else:
            # Single value
            lines.append(f"{self.name} {self.value}")

        return "\n".join(lines)


class PrometheusCounter:
    """Thread-safe Prometheus counter."""

    def __init__(self, name: str, help_text: str, label_names: list[str] = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: dict[str, str] = None):
        """Increment counter."""
        if amount < 0:
            raise ValueError("Counter increment must be non-negative")

        label_key = self._make_label_key(labels or {})

        with self._lock:
            self._values[label_key] += amount

    def get_value(self, labels: dict[str, str] = None) -> float:
        """Get current counter value."""
        label_key = self._make_label_key(labels or {})

        with self._lock:
            return self._values.get(label_key, 0.0)

    def get_all_values(self) -> dict[str, float]:
        """Get all counter values."""
        with self._lock:
            return dict(self._values)

    def _make_label_key(self, labels: dict[str, str]) -> str:
        """Create label key for internal storage."""
        if not labels:
            return ""

        # Ensure label names are in order
        sorted_labels = sorted(labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_labels)

    def to_prometheus_metric(self) -> PrometheusMetric:
        """Convert to Prometheus metric format."""
        with self._lock:
            values = dict(self._values)

        return PrometheusMetric(
            name=self.name,
            metric_type=MetricType.COUNTER,
            help_text=self.help_text,
            labels=self.label_names,
            value=values,
            timestamp=time.time()
        )


class PrometheusGauge:
    """Thread-safe Prometheus gauge."""

    def __init__(self, name: str, help_text: str, label_names: list[str] = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: dict[str, str] = None):
        """Set gauge value."""
        label_key = self._make_label_key(labels or {})

        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, labels: dict[str, str] = None):
        """Increment gauge value."""
        label_key = self._make_label_key(labels or {})

        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] = None):
        """Decrement gauge value."""
        label_key = self._make_label_key(labels or {})

        with self._lock:
            self._values[label_key] -= amount

    def get_value(self, labels: dict[str, str] = None) -> float:
        """Get current gauge value."""
        label_key = self._make_label_key(labels or {})

        with self._lock:
            return self._values.get(label_key, 0.0)

    def get_all_values(self) -> dict[str, float]:
        """Get all gauge values."""
        with self._lock:
            return dict(self._values)

    def _make_label_key(self, labels: dict[str, str]) -> str:
        """Create label key for internal storage."""
        if not labels:
            return ""

        sorted_labels = sorted(labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_labels)

    def to_prometheus_metric(self) -> PrometheusMetric:
        """Convert to Prometheus metric format."""
        with self._lock:
            values = dict(self._values)

        return PrometheusMetric(
            name=self.name,
            metric_type=MetricType.GAUGE,
            help_text=self.help_text,
            labels=self.label_names,
            value=values,
            timestamp=time.time()
        )


class PrometheusHistogram:
    """Thread-safe Prometheus histogram."""

    def __init__(self, name: str, help_text: str, buckets: list[float] = None,
                 label_names: list[str] = None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []

        # Default buckets for latency measurements
        if buckets is None:
            buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]

        self.buckets = sorted(buckets)
        self._bucket_counts = defaultdict(lambda: [0] * len(self.buckets))
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict[str, str] = None):
        """Observe a value."""
        if value < 0:
            raise ValueError("Histogram observation must be non-negative")

        label_key = self._make_label_key(labels or {})

        with self._lock:
            self._sums[label_key] += value
            self._counts[label_key] += 1

            # Update bucket counts
            bucket_counts = self._bucket_counts[label_key]
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    bucket_counts[i] += 1

    def get_statistics(self, labels: dict[str, str] = None) -> dict[str, Any]:
        """Get histogram statistics."""
        label_key = self._make_label_key(labels or {})

        with self._lock:
            count = self._counts.get(label_key, 0)
            sum_value = self._sums.get(label_key, 0.0)
            bucket_counts = self._bucket_counts.get(label_key, [0] * len(self.buckets))

            return {
                'count': count,
                'sum': sum_value,
                'average': sum_value / count if count > 0 else 0.0,
                'buckets': dict(zip(self.buckets, bucket_counts, strict=False))
            }

    def _make_label_key(self, labels: dict[str, str]) -> str:
        """Create label key for internal storage."""
        if not labels:
            return ""

        sorted_labels = sorted(labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_labels)

    def to_prometheus_metric(self) -> PrometheusMetric:
        """Convert to Prometheus metric format."""
        with self._lock:
            all_values = {}

            # Export bucket counts
            for label_key, bucket_counts in self._bucket_counts.items():
                for bucket, count in zip(self.buckets, bucket_counts, strict=False):
                    bucket_label = f"le=\"{bucket}\""
                    if label_key:
                        full_label = f"{label_key},{bucket_label}"
                    else:
                        full_label = bucket_label

                    all_values[full_label] = count

            # Export sums and counts as separate metrics
            sum_values = {}
            count_values = {}

            for label_key in self._sums.keys():
                sum_values[label_key] = self._sums[label_key]
                count_values[label_key] = self._counts[label_key]

        # Return the main histogram metric
        return PrometheusMetric(
            name=f"{self.name}_bucket",
            metric_type=MetricType.HISTOGRAM,
            help_text=self.help_text,
            labels=self.label_names + ["le"],
            value=all_values,
            timestamp=time.time()
        )


class SystemMetricsCollector:
    """Collect system-level metrics."""

    def __init__(self):
        self.process = psutil.Process()

    def collect_cpu_metrics(self) -> dict[str, float]:
        """Collect CPU metrics."""
        return {
            'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
            'load_average_5m': psutil.getloadavg()[1] if hasattr(psutil, 'getloadavg') else 0.0,
            'load_average_15m': psutil.getloadavg()[2] if hasattr(psutil, 'getloadavg') else 0.0,
        }

    def collect_memory_metrics(self) -> dict[str, float]:
        """Collect memory metrics."""
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        return {
            'memory_total_bytes': memory.total,
            'memory_available_bytes': memory.available,
            'memory_used_bytes': memory.used,
            'memory_usage_percent': memory.percent,
            'process_memory_rss_bytes': process_memory.rss,
            'process_memory_vms_bytes': process_memory.vms,
        }

    def collect_disk_metrics(self) -> dict[str, float]:
        """Collect disk metrics."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        metrics = {
            'disk_total_bytes': disk_usage.total,
            'disk_used_bytes': disk_usage.used,
            'disk_free_bytes': disk_usage.free,
            'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
        }

        if disk_io:
            metrics.update({
                'disk_read_bytes_total': disk_io.read_bytes,
                'disk_write_bytes_total': disk_io.write_bytes,
                'disk_read_operations_total': disk_io.read_count,
                'disk_write_operations_total': disk_io.write_count,
            })

        return metrics

    def collect_network_metrics(self) -> dict[str, float]:
        """Collect network metrics."""
        network_io = psutil.net_io_counters()

        if network_io:
            return {
                'network_receive_bytes_total': network_io.bytes_recv,
                'network_transmit_bytes_total': network_io.bytes_sent,
                'network_receive_packets_total': network_io.packets_recv,
                'network_transmit_packets_total': network_io.packets_sent,
                'network_receive_errors_total': network_io.errin,
                'network_transmit_errors_total': network_io.errout,
            }

        return {}

    def collect_gpu_metrics(self) -> dict[str, float]:
        """Collect GPU metrics if PyTorch is available."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        metrics = {}

        try:
            for i in range(torch.cuda.device_count()):
                device_name = f"cuda:{i}"

                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory

                metrics.update({
                    f'gpu_{i}_memory_allocated_bytes': memory_allocated,
                    f'gpu_{i}_memory_cached_bytes': memory_cached,
                    f'gpu_{i}_memory_total_bytes': memory_total,
                    f'gpu_{i}_memory_utilization_percent': (memory_allocated / memory_total) * 100,
                })

                # Device properties
                props = torch.cuda.get_device_properties(i)
                metrics.update({
                    f'gpu_{i}_multiprocessor_count': props.multi_processor_count,
                    f'gpu_{i}_major_capability': props.major,
                    f'gpu_{i}_minor_capability': props.minor,
                })

        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")

        return metrics

    def collect_process_metrics(self) -> dict[str, float]:
        """Collect process-specific metrics."""
        try:
            return {
                'process_cpu_usage_percent': self.process.cpu_percent(),
                'process_threads_count': self.process.num_threads(),
                'process_open_fds_count': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                'process_uptime_seconds': time.time() - self.process.create_time(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}


class AdvancedPrometheusExporter:
    """Advanced Prometheus metrics exporter with comprehensive monitoring."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Metric storage
        self.counters: dict[str, PrometheusCounter] = {}
        self.gauges: dict[str, PrometheusGauge] = {}
        self.histograms: dict[str, PrometheusHistogram] = {}

        # System metrics collector
        self.system_collector = SystemMetricsCollector()

        # Configuration
        self.collection_interval = self.config.get('collection_interval', 15)  # seconds
        self.enable_system_metrics = self.config.get('enable_system_metrics', True)
        self.enable_application_metrics = self.config.get('enable_application_metrics', True)
        self.enable_security_metrics = self.config.get('enable_security_metrics', True)

        # State
        self.last_collection_time = 0
        self._lock = threading.RLock()
        self._collection_task = None

        # Initialize default metrics
        self._initialize_default_metrics()

        # Start collection task
        if self.config.get('auto_start_collection', True):
            self.start_collection()

        logger.info("Advanced Prometheus exporter initialized")

    def _initialize_default_metrics(self):
        """Initialize default application metrics."""

        # HTTP metrics
        self.register_counter(
            "http_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status"]
        )

        self.register_histogram(
            "http_request_duration_seconds",
            "Duration of HTTP requests in seconds",
            label_names=["method", "endpoint", "status"]
        )

        # MPC protocol metrics
        self.register_counter(
            "mpc_operations_total",
            "Total number of MPC operations",
            ["protocol", "operation_type", "status"]
        )

        self.register_histogram(
            "mpc_operation_duration_seconds",
            "Duration of MPC operations in seconds",
            label_names=["protocol", "operation_type"]
        )

        self.register_gauge(
            "mpc_active_sessions",
            "Number of active MPC sessions",
            ["protocol"]
        )

        # Inference metrics
        self.register_counter(
            "inference_requests_total",
            "Total number of inference requests",
            ["model", "status"]
        )

        self.register_histogram(
            "inference_duration_seconds",
            "Duration of inference operations in seconds",
            label_names=["model"]
        )

        self.register_gauge(
            "inference_batch_size",
            "Current inference batch size",
            ["model"]
        )

        # Security metrics
        self.register_counter(
            "security_events_total",
            "Total number of security events",
            ["event_type", "severity"]
        )

        self.register_gauge(
            "active_sessions_count",
            "Number of active user sessions"
        )

        self.register_counter(
            "blocked_requests_total",
            "Total number of blocked requests",
            ["reason"]
        )

        # System health metrics
        self.register_gauge(
            "application_info",
            "Application information",
            ["version", "build_date", "python_version"]
        )

        # Set application info
        self.gauges["application_info"].set(1.0, {
            "version": "0.1.0",
            "build_date": "2024-01-01",
            "python_version": "3.9"
        })

    def register_counter(self, name: str, help_text: str, label_names: list[str] = None) -> PrometheusCounter:
        """Register a new counter metric."""
        if name in self.counters:
            return self.counters[name]

        counter = PrometheusCounter(name, help_text, label_names)

        with self._lock:
            self.counters[name] = counter

        return counter

    def register_gauge(self, name: str, help_text: str, label_names: list[str] = None) -> PrometheusGauge:
        """Register a new gauge metric."""
        if name in self.gauges:
            return self.gauges[name]

        gauge = PrometheusGauge(name, help_text, label_names)

        with self._lock:
            self.gauges[name] = gauge

        return gauge

    def register_histogram(self, name: str, help_text: str, buckets: list[float] = None,
                          label_names: list[str] = None) -> PrometheusHistogram:
        """Register a new histogram metric."""
        if name in self.histograms:
            return self.histograms[name]

        histogram = PrometheusHistogram(name, help_text, buckets, label_names)

        with self._lock:
            self.histograms[name] = histogram

        return histogram

    def increment_counter(self, name: str, amount: float = 1.0, labels: dict[str, str] = None):
        """Increment a counter metric."""
        if name in self.counters:
            self.counters[name].inc(amount, labels)
        else:
            logger.warning(f"Counter {name} not found")

    def set_gauge(self, name: str, value: float, labels: dict[str, str] = None):
        """Set a gauge metric value."""
        if name in self.gauges:
            self.gauges[name].set(value, labels)
        else:
            logger.warning(f"Gauge {name} not found")

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] = None):
        """Observe a histogram metric value."""
        if name in self.histograms:
            self.histograms[name].observe(value, labels)
        else:
            logger.warning(f"Histogram {name} not found")

    def start_collection(self):
        """Start automatic metrics collection."""
        if self._collection_task:
            return

        async def collection_loop():
            while True:
                try:
                    await self._collect_system_metrics()
                    await asyncio.sleep(self.collection_interval)
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    await asyncio.sleep(self.collection_interval)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._collection_task = asyncio.create_task(collection_loop())
        except RuntimeError:
            # Start manual collection
            threading.Timer(self.collection_interval, self._manual_collection).start()

    def stop_collection(self):
        """Stop automatic metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            self._collection_task = None

    async def _collect_system_metrics(self):
        """Collect system metrics."""
        if not self.enable_system_metrics:
            return

        current_time = time.time()

        # CPU metrics
        cpu_metrics = self.system_collector.collect_cpu_metrics()
        for metric_name, value in cpu_metrics.items():
            self.set_gauge(f"system_{metric_name}", value)

        # Memory metrics
        memory_metrics = self.system_collector.collect_memory_metrics()
        for metric_name, value in memory_metrics.items():
            self.set_gauge(f"system_{metric_name}", value)

        # Disk metrics
        disk_metrics = self.system_collector.collect_disk_metrics()
        for metric_name, value in disk_metrics.items():
            if metric_name.endswith('_total'):
                # Convert to counter if it's a total
                counter_name = f"system_{metric_name}"
                if counter_name not in self.counters:
                    self.register_counter(counter_name, f"System {metric_name}")
                # Note: We can't directly set counter values, so we track the difference
                # This is a simplified approach - in production, you'd want proper counter handling
            else:
                self.set_gauge(f"system_{metric_name}", value)

        # Network metrics
        network_metrics = self.system_collector.collect_network_metrics()
        for metric_name, value in network_metrics.items():
            if metric_name.endswith('_total'):
                counter_name = f"system_{metric_name}"
                if counter_name not in self.counters:
                    self.register_counter(counter_name, f"System {metric_name}")
            else:
                self.set_gauge(f"system_{metric_name}", value)

        # GPU metrics
        gpu_metrics = self.system_collector.collect_gpu_metrics()
        for metric_name, value in gpu_metrics.items():
            self.set_gauge(f"system_{metric_name}", value)

        # Process metrics
        process_metrics = self.system_collector.collect_process_metrics()
        for metric_name, value in process_metrics.items():
            self.set_gauge(f"system_{metric_name}", value)

        # Python GC metrics
        gc_stats = gc.get_stats()
        if gc_stats:
            for i, gen_stats in enumerate(gc_stats):
                self.set_gauge("python_gc_collections_total", gen_stats['collections'], {'generation': str(i)})
                self.set_gauge("python_gc_collected_total", gen_stats['collected'], {'generation': str(i)})
                self.set_gauge("python_gc_uncollectable_total", gen_stats['uncollectable'], {'generation': str(i)})

        self.last_collection_time = current_time

    def _manual_collection(self):
        """Manual collection for non-async environments."""
        try:
            asyncio.run(self._collect_system_metrics())
        except Exception as e:
            logger.error(f"Manual metrics collection error: {e}")
        finally:
            threading.Timer(self.collection_interval, self._manual_collection).start()

    def export_metrics(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        current_time = time.time()

        with self._lock:
            # Export counters
            for counter in self.counters.values():
                metric = counter.to_prometheus_metric()
                lines.append(metric.to_prometheus_format())
                lines.append("")  # Empty line between metrics

            # Export gauges
            for gauge in self.gauges.values():
                metric = gauge.to_prometheus_metric()
                lines.append(metric.to_prometheus_format())
                lines.append("")

            # Export histograms
            for histogram in self.histograms.values():
                metric = histogram.to_prometheus_metric()
                lines.append(metric.to_prometheus_format())

                # Also export sum and count
                stats = histogram.get_statistics()

                # Sum metric
                lines.append(f"# HELP {histogram.name}_sum Total sum of observed values")
                lines.append(f"# TYPE {histogram.name}_sum counter")
                lines.append(f"{histogram.name}_sum {stats['sum']}")
                lines.append("")

                # Count metric
                lines.append(f"# HELP {histogram.name}_count Total count of observations")
                lines.append(f"# TYPE {histogram.name}_count counter")
                lines.append(f"{histogram.name}_count {stats['count']}")
                lines.append("")

        # Add metadata
        lines.insert(0, "# HELP prometheus_exporter_build_info Build information")
        lines.insert(1, "# TYPE prometheus_exporter_build_info gauge")
        lines.insert(2, f'prometheus_exporter_build_info{{version="1.0.0",timestamp="{current_time}"}} 1')
        lines.insert(3, "")

        return "\n".join(lines)

    def get_metric_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            return {
                'counters': len(self.counters),
                'gauges': len(self.gauges),
                'histograms': len(self.histograms),
                'last_collection_time': self.last_collection_time,
                'collection_interval': self.collection_interval,
                'system_metrics_enabled': self.enable_system_metrics,
                'application_metrics_enabled': self.enable_application_metrics,
                'security_metrics_enabled': self.enable_security_metrics
            }


# Context manager for timing operations
class PrometheusTimer:
    """Context manager for timing operations with Prometheus histograms."""

    def __init__(self, histogram: PrometheusHistogram, labels: dict[str, str] = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration, self.labels)


# Decorator for timing functions
def prometheus_timer(histogram_name: str, labels_func: Callable = None):
    """Decorator for timing function execution with Prometheus metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            exporter = prometheus_exporter
            if histogram_name in exporter.histograms:
                histogram = exporter.histograms[histogram_name]
                labels = labels_func(*args, **kwargs) if labels_func else {}

                with PrometheusTimer(histogram, labels):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


# Global Prometheus exporter instance
prometheus_exporter = AdvancedPrometheusExporter()
