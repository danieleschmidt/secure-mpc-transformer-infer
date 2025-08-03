"""Metrics collection and monitoring utilities."""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        with self._lock:
            self._value += amount
    
    def get_value(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Histogram:
    """Thread-safe histogram metric."""
    
    def __init__(self, name: str, help_text: str = "", buckets: Optional[List[float]] = None):
        self.name = name
        self.help_text = help_text
        
        # Default buckets for latency measurements
        if buckets is None:
            buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        
        self.buckets = sorted(buckets)
        self._bucket_counts = [0] * len(self.buckets)
        self._sum = 0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        with self._lock:
            self._sum += value
            self._count += 1
            
            # Update bucket counts
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._bucket_counts[i] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            return {
                "count": self._count,
                "sum": self._sum,
                "average": self._sum / self._count if self._count > 0 else 0,
                "buckets": dict(zip(self.buckets, self._bucket_counts))
            }
    
    def reset(self):
        """Reset histogram."""
        with self._lock:
            self._bucket_counts = [0] * len(self.buckets)
            self._sum = 0
            self._count = 0


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self._value = 0
        self._lock = threading.Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment gauge value."""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement gauge value."""
        with self._lock:
            self._value -= amount
    
    def get_value(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value


class TimeSeries:
    """Time series data collector."""
    
    def __init__(self, name: str, max_points: int = 1000):
        self.name = name
        self.max_points = max_points
        self._points = deque(maxlen=max_points)
        self._lock = threading.Lock()
    
    def add_point(self, value: float, timestamp: Optional[float] = None, 
                  labels: Optional[Dict[str, str]] = None):
        """Add a data point."""
        if timestamp is None:
            timestamp = time.time()
        
        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=timestamp,
            labels=labels or {}
        )
        
        with self._lock:
            self._points.append(point)
    
    def get_points(self, start_time: Optional[float] = None, 
                   end_time: Optional[float] = None) -> List[MetricPoint]:
        """Get data points within time range."""
        with self._lock:
            points = list(self._points)
        
        if start_time is not None:
            points = [p for p in points if p.timestamp >= start_time]
        
        if end_time is not None:
            points = [p for p in points if p.timestamp <= end_time]
        
        return points
    
    def get_statistics(self, start_time: Optional[float] = None) -> Dict[str, Any]:
        """Get time series statistics."""
        points = self.get_points(start_time=start_time)
        
        if not points:
            return {"count": 0}
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "average": sum(values) / len(values),
            "latest": values[-1],
            "start_time": points[0].timestamp,
            "end_time": points[-1].timestamp
        }


class MetricsCollector:
    """Central metrics collection and management."""
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._time_series: Dict[str, TimeSeries] = {}
        self._lock = threading.Lock()
        
        # Initialize common metrics
        self._init_default_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def _init_default_metrics(self):
        """Initialize default metrics for the application."""
        # Inference metrics
        self.create_counter("inference_requests_total", "Total number of inference requests")
        self.create_counter("inference_requests_success", "Number of successful inference requests")
        self.create_counter("inference_requests_failed", "Number of failed inference requests")
        
        self.create_histogram("inference_latency_ms", "Inference latency in milliseconds")
        self.create_histogram("model_loading_time_ms", "Model loading time in milliseconds")
        
        # System metrics
        self.create_gauge("active_requests", "Number of active inference requests")
        self.create_gauge("loaded_models", "Number of loaded models in memory")
        self.create_gauge("memory_usage_mb", "Memory usage in megabytes")
        
        # Protocol metrics
        self.create_counter("mpc_operations_total", "Total number of MPC operations")
        self.create_histogram("mpc_round_latency_ms", "MPC round latency in milliseconds")
        self.create_gauge("protocol_parties_active", "Number of active protocol parties")
        
        # Security metrics
        self.create_counter("security_violations", "Number of security violations detected")
        self.create_counter("authentication_failures", "Number of authentication failures")
        
        # Communication metrics
        self.create_histogram("communication_bytes", "Communication data size in bytes")
        self.create_counter("network_errors", "Number of network errors")
    
    def create_counter(self, name: str, help_text: str = "") -> Counter:
        """Create a new counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, help_text)
            return self._counters[name]
    
    def create_histogram(self, name: str, help_text: str = "", 
                        buckets: Optional[List[float]] = None) -> Histogram:
        """Create a new histogram metric."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, help_text, buckets)
            return self._histograms[name]
    
    def create_gauge(self, name: str, help_text: str = "") -> Gauge:
        """Create a new gauge metric."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, help_text)
            return self._gauges[name]
    
    def create_time_series(self, name: str, max_points: int = 1000) -> TimeSeries:
        """Create a new time series collector."""
        with self._lock:
            if name not in self._time_series:
                self._time_series[name] = TimeSeries(name, max_points)
            return self._time_series[name]
    
    def increment_counter(self, name: str, amount: float = 1.0, 
                         labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        counter = self.create_counter(name)
        counter.increment(amount, labels)
    
    def observe_histogram(self, name: str, value: float, 
                         labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram."""
        histogram = self.create_histogram(name)
        histogram.observe(value, labels)
    
    def set_gauge(self, name: str, value: float, 
                  labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        gauge = self.create_gauge(name)
        gauge.set(value, labels)
    
    def record_time_series(self, name: str, value: float, 
                          labels: Optional[Dict[str, str]] = None):
        """Record a time series data point."""
        ts = self.create_time_series(name)
        ts.add_point(value, labels=labels)
    
    def get_counter_value(self, name: str) -> float:
        """Get current counter value."""
        if name in self._counters:
            return self._counters[name].get_value()
        return 0.0
    
    def get_gauge_value(self, name: str) -> float:
        """Get current gauge value."""
        if name in self._gauges:
            return self._gauges[name].get_value()
        return 0.0
    
    def get_histogram_stats(self, name: str) -> Dict[str, Any]:
        """Get histogram statistics."""
        if name in self._histograms:
            return self._histograms[name].get_statistics()
        return {}
    
    def get_time_series_stats(self, name: str, start_time: Optional[float] = None) -> Dict[str, Any]:
        """Get time series statistics."""
        if name in self._time_series:
            return self._time_series[name].get_statistics(start_time)
        return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics data."""
        with self._lock:
            metrics = {
                "counters": {name: counter.get_value() for name, counter in self._counters.items()},
                "gauges": {name: gauge.get_value() for name, gauge in self._gauges.items()},
                "histograms": {name: hist.get_statistics() for name, hist in self._histograms.items()},
                "time_series": {name: ts.get_statistics() for name, ts in self._time_series.items()}
            }
        
        return metrics
    
    def reset_all_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            
            for histogram in self._histograms.values():
                histogram.reset()
            
            for gauge in self._gauges.values():
                gauge.set(0)
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get_value()}")
        
        # Export gauges
        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get_value()}")
        
        # Export histograms
        for name, histogram in self._histograms.items():
            stats = histogram.get_statistics()
            lines.append(f"# HELP {name} {histogram.help_text}")
            lines.append(f"# TYPE {name} histogram")
            
            for bucket, count in stats["buckets"].items():
                lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            
            lines.append(f"{name}_sum {stats['sum']}")
            lines.append(f"{name}_count {stats['count']}")
        
        return "\n".join(lines)
    
    def start_periodic_collection(self, interval: float = 60.0):
        """Start periodic metrics collection."""
        import threading
        
        def collect_system_metrics():
            while True:
                try:
                    # Collect system metrics
                    import psutil
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.set_gauge("memory_usage_mb", memory.used / (1024 * 1024))
                    self.set_gauge("memory_usage_percent", memory.percent)
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge("cpu_usage_percent", cpu_percent)
                    
                    # GPU metrics (if available)
                    try:
                        import torch
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                memory_used = torch.cuda.memory_allocated(i) / (1024 * 1024)
                                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                                
                                self.set_gauge(f"gpu_{i}_memory_used_mb", memory_used)
                                self.set_gauge(f"gpu_{i}_memory_total_mb", memory_total)
                                self.set_gauge(f"gpu_{i}_memory_utilization", memory_used / memory_total * 100)
                    except:
                        pass
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(interval)
        
        # Start collection thread
        collection_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        collection_thread.start()
        
        logger.info(f"Started periodic metrics collection with {interval}s interval")


# Global metrics collector instance
metrics_collector = MetricsCollector()


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str, collector: MetricsCollector = None):
        self.metric_name = metric_name
        self.collector = collector or metrics_collector
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.observe_histogram(self.metric_name, duration_ms)


def time_operation(metric_name: str):
    """Decorator for timing function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(metric_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator