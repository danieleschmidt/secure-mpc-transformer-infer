"""
Performance Monitoring for Quantum Planning

Comprehensive monitoring and metrics collection for quantum-inspired
task planning algorithms, providing insights into performance and optimization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import time
import threading
import asyncio
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Performance alert"""
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    quantum_metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    task_metrics: Dict[str, float]
    alerts: List[Alert] = field(default_factory=list)


class MetricsCollector:
    """
    High-performance metrics collector for quantum planning operations.
    Thread-safe collection with minimal overhead on planning algorithms.
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.collection_start_time = time.time()
        self.total_metrics_collected = 0
        
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record counter metric (monotonically increasing)"""
        with self._lock:
            self.counters[name] += value
            self._add_metric(name, value, MetricType.COUNTER, labels)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record gauge metric (point-in-time value)"""
        with self._lock:
            self.gauges[name] = value
            self._add_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric (distribution of values)"""
        with self._lock:
            self.histograms[name].append(value)
            if len(self.histograms[name]) > self.buffer_size:
                self.histograms[name].pop(0)
            self._add_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record timer metric (duration measurements)"""
        with self._lock:
            self.timers[name].append(duration)
            if len(self.timers[name]) > self.buffer_size:
                self.timers[name].pop(0)
            self._add_metric(name, duration, MetricType.TIMER, labels)
    
    def _add_metric(self, name: str, value: float, metric_type: MetricType, labels: Optional[Dict[str, str]]):
        """Internal method to add metric to buffer"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        self.total_metrics_collected += 1
    
    def get_counter_value(self, name: str) -> float:
        """Get current counter value"""
        with self._lock:
            return self.counters.get(name, 0.0)
    
    def get_gauge_value(self, name: str) -> Optional[float]:
        """Get current gauge value"""
        with self._lock:
            return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            values = self.histograms.get(name, [])
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "std": np.std(values)
            }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics"""
        return self.get_histogram_stats(name)  # Timers are histograms of durations


class QuantumPerformanceMonitor:
    """
    Specialized monitor for quantum planning algorithm performance.
    Tracks quantum-specific metrics and performance characteristics.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.quantum_sessions: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Performance thresholds
        self.thresholds = {
            "quantum_coherence_min": 0.1,
            "optimization_time_max": 30.0,  # seconds
            "convergence_rate_min": 0.5,
            "memory_usage_max": 0.9,  # 90% of available
            "error_rate_max": 0.05   # 5% error rate
        }
        
    def start_quantum_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Start monitoring a quantum planning session"""
        self.quantum_sessions[session_id] = {
            "start_time": time.time(),
            "metadata": metadata or {},
            "quantum_states": [],
            "optimization_steps": 0,
            "convergence_history": [],
            "resource_usage": [],
            "errors": []
        }
        
        self.metrics.record_counter("quantum_sessions_started")
        logger.debug(f"Started quantum session monitoring: {session_id}")
    
    def record_quantum_state(self, session_id: str, quantum_state: np.ndarray, step: int):
        """Record quantum state metrics"""
        if session_id not in self.quantum_sessions:
            logger.warning(f"Unknown quantum session: {session_id}")
            return
        
        session = self.quantum_sessions[session_id]
        
        # Calculate quantum metrics
        norm = np.linalg.norm(quantum_state)
        amplitudes = np.abs(quantum_state) ** 2
        entropy = -np.sum(amplitudes * np.log2(amplitudes + 1e-12))
        purity = np.sum(amplitudes ** 2)
        
        # Coherence (simplified measure)
        n = len(quantum_state)
        density_matrix = np.outer(quantum_state, np.conj(quantum_state))
        off_diagonal = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
        coherence = off_diagonal / (n * n - n) if n > 1 else 0.0
        
        quantum_metrics = {
            "normalization": norm,
            "entropy": entropy,
            "purity": purity,
            "coherence": coherence,
            "step": step
        }
        
        session["quantum_states"].append(quantum_metrics)
        
        # Record individual metrics
        self.metrics.record_gauge("quantum_normalization", norm, {"session": session_id})
        self.metrics.record_gauge("quantum_entropy", entropy, {"session": session_id})
        self.metrics.record_gauge("quantum_purity", purity, {"session": session_id})
        self.metrics.record_gauge("quantum_coherence", coherence, {"session": session_id})
        
        # Check alerts
        if coherence < self.thresholds["quantum_coherence_min"]:
            self._trigger_alert(
                "low_quantum_coherence",
                AlertLevel.WARNING,
                f"Quantum coherence dropped to {coherence:.3f}",
                "quantum_coherence",
                coherence,
                self.thresholds["quantum_coherence_min"]
            )
    
    def record_optimization_step(self, session_id: str, 
                                objective_value: float,
                                convergence_rate: float,
                                step_duration: float):
        """Record optimization algorithm step"""
        if session_id not in self.quantum_sessions:
            return
        
        session = self.quantum_sessions[session_id]
        session["optimization_steps"] += 1
        session["convergence_history"].append({
            "step": session["optimization_steps"],
            "objective": objective_value,
            "convergence_rate": convergence_rate,
            "duration": step_duration
        })
        
        # Record metrics
        self.metrics.record_histogram("optimization_objective", objective_value, {"session": session_id})
        self.metrics.record_histogram("convergence_rate", convergence_rate, {"session": session_id})
        self.metrics.record_timer("optimization_step_duration", step_duration, {"session": session_id})
        
        # Check convergence alerts
        if convergence_rate < self.thresholds["convergence_rate_min"]:
            self._trigger_alert(
                "slow_convergence",
                AlertLevel.INFO,
                f"Slow convergence rate: {convergence_rate:.3f}",
                "convergence_rate",
                convergence_rate,
                self.thresholds["convergence_rate_min"]
            )
    
    def record_resource_usage(self, session_id: str, 
                            cpu_usage: float, 
                            memory_usage: float, 
                            gpu_usage: Optional[float] = None):
        """Record resource usage metrics"""
        if session_id not in self.quantum_sessions:
            return
        
        session = self.quantum_sessions[session_id]
        resource_snapshot = {
            "timestamp": time.time(),
            "cpu": cpu_usage,
            "memory": memory_usage,
            "gpu": gpu_usage
        }
        session["resource_usage"].append(resource_snapshot)
        
        # Record metrics
        self.metrics.record_gauge("cpu_usage", cpu_usage, {"session": session_id})
        self.metrics.record_gauge("memory_usage", memory_usage, {"session": session_id})
        if gpu_usage is not None:
            self.metrics.record_gauge("gpu_usage", gpu_usage, {"session": session_id})
        
        # Memory usage alert
        if memory_usage > self.thresholds["memory_usage_max"]:
            self._trigger_alert(
                "high_memory_usage",
                AlertLevel.WARNING,
                f"High memory usage: {memory_usage:.1%}",
                "memory_usage",
                memory_usage,
                self.thresholds["memory_usage_max"]
            )
    
    def record_error(self, session_id: str, error_type: str, error_message: str):
        """Record error in quantum session"""
        if session_id not in self.quantum_sessions:
            return
        
        session = self.quantum_sessions[session_id]
        error_record = {
            "timestamp": time.time(),
            "type": error_type,
            "message": error_message
        }
        session["errors"].append(error_record)
        
        self.metrics.record_counter("quantum_errors", 1.0, {
            "session": session_id,
            "error_type": error_type
        })
        
        # Calculate error rate
        total_steps = session["optimization_steps"]
        error_rate = len(session["errors"]) / max(total_steps, 1)
        
        if error_rate > self.thresholds["error_rate_max"]:
            self._trigger_alert(
                "high_error_rate",
                AlertLevel.ERROR,
                f"High error rate: {error_rate:.1%}",
                "error_rate",
                error_rate,
                self.thresholds["error_rate_max"]
            )
    
    def end_quantum_session(self, session_id: str) -> Dict[str, Any]:
        """End quantum session and return summary"""
        if session_id not in self.quantum_sessions:
            return {}
        
        session = self.quantum_sessions[session_id]
        end_time = time.time()
        duration = end_time - session["start_time"]
        
        # Calculate summary statistics
        quantum_states = session["quantum_states"]
        convergence_history = session["convergence_history"]
        
        summary = {
            "session_id": session_id,
            "duration": duration,
            "total_steps": session["optimization_steps"],
            "total_errors": len(session["errors"]),
            "error_rate": len(session["errors"]) / max(session["optimization_steps"], 1),
            "quantum_metrics": {
                "final_coherence": quantum_states[-1]["coherence"] if quantum_states else 0,
                "avg_entropy": np.mean([s["entropy"] for s in quantum_states]) if quantum_states else 0,
                "coherence_stability": np.std([s["coherence"] for s in quantum_states]) if quantum_states else 0
            },
            "optimization_metrics": {
                "final_objective": convergence_history[-1]["objective"] if convergence_history else 0,
                "avg_convergence_rate": np.mean([h["convergence_rate"] for h in convergence_history]) if convergence_history else 0,
                "total_optimization_time": sum(h["duration"] for h in convergence_history)
            }
        }
        
        # Record session summary
        self.metrics.record_timer("quantum_session_duration", duration, {"session": session_id})
        self.metrics.record_histogram("session_steps", session["optimization_steps"], {"session": session_id})
        self.metrics.record_histogram("session_error_rate", summary["error_rate"], {"session": session_id})
        
        # Check session-level alerts
        if duration > self.thresholds["optimization_time_max"]:
            self._trigger_alert(
                "long_optimization_time",
                AlertLevel.WARNING,
                f"Long optimization time: {duration:.1f}s",
                "optimization_time",
                duration,
                self.thresholds["optimization_time_max"]
            )
        
        # Clean up session
        del self.quantum_sessions[session_id]
        self.metrics.record_counter("quantum_sessions_completed")
        
        logger.info(f"Completed quantum session {session_id} in {duration:.2f}s with {session['optimization_steps']} steps")
        
        return summary
    
    def _trigger_alert(self, alert_name: str, level: AlertLevel, message: str,
                      metric_name: str, current_value: float, threshold: float):
        """Trigger performance alert"""
        alert = Alert(
            name=alert_name,
            level=level,
            message=message,
            timestamp=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Record alert metric
        self.metrics.record_counter("alerts_triggered", 1.0, {
            "alert_name": alert_name,
            "level": level.value
        })
        
        logger.warning(f"Alert triggered: {alert_name} - {message}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        quantum_metrics = {}
        system_metrics = {}
        task_metrics = {}
        
        # Aggregate quantum metrics
        for session_id, session in self.quantum_sessions.items():
            if session["quantum_states"]:
                latest = session["quantum_states"][-1]
                quantum_metrics[f"{session_id}_coherence"] = latest["coherence"]
                quantum_metrics[f"{session_id}_entropy"] = latest["entropy"]
                quantum_metrics[f"{session_id}_purity"] = latest["purity"]
        
        # System metrics
        system_metrics["active_sessions"] = len(self.quantum_sessions)
        system_metrics["total_metrics_collected"] = self.metrics.total_metrics_collected
        system_metrics["uptime"] = time.time() - self.metrics.collection_start_time
        
        # Task metrics from metrics collector
        task_metrics["optimization_steps_total"] = self.metrics.get_counter_value("optimization_steps")
        task_metrics["sessions_completed"] = self.metrics.get_counter_value("quantum_sessions_completed")
        task_metrics["errors_total"] = self.metrics.get_counter_value("quantum_errors")
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            quantum_metrics=quantum_metrics,
            system_metrics=system_metrics,
            task_metrics=task_metrics
        )


class TimingContextManager:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics = metrics_collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics.record_timer(self.metric_name, duration, self.labels)


def create_default_monitor() -> QuantumPerformanceMonitor:
    """Create default quantum performance monitor"""
    metrics_collector = MetricsCollector()
    monitor = QuantumPerformanceMonitor(metrics_collector)
    
    # Add default alert handler
    def log_alert(alert: Alert):
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        logger.log(level_map.get(alert.level, logging.INFO), 
                  f"Performance Alert [{alert.level.value}]: {alert.message}")
    
    monitor.add_alert_handler(log_alert)
    
    return monitor