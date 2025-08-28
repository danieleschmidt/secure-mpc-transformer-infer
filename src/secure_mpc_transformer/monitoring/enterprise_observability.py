"""
Enterprise Observability Platform

Comprehensive monitoring, metrics collection, alerting, and observability
for the secure MPC transformer system with real-time dashboards.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict, deque
import threading
import psutil
from datetime import datetime, timedelta

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        CollectorRegistry, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes if prometheus_client not available
    class Counter: pass
    class Gauge: pass  
    class Histogram: pass
    class Summary: pass

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Defines a metric."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at
        }


class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
    def _initialize_core_metrics(self):
        """Initialize core system metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['memory_usage'] = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.metrics['disk_usage'] = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Application metrics
        self.metrics['request_count'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['request_duration'] = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.metrics['inference_count'] = Counter(
            'inference_requests_total',
            'Total inference requests',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.metrics['inference_duration'] = Histogram(
            'inference_duration_seconds',
            'Inference duration in seconds',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # MPC-specific metrics
        self.metrics['mpc_protocol_operations'] = Counter(
            'mpc_protocol_operations_total',
            'Total MPC protocol operations',
            ['protocol', 'operation', 'status'],
            registry=self.registry
        )
        
        self.metrics['quantum_coherence'] = Gauge(
            'quantum_coherence_score',
            'Current quantum coherence score',
            registry=self.registry
        )
        
        self.metrics['security_threats'] = Counter(
            'security_threats_total',
            'Total security threats detected',
            ['threat_type', 'severity'],
            registry=self.registry
        )
        
        # Cache metrics
        self.metrics['cache_hits'] = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['cache_misses'] = Counter(
            'cache_misses_total', 
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
    def record_system_metrics(self):
        """Record system-level metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].set(disk.percent)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            self.metrics['request_count'].labels(
                method=method,
                endpoint=endpoint, 
                status=str(status)
            ).inc()
            
            self.metrics['request_duration'].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
    
    def record_inference(self, model: str, duration: float, success: bool):
        """Record inference metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            status = 'success' if success else 'error'
            
            self.metrics['inference_count'].labels(
                model=model,
                status=status
            ).inc()
            
            if success:
                self.metrics['inference_duration'].labels(
                    model=model
                ).observe(duration)
                
        except Exception as e:
            logger.error(f"Failed to record inference metrics: {e}")
    
    def record_mpc_operation(self, protocol: str, operation: str, success: bool):
        """Record MPC protocol operation."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            status = 'success' if success else 'error'
            
            self.metrics['mpc_protocol_operations'].labels(
                protocol=protocol,
                operation=operation,
                status=status
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record MPC operation: {e}")
    
    def set_quantum_coherence(self, coherence: float):
        """Set current quantum coherence score."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            self.metrics['quantum_coherence'].set(coherence)
        except Exception as e:
            logger.error(f"Failed to set quantum coherence: {e}")
    
    def record_security_threat(self, threat_type: str, severity: str):
        """Record security threat detection."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            self.metrics['security_threats'].labels(
                threat_type=threat_type,
                severity=severity
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record security threat: {e}")
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            self.metrics['cache_hits'].labels(cache_type=cache_type).inc()
        except Exception as e:
            logger.error(f"Failed to record cache hit: {e}")
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            self.metrics['cache_misses'].labels(cache_type=cache_type).inc()
        except Exception as e:
            logger.error(f"Failed to record cache miss: {e}")
    
    def get_metrics_data(self) -> str:
        """Get Prometheus metrics data."""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return ""
            
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics data: {e}")
            return ""
    
    def define_custom_metric(self, definition: MetricDefinition) -> bool:
        """Define a custom metric."""
        if not PROMETHEUS_AVAILABLE:
            return False
            
        try:
            with self._lock:
                if definition.name in self.custom_metrics:
                    return False
                
                if definition.type == MetricType.COUNTER:
                    metric = Counter(
                        definition.name,
                        definition.description,
                        definition.labels,
                        registry=self.registry
                    )
                elif definition.type == MetricType.GAUGE:
                    metric = Gauge(
                        definition.name,
                        definition.description,
                        definition.labels,
                        registry=self.registry
                    )
                elif definition.type == MetricType.HISTOGRAM:
                    metric = Histogram(
                        definition.name,
                        definition.description,
                        definition.labels,
                        buckets=definition.buckets,
                        registry=self.registry
                    )
                elif definition.type == MetricType.SUMMARY:
                    metric = Summary(
                        definition.name,
                        definition.description,
                        definition.labels,
                        registry=self.registry
                    )
                else:
                    return False
                
                self.custom_metrics[definition.name] = metric
                return True
                
        except Exception as e:
            logger.error(f"Failed to define custom metric: {e}")
            return False


class AlertManager:
    """Alert management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notifications_enabled = self.config.get('notifications_enabled', True)
        self._lock = threading.Lock()
        
        # Alert history
        self.alert_history: deque = deque(maxlen=10000)
        
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        message_template: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Add an alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'labels': labels or {},
            'enabled': True
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def evaluate_alerts(self, metrics: Dict[str, Any]):
        """Evaluate alert rules against current metrics."""
        for rule in self.alert_rules:
            if not rule['enabled']:
                continue
                
            try:
                if rule['condition'](metrics):
                    self._fire_alert(rule, metrics)
                else:
                    self._resolve_alert(rule['name'])
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    def _fire_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Fire an alert."""
        alert_id = f"{rule['name']}_{int(time.time())}"
        
        with self._lock:
            # Check if alert is already active
            active_alert = None
            for alert in self.alerts.values():
                if alert.name == rule['name'] and not alert.resolved:
                    active_alert = alert
                    break
            
            if active_alert:
                return  # Alert already active
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                name=rule['name'],
                severity=rule['severity'],
                message=rule['message_template'].format(**metrics),
                timestamp=time.time(),
                labels=rule['labels']
            )
            
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            logger.warning(f"Alert fired: {rule['name']} - {alert.message}")
            
            if self.notifications_enabled:
                self._send_notification(alert)
    
    def _resolve_alert(self, alert_name: str):
        """Resolve an active alert."""
        with self._lock:
            for alert in self.alerts.values():
                if alert.name == alert_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    logger.info(f"Alert resolved: {alert_name}")
                    
                    if self.notifications_enabled:
                        self._send_notification(alert, resolved=True)
                    break
    
    def _send_notification(self, alert: Alert, resolved: bool = False):
        """Send alert notification."""
        # This would integrate with notification systems like Slack, email, etc.
        action = "RESOLVED" if resolved else "FIRED"
        logger.info(f"NOTIFICATION [{action}]: {alert.name} - {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_time
        ]


class PerformanceTracker:
    """Tracks performance metrics and trends."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.performance_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._lock = threading.Lock()
        
    def record_performance(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Record performance data for an operation."""
        with self._lock:
            record = {
                'timestamp': time.time(),
                'duration': duration,
                'metadata': metadata or {}
            }
            self.performance_data[operation].append(record)
    
    def get_performance_stats(self, operation: str) -> Dict[str, Any]:
        """Get performance statistics for an operation."""
        with self._lock:
            data = list(self.performance_data[operation])
            
            if not data:
                return {'operation': operation, 'samples': 0}
            
            durations = [record['duration'] for record in data]
            
            return {
                'operation': operation,
                'samples': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'p95_duration': self._percentile(durations, 0.95),
                'p99_duration': self._percentile(durations, 0.99),
                'recent_trend': self._calculate_trend(durations[-50:] if len(durations) >= 50 else durations)
            }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate performance trend."""
        if len(data) < 10:
            return 'insufficient_data'
        
        # Simple trend calculation
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change = (second_avg - first_avg) / first_avg
        
        if change > 0.1:
            return 'degrading'
        elif change < -0.1:
            return 'improving'
        else:
            return 'stable'


class EnterpriseObservability:
    """Main observability orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(config.get('alerting', {}))
        self.performance_tracker = PerformanceTracker()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        logger.info("Enterprise Observability initialized")
    
    async def start(self):
        """Start the observability system."""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Enterprise Observability started")
    
    async def stop(self):
        """Stop the observability system."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Enterprise Observability stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                self.metrics_collector.record_system_metrics()
                
                # Get current metrics for alert evaluation
                # This is a simplified version - in practice you'd get actual values
                current_metrics = {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'timestamp': time.time()
                }
                
                # Evaluate alerts
                self.alert_manager.evaluate_alerts(current_metrics)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        
        # High CPU usage alert
        self.alert_manager.add_alert_rule(
            name="high_cpu_usage",
            condition=lambda metrics: metrics.get('cpu_usage', 0) > 80,
            severity=AlertSeverity.WARNING,
            message_template="High CPU usage detected: {cpu_usage:.1f}%",
            labels={'component': 'system'}
        )
        
        # High memory usage alert
        self.alert_manager.add_alert_rule(
            name="high_memory_usage", 
            condition=lambda metrics: metrics.get('memory_usage', 0) > 85,
            severity=AlertSeverity.WARNING,
            message_template="High memory usage detected: {memory_usage:.1f}%",
            labels={'component': 'system'}
        )
        
        # Critical memory usage alert
        self.alert_manager.add_alert_rule(
            name="critical_memory_usage",
            condition=lambda metrics: metrics.get('memory_usage', 0) > 95,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical memory usage detected: {memory_usage:.1f}%",
            labels={'component': 'system'}
        )
        
        # High disk usage alert
        self.alert_manager.add_alert_rule(
            name="high_disk_usage",
            condition=lambda metrics: metrics.get('disk_usage', 0) > 85,
            severity=AlertSeverity.WARNING,
            message_template="High disk usage detected: {disk_usage:.1f}%",
            labels={'component': 'system'}
        )
    
    def get_observability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive observability dashboard data."""
        
        # System metrics
        try:
            system_metrics = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except:
            system_metrics = {}
        
        # Alert summary
        active_alerts = self.alert_manager.get_active_alerts()
        alert_summary = {
            'total_active': len(active_alerts),
            'by_severity': {}
        }
        
        for severity in AlertSeverity:
            alert_summary['by_severity'][severity.value] = len([
                alert for alert in active_alerts if alert.severity == severity
            ])
        
        # Performance summary
        performance_summary = {}
        common_operations = ['inference', 'mpc_operation', 'http_request']
        for operation in common_operations:
            performance_summary[operation] = self.performance_tracker.get_performance_stats(operation)
        
        return {
            'timestamp': time.time(),
            'system_metrics': system_metrics,
            'alert_summary': alert_summary,
            'active_alerts': [alert.to_dict() for alert in active_alerts[:10]],  # Latest 10
            'performance_summary': performance_summary,
            'observability_status': {
                'metrics_collector_active': True,
                'alert_manager_active': True,
                'performance_tracker_active': True,
                'monitoring_loop_running': self._running
            }
        }


# Global instance
_observability: Optional[EnterpriseObservability] = None


def get_observability() -> EnterpriseObservability:
    """Get the global observability instance."""
    global _observability
    if _observability is None:
        _observability = EnterpriseObservability()
    return _observability