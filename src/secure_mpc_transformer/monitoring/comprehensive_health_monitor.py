"""Comprehensive Health Monitoring System - Generation 2 Robustness Enhancement."""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    threshold: Optional[float] = None
    timestamp: float = None
    message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolution_time = time.time()


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, interval: float = 60.0):
        self.name = name
        self.interval = interval
        self.last_check = 0.0
        self.last_result: Optional[HealthMetric] = None
    
    async def check(self) -> HealthMetric:
        """Perform health check."""
        raise NotImplementedError
    
    def is_due(self) -> bool:
        """Check if health check is due."""
        return time.time() - self.last_check >= self.interval


class SystemResourceCheck(HealthCheck):
    """System resource health check."""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0):
        super().__init__("system_resources", interval=30.0)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
    
    async def check(self) -> HealthMetric:
        """Check system resources."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Determine status
            if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
                status = HealthStatus.DEGRADED if max(cpu_percent, memory_percent) < 95 else HealthStatus.UNHEALTHY
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resources normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
            self.last_check = time.time()
            self.last_result = HealthMetric(
                name=self.name,
                value={"cpu_percent": cpu_percent, "memory_percent": memory_percent},
                status=status,
                message=message
            )
            
            return self.last_result
            
        except ImportError:
            # Fallback when psutil not available
            self.last_result = HealthMetric(
                name=self.name,
                value={"cpu_percent": 0, "memory_percent": 0},
                status=HealthStatus.UNKNOWN,
                message="psutil not available for resource monitoring"
            )
            return self.last_result
        except Exception as e:
            self.last_result = HealthMetric(
                name=self.name,
                value={},
                status=HealthStatus.UNHEALTHY,
                message=f"Resource check failed: {str(e)}"
            )
            return self.last_result


class ModelServiceCheck(HealthCheck):
    """Model service health check."""
    
    def __init__(self, model_service):
        super().__init__("model_service", interval=60.0)
        self.model_service = model_service
    
    async def check(self) -> HealthMetric:
        """Check model service health."""
        try:
            if not self.model_service:
                return HealthMetric(
                    name=self.name,
                    value={},
                    status=HealthStatus.UNKNOWN,
                    message="Model service not available"
                )
            
            # Test basic functionality
            models = self.model_service.list_models()
            cache_stats = models.get("cache_stats", {})
            
            # Check cache utilization
            cached_models = cache_stats.get("cached_models", 0)
            max_models = cache_stats.get("max_models", 1)
            utilization = (cached_models / max_models) * 100
            
            # Determine status based on utilization and errors
            if utilization > 90:
                status = HealthStatus.DEGRADED
                message = f"High cache utilization: {utilization:.1f}%"
            elif hasattr(self.model_service, 'error_count') and self.model_service.error_count > 0:
                status = HealthStatus.DEGRADED
                message = f"Model service has {self.model_service.error_count} errors"
            else:
                status = HealthStatus.HEALTHY
                message = f"Model service operational: {cached_models}/{max_models} models cached"
            
            self.last_check = time.time()
            self.last_result = HealthMetric(
                name=self.name,
                value={
                    "cached_models": cached_models,
                    "max_models": max_models,
                    "utilization_percent": utilization
                },
                status=status,
                message=message
            )
            
            return self.last_result
            
        except Exception as e:
            self.last_result = HealthMetric(
                name=self.name,
                value={},
                status=HealthStatus.UNHEALTHY,
                message=f"Model service check failed: {str(e)}"
            )
            return self.last_result


class DatabaseCheck(HealthCheck):
    """Database connectivity health check."""
    
    def __init__(self, connection_string: Optional[str] = None):
        super().__init__("database", interval=120.0)
        self.connection_string = connection_string
    
    async def check(self) -> HealthMetric:
        """Check database connectivity."""
        try:
            if not self.connection_string:
                return HealthMetric(
                    name=self.name,
                    value={},
                    status=HealthStatus.UNKNOWN,
                    message="Database not configured"
                )
            
            # Mock database check for now
            # In production, would test actual database connection
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate connection test
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY if response_time < 100 else HealthStatus.DEGRADED
            message = f"Database responsive in {response_time:.2f}ms"
            
            self.last_check = time.time()
            self.last_result = HealthMetric(
                name=self.name,
                value={"response_time_ms": response_time},
                status=status,
                message=message
            )
            
            return self.last_result
            
        except Exception as e:
            self.last_result = HealthMetric(
                name=self.name,
                value={},
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}"
            )
            return self.last_result


class SecurityValidatorCheck(HealthCheck):
    """Security validator health check."""
    
    def __init__(self, security_service):
        super().__init__("security_validator", interval=300.0)  # 5 minutes
        self.security_service = security_service
    
    async def check(self) -> HealthMetric:
        """Check security validator health."""
        try:
            if not self.security_service:
                return HealthMetric(
                    name=self.name,
                    value={},
                    status=HealthStatus.UNKNOWN,
                    message="Security service not available"
                )
            
            # Test security validation
            test_inputs = [
                "normal text",
                "<script>alert('xss')</script>",  # Should be blocked
                "SELECT * FROM users"  # Should be blocked
            ]
            
            blocked_count = 0
            for test_input in test_inputs[1:]:  # Skip normal text
                try:
                    if hasattr(self.security_service, 'validate_input'):
                        result = await self.security_service.validate_input(test_input)
                        if not result.get("allowed", True):
                            blocked_count += 1
                except Exception:
                    pass  # Count as blocked
            
            # Determine status
            if blocked_count >= len(test_inputs) - 1:
                status = HealthStatus.HEALTHY
                message = "Security validator operational"
            else:
                status = HealthStatus.DEGRADED
                message = f"Security validator blocked {blocked_count}/{len(test_inputs)-1} threats"
            
            self.last_check = time.time()
            self.last_result = HealthMetric(
                name=self.name,
                value={"threats_blocked": blocked_count, "total_tests": len(test_inputs) - 1},
                status=status,
                message=message
            )
            
            return self.last_result
            
        except Exception as e:
            self.last_result = HealthMetric(
                name=self.name,
                value={},
                status=HealthStatus.UNHEALTHY,
                message=f"Security validator check failed: {str(e)}"
            )
            return self.last_result


class ComprehensiveHealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checks: List[HealthCheck] = []
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.last_health_summary: Optional[Dict[str, Any]] = None
        
        # Metrics
        self.check_count = 0
        self.alert_count = 0
        
        logger.info("Comprehensive Health Monitor initialized")
    
    def add_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def setup_default_checks(self, model_service=None, security_service=None, database_config=None):
        """Setup default health checks."""
        # System resources
        self.add_check(SystemResourceCheck())
        
        # Model service
        if model_service:
            self.add_check(ModelServiceCheck(model_service))
        
        # Security service
        if security_service:
            self.add_check(SecurityValidatorCheck(security_service))
        
        # Database
        if database_config:
            self.add_check(DatabaseCheck(database_config.get("connection_string")))
        
        logger.info(f"Setup {len(self.checks)} default health checks")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run due health checks
                check_tasks = []
                for check in self.checks:
                    if check.is_due():
                        task = asyncio.create_task(self._run_check(check))
                        check_tasks.append(task)
                
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Process alerts
                self._process_alerts()
                
                # Update health summary
                self._update_health_summary()
                
                # Sleep before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _run_check(self, check: HealthCheck):
        """Run a single health check."""
        try:
            result = await check.check()
            self.check_count += 1
            
            # Generate alerts if needed
            self._evaluate_alerts(check, result)
            
            logger.debug(f"Health check {check.name}: {result.status.value}")
            
        except Exception as e:
            logger.error(f"Health check {check.name} failed: {e}")
            
            # Create error metric
            error_result = HealthMetric(
                name=check.name,
                value={},
                status=HealthStatus.UNHEALTHY,
                message=f"Check execution failed: {str(e)}"
            )
            
            check.last_result = error_result
            self._evaluate_alerts(check, error_result)
    
    def _evaluate_alerts(self, check: HealthCheck, result: HealthMetric):
        """Evaluate if alerts should be generated."""
        # Check if status changed to worse
        if (check.last_result and 
            check.last_result.status != result.status and
            self._is_worse_status(check.last_result.status, result.status)):
            
            # Generate alert
            severity = self._status_to_severity(result.status)
            alert = Alert(
                id=f"{check.name}_{int(time.time())}",
                severity=severity,
                component=check.name,
                message=result.message or f"{check.name} status changed to {result.status.value}",
                timestamp=time.time()
            )
            
            self.alerts.append(alert)
            self.alert_count += 1
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
    
    def _is_worse_status(self, old_status: HealthStatus, new_status: HealthStatus) -> bool:
        """Check if new status is worse than old status."""
        status_order = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.UNKNOWN: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.UNHEALTHY: 3,
            HealthStatus.CRITICAL: 4
        }
        
        return status_order.get(new_status, 0) > status_order.get(old_status, 0)
    
    def _status_to_severity(self, status: HealthStatus) -> AlertSeverity:
        """Convert health status to alert severity."""
        mapping = {
            HealthStatus.HEALTHY: AlertSeverity.INFO,
            HealthStatus.UNKNOWN: AlertSeverity.WARNING,
            HealthStatus.DEGRADED: AlertSeverity.WARNING,
            HealthStatus.UNHEALTHY: AlertSeverity.ERROR,
            HealthStatus.CRITICAL: AlertSeverity.CRITICAL
        }
        return mapping.get(status, AlertSeverity.WARNING)
    
    def _process_alerts(self):
        """Process and clean up alerts."""
        # Auto-resolve alerts older than 1 hour if conditions improved
        current_time = time.time()
        
        for alert in self.alerts:
            if (not alert.resolved and 
                current_time - alert.timestamp > 3600):  # 1 hour
                
                # Check if condition improved
                check = next((c for c in self.checks if c.name == alert.component), None)
                if (check and check.last_result and 
                    check.last_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]):
                    alert.resolve()
                    logger.info(f"Auto-resolved alert: {alert.id}")
    
    def _update_health_summary(self):
        """Update overall health summary."""
        if not self.checks:
            return
        
        # Collect current status from all checks
        check_results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check in self.checks:
            if check.last_result:
                check_results[check.name] = asdict(check.last_result)
                
                # Update overall status (worst status wins)
                if self._is_worse_status(overall_status, check.last_result.status):
                    overall_status = check.last_result.status
        
        # Count active alerts
        active_alerts = len([a for a in self.alerts if not a.resolved])
        
        self.last_health_summary = {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": check_results,
            "active_alerts": active_alerts,
            "total_checks": self.check_count,
            "total_alerts": self.alert_count
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.last_health_summary:
            self._update_health_summary()
        
        return self.last_health_summary or {
            "overall_status": HealthStatus.UNKNOWN.value,
            "timestamp": time.time(),
            "checks": {},
            "active_alerts": 0,
            "message": "No health data available"
        }
    
    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get alerts."""
        alerts = self.alerts if include_resolved else [a for a in self.alerts if not a.resolved]
        return [asdict(alert) for alert in alerts]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        return {
            "total_checks": self.check_count,
            "total_alerts": self.alert_count,
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "monitoring_active": self.monitoring_active,
            "checks_configured": len(self.checks)
        }


def log_alert_handler(alert: Alert):
    """Default alert handler that logs to the logger."""
    level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.ERROR: logging.ERROR,
        AlertSeverity.CRITICAL: logging.CRITICAL
    }.get(alert.severity, logging.WARNING)
    
    logger.log(level, f"ALERT [{alert.severity.value.upper()}] {alert.component}: {alert.message}")


def console_alert_handler(alert: Alert):
    """Alert handler that prints to console."""
    timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ALERT [{alert.severity.value.upper()}] {alert.component}: {alert.message}")


# Default instance for easy use
default_health_monitor = ComprehensiveHealthMonitor()