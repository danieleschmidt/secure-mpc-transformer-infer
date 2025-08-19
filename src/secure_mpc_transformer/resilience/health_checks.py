"""Comprehensive health checking system for service monitoring."""

import asyncio
import inspect
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"      # Service is alive
    READINESS = "readiness"    # Service is ready to serve traffic
    STARTUP = "startup"        # Service has started up correctly
    DEPENDENCY = "dependency"   # External dependency check
    RESOURCE = "resource"      # Resource availability check
    FUNCTIONAL = "functional"  # Functional correctness check


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    check_name: str
    status: HealthStatus
    timestamp: float
    duration: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'message': self.message,
            'details': self.details
        }

    @property
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""

    name: str
    check_type: HealthCheckType
    interval: float = 30.0  # seconds
    timeout: float = 10.0   # seconds
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    critical: bool = False  # If true, failure marks entire service as unhealthy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'check_type': self.check_type.value,
            'interval': self.interval,
            'timeout': self.timeout,
            'failure_threshold': self.failure_threshold,
            'success_threshold': self.success_threshold,
            'enabled': self.enabled,
            'critical': self.critical
        }


class HealthCheck:
    """Base health check implementation."""

    def __init__(self, config: HealthCheckConfig, check_func: Callable):
        self.config = config
        self.check_func = check_func

        # State tracking
        self.last_result: HealthCheckResult | None = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.total_executions = 0
        self.total_failures = 0

        # History
        self.result_history = deque(maxlen=100)

        # Scheduling
        self.scheduled_task: asyncio.Task | None = None
        self.stop_event = threading.Event()

        self._lock = threading.Lock()

    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()

        try:
            # Apply timeout
            if inspect.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(
                    self.check_func(),
                    timeout=self.config.timeout
                )
            else:
                result = await asyncio.wait_for(
                    self._run_sync_check(),
                    timeout=self.config.timeout
                )

            # Process result
            if isinstance(result, HealthCheckResult):
                health_result = result
            elif isinstance(result, dict):
                health_result = HealthCheckResult(
                    check_name=self.config.name,
                    status=HealthStatus(result.get('status', 'healthy')),
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    message=result.get('message', 'Check passed'),
                    details=result.get('details', {})
                )
            elif isinstance(result, bool):
                health_result = HealthCheckResult(
                    check_name=self.config.name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    message="Check passed" if result else "Check failed"
                )
            else:
                health_result = HealthCheckResult(
                    check_name=self.config.name,
                    status=HealthStatus.HEALTHY,
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    message=str(result) if result else "Check passed"
                )

        except asyncio.TimeoutError:
            health_result = HealthCheckResult(
                check_name=self.config.name,
                status=HealthStatus.CRITICAL,
                timestamp=start_time,
                duration=self.config.timeout,
                message=f"Health check timed out after {self.config.timeout}s"
            )

        except Exception as e:
            health_result = HealthCheckResult(
                check_name=self.config.name,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                duration=time.time() - start_time,
                message=f"Health check failed: {str(e)}",
                details={'exception_type': type(e).__name__}
            )

        # Update state
        self._update_state(health_result)

        return health_result

    async def _run_sync_check(self):
        """Run synchronous check function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_func)

    def _update_state(self, result: HealthCheckResult):
        """Update health check state based on result."""
        with self._lock:
            self.last_result = result
            self.total_executions += 1
            self.result_history.append(result)

            if result.is_healthy:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0
                self.total_failures += 1

    def get_current_status(self) -> HealthStatus:
        """Get current health status based on thresholds."""
        with self._lock:
            if not self.last_result:
                return HealthStatus.UNKNOWN

            # Apply failure threshold
            if self.consecutive_failures >= self.config.failure_threshold:
                return HealthStatus.UNHEALTHY

            # Apply success threshold (for recovery)
            if (self.consecutive_failures > 0 and
                self.consecutive_successes < self.config.success_threshold):
                return HealthStatus.DEGRADED

            return self.last_result.status

    def get_statistics(self) -> dict[str, Any]:
        """Get health check statistics."""
        with self._lock:
            if self.total_executions == 0:
                success_rate = 0.0
            else:
                success_rate = (self.total_executions - self.total_failures) / self.total_executions

            # Calculate average duration
            if self.result_history:
                avg_duration = sum(r.duration for r in self.result_history) / len(self.result_history)
            else:
                avg_duration = 0.0

            return {
                'config': self.config.to_dict(),
                'current_status': self.get_current_status().value,
                'last_execution': self.last_result.timestamp if self.last_result else None,
                'consecutive_failures': self.consecutive_failures,
                'consecutive_successes': self.consecutive_successes,
                'total_executions': self.total_executions,
                'total_failures': self.total_failures,
                'success_rate': success_rate,
                'average_duration': avg_duration
            }

    def start_scheduled_execution(self):
        """Start scheduled execution of health check."""
        if self.scheduled_task or not self.config.enabled:
            return

        async def scheduled_run():
            while not self.stop_event.is_set():
                try:
                    await self.execute()
                    await asyncio.sleep(self.config.interval)
                except Exception as e:
                    logger.error(f"Scheduled health check {self.config.name} failed: {e}")
                    await asyncio.sleep(self.config.interval)

        try:
            loop = asyncio.get_event_loop()
            self.scheduled_task = asyncio.create_task(scheduled_run())
        except RuntimeError:
            # No event loop, manual scheduling not implemented
            pass

    def stop_scheduled_execution(self):
        """Stop scheduled execution."""
        self.stop_event.set()
        if self.scheduled_task:
            self.scheduled_task.cancel()
            self.scheduled_task = None


class DatabaseHealthCheck:
    """Health check for database connectivity."""

    def __init__(self, connection_string: str, query: str = "SELECT 1"):
        self.connection_string = connection_string
        self.query = query

    async def check(self) -> dict[str, Any]:
        """Perform database health check."""
        try:
            # This would use actual database connection
            # For now, simulate the check
            await asyncio.sleep(0.1)  # Simulate query time

            return {
                'status': 'healthy',
                'message': 'Database connection successful',
                'details': {
                    'query_duration': 0.1,
                    'connection_pool_size': 10
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Database connection failed: {str(e)}',
                'details': {'error_type': type(e).__name__}
            }


class MemoryHealthCheck:
    """Health check for memory usage."""

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0

            if usage_percent >= self.critical_threshold:
                status = 'critical'
                message = f'Critical memory usage: {usage_percent:.1%}'
            elif usage_percent >= self.warning_threshold:
                status = 'degraded'
                message = f'High memory usage: {usage_percent:.1%}'
            else:
                status = 'healthy'
                message = f'Memory usage normal: {usage_percent:.1%}'

            return {
                'status': status,
                'message': message,
                'details': {
                    'usage_percent': usage_percent,
                    'total_mb': memory.total / (1024 * 1024),
                    'available_mb': memory.available / (1024 * 1024),
                    'used_mb': memory.used / (1024 * 1024)
                }
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Failed to check memory usage: {str(e)}'
            }


class DiskHealthCheck:
    """Health check for disk space."""

    def __init__(self, path: str = "/", warning_threshold: float = 0.8,
                 critical_threshold: float = 0.9):
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> dict[str, Any]:
        """Check disk space."""
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.path)
            usage_percent = used / total

            if usage_percent >= self.critical_threshold:
                status = 'critical'
                message = f'Critical disk usage: {usage_percent:.1%}'
            elif usage_percent >= self.warning_threshold:
                status = 'degraded'
                message = f'High disk usage: {usage_percent:.1%}'
            else:
                status = 'healthy'
                message = f'Disk usage normal: {usage_percent:.1%}'

            return {
                'status': status,
                'message': message,
                'details': {
                    'path': self.path,
                    'usage_percent': usage_percent,
                    'total_gb': total / (1024**3),
                    'used_gb': used / (1024**3),
                    'free_gb': free / (1024**3)
                }
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Failed to check disk usage: {str(e)}'
            }


class ServiceHealthCheck:
    """Health check for external service connectivity."""

    def __init__(self, service_url: str, timeout: float = 5.0):
        self.service_url = service_url
        self.timeout = timeout

    async def check(self) -> dict[str, Any]:
        """Check service connectivity."""
        try:
            import aiohttp

            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.service_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    duration = time.time() - start_time

                    if response.status == 200:
                        status = 'healthy'
                        message = f'Service responding normally ({response.status})'
                    else:
                        status = 'degraded'
                        message = f'Service responding with status {response.status}'

                    return {
                        'status': status,
                        'message': message,
                        'details': {
                            'url': self.service_url,
                            'status_code': response.status,
                            'response_time': duration
                        }
                    }

        except asyncio.TimeoutError:
            return {
                'status': 'unhealthy',
                'message': f'Service timeout after {self.timeout}s',
                'details': {'url': self.service_url}
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Service check failed: {str(e)}',
                'details': {'url': self.service_url, 'error_type': type(e).__name__}
            }


class HealthCheckManager:
    """Manager for all health checks."""

    def __init__(self):
        self.health_checks: dict[str, HealthCheck] = {}
        self.last_overall_check = None
        self._lock = threading.Lock()

        # Built-in health checks
        self._register_builtin_checks()

        logger.info("Health check manager initialized")

    def _register_builtin_checks(self):
        """Register built-in system health checks."""

        # Memory check
        memory_check = MemoryHealthCheck()
        self.register_check(
            HealthCheckConfig(
                name="system_memory",
                check_type=HealthCheckType.RESOURCE,
                interval=60.0,
                critical=True
            ),
            memory_check.check
        )

        # Disk check
        disk_check = DiskHealthCheck()
        self.register_check(
            HealthCheckConfig(
                name="system_disk",
                check_type=HealthCheckType.RESOURCE,
                interval=300.0,  # 5 minutes
                critical=True
            ),
            disk_check.check
        )

        # Application liveness check
        self.register_check(
            HealthCheckConfig(
                name="application_liveness",
                check_type=HealthCheckType.LIVENESS,
                interval=30.0,
                critical=True
            ),
            self._application_liveness_check
        )

    async def _application_liveness_check(self) -> dict[str, Any]:
        """Basic application liveness check."""
        try:
            # Simple check that the application is running
            current_time = time.time()

            return {
                'status': 'healthy',
                'message': 'Application is running',
                'details': {
                    'timestamp': current_time,
                    'uptime': current_time - (hasattr(self, '_start_time') and self._start_time or current_time)
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Application liveness check failed: {str(e)}'
            }

    def register_check(self, config: HealthCheckConfig, check_func: Callable):
        """Register a new health check."""
        health_check = HealthCheck(config, check_func)

        with self._lock:
            self.health_checks[config.name] = health_check

        # Start scheduled execution if enabled
        if config.enabled:
            health_check.start_scheduled_execution()

        logger.info(f"Registered health check: {config.name}")

    def unregister_check(self, check_name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if check_name in self.health_checks:
                health_check = self.health_checks[check_name]
                health_check.stop_scheduled_execution()
                del self.health_checks[check_name]
                logger.info(f"Unregistered health check: {check_name}")
                return True
            return False

    async def execute_check(self, check_name: str) -> HealthCheckResult | None:
        """Execute a specific health check."""
        if check_name not in self.health_checks:
            return None

        return await self.health_checks[check_name].execute()

    async def execute_all_checks(self) -> dict[str, HealthCheckResult]:
        """Execute all registered health checks."""
        results = {}

        tasks = []
        check_names = []

        for name, health_check in self.health_checks.items():
            if health_check.config.enabled:
                tasks.append(health_check.execute())
                check_names.append(name)

        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(check_names, check_results, strict=False):
                if isinstance(result, Exception):
                    results[name] = HealthCheckResult(
                        check_name=name,
                        status=HealthStatus.CRITICAL,
                        timestamp=time.time(),
                        duration=0.0,
                        message=f"Check execution failed: {str(result)}"
                    )
                else:
                    results[name] = result

        self.last_overall_check = time.time()
        return results

    def get_overall_status(self) -> dict[str, Any]:
        """Get overall health status."""
        overall_status = HealthStatus.HEALTHY
        critical_failures = []
        total_checks = 0
        healthy_checks = 0

        with self._lock:
            for name, health_check in self.health_checks.items():
                if not health_check.config.enabled:
                    continue

                total_checks += 1
                current_status = health_check.get_current_status()

                if current_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    healthy_checks += 1

                # Check if this is a critical failure
                if (health_check.config.critical and
                    current_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]):
                    critical_failures.append(name)
                    overall_status = HealthStatus.CRITICAL

                # Update overall status based on individual check status
                if overall_status != HealthStatus.CRITICAL:
                    if current_status == HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.UNHEALTHY
                    elif (current_status == HealthStatus.DEGRADED and
                          overall_status == HealthStatus.HEALTHY):
                        overall_status = HealthStatus.DEGRADED

        health_ratio = healthy_checks / total_checks if total_checks > 0 else 1.0

        return {
            'overall_status': overall_status.value,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'health_ratio': health_ratio,
            'critical_failures': critical_failures,
            'last_check_time': self.last_overall_check
        }

    def get_detailed_status(self) -> dict[str, Any]:
        """Get detailed status of all health checks."""
        overall = self.get_overall_status()

        checks = {}
        with self._lock:
            for name, health_check in self.health_checks.items():
                checks[name] = health_check.get_statistics()

        return {
            'overall': overall,
            'checks': checks,
            'timestamp': time.time()
        }

    def get_readiness_status(self) -> dict[str, Any]:
        """Get readiness status (for Kubernetes readiness probe)."""
        readiness_checks = []

        with self._lock:
            for name, health_check in self.health_checks.items():
                if (health_check.config.enabled and
                    health_check.config.check_type in [HealthCheckType.READINESS, HealthCheckType.DEPENDENCY]):
                    current_status = health_check.get_current_status()
                    readiness_checks.append({
                        'name': name,
                        'status': current_status.value,
                        'healthy': current_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
                    })

        all_ready = all(check['healthy'] for check in readiness_checks)

        return {
            'ready': all_ready,
            'checks': readiness_checks,
            'timestamp': time.time()
        }

    def get_liveness_status(self) -> dict[str, Any]:
        """Get liveness status (for Kubernetes liveness probe)."""
        liveness_checks = []

        with self._lock:
            for name, health_check in self.health_checks.items():
                if (health_check.config.enabled and
                    health_check.config.check_type == HealthCheckType.LIVENESS):
                    current_status = health_check.get_current_status()
                    liveness_checks.append({
                        'name': name,
                        'status': current_status.value,
                        'alive': current_status != HealthStatus.CRITICAL
                    })

        all_alive = all(check['alive'] for check in liveness_checks)

        return {
            'alive': all_alive,
            'checks': liveness_checks,
            'timestamp': time.time()
        }

    def shutdown(self):
        """Shutdown all health checks."""
        with self._lock:
            for health_check in self.health_checks.values():
                health_check.stop_scheduled_execution()

        logger.info("Health check manager shutdown")


# Global health check manager
health_manager = HealthCheckManager()


def register_health_check(config: HealthCheckConfig):
    """Decorator for registering health check functions."""
    def decorator(func):
        health_manager.register_check(config, func)
        return func

    return decorator
