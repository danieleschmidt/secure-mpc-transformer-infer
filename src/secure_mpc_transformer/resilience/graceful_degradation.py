"""Graceful degradation system for maintaining service availability."""

import asyncio
import functools
import inspect
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ServiceLevel(Enum):
    """Service availability levels."""
    FULL = "full"                    # All features available
    DEGRADED = "degraded"           # Reduced functionality
    MINIMAL = "minimal"             # Essential features only
    EMERGENCY = "emergency"         # Critical operations only
    MAINTENANCE = "maintenance"      # Service temporarily unavailable


class DegradationTrigger(Enum):
    """Triggers for service degradation."""
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    SECURITY_INCIDENT = "security_incident"
    MANUAL = "manual"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"


@dataclass
class ServiceComponent:
    """Represents a service component that can be degraded."""

    name: str
    description: str
    priority: int  # Lower number = higher priority
    dependencies: list[str] = field(default_factory=list)
    fallback_handler: Callable | None = None
    required_for_minimal: bool = False
    resource_requirements: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'description': self.description,
            'priority': self.priority,
            'dependencies': self.dependencies,
            'has_fallback': self.fallback_handler is not None,
            'required_for_minimal': self.required_for_minimal,
            'resource_requirements': self.resource_requirements
        }


@dataclass
class DegradationEvent:
    """Represents a degradation event."""

    event_id: str
    trigger: DegradationTrigger
    timestamp: float
    from_level: ServiceLevel
    to_level: ServiceLevel
    affected_components: list[str]
    reason: str
    auto_recovery: bool = True
    recovery_condition: Callable[[], bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'event_id': self.event_id,
            'trigger': self.trigger.value,
            'timestamp': self.timestamp,
            'from_level': self.from_level.value,
            'to_level': self.to_level.value,
            'affected_components': self.affected_components,
            'reason': self.reason,
            'auto_recovery': self.auto_recovery
        }


class ServiceHealth:
    """Tracks service health metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.error_rates = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.resource_usage = {}
        self._lock = threading.Lock()

        # Thresholds
        self.error_rate_threshold = 0.1  # 10%
        self.latency_threshold = 5.0     # 5 seconds
        self.resource_thresholds = {
            'cpu': 0.8,    # 80%
            'memory': 0.9,  # 90%
            'gpu': 0.9      # 90%
        }

    def record_request(self, success: bool, latency: float):
        """Record a request outcome."""
        with self._lock:
            self.error_rates.append(0 if success else 1)
            self.latencies.append(latency)

    def update_resource_usage(self, resource: str, usage: float):
        """Update resource usage."""
        with self._lock:
            self.resource_usage[resource] = usage

    def get_health_score(self) -> float:
        """Get overall health score (0.0 = unhealthy, 1.0 = healthy)."""
        with self._lock:
            if not self.error_rates:
                return 1.0

            # Error rate component (0-1)
            error_rate = sum(self.error_rates) / len(self.error_rates)
            error_score = max(0, 1 - (error_rate / self.error_rate_threshold))

            # Latency component (0-1)
            if self.latencies:
                avg_latency = sum(self.latencies) / len(self.latencies)
                latency_score = max(0, 1 - (avg_latency / self.latency_threshold))
            else:
                latency_score = 1.0

            # Resource component (0-1)
            resource_scores = []
            for resource, usage in self.resource_usage.items():
                threshold = self.resource_thresholds.get(resource, 0.9)
                resource_scores.append(max(0, 1 - (usage / threshold)))

            resource_score = min(resource_scores) if resource_scores else 1.0

            # Combined score (weighted)
            return (error_score * 0.4 + latency_score * 0.3 + resource_score * 0.3)

    def get_metrics(self) -> dict[str, Any]:
        """Get current health metrics."""
        with self._lock:
            error_rate = sum(self.error_rates) / len(self.error_rates) if self.error_rates else 0
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0

            return {
                'error_rate': error_rate,
                'average_latency': avg_latency,
                'resource_usage': dict(self.resource_usage),
                'health_score': self.get_health_score(),
                'sample_count': len(self.error_rates)
            }


class FallbackManager:
    """Manages fallback implementations for degraded services."""

    def __init__(self):
        self.fallbacks: dict[str, Callable] = {}
        self.fallback_stats = defaultdict(lambda: {
            'calls': 0,
            'successes': 0,
            'failures': 0,
            'total_duration': 0.0
        })
        self._lock = threading.Lock()

    def register_fallback(self, component_name: str, fallback_func: Callable):
        """Register a fallback function for a component."""
        self.fallbacks[component_name] = fallback_func
        logger.info(f"Registered fallback for component: {component_name}")

    def execute_fallback(self, component_name: str, *args, **kwargs):
        """Execute fallback for a component."""
        if component_name not in self.fallbacks:
            raise ValueError(f"No fallback registered for component: {component_name}")

        fallback_func = self.fallbacks[component_name]
        start_time = time.time()

        try:
            if inspect.iscoroutinefunction(fallback_func):
                return asyncio.create_task(fallback_func(*args, **kwargs))
            else:
                result = fallback_func(*args, **kwargs)

            # Record success
            duration = time.time() - start_time
            with self._lock:
                stats = self.fallback_stats[component_name]
                stats['calls'] += 1
                stats['successes'] += 1
                stats['total_duration'] += duration

            logger.info(f"Fallback executed successfully for {component_name}")
            return result

        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            with self._lock:
                stats = self.fallback_stats[component_name]
                stats['calls'] += 1
                stats['failures'] += 1
                stats['total_duration'] += duration

            logger.error(f"Fallback failed for {component_name}: {e}")
            raise

    async def async_execute_fallback(self, component_name: str, *args, **kwargs):
        """Execute async fallback for a component."""
        if component_name not in self.fallbacks:
            raise ValueError(f"No fallback registered for component: {component_name}")

        fallback_func = self.fallbacks[component_name]
        start_time = time.time()

        try:
            if inspect.iscoroutinefunction(fallback_func):
                result = await fallback_func(*args, **kwargs)
            else:
                result = fallback_func(*args, **kwargs)

            # Record success
            duration = time.time() - start_time
            with self._lock:
                stats = self.fallback_stats[component_name]
                stats['calls'] += 1
                stats['successes'] += 1
                stats['total_duration'] += duration

            logger.info(f"Async fallback executed successfully for {component_name}")
            return result

        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            with self._lock:
                stats = self.fallback_stats[component_name]
                stats['calls'] += 1
                stats['failures'] += 1
                stats['total_duration'] += duration

            logger.error(f"Async fallback failed for {component_name}: {e}")
            raise

    def get_fallback_stats(self) -> dict[str, dict[str, Any]]:
        """Get fallback execution statistics."""
        with self._lock:
            stats = {}
            for component, data in self.fallback_stats.items():
                stats[component] = {
                    'calls': data['calls'],
                    'successes': data['successes'],
                    'failures': data['failures'],
                    'success_rate': data['successes'] / data['calls'] if data['calls'] > 0 else 0,
                    'average_duration': data['total_duration'] / data['calls'] if data['calls'] > 0 else 0
                }
            return stats


class GracefulDegradationManager:
    """Main manager for graceful service degradation."""

    def __init__(self):
        self.service_level = ServiceLevel.FULL
        self.components: dict[str, ServiceComponent] = {}
        self.disabled_components: set[str] = set()
        self.degradation_history: list[DegradationEvent] = []

        # Managers
        self.health = ServiceHealth()
        self.fallback_manager = FallbackManager()

        # Configuration
        self.auto_recovery = True
        self.recovery_delay = 60.0  # seconds
        self.health_check_interval = 30.0  # seconds

        # State
        self.current_event: DegradationEvent | None = None
        self._lock = threading.Lock()

        # Health monitoring
        self._health_monitor_task = None
        self._start_health_monitoring()

        logger.info("Graceful degradation manager initialized")

    def register_component(self, component: ServiceComponent):
        """Register a service component."""
        self.components[component.name] = component

        # Register fallback if provided
        if component.fallback_handler:
            self.fallback_manager.register_fallback(
                component.name, component.fallback_handler
            )

        logger.info(f"Registered component: {component.name}")

    def degrade_service(self, to_level: ServiceLevel, trigger: DegradationTrigger,
                       reason: str, affected_components: list[str] = None):
        """Manually degrade service level."""
        with self._lock:
            if to_level == self.service_level:
                return

            from_level = self.service_level
            event_id = f"deg_{int(time.time())}_{hash(reason) % 10000}"

            # Determine affected components
            if affected_components is None:
                affected_components = self._determine_affected_components(to_level)

            # Create degradation event
            event = DegradationEvent(
                event_id=event_id,
                trigger=trigger,
                timestamp=time.time(),
                from_level=from_level,
                to_level=to_level,
                affected_components=affected_components,
                reason=reason,
                auto_recovery=trigger != DegradationTrigger.MANUAL
            )

            # Apply degradation
            self._apply_degradation(event)

            logger.warning(f"Service degraded from {from_level.value} to {to_level.value}: {reason}")

    def _determine_affected_components(self, target_level: ServiceLevel) -> list[str]:
        """Determine which components should be disabled for target level."""
        affected = []

        if target_level == ServiceLevel.MINIMAL:
            # Disable all non-essential components
            for name, component in self.components.items():
                if not component.required_for_minimal:
                    affected.append(name)

        elif target_level == ServiceLevel.DEGRADED:
            # Disable low priority components
            sorted_components = sorted(
                self.components.items(),
                key=lambda x: x[1].priority,
                reverse=True
            )

            # Disable bottom 50% by priority
            half_point = len(sorted_components) // 2
            for name, component in sorted_components[:half_point]:
                if not component.required_for_minimal:
                    affected.append(name)

        elif target_level == ServiceLevel.EMERGENCY:
            # Disable almost everything except critical
            for name, component in self.components.items():
                if component.priority > 1:  # Only keep priority 0 and 1
                    affected.append(name)

        return affected

    def _apply_degradation(self, event: DegradationEvent):
        """Apply degradation based on event."""
        # Update service level
        self.service_level = event.to_level

        # Disable affected components
        for component_name in event.affected_components:
            self.disabled_components.add(component_name)

        # Store current event
        self.current_event = event

        # Add to history
        self.degradation_history.append(event)

        # Trigger recovery check if auto-recovery is enabled
        if event.auto_recovery and self.auto_recovery:
            threading.Timer(
                self.recovery_delay,
                self._check_recovery
            ).start()

    def recover_service(self, reason: str = "Manual recovery"):
        """Manually recover service to full level."""
        with self._lock:
            if self.service_level == ServiceLevel.FULL:
                return

            from_level = self.service_level
            event_id = f"rec_{int(time.time())}_{hash(reason) % 10000}"

            # Create recovery event
            event = DegradationEvent(
                event_id=event_id,
                trigger=DegradationTrigger.MANUAL,
                timestamp=time.time(),
                from_level=from_level,
                to_level=ServiceLevel.FULL,
                affected_components=list(self.disabled_components),
                reason=reason,
                auto_recovery=False
            )

            # Apply recovery
            self._apply_recovery(event)

            logger.info(f"Service recovered from {from_level.value} to full: {reason}")

    def _apply_recovery(self, event: DegradationEvent):
        """Apply service recovery."""
        # Re-enable all components
        self.disabled_components.clear()

        # Update service level
        self.service_level = ServiceLevel.FULL

        # Clear current event
        self.current_event = None

        # Add to history
        self.degradation_history.append(event)

    def _check_recovery(self):
        """Check if service can be recovered automatically."""
        if not self.current_event or not self.current_event.auto_recovery:
            return

        # Check recovery condition if specified
        if self.current_event.recovery_condition:
            try:
                if not self.current_event.recovery_condition():
                    # Schedule next check
                    threading.Timer(
                        self.recovery_delay,
                        self._check_recovery
                    ).start()
                    return
            except Exception as e:
                logger.error(f"Error checking recovery condition: {e}")
                return

        # Check general health
        health_score = self.health.get_health_score()
        if health_score > 0.8:  # Healthy enough to recover
            self.recover_service("Automatic recovery - health improved")
        else:
            # Schedule next check
            threading.Timer(
                self.recovery_delay,
                self._check_recovery
            ).start()

    def is_component_available(self, component_name: str) -> bool:
        """Check if a component is available."""
        return component_name not in self.disabled_components

    def execute_with_fallback(self, component_name: str, primary_func: Callable,
                             *args, **kwargs):
        """Execute function with fallback if component is unavailable."""
        if self.is_component_available(component_name):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Primary function failed for {component_name}: {e}")
                # Fall through to fallback

        # Use fallback
        return self.fallback_manager.execute_fallback(component_name, *args, **kwargs)

    async def async_execute_with_fallback(self, component_name: str, primary_func: Callable,
                                        *args, **kwargs):
        """Execute async function with fallback if component is unavailable."""
        if self.is_component_available(component_name):
            try:
                if inspect.iscoroutinefunction(primary_func):
                    return await primary_func(*args, **kwargs)
                else:
                    return primary_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Primary async function failed for {component_name}: {e}")
                # Fall through to fallback

        # Use fallback
        return await self.fallback_manager.async_execute_fallback(component_name, *args, **kwargs)

    def _start_health_monitoring(self):
        """Start background health monitoring."""
        async def health_monitor():
            while True:
                try:
                    await self._monitor_health()
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(self.health_check_interval)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._health_monitor_task = asyncio.create_task(health_monitor())
        except RuntimeError:
            # Start manual monitoring
            threading.Timer(self.health_check_interval, self._manual_health_check).start()

    async def _monitor_health(self):
        """Monitor service health and trigger degradation if needed."""
        health_score = self.health.get_health_score()
        metrics = self.health.get_metrics()

        # Determine if degradation is needed
        if self.service_level == ServiceLevel.FULL:
            if health_score < 0.3:
                self.degrade_service(
                    ServiceLevel.EMERGENCY,
                    DegradationTrigger.HIGH_ERROR_RATE,
                    f"Critical health score: {health_score:.2f}"
                )
            elif health_score < 0.5:
                self.degrade_service(
                    ServiceLevel.MINIMAL,
                    DegradationTrigger.HIGH_ERROR_RATE,
                    f"Low health score: {health_score:.2f}"
                )
            elif health_score < 0.7:
                self.degrade_service(
                    ServiceLevel.DEGRADED,
                    DegradationTrigger.HIGH_ERROR_RATE,
                    f"Moderate health score: {health_score:.2f}"
                )

        # Check specific thresholds
        if metrics['error_rate'] > 0.2:  # 20% error rate
            if self.service_level not in [ServiceLevel.MINIMAL, ServiceLevel.EMERGENCY]:
                self.degrade_service(
                    ServiceLevel.MINIMAL,
                    DegradationTrigger.HIGH_ERROR_RATE,
                    f"High error rate: {metrics['error_rate']:.1%}"
                )

        if metrics['average_latency'] > 10.0:  # 10 second average latency
            if self.service_level == ServiceLevel.FULL:
                self.degrade_service(
                    ServiceLevel.DEGRADED,
                    DegradationTrigger.HIGH_LATENCY,
                    f"High latency: {metrics['average_latency']:.1f}s"
                )

    def _manual_health_check(self):
        """Manual health check for non-async environments."""
        try:
            asyncio.run(self._monitor_health())
        except Exception as e:
            logger.error(f"Manual health check error: {e}")
        finally:
            threading.Timer(self.health_check_interval, self._manual_health_check).start()

    def get_status(self) -> dict[str, Any]:
        """Get current degradation status."""
        with self._lock:
            return {
                'service_level': self.service_level.value,
                'disabled_components': list(self.disabled_components),
                'component_count': len(self.components),
                'current_event': self.current_event.to_dict() if self.current_event else None,
                'health_metrics': self.health.get_metrics(),
                'fallback_stats': self.fallback_manager.get_fallback_stats(),
                'auto_recovery_enabled': self.auto_recovery,
                'degradation_events': len(self.degradation_history)
            }

    def get_component_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all components."""
        status = {}
        for name, component in self.components.items():
            status[name] = {
                **component.to_dict(),
                'available': self.is_component_available(name),
                'disabled': name in self.disabled_components
            }
        return status


# Global degradation manager
degradation_manager = GracefulDegradationManager()


def with_fallback(component_name: str, fallback_func: Callable | None = None):
    """Decorator for functions with fallback support."""
    def decorator(func):
        # Register fallback if provided
        if fallback_func:
            degradation_manager.fallback_manager.register_fallback(component_name, fallback_func)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await degradation_manager.async_execute_with_fallback(
                    component_name, func, *args, **kwargs
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return degradation_manager.execute_with_fallback(
                    component_name, func, *args, **kwargs
                )
            return sync_wrapper

    return decorator


def component_health_check(component_name: str, success: bool, latency: float):
    """Record health check result for component."""
    degradation_manager.health.record_request(success, latency)
