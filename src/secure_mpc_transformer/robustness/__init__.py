"""Production robustness framework integration module."""

import logging
from typing import Any, Dict, Optional

from ..hardening.resource_manager import resource_manager
from ..hardening.secrets_manager import secrets_manager
from ..hardening.shutdown_handler import shutdown_manager
from ..monitoring.circuit_breakers import CircuitBreakerConfig, circuit_breaker_registry
from ..monitoring.distributed_tracing import tracer
from ..monitoring.prometheus_metrics import prometheus_exporter
from ..resilience.graceful_degradation import degradation_manager
from ..resilience.health_checks import health_manager
from ..resilience.retry_manager import CommonRetryConfigs, retry_registry
from ..security.ddos_protection import DDosProtectionSystem, ddos_protection
from ..security.key_manager import incident_response, key_manager
from ..security.session_manager import SecureSessionManager

# Import all robustness components
from ..security.threat_detector import AdvancedThreatDetector, threat_detector
from ..validation.schema_validator import CommonSchemas, schema_validator

logger = logging.getLogger(__name__)


class RobustnessFramework:
    """Main robustness framework that integrates all production features."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Initialize all components
        self.threat_detector = threat_detector
        self.ddos_protection = ddos_protection
        self.session_manager = None  # Initialize on demand
        self.key_manager = key_manager
        self.incident_response = incident_response

        self.prometheus_exporter = prometheus_exporter
        self.tracer = tracer
        self.circuit_breakers = circuit_breaker_registry

        self.retry_manager = retry_registry
        self.degradation_manager = degradation_manager
        self.health_manager = health_manager

        self.schema_validator = schema_validator

        self.resource_manager = resource_manager
        self.secrets_manager = secrets_manager
        self.shutdown_manager = shutdown_manager

        # Framework state
        self.initialized = False
        self.framework_stats = {
            'initialization_time': 0.0,
            'components_initialized': 0,
            'components_failed': 0
        }

        logger.info("Robustness framework created")

    async def initialize(self):
        """Initialize all robustness components."""
        if self.initialized:
            return

        import time
        start_time = time.time()

        logger.info("Initializing production robustness framework...")

        try:
            # Initialize session manager if configured
            if self.config.get('enable_session_management', True):
                session_config = self.config.get('session', {})
                self.session_manager = SecureSessionManager(session_config)
                self.framework_stats['components_initialized'] += 1

            # Register default circuit breakers
            self._setup_default_circuit_breakers()

            # Register default retry policies
            self._setup_default_retry_policies()

            # Setup default health checks
            self._setup_default_health_checks()

            # Configure monitoring
            self._configure_monitoring()

            # Configure security components
            self._configure_security()

            # Register shutdown handlers
            self._register_shutdown_handlers()

            self.framework_stats['initialization_time'] = time.time() - start_time
            self.initialized = True

            logger.info(f"Robustness framework initialized successfully in {self.framework_stats['initialization_time']:.2f}s")

        except Exception as e:
            self.framework_stats['components_failed'] += 1
            logger.error(f"Failed to initialize robustness framework: {e}")
            raise

    def _setup_default_circuit_breakers(self):
        """Setup default circuit breaker configurations."""

        # Database operations
        db_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            timeout=10.0
        )
        self.circuit_breakers.register("database", db_config)

        # External API calls
        api_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=5.0
        )
        self.circuit_breakers.register("external_api", api_config)

        # MPC operations
        mpc_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=120.0,
            timeout=300.0
        )
        self.circuit_breakers.register("mpc_computation", mpc_config)

        logger.debug("Default circuit breakers configured")

    def _setup_default_retry_policies(self):
        """Setup default retry policies."""

        # Network operations
        self.retry_manager.register("network", CommonRetryConfigs.network_request())

        # Database operations
        self.retry_manager.register("database", CommonRetryConfigs.database_operation())

        # MPC computations
        self.retry_manager.register("mpc", CommonRetryConfigs.mpc_computation())

        # File operations
        self.retry_manager.register("file", CommonRetryConfigs.file_operation())

        logger.debug("Default retry policies configured")

    def _setup_default_health_checks(self):
        """Setup default health checks."""

        # Register framework health check
        from ..resilience.health_checks import HealthCheckConfig, HealthCheckType

        framework_config = HealthCheckConfig(
            name="robustness_framework",
            check_type=HealthCheckType.LIVENESS,
            interval=60.0,
            timeout=5.0,
            critical=True
        )

        self.health_manager.register_check(framework_config, self._framework_health_check)

        logger.debug("Default health checks configured")

    async def _framework_health_check(self) -> dict[str, Any]:
        """Health check for the robustness framework."""

        healthy_components = 0
        total_components = 0
        issues = []

        # Check threat detector
        total_components += 1
        try:
            threat_summary = self.threat_detector.get_threat_summary()
            if threat_summary.get('total_threats', 0) < 1000:  # Reasonable threshold
                healthy_components += 1
            else:
                issues.append("High threat count detected")
        except Exception as e:
            issues.append(f"Threat detector error: {str(e)}")

        # Check DDoS protection
        total_components += 1
        try:
            ddos_stats = self.ddos_protection.get_protection_stats()
            if ddos_stats.get('protection_level') != 'emergency':
                healthy_components += 1
            else:
                issues.append("DDoS protection in emergency mode")
        except Exception as e:
            issues.append(f"DDoS protection error: {str(e)}")

        # Check resource manager
        total_components += 1
        try:
            resource_stats = self.resource_manager.get_resource_stats()
            if resource_stats['current_usage']['memory_percent'] < 90:
                healthy_components += 1
            else:
                issues.append("High memory usage")
        except Exception as e:
            issues.append(f"Resource manager error: {str(e)}")

        # Check secrets manager
        total_components += 1
        try:
            secrets_health = self.secrets_manager.health_check()
            if secrets_health.get('healthy', False):
                healthy_components += 1
            else:
                issues.append("Secrets manager unhealthy")
        except Exception as e:
            issues.append(f"Secrets manager error: {str(e)}")

        health_ratio = healthy_components / total_components if total_components > 0 else 0

        if health_ratio >= 0.8:
            status = "healthy"
            message = "Robustness framework operating normally"
        elif health_ratio >= 0.6:
            status = "degraded"
            message = "Robustness framework partially degraded"
        else:
            status = "unhealthy"
            message = "Robustness framework significantly degraded"

        return {
            'status': status,
            'message': message,
            'details': {
                'healthy_components': healthy_components,
                'total_components': total_components,
                'health_ratio': health_ratio,
                'issues': issues
            }
        }

    def _configure_monitoring(self):
        """Configure monitoring components."""

        # Start Prometheus metrics collection if configured
        if self.config.get('enable_prometheus', True):
            self.prometheus_exporter.start_collection()

        # Configure distributed tracing
        tracing_config = self.config.get('tracing', {})
        if tracing_config.get('enabled', False):
            self.tracer.config.update(tracing_config)

        logger.debug("Monitoring configured")

    def _configure_security(self):
        """Configure security components."""

        # Configure threat detection thresholds
        security_config = self.config.get('security', {})

        if 'threat_thresholds' in security_config:
            # This would configure threat detection thresholds
            pass

        # Configure DDoS protection levels
        if 'ddos_protection' in security_config:
            ddos_config = security_config['ddos_protection']
            if 'protection_level' in ddos_config:
                # This would set protection level
                pass

        logger.debug("Security components configured")

    def _register_shutdown_handlers(self):
        """Register shutdown handlers for all components."""

        from ..hardening.shutdown_handler import ComponentPriority

        # Register high priority shutdown handlers
        self.shutdown_manager.register_simple(
            "threat_detector_cleanup",
            lambda: self.threat_detector.cleanup_old_data(),
            ComponentPriority.HIGH,
            10.0,
            "Cleanup threat detector data"
        )

        # Register medium priority shutdown handlers
        self.shutdown_manager.register_simple(
            "resource_manager_shutdown",
            self.resource_manager.shutdown,
            ComponentPriority.MEDIUM,
            15.0,
            "Shutdown resource manager"
        )

        # Register low priority shutdown handlers
        self.shutdown_manager.register_simple(
            "prometheus_cleanup",
            self.prometheus_exporter.stop_collection,
            ComponentPriority.LOW,
            5.0,
            "Stop Prometheus collection"
        )

        logger.debug("Shutdown handlers registered")

    def get_framework_status(self) -> dict[str, Any]:
        """Get comprehensive framework status."""

        status = {
            'initialized': self.initialized,
            'framework_stats': self.framework_stats.copy(),
            'components': {}
        }

        try:
            # Security components
            status['components']['threat_detector'] = {
                'active': True,
                'summary': self.threat_detector.get_threat_summary(hours=1)
            }

            status['components']['ddos_protection'] = {
                'active': True,
                'stats': self.ddos_protection.get_protection_stats()
            }

            # Monitoring components
            status['components']['prometheus'] = {
                'active': True,
                'summary': self.prometheus_exporter.get_metric_summary()
            }

            status['components']['tracing'] = {
                'active': True,
                'stats': self.tracer.get_trace_stats()
            }

            status['components']['circuit_breakers'] = {
                'active': True,
                'metrics': self.circuit_breakers.get_all_metrics()
            }

            # Resilience components
            status['components']['retry_manager'] = {
                'active': True,
                'stats': self.retry_manager.get_all_stats()
            }

            status['components']['degradation'] = {
                'active': True,
                'status': self.degradation_manager.get_status()
            }

            status['components']['health_checks'] = {
                'active': True,
                'status': self.health_manager.get_overall_status()
            }

            # Hardening components
            status['components']['resource_manager'] = {
                'active': True,
                'stats': self.resource_manager.get_resource_stats()
            }

            status['components']['secrets_manager'] = {
                'active': True,
                'summary': self.secrets_manager.get_secrets_summary()
            }

        except Exception as e:
            logger.error(f"Error getting framework status: {e}")
            status['error'] = str(e)

        return status

    def validate_request(self, request_data: dict[str, Any], source_ip: str = "unknown") -> dict[str, Any]:
        """Comprehensive request validation using all frameworks."""

        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'sanitized_data': request_data.copy()
        }

        try:
            # Security validation
            security_valid, security_errors = self.threat_detector.auditor.validate_request(
                request_data, source_ip
            )

            if not security_valid:
                validation_results['valid'] = False
                validation_results['errors'].extend(security_errors)

            # DDoS protection check
            ddos_result = self.ddos_protection.check_request(
                source_ip, request_data.get('endpoint', '/'),
                request_data.get('user_agent', ''),
                request_size=len(str(request_data))
            )

            if not ddos_result.get('allowed', True):
                validation_results['valid'] = False
                validation_results['errors'].append(ddos_result.get('reason', 'Rate limited'))

            # Schema validation (if schema provided)
            if 'schema_type' in request_data:
                schema_type = request_data['schema_type']
                if hasattr(CommonSchemas, f"{schema_type}_schema"):
                    schema = getattr(CommonSchemas, f"{schema_type}_schema")()

                    schema_valid, schema_errors, sanitized = self.schema_validator.validate(
                        request_data, schema, sanitize=True
                    )

                    if not schema_valid:
                        validation_results['warnings'].extend([
                            error.message for error in schema_errors
                            if error.severity.value in ['warning', 'info']
                        ])
                        validation_results['errors'].extend([
                            error.message for error in schema_errors
                            if error.severity.value in ['error', 'critical']
                        ])
                    else:
                        validation_results['sanitized_data'] = sanitized

        except Exception as e:
            logger.error(f"Request validation error: {e}")
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation system error: {str(e)}")

        return validation_results

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary."""

        try:
            overall_status = self.health_manager.get_overall_status()
            framework_health = self._framework_health_check()

            return {
                'framework_healthy': self.initialized,
                'overall_status': overall_status,
                'framework_health': framework_health,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {
                'framework_healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }


# Global robustness framework instance
robustness_framework = RobustnessFramework()


# Convenience functions
async def initialize_robustness_framework(config: dict[str, Any] | None = None):
    """Initialize the global robustness framework."""
    if config:
        robustness_framework.config.update(config)
    await robustness_framework.initialize()


def get_framework_status() -> dict[str, Any]:
    """Get status of the robustness framework."""
    return robustness_framework.get_framework_status()


def validate_request(request_data: dict[str, Any], source_ip: str = "unknown") -> dict[str, Any]:
    """Validate request using all robustness frameworks."""
    return robustness_framework.validate_request(request_data, source_ip)


def get_health_summary() -> dict[str, Any]:
    """Get overall health summary."""
    return robustness_framework.get_health_summary()


# Export key components for easy access
__all__ = [
    'RobustnessFramework',
    'robustness_framework',
    'initialize_robustness_framework',
    'get_framework_status',
    'validate_request',
    'get_health_summary',

    # Security components
    'threat_detector',
    'ddos_protection',
    'key_manager',
    'incident_response',

    # Monitoring components
    'prometheus_exporter',
    'tracer',
    'circuit_breaker_registry',

    # Resilience components
    'retry_registry',
    'degradation_manager',
    'health_manager',

    # Validation components
    'schema_validator',

    # Hardening components
    'resource_manager',
    'secrets_manager',
    'shutdown_manager'
]


# Import time for health checks
import time
