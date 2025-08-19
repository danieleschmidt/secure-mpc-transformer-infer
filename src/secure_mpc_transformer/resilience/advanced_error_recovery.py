"""Advanced Error Recovery System - Generation 2 Robustness Enhancement."""

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class ErrorCategory(Enum):
    """Error category classification."""
    NETWORK = "network"
    COMPUTE = "compute"
    MEMORY = "memory"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """Error occurrence record."""
    id: str
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: dict[str, Any]
    recovery_attempted: bool = False
    recovery_action: RecoveryAction | None = None
    recovery_successful: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RecoveryRule:
    """Error recovery rule definition."""
    name: str
    error_pattern: str  # Regex pattern to match error messages
    category: ErrorCategory
    max_retries: int
    retry_delay: float
    recovery_action: RecoveryAction
    fallback_function: Callable | None = None
    escalation_threshold: int = 5  # Number of failures before escalation


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Success - reset on successful call
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            raise e


class RetryManager:
    """Advanced retry mechanism with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")

        raise last_exception

    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry sync function with exponential backoff."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")

        raise last_exception


class FallbackManager:
    """Fallback mechanism for graceful degradation."""

    def __init__(self):
        self.fallbacks: dict[str, Callable] = {}
        self.fallback_used_count: dict[str, int] = {}

    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register fallback function for an operation."""
        self.fallbacks[operation_name] = fallback_func
        self.fallback_used_count[operation_name] = 0
        logger.info(f"Registered fallback for operation: {operation_name}")

    async def execute_with_fallback(self, operation_name: str, primary_func: Callable,
                                  *args, **kwargs) -> Any:
        """Execute function with fallback on failure."""
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)

        except Exception as e:
            logger.warning(f"Primary function failed for {operation_name}: {str(e)}")

            if operation_name in self.fallbacks:
                try:
                    fallback_func = self.fallbacks[operation_name]
                    self.fallback_used_count[operation_name] += 1

                    logger.info(f"Using fallback for {operation_name}")

                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)

                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {operation_name}: {str(fallback_error)}")
                    raise fallback_error
            else:
                logger.error(f"No fallback registered for {operation_name}")
                raise e

    def get_fallback_stats(self) -> dict[str, Any]:
        """Get fallback usage statistics."""
        return {
            "registered_fallbacks": len(self.fallbacks),
            "usage_counts": self.fallback_used_count.copy(),
            "total_fallback_uses": sum(self.fallback_used_count.values())
        }


class ErrorAnalyzer:
    """Error pattern analysis and categorization."""

    def __init__(self):
        self.error_patterns = {
            ErrorCategory.NETWORK: [
                r"connection.*refused",
                r"timeout",
                r"network.*unreachable",
                r"dns.*resolution.*failed",
                r"ssl.*error"
            ],
            ErrorCategory.MEMORY: [
                r"out.*of.*memory",
                r"memory.*error",
                r"allocation.*failed",
                r"insufficient.*memory"
            ],
            ErrorCategory.COMPUTE: [
                r"cuda.*error",
                r"gpu.*error",
                r"computation.*failed",
                r"device.*error"
            ],
            ErrorCategory.VALIDATION: [
                r"validation.*error",
                r"invalid.*input",
                r"schema.*error",
                r"type.*error"
            ],
            ErrorCategory.AUTHENTICATION: [
                r"authentication.*failed",
                r"unauthorized",
                r"access.*denied",
                r"invalid.*credentials"
            ],
            ErrorCategory.RATE_LIMIT: [
                r"rate.*limit",
                r"too.*many.*requests",
                r"quota.*exceeded",
                r"throttled"
            ]
        }

    def categorize_error(self, error_message: str, error_type: str) -> ErrorCategory:
        """Categorize error based on message and type."""
        import re

        error_text = f"{error_type} {error_message}".lower()

        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return category

        return ErrorCategory.UNKNOWN

    def determine_severity(self, error_type: str, category: ErrorCategory,
                         context: dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on type, category, and context."""
        # Critical errors
        if category in [ErrorCategory.MEMORY, ErrorCategory.SYSTEM]:
            return ErrorSeverity.CRITICAL

        # High severity
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.COMPUTE]:
            return ErrorSeverity.HIGH

        # Medium severity
        if category in [ErrorCategory.NETWORK, ErrorCategory.VALIDATION]:
            return ErrorSeverity.MEDIUM

        # Check context for severity indicators
        if context.get("user_facing", False):
            return ErrorSeverity.HIGH

        if context.get("affects_multiple_users", False):
            return ErrorSeverity.CRITICAL

        return ErrorSeverity.LOW


class AdvancedErrorRecovery:
    """Advanced error recovery and resilience system."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.retry_manager = RetryManager(
            max_retries=self.config.get("max_retries", 3),
            base_delay=self.config.get("base_delay", 1.0),
            max_delay=self.config.get("max_delay", 60.0)
        )

        self.fallback_manager = FallbackManager()
        self.error_analyzer = ErrorAnalyzer()

        # Error tracking
        self.error_records: list[ErrorRecord] = []
        self.max_error_records = self.config.get("max_error_records", 1000)

        # Recovery rules
        self.recovery_rules: list[RecoveryRule] = []
        self.setup_default_recovery_rules()

        # Circuit breakers for external services
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Error handling callbacks
        self.error_handlers: list[Callable[[ErrorRecord], None]] = []

        logger.info("Advanced Error Recovery system initialized")

    def setup_default_recovery_rules(self):
        """Setup default error recovery rules."""
        default_rules = [
            RecoveryRule(
                name="network_retry",
                error_pattern=r"connection.*refused|timeout",
                category=ErrorCategory.NETWORK,
                max_retries=3,
                retry_delay=2.0,
                recovery_action=RecoveryAction.RETRY
            ),
            RecoveryRule(
                name="memory_fallback",
                error_pattern=r"out.*of.*memory",
                category=ErrorCategory.MEMORY,
                max_retries=1,
                retry_delay=5.0,
                recovery_action=RecoveryAction.FALLBACK
            ),
            RecoveryRule(
                name="validation_ignore",
                error_pattern=r"validation.*error",
                category=ErrorCategory.VALIDATION,
                max_retries=0,
                retry_delay=0.0,
                recovery_action=RecoveryAction.IGNORE
            ),
            RecoveryRule(
                name="rate_limit_backoff",
                error_pattern=r"rate.*limit|too.*many.*requests",
                category=ErrorCategory.RATE_LIMIT,
                max_retries=5,
                retry_delay=10.0,
                recovery_action=RecoveryAction.RETRY
            )
        ]

        self.recovery_rules.extend(default_rules)
        logger.info(f"Setup {len(default_rules)} default recovery rules")

    def add_recovery_rule(self, rule: RecoveryRule):
        """Add custom recovery rule."""
        self.recovery_rules.append(rule)
        logger.info(f"Added recovery rule: {rule.name}")

    def add_error_handler(self, handler: Callable[[ErrorRecord], None]):
        """Add error handling callback."""
        self.error_handlers.append(handler)

    def register_circuit_breaker(self, service_name: str, failure_threshold: int = 5,
                               recovery_timeout: int = 60):
        """Register circuit breaker for external service."""
        self.circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info(f"Registered circuit breaker for service: {service_name}")

    async def execute_with_recovery(self, operation_name: str, func: Callable,
                                  *args, **kwargs) -> Any:
        """Execute function with comprehensive error recovery."""
        start_time = time.time()
        error_record = None

        try:
            # Check if circuit breaker exists for this operation
            if operation_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation_name]
                return await circuit_breaker.call(func, *args, **kwargs)
            else:
                # Execute with fallback
                return await self.fallback_manager.execute_with_fallback(
                    operation_name, func, *args, **kwargs
                )

        except Exception as e:
            # Create error record
            error_record = self._create_error_record(e, {
                "operation_name": operation_name,
                "execution_time": time.time() - start_time,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })

            # Attempt recovery
            recovery_result = await self._attempt_recovery(error_record, func, *args, **kwargs)

            if recovery_result["success"]:
                return recovery_result["result"]
            else:
                # Recovery failed, re-raise original exception
                raise e

    def _create_error_record(self, exception: Exception, context: dict[str, Any]) -> ErrorRecord:
        """Create error record from exception."""
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()

        # Analyze error
        category = self.error_analyzer.categorize_error(error_message, error_type)
        severity = self.error_analyzer.determine_severity(error_type, category, context)

        # Create record
        record = ErrorRecord(
            id=f"err_{int(time.time() * 1000)}",
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            category=category,
            severity=severity,
            context=context
        )

        # Store record
        self.error_records.append(record)

        # Trim records if too many
        if len(self.error_records) > self.max_error_records:
            self.error_records = self.error_records[-self.max_error_records:]

        # Notify handlers
        for handler in self.error_handlers:
            try:
                handler(record)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")

        return record

    async def _attempt_recovery(self, error_record: ErrorRecord, func: Callable,
                              *args, **kwargs) -> dict[str, Any]:
        """Attempt error recovery based on rules."""
        # Find matching recovery rule
        recovery_rule = self._find_recovery_rule(error_record)

        if not recovery_rule:
            logger.debug(f"No recovery rule found for error: {error_record.error_type}")
            return {"success": False, "reason": "No recovery rule"}

        error_record.recovery_attempted = True
        error_record.recovery_action = recovery_rule.recovery_action

        try:
            if recovery_rule.recovery_action == RecoveryAction.RETRY:
                # Retry with exponential backoff
                retry_manager = RetryManager(
                    max_retries=recovery_rule.max_retries,
                    base_delay=recovery_rule.retry_delay
                )

                result = await retry_manager.retry_async(func, *args, **kwargs)
                error_record.recovery_successful = True

                logger.info(f"Recovery successful for {error_record.error_type} using retry")
                return {"success": True, "result": result}

            elif recovery_rule.recovery_action == RecoveryAction.FALLBACK:
                if recovery_rule.fallback_function:
                    result = await recovery_rule.fallback_function(*args, **kwargs)
                    error_record.recovery_successful = True

                    logger.info(f"Recovery successful for {error_record.error_type} using fallback")
                    return {"success": True, "result": result}
                else:
                    logger.warning(f"No fallback function defined for rule: {recovery_rule.name}")
                    return {"success": False, "reason": "No fallback function"}

            elif recovery_rule.recovery_action == RecoveryAction.IGNORE:
                error_record.recovery_successful = True
                logger.info(f"Ignoring error {error_record.error_type} as per recovery rule")
                return {"success": True, "result": None}

            else:
                logger.warning(f"Unsupported recovery action: {recovery_rule.recovery_action}")
                return {"success": False, "reason": "Unsupported recovery action"}

        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return {"success": False, "reason": str(recovery_error)}

    def _find_recovery_rule(self, error_record: ErrorRecord) -> RecoveryRule | None:
        """Find matching recovery rule for error."""
        import re

        for rule in self.recovery_rules:
            # Check category match
            if rule.category != error_record.category:
                continue

            # Check pattern match
            if re.search(rule.error_pattern, error_record.error_message, re.IGNORECASE):
                return rule

        return None

    def get_error_statistics(self, hours: int = 24) -> dict[str, Any]:
        """Get error statistics for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_errors = [e for e in self.error_records if e.timestamp > cutoff_time]

        # Count by category
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = len([e for e in recent_errors if e.category == category])

        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = len([e for e in recent_errors if e.severity == severity])

        # Recovery statistics
        recovery_attempts = len([e for e in recent_errors if e.recovery_attempted])
        recovery_successes = len([e for e in recent_errors if e.recovery_successful])

        return {
            "total_errors": len(recent_errors),
            "time_period_hours": hours,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
            "recovery_success_rate": recovery_successes / max(recovery_attempts, 1),
            "fallback_stats": self.fallback_manager.get_fallback_stats(),
            "timestamp": time.time()
        }

    def get_recent_errors(self, limit: int = 50, severity: ErrorSeverity = None) -> list[dict[str, Any]]:
        """Get recent error records."""
        errors = self.error_records

        if severity:
            errors = [e for e in errors if e.severity == severity]

        # Sort by timestamp (newest first)
        errors.sort(key=lambda x: x.timestamp, reverse=True)

        return [error.to_dict() for error in errors[:limit]]

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health based on error patterns."""
        recent_stats = self.get_error_statistics(hours=1)  # Last hour

        # Determine health status
        critical_errors = recent_stats["severity_breakdown"]["critical"]
        high_errors = recent_stats["severity_breakdown"]["high"]
        total_errors = recent_stats["total_errors"]

        if critical_errors > 0:
            health_status = "critical"
        elif high_errors > 5:
            health_status = "degraded"
        elif total_errors > 20:
            health_status = "warning"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "recent_errors": total_errors,
            "critical_errors": critical_errors,
            "high_errors": high_errors,
            "recovery_success_rate": recent_stats["recovery_success_rate"],
            "timestamp": time.time()
        }

    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register fallback function for operation."""
        self.fallback_manager.register_fallback(operation_name, fallback_func)

    def clear_error_history(self):
        """Clear error history (for testing/maintenance)."""
        self.error_records.clear()
        logger.info("Error history cleared")


# Default error handler
def log_error_handler(error_record: ErrorRecord):
    """Default error handler that logs errors."""
    logger.error(
        f"Error [{error_record.severity.value}]: {error_record.error_type} - "
        f"{error_record.error_message} (Category: {error_record.category.value})"
    )
