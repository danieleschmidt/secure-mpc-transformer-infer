"""Comprehensive error handling and logging utilities."""

import functools
import logging
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SECURITY = "security"
    PROTOCOL = "protocol"
    COMPUTATION = "computation"
    NETWORK = "network"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    SYSTEM = "system"


@dataclass
class ErrorDetails:
    """Detailed error information."""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    timestamp: float
    function_name: str
    file_name: str
    line_number: int
    stack_trace: str
    context: dict[str, Any]
    user_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "function_name": self.function_name,
            "file_name": self.file_name,
            "line_number": self.line_number,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "user_message": self.user_message
        }


class SecureMPCException(Exception):
    """Base exception for secure MPC operations."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class SecurityException(SecureMPCException):
    """Security-related exceptions."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, **kwargs):
        super().__init__(message, ErrorCategory.SECURITY, severity, **kwargs)


class ProtocolException(SecureMPCException):
    """MPC protocol-related exceptions."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, **kwargs):
        super().__init__(message, ErrorCategory.PROTOCOL, severity, **kwargs)


class ComputationException(SecureMPCException):
    """Computation-related exceptions."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message, ErrorCategory.COMPUTATION, severity, **kwargs)


class NetworkException(SecureMPCException):
    """Network communication exceptions."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message, ErrorCategory.NETWORK, severity, **kwargs)


class ValidationException(SecureMPCException):
    """Input validation exceptions."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.LOW, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, severity, **kwargs)


class ConfigurationException(SecureMPCException):
    """Configuration-related exceptions."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION, severity, **kwargs)


class ResourceException(SecureMPCException):
    """Resource-related exceptions (memory, GPU, etc.)."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message, ErrorCategory.RESOURCE, severity, **kwargs)


class ErrorHandler:
    """Centralized error handling and logging."""

    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.error_count = 0
        self.error_history: list[ErrorDetails] = []
        self.max_history = 1000

    def handle_error(self, error: Exception, context: dict[str, Any] | None = None,
                    user_message: str | None = None) -> ErrorDetails:
        """Handle and log an error with full details."""

        # Generate unique error ID
        self.error_count += 1
        error_id = f"err_{int(time.time())}_{self.error_count:04d}"

        # Extract error details
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback is None:
            # If not in exception context, get current frame info
            frame = sys._getframe(1)
            file_name = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            stack_trace = "".join(traceback.format_stack())
        else:
            # Extract from traceback
            tb_frame = exc_traceback.tb_frame
            file_name = tb_frame.f_code.co_filename
            function_name = tb_frame.f_code.co_name
            line_number = exc_traceback.tb_lineno
            stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # Determine category and severity
        if isinstance(error, SecureMPCException):
            category = error.category
            severity = error.severity
            message = error.message
            error_context = error.context
        else:
            category = self._classify_error(error)
            severity = self._assess_severity(error, category)
            message = str(error)
            error_context = {}

        # Merge contexts
        if context:
            error_context.update(context)

        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            timestamp=time.time(),
            function_name=function_name,
            file_name=file_name.split('/')[-1],  # Just filename, not full path
            line_number=line_number,
            stack_trace=stack_trace,
            context=error_context,
            user_message=user_message
        )

        # Log the error
        self._log_error(error_details)

        # Store in history
        self.error_history.append(error_details)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        return error_details

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error by type."""
        error_type = type(error).__name__.lower()

        if any(keyword in error_type for keyword in ['security', 'auth', 'permission']):
            return ErrorCategory.SECURITY
        elif any(keyword in error_type for keyword in ['network', 'connection', 'timeout']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_type for keyword in ['validation', 'value', 'type']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_type for keyword in ['memory', 'resource', 'gpu', 'cuda']):
            return ErrorCategory.RESOURCE
        elif any(keyword in error_type for keyword in ['config', 'setting']):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.SYSTEM

    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()

        # Critical conditions
        if (category == ErrorCategory.SECURITY or
            any(keyword in error_type for keyword in ['critical', 'fatal']) or
            any(keyword in error_message for keyword in ['critical', 'fatal', 'emergency'])):
            return ErrorSeverity.CRITICAL

        # High severity conditions
        elif (category in [ErrorCategory.PROTOCOL, ErrorCategory.CONFIGURATION] or
              any(keyword in error_type for keyword in ['assertion', 'runtime']) or
              any(keyword in error_message for keyword in ['corruption', 'integrity'])):
            return ErrorSeverity.HIGH

        # Medium severity conditions
        elif (category in [ErrorCategory.COMPUTATION, ErrorCategory.NETWORK, ErrorCategory.RESOURCE] or
              any(keyword in error_type for keyword in ['computation', 'overflow'])):
            return ErrorSeverity.MEDIUM

        # Low severity (validation, minor issues)
        else:
            return ErrorSeverity.LOW

    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_message = f"[{error_details.error_id}] {error_details.message}"

        if error_details.context:
            context_str = ", ".join(f"{k}={v}" for k, v in error_details.context.items())
            log_message += f" | Context: {context_str}"

        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.debug(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.debug(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}

        # Count by category and severity
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        # Recent errors (last hour)
        current_time = time.time()
        recent_errors = [e for e in self.error_history if current_time - e.timestamp < 3600]

        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "error_rate_per_hour": len(recent_errors),
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "highest_severity": max(severity_counts.keys()) if severity_counts else None
        }


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler(logger_name: str | None = None) -> ErrorHandler:
    """Get error handler instance."""
    if logger_name:
        return ErrorHandler(logger_name)
    return _error_handler


def handle_exceptions(category: ErrorCategory = ErrorCategory.SYSTEM,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     user_message: str | None = None,
                     reraise: bool = True):
    """Decorator for automatic exception handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to SecureMPCException if needed
                if not isinstance(e, SecureMPCException):
                    e = SecureMPCException(str(e), category, severity)

                # Handle the error
                error_details = _error_handler.handle_error(
                    e,
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    user_message=user_message
                )

                if reraise:
                    raise e
                else:
                    return {"error": error_details.to_dict()}

        return wrapper
    return decorator


def handle_async_exceptions(category: ErrorCategory = ErrorCategory.SYSTEM,
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           user_message: str | None = None,
                           reraise: bool = True):
    """Decorator for automatic async exception handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Convert to SecureMPCException if needed
                if not isinstance(e, SecureMPCException):
                    e = SecureMPCException(str(e), category, severity)

                # Handle the error
                error_details = _error_handler.handle_error(
                    e,
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    user_message=user_message
                )

                if reraise:
                    raise e
                else:
                    return {"error": error_details.to_dict()}

        return wrapper
    return decorator


def setup_logging(log_level: str = "INFO", log_format: str | None = None,
                 log_file: str | None = None) -> None:
    """Setup comprehensive logging configuration."""

    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        filename=log_file
    )

    # Setup specific loggers for different components
    loggers = [
        "secure_mpc_transformer",
        "secure_mpc_transformer.models",
        "secure_mpc_transformer.protocols",
        "secure_mpc_transformer.services",
        "secure_mpc_transformer.security",
        "secure_mpc_transformer.planning"
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Add console handler if not already present
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(console_handler)

    # Set third-party library log levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def log_function_entry(logger: logging.Logger, func_name: str, *args, **kwargs):
    """Log function entry with parameters."""
    args_str = ", ".join(str(arg)[:100] for arg in args)  # Truncate long args
    kwargs_str = ", ".join(f"{k}={str(v)[:100]}" for k, v in kwargs.items())
    params_str = ", ".join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"Entering {func_name}({params_str})")


def log_function_exit(logger: logging.Logger, func_name: str, result: Any = None,
                     duration_ms: float | None = None):
    """Log function exit with result and duration."""
    result_str = str(result)[:200] if result is not None else "None"
    duration_str = f" (took {duration_ms:.2f}ms)" if duration_ms else ""

    logger.debug(f"Exiting {func_name} -> {result_str}{duration_str}")


class TimedOperation:
    """Context manager for timing operations and logging."""

    def __init__(self, operation_name: str, logger: logging.Logger | None = None,
                 log_level: int = logging.INFO):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.log_level, f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is None:
            self.logger.log(self.log_level,
                          f"Completed {self.operation_name} in {self.duration_ms:.2f}ms")
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {self.duration_ms:.2f}ms: {exc_val}")

        return False  # Don't suppress exceptions
