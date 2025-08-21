"""
Generation 2: Robust Error Handling with Enhanced Recovery and Monitoring
"""

import logging
import time
import traceback
import functools
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for Generation 2."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for Generation 2."""
    SECURITY = "security"
    VALIDATION = "validation"
    COMPUTATION = "computation"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Enhanced error context for Generation 2."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    function_name: str
    file_name: str
    line_number: int
    stack_trace: str
    request_id: Optional[str] = None
    client_id: Optional[str] = None
    retry_count: int = 0
    recovery_attempted: bool = False
    recovery_successful: bool = False
    additional_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_context is None:
            self.additional_context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'function_name': self.function_name,
            'file_name': self.file_name,
            'line_number': self.line_number,
            'request_id': self.request_id,
            'client_id': self.client_id,
            'retry_count': self.retry_count,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'additional_context': self.additional_context
        }


class SecureMPCException(Exception):
    """Enhanced base exception for Generation 2."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Dict[str, Any] = None,
                 request_id: str = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.request_id = request_id
        self.timestamp = time.time()


class SecurityException(SecureMPCException):
    """Security-related exceptions with enhanced context."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, **kwargs):
        super().__init__(message, ErrorCategory.SECURITY, severity, **kwargs)


class ValidationException(SecureMPCException):
    """Input validation exceptions."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, severity, **kwargs)


class TimeoutException(SecureMPCException):
    """Timeout-related exceptions."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, **kwargs):
        super().__init__(message, ErrorCategory.TIMEOUT, severity, **kwargs)


class RateLimitException(SecureMPCException):
    """Rate limiting exceptions."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message, ErrorCategory.RATE_LIMIT, severity, **kwargs)


class DataIntegrityException(SecureMPCException):
    """Data integrity exceptions."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, **kwargs):
        super().__init__(message, ErrorCategory.DATA_INTEGRITY, severity, **kwargs)


class RobustErrorHandler:
    """Enhanced error handler with recovery capabilities for Generation 2."""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.error_count = 0
        self.error_history: List[ErrorContext] = []
        self.max_history = 1000
        self.recovery_strategies = {}
        
        # Set up default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Set up default error recovery strategies."""
        self.recovery_strategies = {
            ErrorCategory.TIMEOUT: self._recover_from_timeout,
            ErrorCategory.NETWORK: self._recover_from_network_error,
            ErrorCategory.RESOURCE: self._recover_from_resource_error,
            ErrorCategory.VALIDATION: self._recover_from_validation_error,
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    request_id: str = None, client_id: str = None,
                    attempt_recovery: bool = True) -> ErrorContext:
        """Enhanced error handling with recovery attempts."""
        
        # Generate unique error ID
        self.error_count += 1
        error_id = f"err_{int(time.time())}_{self.error_count:04d}"
        
        # Extract error details
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        if exc_traceback is None:
            # Get current frame info
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
        
        # Create error context
        error_ctx = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=message,
            function_name=function_name,
            file_name=file_name.split('/')[-1],
            line_number=line_number,
            stack_trace=stack_trace,
            request_id=request_id,
            client_id=client_id,
            additional_context=error_context
        )
        
        # Attempt recovery if enabled and appropriate
        if attempt_recovery and category in self.recovery_strategies:
            try:
                error_ctx.recovery_attempted = True
                recovery_result = self.recovery_strategies[category](error, error_ctx)
                error_ctx.recovery_successful = recovery_result
                
                if recovery_result:
                    self.logger.info(f"Error recovery successful for {error_id}")
                else:
                    self.logger.warning(f"Error recovery failed for {error_id}")
            except Exception as recovery_error:
                self.logger.error(f"Error recovery attempt failed: {recovery_error}")
                error_ctx.recovery_successful = False
        
        # Log the error
        self._log_error(error_ctx)
        
        # Store in history
        self.error_history.append(error_ctx)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        return error_ctx
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error by type and message."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # Classification logic
        if any(keyword in error_type for keyword in ['security', 'auth', 'permission']):
            return ErrorCategory.SECURITY
        elif any(keyword in error_type for keyword in ['timeout', 'time']):
            return ErrorCategory.TIMEOUT
        elif any(keyword in error_type for keyword in ['validation', 'value', 'invalid']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_type for keyword in ['network', 'connection', 'http']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_type for keyword in ['memory', 'resource', 'limit']):
            return ErrorCategory.RESOURCE
        elif any(keyword in error_message for keyword in ['rate limit', 'too many']):
            return ErrorCategory.RATE_LIMIT
        elif any(keyword in error_message for keyword in ['integrity', 'checksum', 'corrupt']):
            return ErrorCategory.DATA_INTEGRITY
        else:
            return ErrorCategory.SYSTEM
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on type and category."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # Critical severity conditions
        if (category == ErrorCategory.SECURITY or
            category == ErrorCategory.DATA_INTEGRITY or
            any(keyword in error_message for keyword in ['critical', 'fatal', 'corruption'])):
            return ErrorSeverity.CRITICAL
        
        # High severity conditions
        elif (category in [ErrorCategory.TIMEOUT, ErrorCategory.CONFIGURATION] or
              any(keyword in error_type for keyword in ['runtime', 'assertion'])):
            return ErrorSeverity.HIGH
        
        # Medium severity conditions
        elif (category in [ErrorCategory.COMPUTATION, ErrorCategory.NETWORK, ErrorCategory.RESOURCE] or
              category == ErrorCategory.VALIDATION):
            return ErrorSeverity.MEDIUM
        
        # Low severity
        else:
            return ErrorSeverity.LOW
    
    def _log_error(self, error_ctx: ErrorContext):
        """Log error with appropriate level and context."""
        log_message = f"[{error_ctx.error_id}] {error_ctx.message}"
        
        if error_ctx.request_id:
            log_message += f" (request: {error_ctx.request_id})"
        
        if error_ctx.additional_context:
            context_items = [f"{k}={v}" for k, v in error_ctx.additional_context.items()]
            log_message += f" | Context: {', '.join(context_items)}"
        
        # Log based on severity
        if error_ctx.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.debug(f"Stack trace: {error_ctx.stack_trace}")
        elif error_ctx.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.debug(f"Stack trace: {error_ctx.stack_trace}")
        elif error_ctx.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _recover_from_timeout(self, error: Exception, error_ctx: ErrorContext) -> bool:
        """Attempt recovery from timeout errors."""
        self.logger.info(f"Attempting timeout recovery for {error_ctx.error_id}")
        
        # Simple recovery: wait and signal that retry should be attempted
        time.sleep(1.0)
        return True
    
    def _recover_from_network_error(self, error: Exception, error_ctx: ErrorContext) -> bool:
        """Attempt recovery from network errors."""
        self.logger.info(f"Attempting network error recovery for {error_ctx.error_id}")
        
        # Simple recovery: wait and signal that retry should be attempted
        time.sleep(2.0)
        return True
    
    def _recover_from_resource_error(self, error: Exception, error_ctx: ErrorContext) -> bool:
        """Attempt recovery from resource errors."""
        self.logger.info(f"Attempting resource error recovery for {error_ctx.error_id}")
        
        # Simple recovery: force garbage collection and wait
        import gc
        gc.collect()
        time.sleep(1.0)
        return True
    
    def _recover_from_validation_error(self, error: Exception, error_ctx: ErrorContext) -> bool:
        """Attempt recovery from validation errors."""
        self.logger.info(f"Validation error recovery not applicable for {error_ctx.error_id}")
        # Validation errors typically can't be recovered from automatically
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count by category and severity
        category_counts = {}
        severity_counts = {}
        recovery_stats = {"attempted": 0, "successful": 0}
        
        current_time = time.time()
        recent_errors = []  # Last hour
        
        for error_ctx in self.error_history:
            category_counts[error_ctx.category.value] = category_counts.get(error_ctx.category.value, 0) + 1
            severity_counts[error_ctx.severity.value] = severity_counts.get(error_ctx.severity.value, 0) + 1
            
            if error_ctx.recovery_attempted:
                recovery_stats["attempted"] += 1
                if error_ctx.recovery_successful:
                    recovery_stats["successful"] += 1
            
            if current_time - error_ctx.timestamp < 3600:  # Last hour
                recent_errors.append(error_ctx)
        
        recovery_rate = (recovery_stats["successful"] / recovery_stats["attempted"] * 100 
                        if recovery_stats["attempted"] > 0 else 0)
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "recovery_statistics": {
                "attempts": recovery_stats["attempted"],
                "successes": recovery_stats["successful"],
                "success_rate_percent": round(recovery_rate, 2)
            },
            "error_rate_per_hour": len(recent_errors),
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "highest_severity": max(severity_counts.keys()) if severity_counts else None
        }


# Global robust error handler
_robust_error_handler = RobustErrorHandler()


def get_robust_error_handler(logger_name: str = None) -> RobustErrorHandler:
    """Get robust error handler instance."""
    if logger_name:
        return RobustErrorHandler(logger_name)
    return _robust_error_handler


def robust_exception_handler(category: ErrorCategory = ErrorCategory.SYSTEM,
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           attempt_recovery: bool = True,
                           reraise: bool = True):
    """Enhanced decorator for robust exception handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to SecureMPCException if needed
                if not isinstance(e, SecureMPCException):
                    e = SecureMPCException(str(e), category, severity)
                
                # Extract request context if available
                request_id = kwargs.get('request_id') or getattr(args[0] if args else None, 'request_id', None)
                client_id = kwargs.get('client_id') or getattr(args[0] if args else None, 'client_id', None)
                
                # Handle the error
                error_ctx = _robust_error_handler.handle_error(
                    e,
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    request_id=request_id,
                    client_id=client_id,
                    attempt_recovery=attempt_recovery
                )
                
                if reraise:
                    raise e
                else:
                    return {"error": error_ctx.to_dict()}
        
        return wrapper
    return decorator


def setup_robust_logging(log_level: str = "INFO", log_file: str = None, 
                        enable_detailed_logging: bool = True):
    """Setup enhanced logging for Generation 2."""
    
    # Basic logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        filename=log_file if log_file else None
    )
    
    if enable_detailed_logging:
        # Add console handler if logging to file
        if log_file:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(console_handler)
    
    # Configure specific loggers
    loggers = [
        "secure_mpc_transformer.models",
        "secure_mpc_transformer.utils",
        "secure_mpc_transformer.security",
        "secure_mpc_transformer.monitoring"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)