"""Advanced retry mechanisms with exponential backoff and jitter."""

import asyncio
import functools
import inspect
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


class JitterType(Enum):
    """Types of jitter for retry delays."""
    NONE = "none"
    FULL = "full"          # Random delay between 0 and calculated delay
    EQUAL = "equal"        # Half fixed + half random
    DECORRELATED = "decorrelated"  # Based on previous delay


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Basic retry parameters
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 300.0  # Maximum delay in seconds

    # Strategy and jitter
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: JitterType = JitterType.EQUAL
    exponential_base: float = 2.0

    # Exception handling
    retryable_exceptions: list[type[Exception]] = field(default_factory=lambda: [Exception])
    non_retryable_exceptions: list[type[Exception]] = field(default_factory=list)

    # Conditional retry
    retry_condition: Callable[[Exception, int], bool] | None = None

    # Timeouts
    operation_timeout: float | None = None
    total_timeout: float | None = None

    # Circuit breaker integration
    circuit_breaker_name: str | None = None

    # Callbacks
    on_retry: Callable[[Exception, int, float], None] | None = None
    on_success: Callable[[Any, int], None] | None = None
    on_failure: Callable[[Exception, int], None] | None = None


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt_number: int
    exception: Exception | None
    duration: float
    delay_before_retry: float | None
    timestamp: float
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'attempt_number': self.attempt_number,
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'exception_message': str(self.exception) if self.exception else None,
            'duration': self.duration,
            'delay_before_retry': self.delay_before_retry,
            'timestamp': self.timestamp,
            'success': self.success
        }


@dataclass
class RetryResult:
    """Result of retry operation."""

    success: bool
    result: Any
    final_exception: Exception | None
    attempts: list[RetryAttempt]
    total_duration: float

    @property
    def attempt_count(self) -> int:
        """Number of attempts made."""
        return len(self.attempts)

    @property
    def success_on_attempt(self) -> int | None:
        """Which attempt succeeded, if any."""
        for attempt in self.attempts:
            if attempt.success:
                return attempt.attempt_number
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'success': self.success,
            'attempt_count': self.attempt_count,
            'success_on_attempt': self.success_on_attempt,
            'total_duration': self.total_duration,
            'attempts': [attempt.to_dict() for attempt in self.attempts],
            'final_exception_type': type(self.final_exception).__name__ if self.final_exception else None,
            'final_exception_message': str(self.final_exception) if self.final_exception else None
        }


class DelayCalculator:
    """Calculate retry delays based on strategy."""

    @staticmethod
    def calculate_delay(strategy: RetryStrategy, attempt: int, base_delay: float,
                       max_delay: float, exponential_base: float = 2.0,
                       previous_delay: float | None = None) -> float:
        """Calculate delay for retry attempt."""

        if strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay

        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * attempt

        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (exponential_base ** (attempt - 1))

        elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = base_delay * DelayCalculator._fibonacci(attempt)

        else:  # CUSTOM or fallback
            delay = base_delay

        return min(delay, max_delay)

    @staticmethod
    def apply_jitter(delay: float, jitter_type: JitterType,
                    previous_delay: float | None = None) -> float:
        """Apply jitter to delay."""

        if jitter_type == JitterType.NONE:
            return delay

        elif jitter_type == JitterType.FULL:
            return random.uniform(0, delay)

        elif jitter_type == JitterType.EQUAL:
            return delay * 0.5 + random.uniform(0, delay * 0.5)

        elif jitter_type == JitterType.DECORRELATED:
            if previous_delay is None:
                return random.uniform(0, delay)
            return random.uniform(delay * 0.5, previous_delay * 3)

        return delay

    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 2:
            return 1
        a, b = 1, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b


class RetryManager:
    """Manages retry operations with comprehensive strategies."""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self.active_retries: dict[str, RetryResult] = {}
        self.retry_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_attempts': 0,
            'operations_succeeded_on_retry': 0
        }
        self._lock = threading.Lock()

    def execute(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute function with retry logic."""
        if inspect.iscoroutinefunction(func):
            raise ValueError("Use async_execute for coroutine functions")

        return self._execute_with_retry(func, args, kwargs)

    async def async_execute(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute async function with retry logic."""
        if not inspect.iscoroutinefunction(func):
            raise ValueError("Use execute for regular functions")

        return await self._async_execute_with_retry(func, args, kwargs)

    def _execute_with_retry(self, func: Callable, args: tuple, kwargs: dict) -> RetryResult:
        """Execute synchronous function with retry logic."""
        start_time = time.time()
        attempts = []
        last_exception = None
        previous_delay = None

        with self._lock:
            self.retry_stats['total_operations'] += 1

        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()

            try:
                # Apply operation timeout if configured
                if self.config.operation_timeout:
                    result = self._execute_with_timeout(func, args, kwargs, self.config.operation_timeout)
                else:
                    result = func(*args, **kwargs)

                # Success!
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    exception=None,
                    duration=time.time() - attempt_start,
                    delay_before_retry=None,
                    timestamp=attempt_start,
                    success=True
                )
                attempts.append(attempt)

                # Update stats
                with self._lock:
                    self.retry_stats['successful_operations'] += 1
                    self.retry_stats['total_attempts'] += attempt_num
                    if attempt_num > 1:
                        self.retry_stats['operations_succeeded_on_retry'] += 1

                # Success callback
                if self.config.on_success:
                    try:
                        self.config.on_success(result, attempt_num)
                    except Exception as e:
                        logger.error(f"Error in success callback: {e}")

                return RetryResult(
                    success=True,
                    result=result,
                    final_exception=None,
                    attempts=attempts,
                    total_duration=time.time() - start_time
                )

            except Exception as e:
                last_exception = e
                attempt_duration = time.time() - attempt_start

                # Check if this exception should be retried
                if not self._should_retry(e, attempt_num):
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        exception=e,
                        duration=attempt_duration,
                        delay_before_retry=None,
                        timestamp=attempt_start,
                        success=False
                    )
                    attempts.append(attempt)
                    break

                # Check total timeout
                if (self.config.total_timeout and
                    time.time() - start_time >= self.config.total_timeout):
                    logger.warning("Total retry timeout exceeded")
                    break

                # Calculate delay for next attempt
                delay = None
                if attempt_num < self.config.max_attempts:
                    delay = DelayCalculator.calculate_delay(
                        self.config.strategy,
                        attempt_num,
                        self.config.base_delay,
                        self.config.max_delay,
                        self.config.exponential_base,
                        previous_delay
                    )

                    delay = DelayCalculator.apply_jitter(
                        delay, self.config.jitter, previous_delay
                    )
                    previous_delay = delay

                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    exception=e,
                    duration=attempt_duration,
                    delay_before_retry=delay,
                    timestamp=attempt_start,
                    success=False
                )
                attempts.append(attempt)

                # Retry callback
                if self.config.on_retry and delay is not None:
                    try:
                        self.config.on_retry(e, attempt_num, delay)
                    except Exception as callback_e:
                        logger.error(f"Error in retry callback: {callback_e}")

                # Wait before next attempt
                if delay is not None and attempt_num < self.config.max_attempts:
                    time.sleep(delay)

        # All attempts failed
        with self._lock:
            self.retry_stats['failed_operations'] += 1
            self.retry_stats['total_attempts'] += len(attempts)

        # Failure callback
        if self.config.on_failure:
            try:
                self.config.on_failure(last_exception, len(attempts))
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")

        return RetryResult(
            success=False,
            result=None,
            final_exception=last_exception,
            attempts=attempts,
            total_duration=time.time() - start_time
        )

    async def _async_execute_with_retry(self, func: Callable, args: tuple, kwargs: dict) -> RetryResult:
        """Execute async function with retry logic."""
        start_time = time.time()
        attempts = []
        last_exception = None
        previous_delay = None

        with self._lock:
            self.retry_stats['total_operations'] += 1

        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()

            try:
                # Apply operation timeout if configured
                if self.config.operation_timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.operation_timeout
                    )
                else:
                    result = await func(*args, **kwargs)

                # Success!
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    exception=None,
                    duration=time.time() - attempt_start,
                    delay_before_retry=None,
                    timestamp=attempt_start,
                    success=True
                )
                attempts.append(attempt)

                # Update stats
                with self._lock:
                    self.retry_stats['successful_operations'] += 1
                    self.retry_stats['total_attempts'] += attempt_num
                    if attempt_num > 1:
                        self.retry_stats['operations_succeeded_on_retry'] += 1

                # Success callback
                if self.config.on_success:
                    try:
                        if inspect.iscoroutinefunction(self.config.on_success):
                            await self.config.on_success(result, attempt_num)
                        else:
                            self.config.on_success(result, attempt_num)
                    except Exception as e:
                        logger.error(f"Error in success callback: {e}")

                return RetryResult(
                    success=True,
                    result=result,
                    final_exception=None,
                    attempts=attempts,
                    total_duration=time.time() - start_time
                )

            except Exception as e:
                last_exception = e
                attempt_duration = time.time() - attempt_start

                # Check if this exception should be retried
                if not self._should_retry(e, attempt_num):
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        exception=e,
                        duration=attempt_duration,
                        delay_before_retry=None,
                        timestamp=attempt_start,
                        success=False
                    )
                    attempts.append(attempt)
                    break

                # Check total timeout
                if (self.config.total_timeout and
                    time.time() - start_time >= self.config.total_timeout):
                    logger.warning("Total retry timeout exceeded")
                    break

                # Calculate delay for next attempt
                delay = None
                if attempt_num < self.config.max_attempts:
                    delay = DelayCalculator.calculate_delay(
                        self.config.strategy,
                        attempt_num,
                        self.config.base_delay,
                        self.config.max_delay,
                        self.config.exponential_base,
                        previous_delay
                    )

                    delay = DelayCalculator.apply_jitter(
                        delay, self.config.jitter, previous_delay
                    )
                    previous_delay = delay

                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    exception=e,
                    duration=attempt_duration,
                    delay_before_retry=delay,
                    timestamp=attempt_start,
                    success=False
                )
                attempts.append(attempt)

                # Retry callback
                if self.config.on_retry and delay is not None:
                    try:
                        if inspect.iscoroutinefunction(self.config.on_retry):
                            await self.config.on_retry(e, attempt_num, delay)
                        else:
                            self.config.on_retry(e, attempt_num, delay)
                    except Exception as callback_e:
                        logger.error(f"Error in retry callback: {callback_e}")

                # Wait before next attempt
                if delay is not None and attempt_num < self.config.max_attempts:
                    await asyncio.sleep(delay)

        # All attempts failed
        with self._lock:
            self.retry_stats['failed_operations'] += 1
            self.retry_stats['total_attempts'] += len(attempts)

        # Failure callback
        if self.config.on_failure:
            try:
                if inspect.iscoroutinefunction(self.config.on_failure):
                    await self.config.on_failure(last_exception, len(attempts))
                else:
                    self.config.on_failure(last_exception, len(attempts))
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")

        return RetryResult(
            success=False,
            result=None,
            final_exception=last_exception,
            attempts=attempts,
            total_duration=time.time() - start_time
        )

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict, timeout: float):
        """Execute function with timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _should_retry(self, exception: Exception, attempt_num: int) -> bool:
        """Determine if exception should trigger retry."""

        # Check if max attempts reached
        if attempt_num >= self.config.max_attempts:
            return False

        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False

        # Check retryable exceptions
        retryable = False
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                retryable = True
                break

        if not retryable:
            return False

        # Apply custom retry condition if provided
        if self.config.retry_condition:
            try:
                return self.config.retry_condition(exception, attempt_num)
            except Exception as e:
                logger.error(f"Error in retry condition: {e}")
                return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get retry statistics."""
        with self._lock:
            stats = self.retry_stats.copy()

        # Calculate derived metrics
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            stats['average_attempts_per_operation'] = stats['total_attempts'] / stats['total_operations']
            stats['retry_success_rate'] = (
                stats['operations_succeeded_on_retry'] / stats['successful_operations']
                if stats['successful_operations'] > 0 else 0.0
            )
        else:
            stats['success_rate'] = 0.0
            stats['average_attempts_per_operation'] = 0.0
            stats['retry_success_rate'] = 0.0

        return stats

    def reset_stats(self):
        """Reset retry statistics."""
        with self._lock:
            self.retry_stats = {
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'total_attempts': 0,
                'operations_succeeded_on_retry': 0
            }


class RetryRegistry:
    """Registry for managing multiple retry managers."""

    def __init__(self):
        self.retry_managers: dict[str, RetryManager] = {}
        self._lock = threading.Lock()

    def register(self, name: str, config: RetryConfig | None = None) -> RetryManager:
        """Register a retry manager."""
        with self._lock:
            if name not in self.retry_managers:
                self.retry_managers[name] = RetryManager(config)
            return self.retry_managers[name]

    def get(self, name: str) -> RetryManager | None:
        """Get retry manager by name."""
        return self.retry_managers.get(name)

    def unregister(self, name: str) -> bool:
        """Unregister retry manager."""
        with self._lock:
            if name in self.retry_managers:
                del self.retry_managers[name]
                return True
            return False

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all retry managers."""
        stats = {}
        for name, manager in self.retry_managers.items():
            stats[name] = manager.get_stats()
        return stats


# Global retry registry
retry_registry = RetryRegistry()


def retry(config: RetryConfig | None = None, manager_name: str = "default"):
    """Decorator for retry functionality."""
    def decorator(func):
        retry_manager = retry_registry.register(manager_name, config)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await retry_manager.async_execute(func, *args, **kwargs)
                if result.success:
                    return result.result
                else:
                    raise result.final_exception
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = retry_manager.execute(func, *args, **kwargs)
                if result.success:
                    return result.result
                else:
                    raise result.final_exception
            return sync_wrapper

    return decorator


def retry_method(config: RetryConfig | None = None, manager_name: str = None):
    """Decorator for retry functionality on class methods."""
    def decorator(func):
        def get_manager_name(*args, **kwargs):
            if manager_name:
                return manager_name
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                method_name = func.__name__
                return f"{class_name}.{method_name}"
            return "default"

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                mgr_name = get_manager_name(*args, **kwargs)
                retry_manager = retry_registry.register(mgr_name, config)
                result = await retry_manager.async_execute(func, *args, **kwargs)
                if result.success:
                    return result.result
                else:
                    raise result.final_exception
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                mgr_name = get_manager_name(*args, **kwargs)
                retry_manager = retry_registry.register(mgr_name, config)
                result = retry_manager.execute(func, *args, **kwargs)
                if result.success:
                    return result.result
                else:
                    raise result.final_exception
            return sync_wrapper

    return decorator


# Predefined common retry configurations
class CommonRetryConfigs:
    """Common retry configurations for different scenarios."""

    @staticmethod
    def network_request() -> RetryConfig:
        """Configuration for network requests."""
        return RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=JitterType.EQUAL,
            retryable_exceptions=[
                ConnectionError, TimeoutError, OSError
            ],
            operation_timeout=30.0,
            total_timeout=180.0
        )

    @staticmethod
    def database_operation() -> RetryConfig:
        """Configuration for database operations."""
        return RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=JitterType.EQUAL,
            retryable_exceptions=[
                ConnectionError, TimeoutError
            ],
            operation_timeout=10.0,
            total_timeout=60.0
        )

    @staticmethod
    def mpc_computation() -> RetryConfig:
        """Configuration for MPC computations."""
        return RetryConfig(
            max_attempts=3,
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=JitterType.DECORRELATED,
            operation_timeout=300.0,  # 5 minutes per attempt
            total_timeout=900.0      # 15 minutes total
        )

    @staticmethod
    def file_operation() -> RetryConfig:
        """Configuration for file operations."""
        return RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=5.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=JitterType.FULL,
            retryable_exceptions=[
                OSError, IOError, PermissionError
            ],
            operation_timeout=30.0
        )
