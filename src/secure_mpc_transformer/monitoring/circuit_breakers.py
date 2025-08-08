"""Circuit breaker patterns for resilient system operation."""

import time
import threading
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
import functools
import inspect

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is broken, rejecting requests
    HALF_OPEN = "half_open" # Testing if circuit can close


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    SLOW_RESPONSE = "slow_response"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    RATE_LIMIT = "rate_limit"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    
    # Failure thresholds
    failure_threshold: int = 5  # Number of failures to open circuit
    failure_rate_threshold: float = 0.5  # Percentage of failures (0.0-1.0)
    slow_call_threshold: float = 5.0  # Seconds
    slow_call_rate_threshold: float = 0.3  # Percentage of slow calls
    
    # Timing configuration
    timeout: Optional[float] = None  # Request timeout in seconds
    recovery_timeout: float = 60.0  # Time to wait before trying half-open
    half_open_max_calls: int = 3  # Max calls to allow in half-open state
    
    # Sliding window configuration
    sliding_window_size: int = 100  # Number of calls to track
    minimum_calls: int = 10  # Minimum calls before evaluating failure rate
    
    # Monitoring
    record_exceptions: List[type] = field(default_factory=lambda: [Exception])
    ignore_exceptions: List[type] = field(default_factory=list)


@dataclass
class CallResult:
    """Result of a circuit breaker protected call."""
    
    timestamp: float
    duration: float
    success: bool
    failure_type: Optional[FailureType] = None
    exception_type: Optional[str] = None
    response_size: Optional[int] = None


class SlidingWindowMetrics:
    """Sliding window metrics for circuit breaker."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.calls = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def add_call(self, result: CallResult):
        """Add call result to sliding window."""
        with self._lock:
            self.calls.append(result)
    
    def get_metrics(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get current metrics from sliding window."""
        with self._lock:
            calls_list = list(self.calls)
        
        if not calls_list:
            return {
                'total_calls': 0,
                'failure_count': 0,
                'failure_rate': 0.0,
                'slow_call_count': 0,
                'slow_call_rate': 0.0,
                'avg_duration': 0.0,
                'success_rate': 0.0
            }
        
        # Filter by time window if specified
        if time_window:
            current_time = time.time()
            calls_list = [
                call for call in calls_list
                if current_time - call.timestamp <= time_window
            ]
        
        total_calls = len(calls_list)
        if total_calls == 0:
            return {
                'total_calls': 0,
                'failure_count': 0,
                'failure_rate': 0.0,
                'slow_call_count': 0,
                'slow_call_rate': 0.0,
                'avg_duration': 0.0,
                'success_rate': 0.0
            }
        
        failure_count = sum(1 for call in calls_list if not call.success)
        slow_call_count = sum(1 for call in calls_list if call.duration > 5.0)  # Configurable
        success_count = total_calls - failure_count
        
        durations = [call.duration for call in calls_list]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        return {
            'total_calls': total_calls,
            'failure_count': failure_count,
            'failure_rate': failure_count / total_calls,
            'slow_call_count': slow_call_count,
            'slow_call_rate': slow_call_count / total_calls,
            'avg_duration': avg_duration,
            'success_rate': success_count / total_calls
        }


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, state: CircuitState):
        self.circuit_name = circuit_name
        self.state = state
        super().__init__(f"Circuit breaker '{circuit_name}' is {state.value}")


class CircuitBreaker:
    """Implementation of circuit breaker pattern."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.consecutive_failures = 0
        
        # Metrics
        self.metrics = SlidingWindowMetrics(self.config.sliding_window_size)
        self.total_calls = 0
        self.total_failures = 0
        self.state_changes = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        self.on_failure: Optional[Callable[[Exception, CallResult], None]] = None
        self.on_success: Optional[Callable[[CallResult], None]] = None
        
        logger.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute a function with circuit breaker protection."""
        if inspect.iscoroutinefunction(func):
            raise ValueError("Use async_call for coroutine functions")
        
        return self._execute_call(func, args, kwargs)
    
    async def async_call(self, func: Callable, *args, **kwargs):
        """Execute an async function with circuit breaker protection."""
        if not inspect.iscoroutinefunction(func):
            raise ValueError("Use call for regular functions")
        
        return await self._execute_async_call(func, args, kwargs)
    
    def _execute_call(self, func: Callable, args: tuple, kwargs: dict):
        """Execute synchronous call with circuit breaker logic."""
        self._check_circuit_state()
        
        start_time = time.time()
        result = None
        exception = None
        
        try:
            # Apply timeout if configured
            if self.config.timeout:
                result = self._call_with_timeout(func, args, kwargs, self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            # Record successful call
            call_result = CallResult(
                timestamp=start_time,
                duration=time.time() - start_time,
                success=True
            )
            
            self._record_success(call_result)
            return result
            
        except Exception as e:
            exception = e
            call_result = CallResult(
                timestamp=start_time,
                duration=time.time() - start_time,
                success=False,
                failure_type=self._classify_failure(e),
                exception_type=type(e).__name__
            )
            
            self._record_failure(call_result, e)
            raise
    
    async def _execute_async_call(self, func: Callable, args: tuple, kwargs: dict):
        """Execute asynchronous call with circuit breaker logic."""
        self._check_circuit_state()
        
        start_time = time.time()
        result = None
        exception = None
        
        try:
            # Apply timeout if configured
            if self.config.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            # Record successful call
            call_result = CallResult(
                timestamp=start_time,
                duration=time.time() - start_time,
                success=True
            )
            
            self._record_success(call_result)
            return result
            
        except Exception as e:
            exception = e
            call_result = CallResult(
                timestamp=start_time,
                duration=time.time() - start_time,
                success=False,
                failure_type=self._classify_failure(e),
                exception_type=type(e).__name__
            )
            
            self._record_failure(call_result, e)
            raise
    
    def _call_with_timeout(self, func: Callable, args: tuple, kwargs: dict, timeout: float):
        """Execute function with timeout (synchronous)."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function call timed out after {timeout} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel the alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _check_circuit_state(self):
        """Check if circuit allows calls and update state if needed."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerException(self.name, self.state)
            
            elif self.state == CircuitState.HALF_OPEN:
                # Check if we've exceeded max calls in half-open
                if self.half_open_calls >= self.config.half_open_max_calls:
                    # Too many calls, evaluate if we should close or open
                    metrics = self.metrics.get_metrics()
                    if self._should_open_circuit(metrics):
                        self._transition_to_open()
                        raise CircuitBreakerException(self.name, self.state)
                    else:
                        self._transition_to_closed()
    
    def _record_success(self, call_result: CallResult):
        """Record successful call."""
        with self._lock:
            self.total_calls += 1
            self.metrics.add_call(call_result)
            self.consecutive_failures = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                # Check if we should close the circuit
                metrics = self.metrics.get_metrics()
                if (self.half_open_calls >= self.config.half_open_max_calls and
                    not self._should_open_circuit(metrics)):
                    self._transition_to_closed()
        
        if self.on_success:
            try:
                self.on_success(call_result)
            except Exception as e:
                logger.error(f"Error in success callback: {e}")
    
    def _record_failure(self, call_result: CallResult, exception: Exception):
        """Record failed call."""
        with self._lock:
            self.total_calls += 1
            self.total_failures += 1
            self.metrics.add_call(call_result)
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
            
            # Check if we should open the circuit
            metrics = self.metrics.get_metrics()
            if self.state == CircuitState.CLOSED and self._should_open_circuit(metrics):
                self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
        
        if self.on_failure:
            try:
                self.on_failure(exception, call_result)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")
    
    def _should_open_circuit(self, metrics: Dict[str, Any]) -> bool:
        """Determine if circuit should be opened based on metrics."""
        if metrics['total_calls'] < self.config.minimum_calls:
            return False
        
        # Check failure rate threshold
        if metrics['failure_rate'] >= self.config.failure_rate_threshold:
            return True
        
        # Check consecutive failures
        if self.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check slow call rate
        if metrics['slow_call_rate'] >= self.config.slow_call_rate_threshold:
            return True
        
        return False
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure."""
        if isinstance(exception, TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif "rate limit" in str(exception).lower():
            return FailureType.RATE_LIMIT
        elif "resource" in str(exception).lower():
            return FailureType.RESOURCE_EXHAUSTION
        else:
            return FailureType.EXCEPTION
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self.state_changes += 1
        
        logger.warning(f"Circuit breaker '{self.name}' opened (failures: {self.consecutive_failures})")
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.state_changes += 1
        
        logger.info(f"Circuit breaker '{self.name}' half-opened for testing")
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        self.consecutive_failures = 0
        self.state_changes += 1
        
        logger.info(f"Circuit breaker '{self.name}' closed (recovered)")
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def force_open(self):
        """Manually open the circuit."""
        with self._lock:
            if self.state != CircuitState.OPEN:
                self._transition_to_open()
    
    def force_close(self):
        """Manually close the circuit."""
        with self._lock:
            if self.state != CircuitState.CLOSED:
                self._transition_to_closed()
    
    def force_half_open(self):
        """Manually set circuit to half-open."""
        with self._lock:
            if self.state != CircuitState.HALF_OPEN:
                self._transition_to_half_open()
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            window_metrics = self.metrics.get_metrics()
            
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'consecutive_failures': self.consecutive_failures,
                'state_changes': self.state_changes,
                'last_failure_time': self.last_failure_time,
                'half_open_calls': self.half_open_calls,
                **window_metrics
            }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.half_open_calls = 0
            self.last_failure_time = 0.0
            
            # Clear metrics
            self.metrics = SlidingWindowMetrics(self.config.sliding_window_size)
            
            logger.info(f"Circuit breaker '{self.name}' reset")
            
            if self.on_state_change and old_state != CircuitState.CLOSED:
                try:
                    self.on_state_change(old_state, self.state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
            
            circuit_breaker = CircuitBreaker(name, config)
            self.circuit_breakers[name] = circuit_breaker
            return circuit_breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def unregister(self, name: str) -> bool:
        """Unregister circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                return True
            return False
    
    def list_all(self) -> List[str]:
        """List all registered circuit breaker names."""
        return list(self.circuit_breakers.keys())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        metrics = {}
        for name, cb in self.circuit_breakers.items():
            metrics[name] = cb.get_metrics()
        return metrics
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        cb = circuit_breaker_registry.register(name, config)
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await cb.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


def circuit_breaker_method(name: str = None, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection of class methods."""
    def decorator(func):
        def get_circuit_name(*args, **kwargs):
            if name:
                return name
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                method_name = func.__name__
                return f"{class_name}.{method_name}"
            return func.__name__
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                circuit_name = get_circuit_name(*args, **kwargs)
                cb = circuit_breaker_registry.register(circuit_name, config)
                return await cb.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                circuit_name = get_circuit_name(*args, **kwargs)
                cb = circuit_breaker_registry.register(circuit_name, config)
                return cb.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


class BulkheadCircuitBreaker(CircuitBreaker):
    """Circuit breaker with bulkhead pattern for resource isolation."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None,
                 max_concurrent_calls: int = 10):
        super().__init__(name, config)
        self.max_concurrent_calls = max_concurrent_calls
        self.current_calls = 0
        self.semaphore = threading.Semaphore(max_concurrent_calls)
        self.async_semaphore = asyncio.Semaphore(max_concurrent_calls)
    
    def _execute_call(self, func: Callable, args: tuple, kwargs: dict):
        """Execute call with bulkhead protection."""
        if not self.semaphore.acquire(blocking=False):
            raise CircuitBreakerException(self.name, CircuitState.OPEN)
        
        try:
            with self._lock:
                self.current_calls += 1
            
            return super()._execute_call(func, args, kwargs)
        finally:
            with self._lock:
                self.current_calls -= 1
            self.semaphore.release()
    
    async def _execute_async_call(self, func: Callable, args: tuple, kwargs: dict):
        """Execute async call with bulkhead protection."""
        try:
            await asyncio.wait_for(
                self.async_semaphore.acquire(), timeout=0.1
            )
        except asyncio.TimeoutError:
            raise CircuitBreakerException(self.name, CircuitState.OPEN)
        
        try:
            with self._lock:
                self.current_calls += 1
            
            return await super()._execute_async_call(func, args, kwargs)
        finally:
            with self._lock:
                self.current_calls -= 1
            self.async_semaphore.release()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including bulkhead information."""
        metrics = super().get_metrics()
        metrics.update({
            'max_concurrent_calls': self.max_concurrent_calls,
            'current_calls': self.current_calls,
            'available_permits': self.semaphore._value
        })
        return metrics