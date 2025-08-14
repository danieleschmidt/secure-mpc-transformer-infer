"""
Autonomous Resilience Framework - Generation 2 Implementation

Comprehensive error handling, recovery mechanisms, and system resilience
for autonomous SDLC execution with defensive security focus.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from functools import wraps
import threading

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures in the system"""
    NETWORK_ERROR = "network_error"
    COMPUTATION_ERROR = "computation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_VIOLATION = "security_violation"
    TIMEOUT_ERROR = "timeout_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types"""
    RETRY_EXPONENTIAL = "retry_exponential"
    RETRY_LINEAR = "retry_linear"
    FALLBACK_MODE = "fallback_mode"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    RESTART_COMPONENT = "restart_component"
    ESCALATE_HUMAN = "escalate_human"


@dataclass
class FailureRecord:
    """Record of a system failure"""
    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_strategy: Optional[RecoveryStrategy] = None


@dataclass
class ResilienceConfig:
    """Configuration for resilience framework"""
    max_retry_attempts: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    timeout_seconds: float = 300.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_graceful_degradation: bool = True
    enable_automatic_recovery: bool = True
    failure_escalation_threshold: int = 10


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise RuntimeError("Circuit breaker is OPEN - calls blocked")
        
        try:
            result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker OPEN due to {self.failure_count} failures")
            
            raise


class AutonomousResilienceManager:
    """
    Comprehensive resilience management for autonomous systems.
    
    Handles failure detection, classification, recovery strategies,
    and system adaptation with defensive security focus.
    """
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.failure_history: List[FailureRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[FailureType, RecoveryStrategy] = {
            FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY_EXPONENTIAL,
            FailureType.COMPUTATION_ERROR: RecoveryStrategy.RETRY_LINEAR,
            FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureType.SECURITY_VIOLATION: RecoveryStrategy.CIRCUIT_BREAKER,
            FailureType.TIMEOUT_ERROR: RecoveryStrategy.RETRY_EXPONENTIAL,
            FailureType.DEPENDENCY_FAILURE: RecoveryStrategy.FALLBACK_MODE,
            FailureType.CONFIGURATION_ERROR: RecoveryStrategy.RESTART_COMPONENT,
            FailureType.UNKNOWN_ERROR: RecoveryStrategy.ESCALATE_HUMAN
        }
        
        # Component health tracking
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.degraded_components: Set[str] = set()
        
        # Performance metrics
        self.recovery_metrics = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "avg_recovery_time": 0.0
        }
        
        logger.info("AutonomousResilienceManager initialized")
    
    def register_component(self, component_name: str, 
                          health_check: Optional[Callable] = None) -> None:
        """Register a component for resilience monitoring"""
        self.component_health[component_name] = {
            "status": "healthy",
            "last_check": time.time(),
            "failure_count": 0,
            "health_check": health_check
        }
        
        # Create circuit breaker for component
        self.circuit_breakers[component_name] = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout
        )
        
        logger.debug(f"Registered component: {component_name}")
    
    def classify_failure(self, exception: Exception, context: Dict[str, Any]) -> FailureType:
        """Classify failure type based on exception and context"""
        error_msg = str(exception).lower()
        
        if "network" in error_msg or "connection" in error_msg:
            return FailureType.NETWORK_ERROR
        elif "timeout" in error_msg:
            return FailureType.TIMEOUT_ERROR
        elif "memory" in error_msg or "resource" in error_msg:
            return FailureType.RESOURCE_EXHAUSTION
        elif "security" in error_msg or "unauthorized" in error_msg:
            return FailureType.SECURITY_VIOLATION
        elif "dependency" in error_msg or "import" in error_msg:
            return FailureType.DEPENDENCY_FAILURE
        elif "config" in error_msg or "configuration" in error_msg:
            return FailureType.CONFIGURATION_ERROR
        elif isinstance(exception, (ValueError, TypeError, AttributeError)):
            return FailureType.COMPUTATION_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    async def handle_failure(self, exception: Exception, component: str, 
                           context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Handle a failure with appropriate recovery strategy.
        
        Returns recovered result or raises if recovery fails.
        """
        context = context or {}
        failure_type = self.classify_failure(exception, context)
        
        # Record failure
        failure_record = FailureRecord(
            timestamp=time.time(),
            failure_type=failure_type,
            component=component,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        self.failure_history.append(failure_record)
        self.recovery_metrics["total_failures"] += 1
        
        # Update component health
        if component in self.component_health:
            self.component_health[component]["failure_count"] += 1
            self.component_health[component]["status"] = "degraded"
        
        logger.warning(f"Failure detected in {component}: {failure_type.value} - {exception}")
        
        # Get recovery strategy
        strategy = self.recovery_strategies.get(failure_type, RecoveryStrategy.ESCALATE_HUMAN)
        failure_record.resolution_strategy = strategy
        
        # Execute recovery
        try:
            recovery_start = time.time()
            result = await self._execute_recovery_strategy(
                strategy, exception, component, context, failure_record
            )
            
            recovery_time = time.time() - recovery_start
            self._update_recovery_metrics(True, recovery_time)
            
            failure_record.resolved = True
            logger.info(f"Recovery successful for {component} using {strategy.value}")
            
            return result
            
        except Exception as recovery_error:
            self._update_recovery_metrics(False, 0)
            logger.error(f"Recovery failed for {component}: {recovery_error}")
            
            # Escalate if multiple recovery attempts fail
            if failure_record.recovery_attempts >= self.config.max_retry_attempts:
                await self._escalate_failure(failure_record)
            
            raise recovery_error
    
    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy, 
                                       original_exception: Exception,
                                       component: str, context: Dict[str, Any],
                                       failure_record: FailureRecord) -> Any:
        """Execute specific recovery strategy"""
        
        if strategy == RecoveryStrategy.RETRY_EXPONENTIAL:
            return await self._retry_with_exponential_backoff(
                context.get("retry_function"), failure_record
            )
        
        elif strategy == RecoveryStrategy.RETRY_LINEAR:
            return await self._retry_with_linear_backoff(
                context.get("retry_function"), failure_record
            )
        
        elif strategy == RecoveryStrategy.FALLBACK_MODE:
            return await self._activate_fallback_mode(component, context)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._enable_graceful_degradation(component, context)
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._activate_circuit_breaker(component, context)
        
        elif strategy == RecoveryStrategy.RESTART_COMPONENT:
            return await self._restart_component(component, context)
        
        elif strategy == RecoveryStrategy.ESCALATE_HUMAN:
            await self._escalate_failure(failure_record)
            raise RuntimeError("Manual intervention required")
        
        else:
            raise ValueError(f"Unknown recovery strategy: {strategy}")
    
    async def _retry_with_exponential_backoff(self, retry_func: Optional[Callable],
                                            failure_record: FailureRecord) -> Any:
        """Retry with exponential backoff"""
        if not retry_func:
            raise ValueError("No retry function provided")
        
        for attempt in range(self.config.max_retry_attempts):
            if attempt > 0:
                delay = min(
                    self.config.base_retry_delay * (2 ** (attempt - 1)),
                    self.config.max_retry_delay
                )
                logger.info(f"Retrying in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
            
            try:
                failure_record.recovery_attempts += 1
                
                if asyncio.iscoroutinefunction(retry_func):
                    return await retry_func()
                else:
                    return retry_func()
                    
            except Exception as e:
                if attempt == self.config.max_retry_attempts - 1:
                    raise e
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        raise RuntimeError("Max retry attempts exceeded")
    
    async def _retry_with_linear_backoff(self, retry_func: Optional[Callable],
                                       failure_record: FailureRecord) -> Any:
        """Retry with linear backoff"""
        if not retry_func:
            raise ValueError("No retry function provided")
        
        for attempt in range(self.config.max_retry_attempts):
            if attempt > 0:
                delay = self.config.base_retry_delay * attempt
                logger.info(f"Retrying in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
            
            try:
                failure_record.recovery_attempts += 1
                
                if asyncio.iscoroutinefunction(retry_func):
                    return await retry_func()
                else:
                    return retry_func()
                    
            except Exception as e:
                if attempt == self.config.max_retry_attempts - 1:
                    raise e
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        raise RuntimeError("Max retry attempts exceeded")
    
    async def _activate_fallback_mode(self, component: str, 
                                    context: Dict[str, Any]) -> Any:
        """Activate fallback mode for component"""
        fallback_func = context.get("fallback_function")
        
        if not fallback_func:
            # Provide default fallback behavior
            logger.warning(f"No fallback function for {component}, using default")
            return {"status": "fallback", "component": component, "message": "Using fallback mode"}
        
        logger.info(f"Activating fallback mode for {component}")
        
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func()
        else:
            return fallback_func()
    
    async def _enable_graceful_degradation(self, component: str,
                                         context: Dict[str, Any]) -> Any:
        """Enable graceful degradation for component"""
        self.degraded_components.add(component)
        
        # Reduce functionality while maintaining core operations
        degraded_func = context.get("degraded_function")
        
        if degraded_func:
            logger.info(f"Enabling graceful degradation for {component}")
            
            if asyncio.iscoroutinefunction(degraded_func):
                return await degraded_func()
            else:
                return degraded_func()
        else:
            # Default degraded mode
            return {
                "status": "degraded",
                "component": component,
                "message": "Operating in degraded mode"
            }
    
    async def _activate_circuit_breaker(self, component: str,
                                      context: Dict[str, Any]) -> Any:
        """Activate circuit breaker for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        
        breaker = self.circuit_breakers[component]
        breaker.state = "OPEN"
        
        logger.warning(f"Circuit breaker activated for {component}")
        
        # Return safe fallback
        return {
            "status": "circuit_breaker_open",
            "component": component,
            "message": "Component temporarily disabled for safety"
        }
    
    async def _restart_component(self, component: str,
                               context: Dict[str, Any]) -> Any:
        """Restart component"""
        restart_func = context.get("restart_function")
        
        if restart_func:
            logger.info(f"Restarting component: {component}")
            
            # Mark component as restarting
            if component in self.component_health:
                self.component_health[component]["status"] = "restarting"
            
            if asyncio.iscoroutinefunction(restart_func):
                result = await restart_func()
            else:
                result = restart_func()
            
            # Reset component health
            if component in self.component_health:
                self.component_health[component]["status"] = "healthy"
                self.component_health[component]["failure_count"] = 0
            
            return result
        else:
            raise ValueError(f"No restart function provided for {component}")
    
    async def _escalate_failure(self, failure_record: FailureRecord) -> None:
        """Escalate failure for human intervention"""
        logger.critical(f"Escalating failure in {failure_record.component}")
        
        # In a real system, this would:
        # - Send alerts to operations team
        # - Create incident tickets
        # - Trigger emergency procedures
        
        escalation_data = {
            "timestamp": failure_record.timestamp,
            "component": failure_record.component,
            "failure_type": failure_record.failure_type.value,
            "error_message": failure_record.error_message,
            "recovery_attempts": failure_record.recovery_attempts,
            "context": failure_record.context
        }
        
        logger.critical(f"ESCALATION REQUIRED: {json.dumps(escalation_data, indent=2)}")
    
    def _update_recovery_metrics(self, success: bool, recovery_time: float) -> None:
        """Update recovery performance metrics"""
        if success:
            self.recovery_metrics["successful_recoveries"] += 1
        else:
            self.recovery_metrics["failed_recoveries"] += 1
        
        # Update average recovery time
        total_recoveries = (
            self.recovery_metrics["successful_recoveries"] +
            self.recovery_metrics["failed_recoveries"]
        )
        
        if total_recoveries > 0:
            current_avg = self.recovery_metrics["avg_recovery_time"]
            self.recovery_metrics["avg_recovery_time"] = (
                (current_avg * (total_recoveries - 1) + recovery_time) / total_recoveries
            )
    
    async def health_check_all_components(self) -> Dict[str, Any]:
        """Perform health checks on all registered components"""
        health_status = {}
        
        for component, health_info in self.component_health.items():
            try:
                if health_info.get("health_check"):
                    health_func = health_info["health_check"]
                    
                    if asyncio.iscoroutinefunction(health_func):
                        is_healthy = await health_func()
                    else:
                        is_healthy = health_func()
                    
                    status = "healthy" if is_healthy else "unhealthy"
                else:
                    # Default health check based on failure count
                    failure_count = health_info.get("failure_count", 0)
                    status = "healthy" if failure_count < 3 else "degraded"
                
                health_status[component] = {
                    "status": status,
                    "failure_count": health_info.get("failure_count", 0),
                    "last_check": time.time()
                }
                
                # Update stored health info
                self.component_health[component]["status"] = status
                self.component_health[component]["last_check"] = time.time()
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                health_status[component] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": time.time()
                }
        
        return health_status
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        recent_failures = [
            f for f in self.failure_history
            if time.time() - f.timestamp < 3600  # Last hour
        ]
        
        failure_by_type = {}
        for failure in recent_failures:
            failure_type = failure.failure_type.value
            failure_by_type[failure_type] = failure_by_type.get(failure_type, 0) + 1
        
        return {
            "recovery_metrics": self.recovery_metrics.copy(),
            "component_health": {
                name: {
                    "status": info["status"],
                    "failure_count": info["failure_count"]
                }
                for name, info in self.component_health.items()
            },
            "degraded_components": list(self.degraded_components),
            "recent_failures": len(recent_failures),
            "failure_breakdown": failure_by_type,
            "circuit_breaker_status": {
                name: breaker.state
                for name, breaker in self.circuit_breakers.items()
            },
            "total_failures_recorded": len(self.failure_history)
        }


def resilient_execution(component_name: str, resilience_manager: AutonomousResilienceManager,
                       retry_function: Optional[Callable] = None,
                       fallback_function: Optional[Callable] = None):
    """
    Decorator for resilient execution of functions.
    
    Automatically handles failures and applies recovery strategies.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = {
                "retry_function": retry_function or func,
                "fallback_function": fallback_function,
                "args": args,
                "kwargs": kwargs
            }
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                return await resilience_manager.handle_failure(e, component_name, context)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                # For async functions called in sync context
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            else:
                context = {
                    "retry_function": retry_function or func,
                    "fallback_function": fallback_function,
                    "args": args,
                    "kwargs": kwargs
                }
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Convert to async for consistency
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        resilience_manager.handle_failure(e, component_name, context)
                    )
        
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator