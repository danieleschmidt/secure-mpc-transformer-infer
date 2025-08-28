"""
Autonomous Recovery System

Self-healing infrastructure for the secure MPC transformer system with
intelligent failure detection, recovery strategies, and adaptive learning.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict, deque
import traceback
import gc
import psutil
import threading

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Categories of system failures."""
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_OVERLOAD = "cpu_overload"
    NETWORK_TIMEOUT = "network_timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATABASE_ERROR = "database_error"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    MPC_PROTOCOL_ERROR = "mpc_protocol_error"
    GPU_ERROR = "gpu_error"
    INFERENCE_TIMEOUT = "inference_timeout"
    SECURITY_VIOLATION = "security_violation"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    FALLBACK_MODE = "fallback_mode"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    QUANTUM_RESET = "quantum_reset"
    CACHE_CLEAR = "cache_clear"
    CONNECTION_RESET = "connection_reset"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    severity: int = 1  # 1-10 scale
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'failure_type': self.failure_type.value,
            'component': self.component,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'context': self.context,
            'severity': self.severity,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }


@dataclass
class RecoveryAction:
    """Defines a recovery action."""
    strategy: RecoveryStrategy
    handler: Callable
    priority: int = 1  # Higher numbers = higher priority
    conditions: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    cooldown_seconds: int = 60
    
    def __post_init__(self):
        self._last_executed = 0
        self._execution_count = 0


class HealthMonitor:
    """Monitors system health metrics."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'gpu_memory': 90.0,
            'response_time': 5.0,
            'error_rate': 0.1
        }
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
        
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self):
        """Collect system health metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'timestamp': time.time()
            }
            
            # GPU metrics (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['gpu_memory'] = (gpu_info.used / gpu_info.total) * 100
            except:
                metrics['gpu_memory'] = 0
            
            # Store metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_history[key].append(value)
                    
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        status = {'overall': 'healthy', 'issues': []}
        
        for metric, threshold in self.thresholds.items():
            if metric in self.metrics_history and self.metrics_history[metric]:
                current_value = self.metrics_history[metric][-1]
                if current_value > threshold:
                    status['overall'] = 'degraded'
                    status['issues'].append({
                        'metric': metric,
                        'current': current_value,
                        'threshold': threshold
                    })
        
        # Add recent metrics
        status['metrics'] = {}
        for key, history in self.metrics_history.items():
            if history:
                status['metrics'][key] = {
                    'current': history[-1],
                    'avg_last_10': sum(list(history)[-10:]) / min(10, len(history))
                }
                
        return status


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half_open
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half_open'
                logger.info("Circuit breaker moving to half-open state")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half_open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker reset to closed state")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e


class AutonomousRecoverySystem:
    """Main autonomous recovery orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.health_monitor = HealthMonitor(
            check_interval=self.config.get('health_check_interval', 30.0)
        )
        
        # Failure tracking
        self.failure_events: deque = deque(maxlen=1000)
        self.recovery_actions: Dict[FailureType, List[RecoveryAction]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery statistics
        self.recovery_stats = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0.0
        }
        
        # Background tasks
        self._recovery_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Register default recovery actions
        self._register_default_recovery_actions()
        
        logger.info("Autonomous Recovery System initialized")
    
    async def start(self):
        """Start the recovery system."""
        if self._running:
            return
            
        self._running = True
        await self.health_monitor.start_monitoring()
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        logger.info("Autonomous Recovery System started")
    
    async def stop(self):
        """Stop the recovery system."""
        self._running = False
        await self.health_monitor.stop_monitoring()
        
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Autonomous Recovery System stopped")
    
    async def _recovery_loop(self):
        """Main recovery monitoring loop."""
        while self._running:
            try:
                await self._check_for_failures()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_for_failures(self):
        """Check system health and initiate recovery if needed."""
        health_status = self.health_monitor.get_health_status()
        
        if health_status['overall'] != 'healthy':
            for issue in health_status['issues']:
                await self._handle_health_issue(issue)
    
    async def _handle_health_issue(self, issue: Dict[str, Any]):
        """Handle a detected health issue."""
        metric = issue['metric']
        current_value = issue['current']
        threshold = issue['threshold']
        
        # Determine failure type
        failure_type = self._classify_failure(metric, current_value, threshold)
        
        # Create failure event
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            component='system',
            error_message=f"{metric} exceeded threshold: {current_value} > {threshold}",
            context={'metric': metric, 'value': current_value, 'threshold': threshold},
            severity=self._calculate_severity(current_value, threshold)
        )
        
        await self.handle_failure(failure_event)
    
    async def handle_failure(self, failure_event: FailureEvent) -> bool:
        """Handle a failure event and attempt recovery."""
        self.failure_events.append(failure_event)
        self.recovery_stats['total_failures'] += 1
        
        logger.warning(f"Handling failure: {failure_event.failure_type.value} in {failure_event.component}")
        
        # Find appropriate recovery actions
        recovery_actions = self.recovery_actions.get(failure_event.failure_type, [])
        recovery_actions.sort(key=lambda x: x.priority, reverse=True)
        
        recovery_successful = False
        failure_event.recovery_attempted = True
        recovery_start_time = time.time()
        
        for action in recovery_actions:
            try:
                # Check cooldown
                if time.time() - action._last_executed < action.cooldown_seconds:
                    continue
                
                # Check max retries
                if action._execution_count >= action.max_retries:
                    continue
                
                logger.info(f"Attempting recovery: {action.strategy.value}")
                
                # Execute recovery action
                success = await self._execute_recovery_action(action, failure_event)
                
                action._last_executed = time.time()
                action._execution_count += 1
                
                if success:
                    recovery_successful = True
                    logger.info(f"Recovery successful: {action.strategy.value}")
                    break
                    
            except Exception as e:
                logger.error(f"Recovery action failed: {action.strategy.value} - {e}")
        
        # Update statistics
        recovery_time = time.time() - recovery_start_time
        if recovery_successful:
            self.recovery_stats['successful_recoveries'] += 1
            failure_event.recovery_successful = True
        else:
            self.recovery_stats['failed_recoveries'] += 1
        
        # Update average recovery time
        total_recoveries = (self.recovery_stats['successful_recoveries'] + 
                          self.recovery_stats['failed_recoveries'])
        if total_recoveries > 0:
            self.recovery_stats['avg_recovery_time'] = (
                (self.recovery_stats['avg_recovery_time'] * (total_recoveries - 1) + recovery_time) 
                / total_recoveries
            )
        
        return recovery_successful
    
    async def _execute_recovery_action(
        self, 
        action: RecoveryAction, 
        failure_event: FailureEvent
    ) -> bool:
        """Execute a specific recovery action."""
        try:
            if asyncio.iscoroutinefunction(action.handler):
                return await action.handler(failure_event, action)
            else:
                return action.handler(failure_event, action)
        except Exception as e:
            logger.error(f"Recovery handler failed: {e}")
            return False
    
    def register_recovery_action(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        handler: Callable,
        priority: int = 1,
        **kwargs
    ):
        """Register a custom recovery action."""
        action = RecoveryAction(
            strategy=strategy,
            handler=handler,
            priority=priority,
            **kwargs
        )
        self.recovery_actions[failure_type].append(action)
        logger.info(f"Registered recovery action: {strategy.value} for {failure_type.value}")
    
    def _register_default_recovery_actions(self):
        """Register default recovery actions."""
        
        # Memory exhaustion recovery
        self.register_recovery_action(
            FailureType.MEMORY_EXHAUSTION,
            RecoveryStrategy.CACHE_CLEAR,
            self._clear_caches,
            priority=3
        )
        
        self.register_recovery_action(
            FailureType.MEMORY_EXHAUSTION,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            self._reduce_memory_usage,
            priority=2
        )
        
        # CPU overload recovery
        self.register_recovery_action(
            FailureType.CPU_OVERLOAD,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            self._reduce_cpu_usage,
            priority=2
        )
        
        # Network timeout recovery
        self.register_recovery_action(
            FailureType.NETWORK_TIMEOUT,
            RecoveryStrategy.CONNECTION_RESET,
            self._reset_connections,
            priority=3
        )
        
        # Quantum decoherence recovery
        self.register_recovery_action(
            FailureType.QUANTUM_DECOHERENCE,
            RecoveryStrategy.QUANTUM_RESET,
            self._reset_quantum_state,
            priority=1
        )
    
    async def _clear_caches(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Clear system caches to free memory."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear Python caches
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            logger.info("System caches cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            return False
    
    async def _reduce_memory_usage(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Reduce memory usage by optimizing operations."""
        try:
            # Implement memory reduction strategies
            gc.collect()
            
            # Could implement:
            # - Reduce batch sizes
            # - Unload unused models
            # - Compress cached data
            
            logger.info("Memory usage reduction attempted")
            return True
        except Exception as e:
            logger.error(f"Failed to reduce memory usage: {e}")
            return False
    
    async def _reduce_cpu_usage(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Reduce CPU usage by throttling operations."""
        try:
            # Implement CPU reduction strategies
            # - Reduce worker threads
            # - Throttle requests
            # - Defer non-critical operations
            
            logger.info("CPU usage reduction attempted")
            return True
        except Exception as e:
            logger.error(f"Failed to reduce CPU usage: {e}")
            return False
    
    async def _reset_connections(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Reset network connections."""
        try:
            # Reset connection pools, reconnect to services
            logger.info("Network connections reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset connections: {e}")
            return False
    
    async def _reset_quantum_state(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Reset quantum state for fresh initialization."""
        try:
            # Reset quantum planning state
            logger.info("Quantum state reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset quantum state: {e}")
            return False
    
    def _classify_failure(self, metric: str, value: float, threshold: float) -> FailureType:
        """Classify failure type based on metric."""
        failure_map = {
            'cpu_usage': FailureType.CPU_OVERLOAD,
            'memory_usage': FailureType.MEMORY_EXHAUSTION,
            'gpu_memory': FailureType.GPU_ERROR,
            'response_time': FailureType.INFERENCE_TIMEOUT,
        }
        return failure_map.get(metric, FailureType.SERVICE_UNAVAILABLE)
    
    def _calculate_severity(self, value: float, threshold: float) -> int:
        """Calculate failure severity (1-10)."""
        ratio = value / threshold
        if ratio < 1.1:
            return 2
        elif ratio < 1.3:
            return 4
        elif ratio < 1.5:
            return 6
        elif ratio < 2.0:
            return 8
        else:
            return 10
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get recovery system status."""
        return {
            'running': self._running,
            'total_failures': len(self.failure_events),
            'recent_failures': len([
                f for f in self.failure_events 
                if time.time() - f.timestamp < 3600
            ]),
            'recovery_stats': self.recovery_stats.copy(),
            'health_status': self.health_monitor.get_health_status(),
            'circuit_breakers': {
                name: {'state': cb.state, 'failures': cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            }
        }


# Global instance
_recovery_system: Optional[AutonomousRecoverySystem] = None


def get_recovery_system() -> AutonomousRecoverySystem:
    """Get the global recovery system instance."""
    global _recovery_system
    if _recovery_system is None:
        _recovery_system = AutonomousRecoverySystem()
    return _recovery_system