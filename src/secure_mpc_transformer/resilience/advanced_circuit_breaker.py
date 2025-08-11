"""
Advanced Circuit Breaker with ML-based Failure Prediction
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import statistics
import logging

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class FailureMetrics:
    failure_rate: float
    average_response_time: float
    consecutive_failures: int
    total_requests: int
    success_rate: float
    error_types: Dict[str, int]

@dataclass
class PredictionMetrics:
    failure_probability: float
    confidence_score: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    predicted_recovery_time: Optional[float]

class AdvancedCircuitBreaker:
    """
    ML-enhanced circuit breaker with predictive failure detection
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 5,
        window_size: int = 100,
        min_requests: int = 10,
        enable_ml_prediction: bool = True
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.window_size = window_size
        self.min_requests = min_requests
        self.enable_ml_prediction = enable_ml_prediction
        
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time = 0
        self.consecutive_failures = 0
        self.half_open_calls = 0
        
        # Metrics tracking
        self.request_history = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.error_types = {}
        
        # ML prediction components
        self.ml_model = None
        self.feature_history = deque(maxlen=50)
        
        if enable_ml_prediction:
            self._initialize_ml_model()
    
    def _initialize_ml_model(self):
        """Initialize simple ML model for failure prediction"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            self.ml_model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.scaler = StandardScaler()
            self._ml_initialized = False
            
        except ImportError:
            logger.warning("scikit-learn not available, disabling ML prediction")
            self.enable_ml_prediction = False
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker {self.name} is OPEN"
                )
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.OPEN
                self.last_failure_time = time.time()
                raise CircuitBreakerOpenException(
                    f"Circuit breaker {self.name} failed recovery attempt"
                )
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_failure(e, response_time)
            raise
    
    def _record_success(self, response_time: float):
        """Record successful request"""
        self.request_history.append({
            'timestamp': time.time(),
            'success': True,
            'response_time': response_time,
            'error_type': None
        })
        self.response_times.append(response_time)
        self.consecutive_failures = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
        
        self._update_ml_features()
    
    def _record_failure(self, error: Exception, response_time: float):
        """Record failed request"""
        error_type = type(error).__name__
        
        self.request_history.append({
            'timestamp': time.time(),
            'success': False,
            'response_time': response_time,
            'error_type': error_type
        })
        
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} failed during recovery")
        elif self._should_open_circuit():
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened due to failures")
        
        self._update_ml_features()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on failure metrics"""
        if len(self.request_history) < self.min_requests:
            return False
        
        metrics = self.get_current_metrics()
        
        # Traditional threshold check
        if metrics.failure_rate >= self.failure_threshold:
            return True
        
        # ML-based prediction check
        if self.enable_ml_prediction:
            prediction = self._predict_failure()
            if prediction and prediction.failure_probability > 0.8:
                logger.info(
                    f"ML prediction suggests high failure probability: "
                    f"{prediction.failure_probability:.2f}"
                )
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        time_since_failure = time.time() - self.last_failure_time
        
        # ML-based recovery prediction
        if self.enable_ml_prediction:
            prediction = self._predict_failure()
            if prediction and prediction.predicted_recovery_time:
                return time_since_failure >= prediction.predicted_recovery_time
        
        return time_since_failure >= self.recovery_timeout
    
    def get_current_metrics(self) -> FailureMetrics:
        """Get current failure metrics"""
        if not self.request_history:
            return FailureMetrics(
                failure_rate=0.0,
                average_response_time=0.0,
                consecutive_failures=0,
                total_requests=0,
                success_rate=1.0,
                error_types={}
            )
        
        total_requests = len(self.request_history)
        failures = sum(1 for req in self.request_history if not req['success'])
        successes = total_requests - failures
        
        failure_rate = failures / total_requests if total_requests > 0 else 0.0
        success_rate = successes / total_requests if total_requests > 0 else 1.0
        
        avg_response_time = (
            statistics.mean(self.response_times) 
            if self.response_times else 0.0
        )
        
        return FailureMetrics(
            failure_rate=failure_rate,
            average_response_time=avg_response_time,
            consecutive_failures=self.consecutive_failures,
            total_requests=total_requests,
            success_rate=success_rate,
            error_types=self.error_types.copy()
        )
    
    def _update_ml_features(self):
        """Update ML model features"""
        if not self.enable_ml_prediction:
            return
        
        metrics = self.get_current_metrics()
        
        features = [
            metrics.failure_rate,
            metrics.average_response_time,
            metrics.consecutive_failures,
            len(self.error_types),
            time.time() % 86400,  # Time of day feature
        ]
        
        self.feature_history.append(features)
        
        # Train model if we have enough data
        if len(self.feature_history) >= 20 and not self._ml_initialized:
            self._train_ml_model()
    
    def _train_ml_model(self):
        """Train ML model with historical data"""
        try:
            import numpy as np
            
            if len(self.feature_history) < 10:
                return
            
            X = np.array(list(self.feature_history))
            X_scaled = self.scaler.fit_transform(X)
            
            self.ml_model.fit(X_scaled)
            self._ml_initialized = True
            
            logger.info(f"ML model trained for circuit breaker {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
    
    def _predict_failure(self) -> Optional[PredictionMetrics]:
        """Predict failure probability using ML model"""
        if not self.enable_ml_prediction or not self._ml_initialized:
            return None
        
        try:
            import numpy as np
            
            if len(self.feature_history) < 5:
                return None
            
            # Get recent features
            recent_features = list(self.feature_history)[-5:]
            X = np.array(recent_features)
            X_scaled = self.scaler.transform(X)
            
            # Predict anomaly score
            anomaly_scores = self.ml_model.decision_function(X_scaled)
            avg_anomaly_score = np.mean(anomaly_scores)
            
            # Convert anomaly score to failure probability
            failure_probability = max(0, min(1, (1 - avg_anomaly_score) / 2))
            
            # Calculate trend
            recent_failure_rates = [
                sum(1 for req in list(self.request_history)[-10:] if not req['success']) / 10
                if len(self.request_history) >= 10 else 0
            ]
            
            trend_direction = "stable"
            if len(recent_failure_rates) >= 2:
                if recent_failure_rates[-1] > recent_failure_rates[0]:
                    trend_direction = "increasing"
                elif recent_failure_rates[-1] < recent_failure_rates[0]:
                    trend_direction = "decreasing"
            
            # Predict recovery time
            predicted_recovery_time = None
            if failure_probability > 0.5:
                predicted_recovery_time = self.recovery_timeout * (1 + failure_probability)
            
            return PredictionMetrics(
                failure_probability=failure_probability,
                confidence_score=min(1.0, len(self.feature_history) / 50),
                trend_direction=trend_direction,
                predicted_recovery_time=predicted_recovery_time
            )
            
        except Exception as e:
            logger.error(f"Failed to predict failure: {e}")
            return None
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive state information"""
        metrics = self.get_current_metrics()
        prediction = self._predict_failure()
        
        return {
            'name': self.name,
            'state': self.state.value,
            'metrics': {
                'failure_rate': metrics.failure_rate,
                'success_rate': metrics.success_rate,
                'average_response_time': metrics.average_response_time,
                'consecutive_failures': metrics.consecutive_failures,
                'total_requests': metrics.total_requests,
                'error_types': metrics.error_types
            },
            'prediction': {
                'failure_probability': prediction.failure_probability if prediction else None,
                'confidence_score': prediction.confidence_score if prediction else None,
                'trend_direction': prediction.trend_direction if prediction else None,
                'predicted_recovery_time': prediction.predicted_recovery_time if prediction else None
            } if prediction else None,
            'configuration': {
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
                'window_size': self.window_size,
                'ml_enabled': self.enable_ml_prediction
            }
        }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerManager:
    """Manage multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def register(
        self,
        name: str,
        **circuit_breaker_kwargs
    ) -> AdvancedCircuitBreaker:
        """Register new circuit breaker"""
        async with self._lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
            
            circuit_breaker = AdvancedCircuitBreaker(name, **circuit_breaker_kwargs)
            self.circuit_breakers[name] = circuit_breaker
            
            logger.info(f"Registered circuit breaker: {name}")
            return circuit_breaker
    
    async def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with named circuit breaker"""
        if name not in self.circuit_breakers:
            await self.register(name)
        
        return await self.circuit_breakers[name].call(func, *args, **kwargs)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        return {
            name: cb.get_state_info()
            for name, cb in self.circuit_breakers.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_breakers = len(self.circuit_breakers)
        
        if total_breakers == 0:
            return {
                'total_circuit_breakers': 0,
                'healthy_count': 0,
                'degraded_count': 0,
                'failed_count': 0,
                'overall_health': 'unknown'
            }
        
        healthy_count = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitBreakerState.CLOSED
        )
        
        degraded_count = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitBreakerState.HALF_OPEN
        )
        
        failed_count = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitBreakerState.OPEN
        )
        
        if failed_count > total_breakers * 0.3:
            overall_health = 'critical'
        elif degraded_count + failed_count > total_breakers * 0.1:
            overall_health = 'degraded'
        else:
            overall_health = 'healthy'
        
        return {
            'total_circuit_breakers': total_breakers,
            'healthy_count': healthy_count,
            'degraded_count': degraded_count,
            'failed_count': failed_count,
            'overall_health': overall_health,
            'health_percentage': (healthy_count / total_breakers) * 100
        }

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()