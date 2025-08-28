"""
Intelligent Auto-Scaling System

AI-powered auto-scaling for the secure MPC transformer system with
predictive scaling, resource optimization, and multi-dimensional scaling.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import threading
import statistics
import psutil

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of scalable resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    WORKERS = "workers"
    REPLICAS = "replicas"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE_IOPS = "storage_iops"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    QUANTUM_OPTIMIZED = "quantum_optimized"


@dataclass
class ScalingMetric:
    """Represents a metric for scaling decisions."""
    name: str
    current_value: float
    target_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_value(self, value: float):
        """Add a new value to history."""
        self.current_value = value
        self.history.append({
            'value': value,
            'timestamp': time.time()
        })
    
    def get_scaling_signal(self) -> ScalingDirection:
        """Get scaling signal based on current value."""
        if self.current_value > self.threshold_up:
            return ScalingDirection.UP
        elif self.current_value < self.threshold_down:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def get_trend(self, window_size: int = 10) -> float:
        """Calculate trend over window (positive = increasing)."""
        if len(self.history) < window_size:
            return 0.0
            
        recent_values = [h['value'] for h in list(self.history)[-window_size:]]
        if len(recent_values) < 2:
            return 0.0
        
        # Simple linear regression for trend
        x = np.arange(len(recent_values)).reshape(-1, 1)
        y = np.array(recent_values)
        
        try:
            model = LinearRegression()
            model.fit(x, y)
            return model.coef_[0]  # Slope indicates trend
        except:
            return 0.0


@dataclass
class ScalingAction:
    """Represents a scaling action."""
    resource_type: ResourceType
    direction: ScalingDirection
    magnitude: float  # How much to scale (e.g., 1.5x, +2 workers)
    reason: str
    timestamp: float = field(default_factory=time.time)
    executed: bool = False
    result: Optional[str] = None


class PredictiveScaler:
    """Predictive scaling using machine learning."""
    
    def __init__(self, prediction_window: int = 300):  # 5 minutes
        self.prediction_window = prediction_window
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._trained_models: set = set()
        
    def record_features(self, features: Dict[str, float]):
        """Record features for training."""
        timestamp = time.time()
        
        for name, value in features.items():
            self.feature_history[name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def predict_load(self, metric_name: str, lookahead_seconds: int = 300) -> Optional[float]:
        """Predict future load for a metric."""
        if metric_name not in self._trained_models:
            self._train_model(metric_name)
            
        if metric_name not in self.models:
            return None
            
        # Get recent history for prediction
        history = list(self.feature_history[metric_name])
        if len(history) < 20:
            return None
            
        try:
            # Prepare features (time-based + recent values)
            current_time = time.time()
            recent_values = [h['value'] for h in history[-10:]]
            
            # Feature vector: [hour_of_day, minute_of_hour, recent_trend, recent_avg, recent_std]
            hour_of_day = (current_time % 86400) / 3600  # 0-24
            minute_of_hour = ((current_time % 3600) / 60)  # 0-60
            recent_trend = self._calculate_trend(recent_values)
            recent_avg = statistics.mean(recent_values)
            recent_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            features = np.array([[
                hour_of_day, minute_of_hour, recent_trend, recent_avg, recent_std
            ]])
            
            # Scale features
            if metric_name in self.scalers:
                features = self.scalers[metric_name].transform(features)
            
            # Predict
            prediction = self.models[metric_name].predict(features)[0]
            return max(0, prediction)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Prediction failed for {metric_name}: {e}")
            return None
    
    def _train_model(self, metric_name: str):
        """Train predictive model for a metric."""
        history = list(self.feature_history[metric_name])
        if len(history) < 50:  # Need enough data
            return
            
        try:
            # Prepare training data
            X = []
            y = []
            
            for i in range(10, len(history) - 5):  # Need lookback and lookahead
                # Features at time i
                timestamp = history[i]['timestamp']
                hour_of_day = (timestamp % 86400) / 3600
                minute_of_hour = ((timestamp % 3600) / 60)
                
                # Recent values for trend calculation
                recent_values = [history[j]['value'] for j in range(i-10, i)]
                recent_trend = self._calculate_trend(recent_values)
                recent_avg = statistics.mean(recent_values)
                recent_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
                
                X.append([hour_of_day, minute_of_hour, recent_trend, recent_avg, recent_std])
                
                # Target: value 5 minutes later
                future_idx = min(i + 5, len(history) - 1)
                y.append(history[future_idx]['value'])
            
            if len(X) < 20:
                return
                
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            self._trained_models.add(metric_name)
            
            logger.info(f"Trained predictive model for {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to train model for {metric_name}: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        try:
            return np.polyfit(x, values, 1)[0]
        except:
            return 0.0


class ResourceOptimizer:
    """Optimizes resource allocation."""
    
    def __init__(self):
        self.resource_costs = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 0.8,
            ResourceType.GPU: 10.0,
            ResourceType.WORKERS: 2.0,
            ResourceType.REPLICAS: 5.0,
            ResourceType.NETWORK_BANDWIDTH: 1.5,
            ResourceType.STORAGE_IOPS: 1.2
        }
        
        self.resource_constraints = {
            ResourceType.CPU: {'min': 1, 'max': 64},
            ResourceType.MEMORY: {'min': 1, 'max': 256},  # GB
            ResourceType.GPU: {'min': 0, 'max': 8},
            ResourceType.WORKERS: {'min': 1, 'max': 100},
            ResourceType.REPLICAS: {'min': 1, 'max': 50},
            ResourceType.NETWORK_BANDWIDTH: {'min': 100, 'max': 10000},  # Mbps
            ResourceType.STORAGE_IOPS: {'min': 100, 'max': 50000}
        }
    
    def optimize_scaling_plan(self, actions: List[ScalingAction]) -> List[ScalingAction]:
        """Optimize a list of scaling actions for cost and efficiency."""
        if not actions:
            return actions
            
        # Sort by cost-effectiveness (impact/cost ratio)
        scored_actions = []
        for action in actions:
            cost = self._calculate_cost(action)
            impact = self._calculate_impact(action)
            efficiency = impact / cost if cost > 0 else 0
            scored_actions.append((action, efficiency))
        
        # Sort by efficiency (highest first)
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        # Apply constraints and dependencies
        optimized_actions = []
        current_resources = self._get_current_resources()
        
        for action, _ in scored_actions:
            if self._can_apply_action(action, current_resources):
                optimized_actions.append(action)
                self._apply_action_to_resources(action, current_resources)
        
        return optimized_actions
    
    def _calculate_cost(self, action: ScalingAction) -> float:
        """Calculate cost of a scaling action."""
        base_cost = self.resource_costs.get(action.resource_type, 1.0)
        
        if action.direction == ScalingDirection.UP:
            return base_cost * action.magnitude
        else:
            return base_cost * (1.0 / action.magnitude)  # Cost savings
    
    def _calculate_impact(self, action: ScalingAction) -> float:
        """Calculate expected impact of a scaling action."""
        # This would be based on historical data and resource relationships
        impact_weights = {
            ResourceType.CPU: 0.8,
            ResourceType.MEMORY: 0.7,
            ResourceType.GPU: 1.5,
            ResourceType.WORKERS: 1.0,
            ResourceType.REPLICAS: 1.2,
            ResourceType.NETWORK_BANDWIDTH: 0.6,
            ResourceType.STORAGE_IOPS: 0.5
        }
        
        return impact_weights.get(action.resource_type, 1.0) * action.magnitude
    
    def _get_current_resources(self) -> Dict[ResourceType, float]:
        """Get current resource allocation."""
        try:
            return {
                ResourceType.CPU: psutil.cpu_count(),
                ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),  # GB
                ResourceType.GPU: 0,  # Would query actual GPU count
                ResourceType.WORKERS: 4,  # Current worker count
                ResourceType.REPLICAS: 1,  # Current replica count
                ResourceType.NETWORK_BANDWIDTH: 1000,  # Mbps
                ResourceType.STORAGE_IOPS: 1000
            }
        except:
            return {}
    
    def _can_apply_action(self, action: ScalingAction, current_resources: Dict[ResourceType, float]) -> bool:
        """Check if action can be applied given constraints."""
        current_value = current_resources.get(action.resource_type, 0)
        constraints = self.resource_constraints.get(action.resource_type, {})
        
        if action.direction == ScalingDirection.UP:
            new_value = current_value * action.magnitude
            return new_value <= constraints.get('max', float('inf'))
        else:
            new_value = current_value / action.magnitude
            return new_value >= constraints.get('min', 0)
    
    def _apply_action_to_resources(self, action: ScalingAction, resources: Dict[ResourceType, float]):
        """Apply action to resource state (for simulation)."""
        current_value = resources.get(action.resource_type, 0)
        
        if action.direction == ScalingDirection.UP:
            resources[action.resource_type] = current_value * action.magnitude
        else:
            resources[action.resource_type] = current_value / action.magnitude


class IntelligentAutoScaler:
    """Main intelligent auto-scaling orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration
        self.scaling_strategy = ScalingStrategy(
            self.config.get('scaling_strategy', 'hybrid')
        )
        self.scaling_interval = self.config.get('scaling_interval', 60.0)
        self.cooldown_period = self.config.get('cooldown_period', 300.0)
        
        # Components
        self.predictive_scaler = PredictiveScaler()
        self.resource_optimizer = ResourceOptimizer()
        
        # Scaling metrics
        self.metrics: Dict[str, ScalingMetric] = {}
        self._initialize_default_metrics()
        
        # Action tracking
        self.scaling_actions: deque = deque(maxlen=1000)
        self.last_scaling_time = 0
        
        # Background tasks
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Intelligent Auto-Scaler initialized with {self.scaling_strategy.value} strategy")
    
    def _initialize_default_metrics(self):
        """Initialize default scaling metrics."""
        
        # CPU utilization
        self.metrics['cpu_utilization'] = ScalingMetric(
            name='cpu_utilization',
            current_value=0.0,
            target_value=70.0,
            threshold_up=80.0,
            threshold_down=50.0,
            weight=1.0
        )
        
        # Memory utilization
        self.metrics['memory_utilization'] = ScalingMetric(
            name='memory_utilization',
            current_value=0.0,
            target_value=75.0,
            threshold_up=85.0,
            threshold_down=60.0,
            weight=1.2
        )
        
        # Request rate
        self.metrics['request_rate'] = ScalingMetric(
            name='request_rate',
            current_value=0.0,
            target_value=100.0,
            threshold_up=150.0,
            threshold_down=50.0,
            weight=0.8
        )
        
        # Response latency
        self.metrics['response_latency'] = ScalingMetric(
            name='response_latency',
            current_value=0.0,
            target_value=500.0,  # ms
            threshold_up=1000.0,
            threshold_down=200.0,
            weight=1.5
        )
        
        # Queue depth
        self.metrics['queue_depth'] = ScalingMetric(
            name='queue_depth',
            current_value=0.0,
            target_value=10.0,
            threshold_up=25.0,
            threshold_down=5.0,
            weight=1.0
        )
        
        # GPU utilization
        self.metrics['gpu_utilization'] = ScalingMetric(
            name='gpu_utilization',
            current_value=0.0,
            target_value=80.0,
            threshold_up=90.0,
            threshold_down=60.0,
            weight=2.0
        )
    
    async def start(self):
        """Start the auto-scaling system."""
        if self._running:
            return
            
        self._running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Intelligent Auto-Scaler started")
    
    async def stop(self):
        """Stop the auto-scaling system."""
        self._running = False
        
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Intelligent Auto-Scaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling decision loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await self._make_scaling_decisions()
                await asyncio.sleep(self.scaling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.scaling_interval)
    
    async def _collect_metrics(self):
        """Collect current metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Update metrics
            self.metrics['cpu_utilization'].add_value(cpu_percent)
            self.metrics['memory_utilization'].add_value(memory_percent)
            
            # Record for predictive model
            features = {
                'cpu_utilization': cpu_percent,
                'memory_utilization': memory_percent,
                'request_rate': self.metrics['request_rate'].current_value,
                'response_latency': self.metrics['response_latency'].current_value
            }
            self.predictive_scaler.record_features(features)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _make_scaling_decisions(self):
        """Make scaling decisions based on current and predicted metrics."""
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.cooldown_period:
            return
        
        actions = []
        
        if self.scaling_strategy in [ScalingStrategy.REACTIVE, ScalingStrategy.HYBRID]:
            actions.extend(self._get_reactive_actions())
        
        if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            actions.extend(await self._get_predictive_actions())
        
        if self.scaling_strategy == ScalingStrategy.QUANTUM_OPTIMIZED:
            actions.extend(await self._get_quantum_optimized_actions())
        
        if actions:
            # Optimize actions
            optimized_actions = self.resource_optimizer.optimize_scaling_plan(actions)
            
            # Execute actions
            for action in optimized_actions:
                await self._execute_scaling_action(action)
                
            self.last_scaling_time = time.time()
    
    def _get_reactive_actions(self) -> List[ScalingAction]:
        """Get reactive scaling actions based on current metrics."""
        actions = []
        
        for metric in self.metrics.values():
            signal = metric.get_scaling_signal()
            
            if signal == ScalingDirection.UP:
                # Scale up based on metric type
                if metric.name in ['cpu_utilization', 'memory_utilization']:
                    actions.append(ScalingAction(
                        resource_type=ResourceType.REPLICAS,
                        direction=ScalingDirection.UP,
                        magnitude=1.5,
                        reason=f"High {metric.name}: {metric.current_value:.1f}%"
                    ))
                
                elif metric.name == 'request_rate':
                    actions.append(ScalingAction(
                        resource_type=ResourceType.WORKERS,
                        direction=ScalingDirection.UP,
                        magnitude=1.3,
                        reason=f"High request rate: {metric.current_value:.1f}/s"
                    ))
                
                elif metric.name == 'gpu_utilization':
                    actions.append(ScalingAction(
                        resource_type=ResourceType.GPU,
                        direction=ScalingDirection.UP,
                        magnitude=2.0,
                        reason=f"High GPU utilization: {metric.current_value:.1f}%"
                    ))
            
            elif signal == ScalingDirection.DOWN:
                # Scale down with more conservative approach
                trend = metric.get_trend()
                if trend < -0.5:  # Decreasing trend
                    if metric.name in ['cpu_utilization', 'memory_utilization']:
                        actions.append(ScalingAction(
                            resource_type=ResourceType.REPLICAS,
                            direction=ScalingDirection.DOWN,
                            magnitude=1.2,
                            reason=f"Low {metric.name}: {metric.current_value:.1f}%"
                        ))
        
        return actions
    
    async def _get_predictive_actions(self) -> List[ScalingAction]:
        """Get predictive scaling actions based on forecasted load."""
        actions = []
        
        # Predict metrics 5 minutes ahead
        prediction_horizon = 300  # seconds
        
        for metric_name, metric in self.metrics.items():
            predicted_value = self.predictive_scaler.predict_load(
                metric_name, prediction_horizon
            )
            
            if predicted_value is None:
                continue
            
            # Check if predicted value would trigger scaling
            if predicted_value > metric.threshold_up:
                actions.append(ScalingAction(
                    resource_type=ResourceType.REPLICAS,
                    direction=ScalingDirection.UP,
                    magnitude=1.3,
                    reason=f"Predicted high {metric_name}: {predicted_value:.1f}"
                ))
            
            elif predicted_value < metric.threshold_down * 0.8:  # More conservative
                current_trend = metric.get_trend()
                if current_trend < -0.3:  # Confirming downward trend
                    actions.append(ScalingAction(
                        resource_type=ResourceType.REPLICAS,
                        direction=ScalingDirection.DOWN,
                        magnitude=1.15,
                        reason=f"Predicted low {metric_name}: {predicted_value:.1f}"
                    ))
        
        return actions
    
    async def _get_quantum_optimized_actions(self) -> List[ScalingAction]:
        """Get quantum-optimized scaling actions."""
        # This would implement quantum optimization algorithms
        # For now, return hybrid approach with quantum-inspired weights
        
        actions = []
        
        # Calculate quantum coherence impact on scaling
        quantum_coherence = 0.8  # Would get from quantum system
        
        # Adjust scaling sensitivity based on quantum coherence
        coherence_factor = 1.0 + (1.0 - quantum_coherence) * 0.5
        
        for metric in self.metrics.values():
            adjusted_threshold_up = metric.threshold_up * coherence_factor
            adjusted_threshold_down = metric.threshold_down * coherence_factor
            
            if metric.current_value > adjusted_threshold_up:
                actions.append(ScalingAction(
                    resource_type=ResourceType.WORKERS,
                    direction=ScalingDirection.UP,
                    magnitude=1.0 + coherence_factor * 0.3,
                    reason=f"Quantum-optimized scaling for {metric.name}"
                ))
        
        return actions
    
    async def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action."""
        try:
            logger.info(f"Executing scaling action: {action.direction.value} "
                       f"{action.resource_type.value} by {action.magnitude}x - {action.reason}")
            
            # This would interface with actual scaling infrastructure
            # For now, just log and mark as executed
            
            success = await self._simulate_scaling_execution(action)
            
            action.executed = True
            action.result = "success" if success else "failed"
            
            self.scaling_actions.append(action)
            
            if success:
                logger.info(f"Scaling action completed successfully")
            else:
                logger.error(f"Scaling action failed")
                
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            action.executed = True
            action.result = f"error: {e}"
            self.scaling_actions.append(action)
    
    async def _simulate_scaling_execution(self, action: ScalingAction) -> bool:
        """Simulate scaling execution (replace with actual implementation)."""
        # Simulate execution delay
        await asyncio.sleep(1)
        
        # Simulate 90% success rate
        import random
        return random.random() < 0.9
    
    def update_metric(self, name: str, value: float):
        """Update a metric value."""
        if name in self.metrics:
            self.metrics[name].add_value(value)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        
        # Recent actions (last hour)
        recent_actions = [
            {
                'resource_type': action.resource_type.value,
                'direction': action.direction.value,
                'magnitude': action.magnitude,
                'reason': action.reason,
                'timestamp': action.timestamp,
                'executed': action.executed,
                'result': action.result
            }
            for action in self.scaling_actions 
            if time.time() - action.timestamp < 3600
        ]
        
        # Current metrics
        current_metrics = {}
        for name, metric in self.metrics.items():
            current_metrics[name] = {
                'current_value': metric.current_value,
                'target_value': metric.target_value,
                'threshold_up': metric.threshold_up,
                'threshold_down': metric.threshold_down,
                'scaling_signal': metric.get_scaling_signal().value,
                'trend': metric.get_trend()
            }
        
        return {
            'running': self._running,
            'strategy': self.scaling_strategy.value,
            'scaling_interval': self.scaling_interval,
            'cooldown_period': self.cooldown_period,
            'last_scaling_time': self.last_scaling_time,
            'current_metrics': current_metrics,
            'recent_actions': recent_actions[-10:],  # Last 10 actions
            'total_actions': len(self.scaling_actions)
        }


# Global instance
_autoscaler: Optional[IntelligentAutoScaler] = None


def get_autoscaler() -> IntelligentAutoScaler:
    """Get the global auto-scaler instance."""
    global _autoscaler
    if _autoscaler is None:
        _autoscaler = IntelligentAutoScaler()
    return _autoscaler