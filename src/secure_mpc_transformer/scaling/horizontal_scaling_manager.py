"""
Horizontal Scaling Manager with AI-Driven Auto-Scaling
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

class ScalingTrigger(Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    THREAT_LEVEL = "threat_level"
    PREDICTIVE = "predictive"
    CUSTOM_METRIC = "custom_metric"

class InstanceState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"

@dataclass
class ScalingMetrics:
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    request_rate: float = 0.0
    avg_response_time: float = 0.0
    queue_depth: int = 0
    threat_level: float = 0.0
    custom_metrics: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingRule:
    id: str
    name: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    scale_up_step: int = 1
    scale_down_step: int = 1
    cooldown_minutes: int = 5
    evaluation_window_minutes: int = 5
    enabled: bool = True
    priority: int = 1  # Higher priority rules are evaluated first

@dataclass
class InstanceInfo:
    instance_id: str
    state: InstanceState
    launch_time: datetime
    instance_type: str
    zone: str
    private_ip: str | None = None
    public_ip: str | None = None
    metrics: ScalingMetrics = field(default_factory=ScalingMetrics)
    tags: dict[str, str] = field(default_factory=dict)

@dataclass
class ScalingDecision:
    direction: ScalingDirection
    target_count: int
    current_count: int
    reasoning: list[str]
    confidence: float
    triggered_by: list[ScalingTrigger]
    timestamp: datetime
    estimated_cost_impact: float = 0.0
    estimated_time_to_effect: int = 300  # seconds

class PredictiveScaler:
    """AI-driven predictive scaling based on historical patterns"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.historical_metrics = deque(maxlen=10000)
        self.prediction_models = {}
        self.pattern_detectors = {}

        # Time-based patterns
        self.time_patterns = {
            'hourly': defaultdict(list),
            'daily': defaultdict(list),
            'weekly': defaultdict(list)
        }

    async def add_metrics(self, metrics: ScalingMetrics) -> None:
        """Add metrics for pattern learning"""
        self.historical_metrics.append(metrics)

        # Update time-based patterns
        timestamp = datetime.fromtimestamp(metrics.timestamp)
        hour = timestamp.hour
        day = timestamp.weekday()
        week = timestamp.isocalendar()[1]

        self.time_patterns['hourly'][hour].append(metrics.request_rate)
        self.time_patterns['daily'][day].append(metrics.request_rate)
        self.time_patterns['weekly'][week % 4].append(metrics.request_rate)  # 4-week cycle

        # Limit pattern history
        for pattern_dict in self.time_patterns.values():
            for key, values in pattern_dict.items():
                if len(values) > 1000:  # Keep last 1000 samples
                    pattern_dict[key] = values[-1000:]

    async def predict_scaling_need(self, current_metrics: ScalingMetrics, horizon_minutes: int = 15) -> dict[str, Any]:
        """Predict scaling needs for the given time horizon"""
        try:
            if len(self.historical_metrics) < 20:
                return {
                    'predicted_direction': ScalingDirection.MAINTAIN,
                    'confidence': 0.0,
                    'reasoning': ['Insufficient historical data for prediction']
                }

            # Time-based prediction
            time_prediction = await self._time_based_prediction(horizon_minutes)

            # Trend-based prediction
            trend_prediction = await self._trend_based_prediction(horizon_minutes)

            # Pattern-based prediction
            pattern_prediction = await self._pattern_based_prediction(horizon_minutes)

            # Combine predictions
            combined = await self._combine_predictions(
                time_prediction, trend_prediction, pattern_prediction
            )

            return combined

        except Exception as e:
            logger.error(f"Predictive scaling failed: {e}")
            return {
                'predicted_direction': ScalingDirection.MAINTAIN,
                'confidence': 0.0,
                'reasoning': [f'Prediction error: {str(e)}']
            }

    async def _time_based_prediction(self, horizon_minutes: int) -> dict[str, Any]:
        """Predict based on time-of-day patterns"""
        try:
            current_time = datetime.now()
            target_time = current_time + timedelta(minutes=horizon_minutes)

            target_hour = target_time.hour
            target_day = target_time.weekday()

            # Get historical patterns for this time
            hourly_rates = self.time_patterns['hourly'].get(target_hour, [])
            daily_rates = self.time_patterns['daily'].get(target_day, [])

            if not hourly_rates or not daily_rates:
                return {'confidence': 0.0, 'predicted_rate': 0.0}

            # Calculate expected request rate
            hourly_avg = sum(hourly_rates[-50:]) / min(len(hourly_rates), 50)  # Last 50 samples
            daily_avg = sum(daily_rates[-20:]) / min(len(daily_rates), 20)    # Last 20 samples

            # Weighted combination (hourly patterns more important)
            predicted_rate = 0.7 * hourly_avg + 0.3 * daily_avg
            current_rate = len(self.historical_metrics) and self.historical_metrics[-1].request_rate or 0

            # Determine scaling direction
            rate_change = (predicted_rate - current_rate) / max(current_rate, 1)

            if rate_change > 0.3:  # 30% increase
                direction = ScalingDirection.SCALE_UP
            elif rate_change < -0.3:  # 30% decrease
                direction = ScalingDirection.SCALE_DOWN
            else:
                direction = ScalingDirection.MAINTAIN

            confidence = min(1.0, (len(hourly_rates) + len(daily_rates)) / 100)

            return {
                'predicted_direction': direction,
                'predicted_rate': predicted_rate,
                'current_rate': current_rate,
                'rate_change': rate_change,
                'confidence': confidence,
                'reasoning': [f'Time-based prediction: {rate_change:.1%} change in request rate']
            }

        except Exception as e:
            logger.error(f"Time-based prediction failed: {e}")
            return {'confidence': 0.0}

    async def _trend_based_prediction(self, horizon_minutes: int) -> dict[str, Any]:
        """Predict based on recent trends"""
        try:
            if len(self.historical_metrics) < 10:
                return {'confidence': 0.0}

            # Get recent metrics (last hour)
            recent_cutoff = time.time() - 3600
            recent_metrics = [
                m for m in self.historical_metrics
                if m.timestamp > recent_cutoff
            ]

            if len(recent_metrics) < 5:
                return {'confidence': 0.0}

            # Calculate trends
            timestamps = [m.timestamp for m in recent_metrics]
            request_rates = [m.request_rate for m in recent_metrics]
            response_times = [m.avg_response_time for m in recent_metrics]
            cpu_utils = [m.cpu_utilization for m in recent_metrics]

            # Linear trend analysis
            rate_trend = self._calculate_linear_trend(timestamps, request_rates)
            response_trend = self._calculate_linear_trend(timestamps, response_times)
            cpu_trend = self._calculate_linear_trend(timestamps, cpu_utils)

            # Project trends forward
            future_timestamp = time.time() + (horizon_minutes * 60)

            projected_rate = request_rates[-1] + rate_trend * (horizon_minutes * 60)
            projected_response_time = response_times[-1] + response_trend * (horizon_minutes * 60)
            projected_cpu = cpu_utils[-1] + cpu_trend * (horizon_minutes * 60)

            # Determine scaling need
            scaling_signals = []

            if projected_rate > request_rates[-1] * 1.2:  # 20% rate increase
                scaling_signals.append("increasing_request_rate")

            if projected_response_time > response_times[-1] * 1.3:  # 30% response time increase
                scaling_signals.append("increasing_response_time")

            if projected_cpu > 0.8:  # High CPU projected
                scaling_signals.append("high_cpu_projected")

            # Determine direction
            if len(scaling_signals) >= 2:
                direction = ScalingDirection.SCALE_UP
            elif projected_rate < request_rates[-1] * 0.7 and projected_cpu < 0.3:
                direction = ScalingDirection.SCALE_DOWN
            else:
                direction = ScalingDirection.MAINTAIN

            confidence = min(1.0, len(recent_metrics) / 20)

            return {
                'predicted_direction': direction,
                'confidence': confidence,
                'reasoning': [f'Trend analysis: {", ".join(scaling_signals)}' if scaling_signals else 'Stable trends'],
                'projected_metrics': {
                    'request_rate': projected_rate,
                    'response_time': projected_response_time,
                    'cpu_utilization': projected_cpu
                }
            }

        except Exception as e:
            logger.error(f"Trend-based prediction failed: {e}")
            return {'confidence': 0.0}

    def _calculate_linear_trend(self, x_values: list[float], y_values: list[float]) -> float:
        """Calculate linear trend (slope) of data points"""
        try:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                return 0.0

            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
            sum_x_squared = sum(x * x for x in x_values)

            # Calculate slope using least squares
            denominator = n * sum_x_squared - sum_x * sum_x
            if abs(denominator) < 1e-10:  # Avoid division by zero
                return 0.0

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope

        except Exception:
            return 0.0

    async def _pattern_based_prediction(self, horizon_minutes: int) -> dict[str, Any]:
        """Predict based on recurring patterns"""
        try:
            if len(self.historical_metrics) < 50:
                return {'confidence': 0.0}

            # Look for cyclical patterns in request rates
            recent_rates = [m.request_rate for m in list(self.historical_metrics)[-100:]]

            # Simple pattern detection - look for repeating sequences
            pattern_strength = await self._detect_cyclical_patterns(recent_rates)

            if pattern_strength > 0.6:  # Strong pattern detected
                # Project pattern forward
                cycle_length = await self._estimate_cycle_length(recent_rates)

                if cycle_length and cycle_length > 0:
                    future_index = len(recent_rates) + (horizon_minutes // 5)  # Assuming 5-minute intervals
                    pattern_index = future_index % cycle_length

                    if pattern_index < len(recent_rates):
                        predicted_rate = recent_rates[pattern_index]
                        current_rate = recent_rates[-1]

                        rate_change = (predicted_rate - current_rate) / max(current_rate, 1)

                        if rate_change > 0.2:
                            direction = ScalingDirection.SCALE_UP
                        elif rate_change < -0.2:
                            direction = ScalingDirection.SCALE_DOWN
                        else:
                            direction = ScalingDirection.MAINTAIN

                        return {
                            'predicted_direction': direction,
                            'confidence': pattern_strength,
                            'reasoning': [f'Cyclical pattern detected (strength: {pattern_strength:.2f})'],
                            'pattern_strength': pattern_strength,
                            'cycle_length': cycle_length
                        }

            return {'confidence': 0.0}

        except Exception as e:
            logger.error(f"Pattern-based prediction failed: {e}")
            return {'confidence': 0.0}

    async def _detect_cyclical_patterns(self, data: list[float]) -> float:
        """Detect cyclical patterns in time series data"""
        try:
            if len(data) < 20:
                return 0.0

            # Simple autocorrelation approach
            max_correlation = 0.0

            for lag in range(2, min(len(data) // 3, 20)):  # Check lags up to 20
                correlation = self._calculate_autocorrelation(data, lag)
                max_correlation = max(max_correlation, abs(correlation))

            return min(1.0, max_correlation)

        except Exception:
            return 0.0

    def _calculate_autocorrelation(self, data: list[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        try:
            if lag >= len(data):
                return 0.0

            n = len(data) - lag
            if n <= 0:
                return 0.0

            mean_val = sum(data) / len(data)

            numerator = sum((data[i] - mean_val) * (data[i + lag] - mean_val) for i in range(n))
            denominator = sum((x - mean_val) ** 2 for x in data)

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception:
            return 0.0

    async def _estimate_cycle_length(self, data: list[float]) -> int | None:
        """Estimate the length of recurring cycles in data"""
        try:
            best_cycle_length = None
            best_correlation = 0.0

            for cycle_length in range(3, min(len(data) // 2, 30)):
                correlation = self._calculate_autocorrelation(data, cycle_length)

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_cycle_length = cycle_length

            return best_cycle_length if best_correlation > 0.5 else None

        except Exception:
            return None

    async def _combine_predictions(self, *predictions: dict[str, Any]) -> dict[str, Any]:
        """Combine multiple prediction results"""
        try:
            valid_predictions = [p for p in predictions if p.get('confidence', 0) > 0.1]

            if not valid_predictions:
                return {
                    'predicted_direction': ScalingDirection.MAINTAIN,
                    'confidence': 0.0,
                    'reasoning': ['No reliable predictions available']
                }

            # Weight by confidence
            total_confidence = sum(p.get('confidence', 0) for p in valid_predictions)

            if total_confidence == 0:
                return {
                    'predicted_direction': ScalingDirection.MAINTAIN,
                    'confidence': 0.0,
                    'reasoning': ['All predictions have zero confidence']
                }

            # Vote on direction weighted by confidence
            direction_votes = defaultdict(float)
            all_reasoning = []

            for prediction in valid_predictions:
                direction = prediction.get('predicted_direction', ScalingDirection.MAINTAIN)
                confidence = prediction.get('confidence', 0)
                weight = confidence / total_confidence

                direction_votes[direction] += weight
                all_reasoning.extend(prediction.get('reasoning', []))

            # Select highest voted direction
            best_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            combined_confidence = direction_votes[best_direction]

            return {
                'predicted_direction': best_direction,
                'confidence': combined_confidence,
                'reasoning': all_reasoning,
                'prediction_breakdown': {
                    direction.value: votes for direction, votes in direction_votes.items()
                }
            }

        except Exception as e:
            logger.error(f"Failed to combine predictions: {e}")
            return {
                'predicted_direction': ScalingDirection.MAINTAIN,
                'confidence': 0.0,
                'reasoning': [f'Combination error: {str(e)}']
            }

class HorizontalScalingManager:
    """
    AI-driven horizontal scaling manager with predictive capabilities,
    cost optimization, and intelligent load balancing across multiple zones.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

        # Scaling configuration
        self.min_instances = config.get("min_instances", 2)
        self.max_instances = config.get("max_instances", 50)
        self.target_zones = config.get("target_zones", ["us-east-1a", "us-east-1b", "us-east-1c"])
        self.instance_type = config.get("instance_type", "t3.medium")

        # Instance management
        self.instances: dict[str, InstanceInfo] = {}
        self.scaling_rules: dict[str, ScalingRule] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=10000)

        # AI components
        self.predictive_scaler = PredictiveScaler(config.get("predictive_scaling", {}))

        # State management
        self.last_scaling_action = 0
        self.cooldown_period = config.get("cooldown_minutes", 5) * 60
        self.scaling_lock = threading.RLock()

        # Cost optimization
        self.cost_per_instance_hour = config.get("cost_per_instance_hour", 0.05)
        self.cost_optimization_enabled = config.get("cost_optimization_enabled", True)

        # Health monitoring
        self.health_check_interval = config.get("health_check_interval", 30)
        self.unhealthy_threshold = config.get("unhealthy_threshold", 3)

        # Background tasks
        self._scaling_task = None
        self._health_monitoring_task = None
        self._metrics_collection_task = None

    async def start_scaling_manager(self) -> None:
        """Start the horizontal scaling manager"""
        logger.info("Starting horizontal scaling manager")

        try:
            # Load default scaling rules
            await self._load_default_scaling_rules()

            # Initialize with minimum instances
            await self._ensure_minimum_instances()

            # Start background tasks
            self._scaling_task = asyncio.create_task(self._scaling_loop())
            self._health_monitoring_task = asyncio.create_task(self._health_monitoring_loop())
            self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())

            logger.info("Horizontal scaling manager started successfully")

        except Exception as e:
            logger.error(f"Failed to start scaling manager: {e}")
            raise

    async def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a scaling rule"""
        try:
            self.scaling_rules[rule.id] = rule
            logger.info(f"Added scaling rule: {rule.name}")

        except Exception as e:
            logger.error(f"Failed to add scaling rule {rule.id}: {e}")

    async def update_metrics(self, metrics: ScalingMetrics) -> None:
        """Update current system metrics"""
        try:
            # Store metrics history
            self.metrics_history.append(metrics)

            # Update predictive scaler
            await self.predictive_scaler.add_metrics(metrics)

            # Update instance metrics
            await self._distribute_metrics_to_instances(metrics)

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    async def trigger_scaling_evaluation(self) -> ScalingDecision:
        """Trigger immediate scaling evaluation"""
        try:
            with self.scaling_lock:
                return await self._evaluate_scaling_decision()

        except Exception as e:
            logger.error(f"Scaling evaluation failed: {e}")
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                target_count=len(self.instances),
                current_count=len(self.instances),
                reasoning=[f"Evaluation error: {str(e)}"],
                confidence=0.0,
                triggered_by=[],
                timestamp=datetime.now(timezone.utc)
            )

    async def scale_to_target(self, target_count: int, reason: str = "Manual scaling") -> bool:
        """Scale to specific target instance count"""
        try:
            current_count = len([i for i in self.instances.values() if i.state == InstanceState.RUNNING])

            if target_count == current_count:
                logger.info(f"Already at target count {target_count}")
                return True

            # Validate bounds
            target_count = max(self.min_instances, min(self.max_instances, target_count))

            if target_count > current_count:
                # Scale up
                instances_to_add = target_count - current_count
                success = await self._scale_up(instances_to_add, reason)
            else:
                # Scale down
                instances_to_remove = current_count - target_count
                success = await self._scale_down(instances_to_remove, reason)

            return success

        except Exception as e:
            logger.error(f"Failed to scale to target {target_count}: {e}")
            return False

    async def get_scaling_status(self) -> dict[str, Any]:
        """Get current scaling status"""
        try:
            running_instances = [i for i in self.instances.values() if i.state == InstanceState.RUNNING]
            pending_instances = [i for i in self.instances.values() if i.state == InstanceState.PENDING]

            # Get latest metrics
            latest_metrics = self.metrics_history[-1] if self.metrics_history else ScalingMetrics()

            # Calculate zone distribution
            zone_distribution = defaultdict(int)
            for instance in running_instances:
                zone_distribution[instance.zone] += 1

            return {
                'total_instances': len(self.instances),
                'running_instances': len(running_instances),
                'pending_instances': len(pending_instances),
                'target_instances': len(running_instances) + len(pending_instances),
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'zone_distribution': dict(zone_distribution),
                'latest_metrics': {
                    'cpu_utilization': latest_metrics.cpu_utilization,
                    'memory_utilization': latest_metrics.memory_utilization,
                    'request_rate': latest_metrics.request_rate,
                    'avg_response_time': latest_metrics.avg_response_time,
                    'queue_depth': latest_metrics.queue_depth,
                    'threat_level': latest_metrics.threat_level
                },
                'scaling_rules': len(self.scaling_rules),
                'enabled_rules': sum(1 for rule in self.scaling_rules.values() if rule.enabled),
                'last_scaling_action': self.last_scaling_action,
                'cooldown_remaining': max(0, self.cooldown_period - (time.time() - self.last_scaling_action))
            }

        except Exception as e:
            logger.error(f"Failed to get scaling status: {e}")
            return {'error': str(e)}

    async def _load_default_scaling_rules(self) -> None:
        """Load default scaling rules"""
        try:
            default_rules = [
                ScalingRule(
                    id="cpu_utilization",
                    name="CPU Utilization Scaling",
                    trigger=ScalingTrigger.CPU_UTILIZATION,
                    threshold_up=75.0,  # 75%
                    threshold_down=30.0,  # 30%
                    scale_up_step=2,
                    scale_down_step=1,
                    cooldown_minutes=5,
                    priority=1
                ),
                ScalingRule(
                    id="memory_utilization",
                    name="Memory Utilization Scaling",
                    trigger=ScalingTrigger.MEMORY_UTILIZATION,
                    threshold_up=80.0,  # 80%
                    threshold_down=40.0,  # 40%
                    scale_up_step=2,
                    scale_down_step=1,
                    cooldown_minutes=5,
                    priority=2
                ),
                ScalingRule(
                    id="request_rate",
                    name="Request Rate Scaling",
                    trigger=ScalingTrigger.REQUEST_RATE,
                    threshold_up=1000.0,  # 1000 RPS
                    threshold_down=200.0,  # 200 RPS
                    scale_up_step=3,
                    scale_down_step=1,
                    cooldown_minutes=3,
                    priority=1
                ),
                ScalingRule(
                    id="response_time",
                    name="Response Time Scaling",
                    trigger=ScalingTrigger.RESPONSE_TIME,
                    threshold_up=2.0,  # 2 seconds
                    threshold_down=0.5,  # 0.5 seconds
                    scale_up_step=2,
                    scale_down_step=1,
                    cooldown_minutes=5,
                    priority=1
                ),
                ScalingRule(
                    id="threat_level",
                    name="Threat Level Scaling",
                    trigger=ScalingTrigger.THREAT_LEVEL,
                    threshold_up=0.7,  # 70% threat level
                    threshold_down=0.3,  # 30% threat level
                    scale_up_step=4,  # Aggressive scaling for security
                    scale_down_step=1,
                    cooldown_minutes=2,
                    priority=0  # Highest priority
                )
            ]

            for rule in default_rules:
                await self.add_scaling_rule(rule)

            logger.info(f"Loaded {len(default_rules)} default scaling rules")

        except Exception as e:
            logger.error(f"Failed to load default scaling rules: {e}")

    async def _ensure_minimum_instances(self) -> None:
        """Ensure minimum number of instances are running"""
        try:
            running_count = len([i for i in self.instances.values() if i.state == InstanceState.RUNNING])

            if running_count < self.min_instances:
                instances_needed = self.min_instances - running_count
                await self._scale_up(instances_needed, "Ensuring minimum instances")

        except Exception as e:
            logger.error(f"Failed to ensure minimum instances: {e}")

    async def _evaluate_scaling_decision(self) -> ScalingDecision:
        """Evaluate whether scaling is needed based on current conditions"""
        try:
            current_time = time.time()

            # Check cooldown
            if current_time - self.last_scaling_action < self.cooldown_period:
                cooldown_remaining = self.cooldown_period - (current_time - self.last_scaling_action)
                return ScalingDecision(
                    direction=ScalingDirection.MAINTAIN,
                    target_count=len(self.instances),
                    current_count=len(self.instances),
                    reasoning=[f"Scaling cooldown active ({cooldown_remaining:.1f}s remaining)"],
                    confidence=1.0,
                    triggered_by=[],
                    timestamp=datetime.now(timezone.utc)
                )

            if not self.metrics_history:
                return ScalingDecision(
                    direction=ScalingDirection.MAINTAIN,
                    target_count=len(self.instances),
                    current_count=len(self.instances),
                    reasoning=["No metrics available"],
                    confidence=0.0,
                    triggered_by=[],
                    timestamp=datetime.now(timezone.utc)
                )

            latest_metrics = self.metrics_history[-1]
            current_count = len([i for i in self.instances.values() if i.state == InstanceState.RUNNING])

            # Rule-based evaluation
            rule_decisions = await self._evaluate_scaling_rules(latest_metrics)

            # Predictive evaluation
            predictive_decision = await self.predictive_scaler.predict_scaling_need(latest_metrics, 15)

            # Combine decisions
            final_decision = await self._combine_scaling_decisions(
                rule_decisions, predictive_decision, current_count
            )

            return final_decision

        except Exception as e:
            logger.error(f"Scaling decision evaluation failed: {e}")
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                target_count=len(self.instances),
                current_count=len(self.instances),
                reasoning=[f"Evaluation error: {str(e)}"],
                confidence=0.0,
                triggered_by=[],
                timestamp=datetime.now(timezone.utc)
            )

    async def _evaluate_scaling_rules(self, metrics: ScalingMetrics) -> list[dict[str, Any]]:
        """Evaluate all scaling rules against current metrics"""
        rule_decisions = []

        # Sort rules by priority (lower number = higher priority)
        sorted_rules = sorted(self.scaling_rules.values(), key=lambda r: r.priority)

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            try:
                decision = await self._evaluate_single_rule(rule, metrics)
                if decision:
                    rule_decisions.append(decision)

            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.id}: {e}")

        return rule_decisions

    async def _evaluate_single_rule(self, rule: ScalingRule, metrics: ScalingMetrics) -> dict[str, Any] | None:
        """Evaluate a single scaling rule"""
        try:
            # Get metric value based on trigger type
            if rule.trigger == ScalingTrigger.CPU_UTILIZATION:
                current_value = metrics.cpu_utilization
            elif rule.trigger == ScalingTrigger.MEMORY_UTILIZATION:
                current_value = metrics.memory_utilization
            elif rule.trigger == ScalingTrigger.REQUEST_RATE:
                current_value = metrics.request_rate
            elif rule.trigger == ScalingTrigger.RESPONSE_TIME:
                current_value = metrics.avg_response_time
            elif rule.trigger == ScalingTrigger.QUEUE_DEPTH:
                current_value = float(metrics.queue_depth)
            elif rule.trigger == ScalingTrigger.THREAT_LEVEL:
                current_value = metrics.threat_level
            else:
                return None

            # Evaluate rule windows (use simple current value for now)
            # In production, would evaluate over the specified time window

            # Check thresholds
            if current_value >= rule.threshold_up:
                return {
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'trigger': rule.trigger,
                    'direction': ScalingDirection.SCALE_UP,
                    'current_value': current_value,
                    'threshold': rule.threshold_up,
                    'scale_step': rule.scale_up_step,
                    'priority': rule.priority,
                    'reasoning': f"{rule.trigger.value} ({current_value:.2f}) exceeds threshold ({rule.threshold_up:.2f})"
                }

            elif current_value <= rule.threshold_down:
                return {
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'trigger': rule.trigger,
                    'direction': ScalingDirection.SCALE_DOWN,
                    'current_value': current_value,
                    'threshold': rule.threshold_down,
                    'scale_step': rule.scale_down_step,
                    'priority': rule.priority,
                    'reasoning': f"{rule.trigger.value} ({current_value:.2f}) below threshold ({rule.threshold_down:.2f})"
                }

            return None

        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.id}: {e}")
            return None

    async def _combine_scaling_decisions(
        self,
        rule_decisions: list[dict[str, Any]],
        predictive_decision: dict[str, Any],
        current_count: int
    ) -> ScalingDecision:
        """Combine rule-based and predictive scaling decisions"""
        try:
            reasoning = []
            triggered_by = []
            confidence = 0.0
            direction = ScalingDirection.MAINTAIN
            target_count = current_count

            # Process rule-based decisions (prioritize highest priority rules)
            scale_up_votes = 0
            scale_down_votes = 0
            max_scale_up = 0
            max_scale_down = 0

            for decision in rule_decisions:
                trigger = decision['trigger']
                triggered_by.append(trigger)
                reasoning.append(decision['reasoning'])

                if decision['direction'] == ScalingDirection.SCALE_UP:
                    scale_up_votes += 1
                    max_scale_up = max(max_scale_up, decision['scale_step'])
                elif decision['direction'] == ScalingDirection.SCALE_DOWN:
                    scale_down_votes += 1
                    max_scale_down = max(max_scale_down, decision['scale_step'])

            # Apply rule-based decision
            if scale_up_votes > scale_down_votes:
                direction = ScalingDirection.SCALE_UP
                target_count = min(self.max_instances, current_count + max_scale_up)
                confidence = 0.8
            elif scale_down_votes > scale_up_votes:
                direction = ScalingDirection.SCALE_DOWN
                target_count = max(self.min_instances, current_count - max_scale_down)
                confidence = 0.7

            # Consider predictive input
            pred_confidence = predictive_decision.get('confidence', 0.0)
            pred_direction = predictive_decision.get('predicted_direction', ScalingDirection.MAINTAIN)

            if pred_confidence > 0.6:  # High confidence prediction
                pred_reasoning = predictive_decision.get('reasoning', [])
                reasoning.extend([f"Predictive: {r}" for r in pred_reasoning])

                # If prediction agrees with rules, boost confidence
                if pred_direction == direction:
                    confidence = min(1.0, confidence + 0.2)

                # If no rule-based decision but high confidence prediction
                elif direction == ScalingDirection.MAINTAIN and pred_confidence > 0.8:
                    direction = pred_direction
                    confidence = pred_confidence
                    triggered_by.append(ScalingTrigger.PREDICTIVE)

                    if direction == ScalingDirection.SCALE_UP:
                        target_count = min(self.max_instances, current_count + 1)
                    elif direction == ScalingDirection.SCALE_DOWN:
                        target_count = max(self.min_instances, current_count - 1)

            # Cost optimization consideration
            if self.cost_optimization_enabled and direction == ScalingDirection.SCALE_UP:
                estimated_cost_impact = (target_count - current_count) * self.cost_per_instance_hour

                # If cost impact is high and scaling reason is not critical, reduce scaling
                if estimated_cost_impact > 10.0:  # $10/hour threshold
                    critical_triggers = [ScalingTrigger.THREAT_LEVEL, ScalingTrigger.RESPONSE_TIME]
                    is_critical = any(trigger in critical_triggers for trigger in triggered_by)

                    if not is_critical:
                        # Reduce scaling step
                        target_count = current_count + max(1, (target_count - current_count) // 2)
                        reasoning.append("Reduced scaling due to cost optimization")

            # Final bounds check
            target_count = max(self.min_instances, min(self.max_instances, target_count))

            return ScalingDecision(
                direction=direction,
                target_count=target_count,
                current_count=current_count,
                reasoning=reasoning or ["No scaling triggers detected"],
                confidence=confidence,
                triggered_by=triggered_by,
                timestamp=datetime.now(timezone.utc),
                estimated_cost_impact=(target_count - current_count) * self.cost_per_instance_hour,
                estimated_time_to_effect=300  # 5 minutes
            )

        except Exception as e:
            logger.error(f"Failed to combine scaling decisions: {e}")
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                target_count=current_count,
                current_count=current_count,
                reasoning=[f"Decision combination error: {str(e)}"],
                confidence=0.0,
                triggered_by=[],
                timestamp=datetime.now(timezone.utc)
            )

    async def _scale_up(self, instances_to_add: int, reason: str) -> bool:
        """Scale up by adding instances"""
        try:
            logger.info(f"Scaling up: adding {instances_to_add} instances. Reason: {reason}")

            # Distribute instances across zones for high availability
            zones_cycle = self.target_zones * (instances_to_add // len(self.target_zones) + 1)

            tasks = []
            for i in range(instances_to_add):
                zone = zones_cycle[i]
                task = asyncio.create_task(self._launch_instance(zone))
                tasks.append(task)

            # Wait for all launches to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_launches = sum(1 for result in results if not isinstance(result, Exception))

            self.last_scaling_action = time.time()

            # Record scaling action
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'instances_requested': instances_to_add,
                'instances_launched': successful_launches,
                'reason': reason,
                'success': successful_launches > 0
            })

            logger.info(f"Scale up completed: {successful_launches}/{instances_to_add} instances launched")
            return successful_launches > 0

        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            return False

    async def _scale_down(self, instances_to_remove: int, reason: str) -> bool:
        """Scale down by terminating instances"""
        try:
            logger.info(f"Scaling down: removing {instances_to_remove} instances. Reason: {reason}")

            # Select instances to terminate (prefer older instances, distribute across zones)
            running_instances = [i for i in self.instances.values() if i.state == InstanceState.RUNNING]

            if len(running_instances) <= self.min_instances:
                logger.warning("Cannot scale down below minimum instances")
                return False

            # Sort by launch time (oldest first)
            instances_to_terminate = sorted(running_instances, key=lambda x: x.launch_time)
            instances_to_terminate = instances_to_terminate[:instances_to_remove]

            tasks = []
            for instance in instances_to_terminate:
                task = asyncio.create_task(self._terminate_instance(instance.instance_id))
                tasks.append(task)

            # Wait for all terminations to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_terminations = sum(1 for result in results if not isinstance(result, Exception))

            self.last_scaling_action = time.time()

            # Record scaling action
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'instances_requested': instances_to_remove,
                'instances_terminated': successful_terminations,
                'reason': reason,
                'success': successful_terminations > 0
            })

            logger.info(f"Scale down completed: {successful_terminations}/{instances_to_remove} instances terminated")
            return successful_terminations > 0

        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            return False

    async def _launch_instance(self, zone: str) -> str:
        """Launch a new instance in the specified zone"""
        try:
            # Generate instance ID (in production, would use cloud provider API)
            instance_id = f"i-{int(time.time() * 1000) % 1000000:06d}"

            # Create instance info
            instance = InstanceInfo(
                instance_id=instance_id,
                state=InstanceState.PENDING,
                launch_time=datetime.now(timezone.utc),
                instance_type=self.instance_type,
                zone=zone,
                private_ip=f"10.0.{len(self.instances) % 255}.{(len(self.instances) // 255) % 255}",
                tags={'managed_by': 'horizontal_scaling_manager', 'zone': zone}
            )

            self.instances[instance_id] = instance

            # Simulate instance launch delay
            await asyncio.sleep(2)

            # Mark as running (in production, would wait for actual readiness)
            instance.state = InstanceState.RUNNING

            logger.info(f"Instance {instance_id} launched successfully in zone {zone}")
            return instance_id

        except Exception as e:
            logger.error(f"Failed to launch instance in zone {zone}: {e}")
            raise

    async def _terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance"""
        try:
            if instance_id not in self.instances:
                logger.warning(f"Instance {instance_id} not found")
                return False

            instance = self.instances[instance_id]

            # Mark as terminating
            instance.state = InstanceState.TERMINATING

            # Simulate termination delay
            await asyncio.sleep(1)

            # Remove from instances (in production, would wait for actual termination)
            instance.state = InstanceState.TERMINATED
            del self.instances[instance_id]

            logger.info(f"Instance {instance_id} terminated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False

    async def _distribute_metrics_to_instances(self, metrics: ScalingMetrics) -> None:
        """Distribute system metrics to individual instances"""
        try:
            running_instances = [i for i in self.instances.values() if i.state == InstanceState.RUNNING]

            if not running_instances:
                return

            # Simple distribution - divide metrics evenly
            per_instance_metrics = ScalingMetrics(
                cpu_utilization=metrics.cpu_utilization,
                memory_utilization=metrics.memory_utilization,
                request_rate=metrics.request_rate / len(running_instances),
                avg_response_time=metrics.avg_response_time,
                queue_depth=metrics.queue_depth // len(running_instances),
                threat_level=metrics.threat_level,
                custom_metrics=metrics.custom_metrics,
                timestamp=metrics.timestamp
            )

            for instance in running_instances:
                instance.metrics = per_instance_metrics

        except Exception as e:
            logger.error(f"Failed to distribute metrics to instances: {e}")

    async def _scaling_loop(self) -> None:
        """Background scaling evaluation loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Evaluate every minute

                decision = await self.trigger_scaling_evaluation()

                if decision.direction != ScalingDirection.MAINTAIN:
                    logger.info(f"Scaling decision: {decision.direction.value} to {decision.target_count} instances")
                    logger.info(f"Reasoning: {'; '.join(decision.reasoning)}")

                    # Execute scaling decision
                    if decision.direction == ScalingDirection.SCALE_UP:
                        instances_to_add = decision.target_count - decision.current_count
                        await self._scale_up(instances_to_add, "Automatic scaling")

                    elif decision.direction == ScalingDirection.SCALE_DOWN:
                        instances_to_remove = decision.current_count - decision.target_count
                        await self._scale_down(instances_to_remove, "Automatic scaling")

            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(120)  # Back off on error

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check instance health (placeholder - would use actual health checks)
                unhealthy_instances = []

                for instance in self.instances.values():
                    if instance.state == InstanceState.RUNNING:
                        # Simulate health check (in production, would ping health endpoints)
                        if instance.metrics.avg_response_time > 10.0:  # Unhealthy threshold
                            unhealthy_instances.append(instance)

                # Replace unhealthy instances
                for instance in unhealthy_instances:
                    logger.warning(f"Replacing unhealthy instance {instance.instance_id}")

                    # Launch replacement
                    await self._launch_instance(instance.zone)

                    # Terminate unhealthy instance
                    await self._terminate_instance(instance.instance_id)

            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect metrics every 30 seconds

                # In production, would collect real metrics from instances/load balancer
                # For now, generate synthetic metrics for demonstration

                running_count = len([i for i in self.instances.values() if i.state == InstanceState.RUNNING])

                if running_count > 0:
                    # Generate realistic synthetic metrics
                    import random

                    base_cpu = 40.0 + (running_count - self.min_instances) * 10.0
                    cpu_variation = random.uniform(-20.0, 20.0)
                    cpu_utilization = max(0.0, min(100.0, base_cpu + cpu_variation))

                    synthetic_metrics = ScalingMetrics(
                        cpu_utilization=cpu_utilization,
                        memory_utilization=random.uniform(30.0, 80.0),
                        request_rate=random.uniform(100.0, 1500.0),
                        avg_response_time=random.uniform(0.1, 3.0),
                        queue_depth=random.randint(0, 100),
                        threat_level=random.uniform(0.1, 0.9),
                        timestamp=time.time()
                    )

                    await self.update_metrics(synthetic_metrics)

            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
