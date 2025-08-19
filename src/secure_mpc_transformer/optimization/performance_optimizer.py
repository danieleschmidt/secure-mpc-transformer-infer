"""
Advanced Performance Optimizer with ML-based Auto-Tuning
"""

import asyncio
import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    COST_EFFICIENCY = "cost_efficiency"
    BALANCED = "balanced"

class OptimizationStrategy(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"
    ML_GUIDED = "ml_guided"

@dataclass
class PerformanceMetrics:
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_rps: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    active_connections: int = 0
    timestamp: float = field(default_factory=time.time)
    custom_metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class OptimizationParameter:
    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    parameter_type: str  # "int", "float", "string", "bool"
    description: str
    impact_weight: float = 1.0  # How much this parameter affects performance

@dataclass
class OptimizationResult:
    parameters: dict[str, Any]
    performance_improvement: float
    confidence: float
    metrics_before: PerformanceMetrics
    metrics_after: PerformanceMetrics
    timestamp: float
    optimization_time: float
    successful: bool
    error_message: str | None = None

class MLPerformanceOptimizer:
    """ML-based performance optimizer using Bayesian optimization"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.parameter_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)

        # ML components (if available)
        self.ml_available = False
        self._initialize_ml_components()

        # Optimization state
        self.current_best_params = {}
        self.current_best_score = 0.0
        self.exploration_factor = config.get("exploration_factor", 0.1)

    def _initialize_ml_components(self):
        """Initialize ML components for optimization"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from sklearn.preprocessing import StandardScaler

            # Gaussian Process for Bayesian optimization
            kernel = Matern(length_scale=1.0, nu=2.5)
            self.gp_regressor = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )

            self.scaler = StandardScaler()
            self.ml_available = True

            logger.info("ML performance optimizer initialized")

        except ImportError:
            logger.warning("scikit-learn not available, using heuristic optimization")

    async def suggest_parameters(
        self,
        parameters: dict[str, OptimizationParameter],
        target: OptimizationTarget
    ) -> dict[str, Any]:
        """Suggest next parameter values to try"""
        try:
            if not self.ml_available or len(self.parameter_history) < 5:
                # Use random/heuristic approach
                return await self._heuristic_parameter_suggestion(parameters, target)

            # Use ML-based approach
            return await self._ml_parameter_suggestion(parameters, target)

        except Exception as e:
            logger.error(f"Parameter suggestion failed: {e}")
            return await self._heuristic_parameter_suggestion(parameters, target)

    async def record_optimization_result(
        self,
        parameters: dict[str, Any],
        metrics: PerformanceMetrics,
        target: OptimizationTarget
    ) -> None:
        """Record the result of an optimization trial"""
        try:
            self.parameter_history.append(parameters.copy())

            # Calculate performance score based on target
            score = await self._calculate_performance_score(metrics, target)
            self.performance_history.append(score)

            # Update best parameters if this is better
            if score > self.current_best_score:
                self.current_best_score = score
                self.current_best_params = parameters.copy()
                logger.info(f"New best performance score: {score:.4f}")

            # Retrain ML model if available
            if self.ml_available and len(self.parameter_history) >= 10:
                await self._retrain_model()

        except Exception as e:
            logger.error(f"Failed to record optimization result: {e}")

    async def _calculate_performance_score(
        self,
        metrics: PerformanceMetrics,
        target: OptimizationTarget
    ) -> float:
        """Calculate performance score based on optimization target"""
        try:
            if target == OptimizationTarget.LATENCY:
                # Lower latency is better (invert score)
                return 1.0 / max(metrics.latency_p95, 0.001)

            elif target == OptimizationTarget.THROUGHPUT:
                # Higher throughput is better
                return metrics.throughput_rps / 1000.0  # Normalize

            elif target == OptimizationTarget.RESOURCE_EFFICIENCY:
                # Balance throughput and resource usage
                resource_usage = (metrics.cpu_utilization + metrics.memory_utilization) / 200.0
                efficiency = metrics.throughput_rps / max(resource_usage, 0.1)
                return efficiency / 1000.0  # Normalize

            elif target == OptimizationTarget.COST_EFFICIENCY:
                # Consider all resource usage
                total_usage = (
                    metrics.cpu_utilization +
                    metrics.memory_utilization +
                    metrics.gpu_utilization
                ) / 300.0
                cost_efficiency = metrics.throughput_rps / max(total_usage, 0.1)
                return cost_efficiency / 1000.0  # Normalize

            elif target == OptimizationTarget.BALANCED:
                # Weighted combination of multiple factors
                latency_score = 1.0 / max(metrics.latency_p95, 0.001)
                throughput_score = metrics.throughput_rps / 1000.0
                efficiency_score = metrics.cache_hit_rate
                error_penalty = 1.0 - metrics.error_rate

                balanced_score = (
                    0.3 * latency_score +
                    0.3 * throughput_score +
                    0.2 * efficiency_score +
                    0.2 * error_penalty
                )
                return balanced_score

            return 0.0

        except Exception as e:
            logger.error(f"Performance score calculation failed: {e}")
            return 0.0

    async def _heuristic_parameter_suggestion(
        self,
        parameters: dict[str, OptimizationParameter],
        target: OptimizationTarget
    ) -> dict[str, Any]:
        """Suggest parameters using heuristic approach"""
        try:
            suggestions = {}

            for name, param in parameters.items():
                if param.parameter_type == "int":
                    if not self.parameter_history:
                        # Start with mid-range value
                        value = (param.min_value + param.max_value) // 2
                    else:
                        # Random walk from current best or random exploration
                        if len(self.parameter_history) > 0 and name in self.current_best_params:
                            base_value = self.current_best_params[name]
                        else:
                            base_value = param.current_value

                        # Add some randomness
                        import random
                        delta = random.randint(-param.step_size, param.step_size)
                        value = max(param.min_value, min(param.max_value, base_value + delta))

                    suggestions[name] = value

                elif param.parameter_type == "float":
                    if not self.parameter_history:
                        value = (param.min_value + param.max_value) / 2.0
                    else:
                        if len(self.parameter_history) > 0 and name in self.current_best_params:
                            base_value = self.current_best_params[name]
                        else:
                            base_value = param.current_value

                        import random
                        delta = random.uniform(-param.step_size, param.step_size)
                        value = max(param.min_value, min(param.max_value, base_value + delta))

                    suggestions[name] = value

                elif param.parameter_type == "bool":
                    # Random boolean choice
                    import random
                    suggestions[name] = random.choice([True, False])

                else:
                    # Keep current value for unknown types
                    suggestions[name] = param.current_value

            return suggestions

        except Exception as e:
            logger.error(f"Heuristic parameter suggestion failed: {e}")
            return {name: param.current_value for name, param in parameters.items()}

    async def _ml_parameter_suggestion(
        self,
        parameters: dict[str, OptimizationParameter],
        target: OptimizationTarget
    ) -> dict[str, Any]:
        """Suggest parameters using ML-based approach"""
        try:
            if len(self.parameter_history) < 5:
                return await self._heuristic_parameter_suggestion(parameters, target)

            import numpy as np

            # Prepare training data
            X = []
            y = list(self.performance_history)

            param_names = sorted(parameters.keys())

            for param_set in self.parameter_history:
                feature_vector = []
                for name in param_names:
                    value = param_set.get(name, parameters[name].current_value)

                    # Normalize parameter values
                    param_info = parameters[name]
                    if param_info.parameter_type in ["int", "float"]:
                        normalized = (value - param_info.min_value) / max(
                            param_info.max_value - param_info.min_value, 1e-6
                        )
                        feature_vector.append(normalized)
                    elif param_info.parameter_type == "bool":
                        feature_vector.append(float(value))
                    else:
                        feature_vector.append(0.5)  # Default for unknown types

                X.append(feature_vector)

            X = np.array(X)
            y = np.array(y)

            # Train/update model
            X_scaled = self.scaler.fit_transform(X)
            self.gp_regressor.fit(X_scaled, y)

            # Generate candidate points and select best
            n_candidates = 100
            best_candidate = None
            best_acquisition = -float('inf')

            for _ in range(n_candidates):
                candidate = {}
                candidate_vector = []

                for name in param_names:
                    param_info = parameters[name]

                    if param_info.parameter_type == "int":
                        value = np.random.randint(param_info.min_value, param_info.max_value + 1)
                        normalized = (value - param_info.min_value) / max(
                            param_info.max_value - param_info.min_value, 1e-6
                        )
                    elif param_info.parameter_type == "float":
                        value = np.random.uniform(param_info.min_value, param_info.max_value)
                        normalized = (value - param_info.min_value) / max(
                            param_info.max_value - param_info.min_value, 1e-6
                        )
                    elif param_info.parameter_type == "bool":
                        value = np.random.choice([True, False])
                        normalized = float(value)
                    else:
                        value = param_info.current_value
                        normalized = 0.5

                    candidate[name] = value
                    candidate_vector.append(normalized)

                # Calculate acquisition function (Upper Confidence Bound)
                candidate_scaled = self.scaler.transform([candidate_vector])
                mean, std = self.gp_regressor.predict(candidate_scaled, return_std=True)

                acquisition = mean[0] + self.exploration_factor * std[0]

                if acquisition > best_acquisition:
                    best_acquisition = acquisition
                    best_candidate = candidate.copy()

            return best_candidate or await self._heuristic_parameter_suggestion(parameters, target)

        except Exception as e:
            logger.error(f"ML parameter suggestion failed: {e}")
            return await self._heuristic_parameter_suggestion(parameters, target)

    async def _retrain_model(self) -> None:
        """Retrain the ML model with latest data"""
        try:
            if not self.ml_available or len(self.parameter_history) < 10:
                return

            # Model is retrained in _ml_parameter_suggestion
            logger.debug("ML model retrained with latest optimization data")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

class PerformanceOptimizer:
    """
    Advanced performance optimizer with ML-based auto-tuning,
    A/B testing capabilities, and intelligent parameter exploration.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.optimization_target = OptimizationTarget(
            config.get("optimization_target", "balanced")
        )
        self.optimization_strategy = OptimizationStrategy(
            config.get("optimization_strategy", "adaptive")
        )

        # Optimization components
        self.ml_optimizer = MLPerformanceOptimizer(config.get("ml_optimizer", {}))

        # Parameter management
        self.parameters: dict[str, OptimizationParameter] = {}
        self.current_config = {}
        self.baseline_metrics = None

        # Optimization state
        self.optimization_history = deque(maxlen=1000)
        self.active_experiments = {}
        self.optimization_lock = threading.RLock()

        # A/B testing
        self.ab_testing_enabled = config.get("ab_testing_enabled", True)
        self.ab_test_duration = config.get("ab_test_duration", 300)  # 5 minutes
        self.ab_test_traffic_split = config.get("ab_test_traffic_split", 0.1)  # 10%

        # Background optimization
        self.auto_optimization_enabled = config.get("auto_optimization_enabled", True)
        self.optimization_interval = config.get("optimization_interval", 1800)  # 30 minutes

        # Performance thresholds
        self.performance_thresholds = {
            'latency_p95_ms': config.get('max_latency_p95', 2000),
            'error_rate': config.get('max_error_rate', 0.01),
            'cpu_utilization': config.get('max_cpu_utilization', 80.0),
            'memory_utilization': config.get('max_memory_utilization', 85.0)
        }

        # Background tasks
        self._optimization_task = None
        self._monitoring_task = None

    async def start_optimizer(self) -> None:
        """Start the performance optimizer"""
        logger.info("Starting performance optimizer")

        try:
            # Initialize default parameters
            await self._initialize_default_parameters()

            # Start background tasks
            if self.auto_optimization_enabled:
                self._optimization_task = asyncio.create_task(self._optimization_loop())

            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Performance optimizer started successfully")

        except Exception as e:
            logger.error(f"Failed to start performance optimizer: {e}")
            raise

    async def add_optimization_parameter(self, parameter: OptimizationParameter) -> None:
        """Add a parameter to optimize"""
        try:
            self.parameters[parameter.name] = parameter
            self.current_config[parameter.name] = parameter.current_value

            logger.info(f"Added optimization parameter: {parameter.name}")

        except Exception as e:
            logger.error(f"Failed to add parameter {parameter.name}: {e}")

    async def trigger_optimization(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Trigger an optimization cycle"""
        try:
            with self.optimization_lock:
                return await self._run_optimization_cycle(metrics)

        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            return OptimizationResult(
                parameters=self.current_config.copy(),
                performance_improvement=0.0,
                confidence=0.0,
                metrics_before=metrics,
                metrics_after=metrics,
                timestamp=time.time(),
                optimization_time=0.0,
                successful=False,
                error_message=str(e)
            )

    async def apply_configuration(self, parameters: dict[str, Any]) -> bool:
        """Apply a parameter configuration"""
        try:
            # Validate parameters
            for name, value in parameters.items():
                if name in self.parameters:
                    param = self.parameters[name]

                    # Type and range validation
                    if param.parameter_type == "int":
                        if not isinstance(value, int):
                            value = int(value)
                        value = max(param.min_value, min(param.max_value, value))
                    elif param.parameter_type == "float":
                        if not isinstance(value, (int, float)):
                            value = float(value)
                        value = max(param.min_value, min(param.max_value, value))
                    elif param.parameter_type == "bool":
                        value = bool(value)

                    parameters[name] = value

            # Apply parameters (placeholder - would integrate with actual systems)
            await self._apply_parameters_to_system(parameters)

            # Update current configuration
            self.current_config.update(parameters)

            logger.info(f"Applied configuration: {parameters}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply configuration: {e}")
            return False

    async def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status"""
        try:
            active_experiments_count = len(self.active_experiments)

            # Calculate recent performance improvement
            recent_results = list(self.optimization_history)[-10:]
            avg_improvement = 0.0
            if recent_results:
                improvements = [r.performance_improvement for r in recent_results if r.successful]
                avg_improvement = statistics.mean(improvements) if improvements else 0.0

            return {
                'optimization_target': self.optimization_target.value,
                'optimization_strategy': self.optimization_strategy.value,
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'ab_testing_enabled': self.ab_testing_enabled,
                'total_parameters': len(self.parameters),
                'current_config': self.current_config.copy(),
                'optimization_cycles': len(self.optimization_history),
                'active_experiments': active_experiments_count,
                'recent_avg_improvement': avg_improvement,
                'ml_optimizer_available': self.ml_optimizer.ml_available,
                'performance_thresholds': self.performance_thresholds.copy()
            }

        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {'error': str(e)}

    async def _initialize_default_parameters(self) -> None:
        """Initialize default optimization parameters"""
        try:
            # Connection pool parameters
            await self.add_optimization_parameter(OptimizationParameter(
                name="max_connections",
                current_value=100,
                min_value=50,
                max_value=500,
                step_size=25,
                parameter_type="int",
                description="Maximum number of concurrent connections",
                impact_weight=1.5
            ))

            # Thread pool parameters
            await self.add_optimization_parameter(OptimizationParameter(
                name="worker_threads",
                current_value=8,
                min_value=4,
                max_value=32,
                step_size=2,
                parameter_type="int",
                description="Number of worker threads",
                impact_weight=1.3
            ))

            # Cache parameters
            await self.add_optimization_parameter(OptimizationParameter(
                name="cache_size_mb",
                current_value=512,
                min_value=128,
                max_value=2048,
                step_size=128,
                parameter_type="int",
                description="Cache size in megabytes",
                impact_weight=1.2
            ))

            # Timeout parameters
            await self.add_optimization_parameter(OptimizationParameter(
                name="request_timeout_ms",
                current_value=5000,
                min_value=1000,
                max_value=30000,
                step_size=1000,
                parameter_type="int",
                description="Request timeout in milliseconds",
                impact_weight=1.0
            ))

            # Batch size parameters
            await self.add_optimization_parameter(OptimizationParameter(
                name="batch_size",
                current_value=32,
                min_value=8,
                max_value=128,
                step_size=8,
                parameter_type="int",
                description="Processing batch size",
                impact_weight=1.4
            ))

            # GPU parameters (if available)
            await self.add_optimization_parameter(OptimizationParameter(
                name="gpu_memory_fraction",
                current_value=0.8,
                min_value=0.3,
                max_value=0.95,
                step_size=0.05,
                parameter_type="float",
                description="Fraction of GPU memory to use",
                impact_weight=1.1
            ))

            # Compression parameters
            await self.add_optimization_parameter(OptimizationParameter(
                name="enable_compression",
                current_value=True,
                min_value=False,
                max_value=True,
                step_size=None,
                parameter_type="bool",
                description="Enable response compression",
                impact_weight=0.8
            ))

            logger.info(f"Initialized {len(self.parameters)} default optimization parameters")

        except Exception as e:
            logger.error(f"Failed to initialize default parameters: {e}")

    async def _run_optimization_cycle(self, current_metrics: PerformanceMetrics) -> OptimizationResult:
        """Run a complete optimization cycle"""
        start_time = time.time()

        try:
            logger.info("Starting optimization cycle")

            # Store baseline metrics if not available
            if self.baseline_metrics is None:
                self.baseline_metrics = current_metrics

            # Get parameter suggestions from ML optimizer
            suggested_params = await self.ml_optimizer.suggest_parameters(
                self.parameters, self.optimization_target
            )

            # Apply suggested configuration
            success = await self.apply_configuration(suggested_params)
            if not success:
                raise Exception("Failed to apply suggested configuration")

            # Wait for configuration to take effect
            await asyncio.sleep(30)  # 30 second settling time

            # Run A/B test if enabled
            if self.ab_testing_enabled:
                test_metrics = await self._run_ab_test(suggested_params)
            else:
                # Wait for metrics collection and use simple before/after comparison
                await asyncio.sleep(60)  # 1 minute measurement period
                test_metrics = await self._collect_current_metrics()

            # Calculate performance improvement
            improvement = await self._calculate_improvement(current_metrics, test_metrics)

            # Record result with ML optimizer
            await self.ml_optimizer.record_optimization_result(
                suggested_params, test_metrics, self.optimization_target
            )

            # Create optimization result
            result = OptimizationResult(
                parameters=suggested_params,
                performance_improvement=improvement,
                confidence=0.8,  # Would calculate based on statistical significance
                metrics_before=current_metrics,
                metrics_after=test_metrics,
                timestamp=time.time(),
                optimization_time=time.time() - start_time,
                successful=True
            )

            # Store result
            self.optimization_history.append(result)

            # Decide whether to keep the configuration
            if improvement > 0.05:  # 5% improvement threshold
                logger.info(f"Keeping optimization with {improvement:.1%} improvement")
            else:
                logger.info(f"Reverting optimization (only {improvement:.1%} improvement)")
                # Revert to previous configuration
                await self._revert_configuration()

            logger.info(f"Optimization cycle completed in {result.optimization_time:.1f}s")
            return result

        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")

            # Revert any changes on failure
            await self._revert_configuration()

            return OptimizationResult(
                parameters=self.current_config.copy(),
                performance_improvement=0.0,
                confidence=0.0,
                metrics_before=current_metrics,
                metrics_after=current_metrics,
                timestamp=time.time(),
                optimization_time=time.time() - start_time,
                successful=False,
                error_message=str(e)
            )

    async def _run_ab_test(self, test_parameters: dict[str, Any]) -> PerformanceMetrics:
        """Run A/B test between current and suggested configuration"""
        try:
            logger.info("Running A/B test for optimization")

            # Split traffic (simplified simulation)
            control_metrics_list = []
            test_metrics_list = []

            test_duration = self.ab_test_duration
            measurement_interval = 10  # 10 seconds
            measurements = test_duration // measurement_interval

            for i in range(measurements):
                # Simulate collecting metrics from both configurations
                # In production, would collect real metrics from traffic split

                control_metrics = await self._collect_current_metrics()

                # Simulate test configuration metrics with some variation
                test_metrics = await self._simulate_test_metrics(control_metrics, test_parameters)

                control_metrics_list.append(control_metrics)
                test_metrics_list.append(test_metrics)

                await asyncio.sleep(measurement_interval)

            # Aggregate results
            final_test_metrics = await self._aggregate_metrics(test_metrics_list)

            logger.info("A/B test completed")
            return final_test_metrics

        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            return await self._collect_current_metrics()

    async def _simulate_test_metrics(
        self,
        baseline: PerformanceMetrics,
        test_parameters: dict[str, Any]
    ) -> PerformanceMetrics:
        """Simulate test metrics based on parameter changes (for demo)"""
        try:
            import random

            # Simulate parameter impacts on metrics
            latency_factor = 1.0
            throughput_factor = 1.0
            resource_factor = 1.0

            # Connection pool impact
            if "max_connections" in test_parameters:
                conn_change = test_parameters["max_connections"] / self.current_config.get("max_connections", 100)
                throughput_factor *= (1.0 + (conn_change - 1.0) * 0.3)
                resource_factor *= (1.0 + (conn_change - 1.0) * 0.2)

            # Worker threads impact
            if "worker_threads" in test_parameters:
                thread_change = test_parameters["worker_threads"] / self.current_config.get("worker_threads", 8)
                latency_factor *= (1.0 + (1.0 - thread_change) * 0.2)
                throughput_factor *= (1.0 + (thread_change - 1.0) * 0.4)

            # Cache size impact
            if "cache_size_mb" in test_parameters:
                cache_change = test_parameters["cache_size_mb"] / self.current_config.get("cache_size_mb", 512)
                cache_impact = min(2.0, cache_change)  # Diminishing returns
                latency_factor *= (1.0 + (1.0 - cache_impact) * 0.1)
                resource_factor *= cache_change

            # Add some random variation
            latency_factor *= random.uniform(0.9, 1.1)
            throughput_factor *= random.uniform(0.9, 1.1)
            resource_factor *= random.uniform(0.9, 1.1)

            # Apply factors to baseline metrics
            return PerformanceMetrics(
                latency_p50=baseline.latency_p50 * latency_factor,
                latency_p95=baseline.latency_p95 * latency_factor,
                latency_p99=baseline.latency_p99 * latency_factor,
                throughput_rps=baseline.throughput_rps * throughput_factor,
                cpu_utilization=min(100.0, baseline.cpu_utilization * resource_factor),
                memory_utilization=min(100.0, baseline.memory_utilization * resource_factor),
                gpu_utilization=baseline.gpu_utilization,
                cache_hit_rate=min(1.0, baseline.cache_hit_rate * (1.0 + (cache_change - 1.0) * 0.1)),
                error_rate=baseline.error_rate,
                queue_depth=max(0, int(baseline.queue_depth / throughput_factor)),
                active_connections=baseline.active_connections,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Failed to simulate test metrics: {e}")
            return baseline

    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        try:
            # In production, would collect real metrics from monitoring systems
            # For demo, generate realistic synthetic metrics

            import random

            return PerformanceMetrics(
                latency_p50=random.uniform(50, 200),
                latency_p95=random.uniform(200, 800),
                latency_p99=random.uniform(500, 2000),
                throughput_rps=random.uniform(100, 1000),
                cpu_utilization=random.uniform(30, 80),
                memory_utilization=random.uniform(40, 75),
                gpu_utilization=random.uniform(20, 90),
                cache_hit_rate=random.uniform(0.7, 0.95),
                error_rate=random.uniform(0.001, 0.01),
                queue_depth=random.randint(0, 50),
                active_connections=random.randint(10, 200),
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return PerformanceMetrics()

    async def _aggregate_metrics(self, metrics_list: list[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate multiple metrics into single result"""
        try:
            if not metrics_list:
                return PerformanceMetrics()

            # Calculate averages
            return PerformanceMetrics(
                latency_p50=statistics.mean(m.latency_p50 for m in metrics_list),
                latency_p95=statistics.mean(m.latency_p95 for m in metrics_list),
                latency_p99=statistics.mean(m.latency_p99 for m in metrics_list),
                throughput_rps=statistics.mean(m.throughput_rps for m in metrics_list),
                cpu_utilization=statistics.mean(m.cpu_utilization for m in metrics_list),
                memory_utilization=statistics.mean(m.memory_utilization for m in metrics_list),
                gpu_utilization=statistics.mean(m.gpu_utilization for m in metrics_list),
                cache_hit_rate=statistics.mean(m.cache_hit_rate for m in metrics_list),
                error_rate=statistics.mean(m.error_rate for m in metrics_list),
                queue_depth=int(statistics.mean(m.queue_depth for m in metrics_list)),
                active_connections=int(statistics.mean(m.active_connections for m in metrics_list)),
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
            return metrics_list[0] if metrics_list else PerformanceMetrics()

    async def _calculate_improvement(
        self,
        before: PerformanceMetrics,
        after: PerformanceMetrics
    ) -> float:
        """Calculate performance improvement between two metric sets"""
        try:
            if self.optimization_target == OptimizationTarget.LATENCY:
                # Lower latency is better
                improvement = (before.latency_p95 - after.latency_p95) / before.latency_p95
                return improvement

            elif self.optimization_target == OptimizationTarget.THROUGHPUT:
                # Higher throughput is better
                improvement = (after.throughput_rps - before.throughput_rps) / before.throughput_rps
                return improvement

            elif self.optimization_target == OptimizationTarget.RESOURCE_EFFICIENCY:
                # Better resource utilization
                before_efficiency = before.throughput_rps / max(before.cpu_utilization, 1)
                after_efficiency = after.throughput_rps / max(after.cpu_utilization, 1)
                improvement = (after_efficiency - before_efficiency) / before_efficiency
                return improvement

            elif self.optimization_target == OptimizationTarget.BALANCED:
                # Weighted improvement across multiple metrics
                latency_improvement = (before.latency_p95 - after.latency_p95) / before.latency_p95
                throughput_improvement = (after.throughput_rps - before.throughput_rps) / before.throughput_rps
                error_improvement = (before.error_rate - after.error_rate) / max(before.error_rate, 0.001)

                balanced_improvement = (
                    0.4 * latency_improvement +
                    0.4 * throughput_improvement +
                    0.2 * error_improvement
                )
                return balanced_improvement

            return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate improvement: {e}")
            return 0.0

    async def _apply_parameters_to_system(self, parameters: dict[str, Any]) -> None:
        """Apply parameters to the actual system (placeholder)"""
        try:
            # In production, would apply parameters to:
            # - Connection pools
            # - Thread pools
            # - Cache configurations
            # - Timeout settings
            # - GPU memory allocation
            # - Compression settings
            # etc.

            logger.debug(f"Applied parameters to system: {parameters}")

        except Exception as e:
            logger.error(f"Failed to apply parameters to system: {e}")

    async def _revert_configuration(self) -> None:
        """Revert to previous known good configuration"""
        try:
            # Find last successful configuration
            for result in reversed(self.optimization_history):
                if result.successful and result.performance_improvement > 0:
                    await self.apply_configuration(result.parameters)
                    logger.info("Reverted to previous successful configuration")
                    return

            # If no successful configuration found, use baseline
            if self.baseline_metrics:
                baseline_params = {
                    name: param.current_value
                    for name, param in self.parameters.items()
                }
                await self.apply_configuration(baseline_params)
                logger.info("Reverted to baseline configuration")

        except Exception as e:
            logger.error(f"Failed to revert configuration: {e}")

    async def _optimization_loop(self) -> None:
        """Background optimization loop"""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)

                # Collect current metrics
                current_metrics = await self._collect_current_metrics()

                # Check if optimization is needed
                if await self._should_optimize(current_metrics):
                    logger.info("Triggering automatic optimization")
                    await self.trigger_optimization(current_metrics)

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)  # Back off on error

    async def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Determine if optimization should be triggered"""
        try:
            # Check performance thresholds
            if metrics.latency_p95 > self.performance_thresholds['latency_p95_ms']:
                return True

            if metrics.error_rate > self.performance_thresholds['error_rate']:
                return True

            if metrics.cpu_utilization > self.performance_thresholds['cpu_utilization']:
                return True

            if metrics.memory_utilization > self.performance_thresholds['memory_utilization']:
                return True

            # Check if enough time has passed since last optimization
            if self.optimization_history:
                last_optimization = self.optimization_history[-1].timestamp
                if time.time() - last_optimization < self.optimization_interval:
                    return False

            # Periodic optimization even if thresholds are met
            return True

        except Exception as e:
            logger.error(f"Failed to check optimization need: {e}")
            return False

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                # Check system health
                metrics = await self._collect_current_metrics()

                # Alert on performance degradation
                if metrics.latency_p95 > self.performance_thresholds['latency_p95_ms'] * 1.5:
                    logger.warning(f"High latency detected: {metrics.latency_p95:.1f}ms")

                if metrics.error_rate > self.performance_thresholds['error_rate'] * 2:
                    logger.warning(f"High error rate detected: {metrics.error_rate:.3f}")

                # Clean up old optimization history
                if len(self.optimization_history) > 500:
                    # Keep most recent 500 results
                    recent_results = list(self.optimization_history)[-500:]
                    self.optimization_history.clear()
                    self.optimization_history.extend(recent_results)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(120)  # Back off on error
