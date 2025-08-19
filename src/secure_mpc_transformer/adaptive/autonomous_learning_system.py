"""
Autonomous Learning System - Self-Improving Patterns Implementation

Advanced self-learning, adaptive optimization, and evolutionary algorithms
for autonomous SDLC execution with defensive security focus.
"""

import asyncio
import logging
import math
import pickle
import random
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Types of learning strategies"""
    REINFORCEMENT = "reinforcement"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN = "bayesian"
    QUANTUM_INSPIRED = "quantum_inspired"
    ENSEMBLE = "ensemble"


class AdaptationTrigger(Enum):
    """Events that trigger adaptation"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PATTERN_CHANGE = "pattern_change"
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"
    SCHEDULED_OPTIMIZATION = "scheduled_optimization"
    EXTERNAL_FEEDBACK = "external_feedback"
    RESOURCE_CONSTRAINT = "resource_constraint"


@dataclass
class LearningExperience:
    """Individual learning experience record"""
    timestamp: float
    context: dict[str, Any]
    action: str
    outcome: dict[str, float]  # metrics like success_rate, execution_time, etc.
    reward: float
    environment_state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationRule:
    """Rule for system adaptation"""
    id: str
    name: str
    trigger: AdaptationTrigger
    condition: str  # Expression to evaluate
    action: str  # Action to take
    priority: int  # Higher number = higher priority
    cooldown_seconds: float = 300.0  # Minimum time between applications
    last_applied: float | None = None


@dataclass
class LearningConfig:
    """Configuration for the learning system"""
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    memory_size: int = 10000
    adaptation_threshold: float = 0.05
    min_samples_for_learning: int = 10
    learning_strategies: list[LearningStrategy] = field(default_factory=lambda: [
        LearningStrategy.REINFORCEMENT, LearningStrategy.EVOLUTIONARY
    ])
    enable_quantum_learning: bool = True
    enable_meta_learning: bool = True
    auto_save_interval: float = 300.0  # 5 minutes


class AutonomousLearningSystem:
    """
    Advanced autonomous learning and adaptation system.
    
    Implements multiple learning strategies, adaptive optimization,
    and self-improving patterns for autonomous systems.
    """

    def __init__(self, config: LearningConfig | None = None,
                 persistence_path: Path | None = None):
        self.config = config or LearningConfig()
        self.persistence_path = persistence_path or Path("learning_state.pkl")

        # Learning memory
        self.experiences: list[LearningExperience] = []
        self.adaptation_rules: dict[str, AdaptationRule] = {}

        # Performance tracking
        self.performance_history: list[dict[str, float]] = []
        self.baseline_metrics: dict[str, float] = {}
        self.current_metrics: dict[str, float] = {}

        # Learning state
        self.knowledge_base: dict[str, Any] = {}
        self.learned_patterns: dict[str, Any] = {}
        self.optimization_parameters: dict[str, float] = {}

        # Quantum-inspired learning state
        self.quantum_learning_state = self._initialize_quantum_learning()

        # Meta-learning state
        self.meta_learning_state = {
            "strategy_performance": {strategy.value: 0.0 for strategy in LearningStrategy},
            "adaptation_history": [],
            "learning_efficiency": 1.0
        }

        # Evolutionary algorithm state
        self.population: list[dict[str, Any]] = []
        self.generation: int = 0

        # Auto-save task
        self.auto_save_task: asyncio.Task | None = None

        logger.info("AutonomousLearningSystem initialized")

        # Initialize default adaptation rules
        self._initialize_adaptation_rules()

        # Load previous state if available
        asyncio.create_task(self._load_state())

    def _initialize_quantum_learning(self) -> dict[str, Any]:
        """Initialize quantum-inspired learning parameters"""
        import numpy as np

        return {
            "superposition_weights": np.random.random(8) + 1j * np.random.random(8),
            "entanglement_matrix": np.random.random((8, 8)),
            "coherence_time": 100.0,
            "decoherence_rate": 0.01,
            "measurement_history": [],
            "quantum_advantage_score": 0.0
        }

    def _initialize_adaptation_rules(self) -> None:
        """Initialize default adaptation rules"""

        default_rules = [
            AdaptationRule(
                id="performance_degradation_response",
                name="Response to Performance Degradation",
                trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
                condition="current_performance < baseline_performance * 0.8",
                action="optimize_parameters",
                priority=10,
                cooldown_seconds=600.0
            ),
            AdaptationRule(
                id="error_rate_adaptation",
                name="Adapt to High Error Rates",
                trigger=AdaptationTrigger.ERROR_THRESHOLD_EXCEEDED,
                condition="error_rate > 0.1",
                action="increase_robustness",
                priority=9,
                cooldown_seconds=300.0
            ),
            AdaptationRule(
                id="resource_optimization",
                name="Optimize Resource Usage",
                trigger=AdaptationTrigger.RESOURCE_CONSTRAINT,
                condition="resource_utilization > 0.9",
                action="optimize_resource_allocation",
                priority=7,
                cooldown_seconds=300.0
            ),
            AdaptationRule(
                id="scheduled_learning",
                name="Scheduled Learning Optimization",
                trigger=AdaptationTrigger.SCHEDULED_OPTIMIZATION,
                condition="time_since_last_optimization > 3600",
                action="run_learning_cycle",
                priority=5,
                cooldown_seconds=3600.0
            )
        ]

        for rule in default_rules:
            self.adaptation_rules[rule.id] = rule

    async def record_experience(self, context: dict[str, Any], action: str,
                              outcome: dict[str, float], environment_state: dict[str, Any],
                              metadata: dict[str, Any] | None = None) -> None:
        """
        Record a learning experience.
        
        Args:
            context: Context in which the action was taken
            action: Action that was performed
            outcome: Measured outcomes/metrics
            environment_state: State of the environment
            metadata: Additional metadata
        """

        # Calculate reward based on outcome
        reward = self._calculate_reward(outcome, context)

        experience = LearningExperience(
            timestamp=time.time(),
            context=context,
            action=action,
            outcome=outcome,
            reward=reward,
            environment_state=environment_state,
            metadata=metadata or {}
        )

        # Add to memory
        self.experiences.append(experience)

        # Limit memory size
        if len(self.experiences) > self.config.memory_size:
            self.experiences = self.experiences[-self.config.memory_size:]

        # Update current metrics
        self.current_metrics.update(outcome)

        logger.debug(f"Recorded experience: {action} -> reward: {reward:.3f}")

        # Trigger learning if we have enough samples
        if len(self.experiences) >= self.config.min_samples_for_learning:
            await self._check_adaptation_triggers(experience)

    def _calculate_reward(self, outcome: dict[str, float], context: dict[str, Any]) -> float:
        """Calculate reward from outcome metrics"""

        # Define reward weights for different metrics
        reward_weights = {
            "success_rate": 1.0,
            "execution_time": -0.5,  # Lower is better
            "error_rate": -2.0,  # Lower is better
            "quality_score": 0.8,
            "performance_score": 0.7,
            "security_score": 1.2,  # Security is highly valued
            "efficiency_score": 0.6
        }

        total_reward = 0.0
        total_weight = 0.0

        for metric, value in outcome.items():
            if metric in reward_weights:
                weight = reward_weights[metric]

                # Normalize execution_time and error_rate (lower is better)
                if metric in ["execution_time", "error_rate"]:
                    # Convert to reward (inverse relationship)
                    normalized_value = 1.0 / (1.0 + value)
                else:
                    normalized_value = value

                total_reward += weight * normalized_value
                total_weight += abs(weight)

        # Normalize final reward
        if total_weight > 0:
            reward = total_reward / total_weight
        else:
            reward = 0.0

        # Apply context-based adjustments
        if context.get("critical_task", False):
            reward *= 1.5  # Amplify reward for critical tasks

        if context.get("security_sensitive", False):
            reward *= 1.3  # Amplify reward for security-sensitive tasks

        return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]

    async def _check_adaptation_triggers(self, latest_experience: LearningExperience) -> None:
        """Check if any adaptation rules should be triggered"""

        current_time = time.time()

        for rule in self.adaptation_rules.values():
            # Check cooldown
            if (rule.last_applied is not None and
                current_time - rule.last_applied < rule.cooldown_seconds):
                continue

            # Evaluate condition
            try:
                should_trigger = await self._evaluate_adaptation_condition(rule, latest_experience)

                if should_trigger:
                    logger.info(f"Triggering adaptation rule: {rule.name}")
                    await self._execute_adaptation_action(rule)
                    rule.last_applied = current_time

            except Exception as e:
                logger.error(f"Failed to evaluate adaptation rule {rule.id}: {e}")

    async def _evaluate_adaptation_condition(self, rule: AdaptationRule,
                                           experience: LearningExperience) -> bool:
        """Evaluate whether an adaptation rule condition is met"""

        # Create evaluation context
        eval_context = {
            "current_performance": self._get_current_performance(),
            "baseline_performance": self._get_baseline_performance(),
            "error_rate": self._get_recent_error_rate(),
            "resource_utilization": self._get_resource_utilization(),
            "time_since_last_optimization": self._get_time_since_last_optimization(),
            "recent_reward": experience.reward,
            "current_metrics": self.current_metrics
        }

        # Simple condition evaluation (in production, use more sophisticated parser)
        condition = rule.condition

        try:
            # Replace variables in condition
            for var, value in eval_context.items():
                condition = condition.replace(var, str(value))

            # Evaluate the condition
            result = eval(condition)
            return bool(result)

        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{rule.condition}': {e}")
            return False

    async def _execute_adaptation_action(self, rule: AdaptationRule) -> None:
        """Execute an adaptation action"""

        action = rule.action

        try:
            if action == "optimize_parameters":
                await self._optimize_parameters()
            elif action == "increase_robustness":
                await self._increase_robustness()
            elif action == "optimize_resource_allocation":
                await self._optimize_resource_allocation()
            elif action == "run_learning_cycle":
                await self._run_learning_cycle()
            else:
                logger.warning(f"Unknown adaptation action: {action}")

            # Record adaptation in meta-learning
            self.meta_learning_state["adaptation_history"].append({
                "timestamp": time.time(),
                "rule_id": rule.id,
                "action": action,
                "trigger": rule.trigger.value
            })

        except Exception as e:
            logger.error(f"Failed to execute adaptation action '{action}': {e}")

    async def _optimize_parameters(self) -> None:
        """Optimize system parameters using learned patterns"""
        logger.info("Optimizing parameters based on learned patterns")

        # Use different optimization strategies
        if LearningStrategy.REINFORCEMENT in self.config.learning_strategies:
            await self._reinforcement_learning_optimization()

        if LearningStrategy.EVOLUTIONARY in self.config.learning_strategies:
            await self._evolutionary_optimization()

        if LearningStrategy.QUANTUM_INSPIRED in self.config.learning_strategies and self.config.enable_quantum_learning:
            await self._quantum_inspired_optimization()

    async def _reinforcement_learning_optimization(self) -> None:
        """Use reinforcement learning to optimize parameters"""

        if not self.experiences:
            return

        # Q-learning style update
        recent_experiences = self.experiences[-100:]  # Last 100 experiences

        # Group experiences by action
        action_rewards = {}
        for exp in recent_experiences:
            action = exp.action
            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(exp.reward)

        # Update action values
        for action, rewards in action_rewards.items():
            if action not in self.knowledge_base:
                self.knowledge_base[action] = 0.0

            avg_reward = statistics.mean(rewards)

            # Q-learning update
            self.knowledge_base[action] += self.config.learning_rate * (
                avg_reward - self.knowledge_base[action]
            )

        logger.debug(f"Updated action values: {self.knowledge_base}")

    async def _evolutionary_optimization(self) -> None:
        """Use evolutionary algorithms to optimize parameters"""

        population_size = 20

        # Initialize population if empty
        if not self.population:
            self.population = self._initialize_population(population_size)

        # Evaluate fitness of current population
        fitness_scores = []
        for individual in self.population:
            fitness = await self._evaluate_individual_fitness(individual)
            fitness_scores.append(fitness)

        # Selection, crossover, and mutation
        new_population = []

        # Keep best individuals (elitism)
        elite_count = population_size // 4
        elite_indices = sorted(range(len(fitness_scores)),
                             key=lambda i: fitness_scores[i], reverse=True)[:elite_count]

        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # Generate new individuals through crossover and mutation
        while len(new_population) < population_size:
            parent1 = self._tournament_selection(self.population, fitness_scores)
            parent2 = self._tournament_selection(self.population, fitness_scores)

            child = self._crossover(parent1, parent2)
            child = self._mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Update optimization parameters with best individual
        best_individual = self.population[elite_indices[0]]
        self.optimization_parameters.update(best_individual)

        logger.info(f"Evolutionary optimization completed (Generation {self.generation})")

    def _initialize_population(self, size: int) -> list[dict[str, Any]]:
        """Initialize population for evolutionary algorithm"""
        population = []

        parameter_ranges = {
            "learning_rate": (0.001, 0.1),
            "exploration_rate": (0.01, 0.3),
            "batch_size": (8, 64),
            "timeout_multiplier": (0.5, 3.0),
            "retry_count": (1, 5),
            "cache_size": (100, 2000),
            "parallelism_factor": (0.5, 2.0)
        }

        for _ in range(size):
            individual = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                if param in ["retry_count", "cache_size"]:
                    individual[param] = random.randint(int(min_val), int(max_val))
                else:
                    individual[param] = random.uniform(min_val, max_val)

            population.append(individual)

        return population

    async def _evaluate_individual_fitness(self, individual: dict[str, Any]) -> float:
        """Evaluate fitness of an individual in the population"""

        # Simulate performance with these parameters
        # In a real implementation, this would run actual tests

        fitness = 0.0

        # Evaluate based on recent experiences and expected performance
        if self.experiences:
            recent_rewards = [exp.reward for exp in self.experiences[-20:]]
            base_performance = statistics.mean(recent_rewards) if recent_rewards else 0.0

            # Adjust based on parameter values
            learning_rate = individual.get("learning_rate", 0.01)
            exploration_rate = individual.get("exploration_rate", 0.1)

            # Optimal ranges give higher fitness
            lr_fitness = 1.0 - abs(learning_rate - 0.02) / 0.02  # Optimal around 0.02
            er_fitness = 1.0 - abs(exploration_rate - 0.1) / 0.1  # Optimal around 0.1

            fitness = base_performance + 0.2 * lr_fitness + 0.1 * er_fitness

        return max(0.0, fitness)

    def _tournament_selection(self, population: list[dict[str, Any]],
                            fitness_scores: list[float], tournament_size: int = 3) -> dict[str, Any]:
        """Tournament selection for evolutionary algorithm"""

        tournament_indices = random.sample(range(len(population)),
                                         min(tournament_size, len(population)))

        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])

        return population[best_idx].copy()

    def _crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> dict[str, Any]:
        """Crossover operation for evolutionary algorithm"""

        child = {}

        for key in parent1:
            if key in parent2:
                # Random blend crossover
                alpha = random.random()
                if isinstance(parent1[key], (int, float)):
                    child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]

                    # Ensure integer types remain integers
                    if isinstance(parent1[key], int):
                        child[key] = int(round(child[key]))
                else:
                    # Random choice for non-numeric values
                    child[key] = random.choice([parent1[key], parent2[key]])
            else:
                child[key] = parent1[key]

        return child

    def _mutate(self, individual: dict[str, Any], mutation_rate: float = 0.1) -> dict[str, Any]:
        """Mutation operation for evolutionary algorithm"""

        mutated = individual.copy()

        for key, value in mutated.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    # Gaussian mutation
                    mutated[key] = value + random.gauss(0, value * 0.1)
                    mutated[key] = max(0.0, mutated[key])  # Ensure positive
                elif isinstance(value, int):
                    # Random increment/decrement
                    change = random.choice([-1, 0, 1])
                    mutated[key] = max(1, value + change)

        return mutated

    async def _quantum_inspired_optimization(self) -> None:
        """Use quantum-inspired algorithms for optimization"""

        import numpy as np

        # Quantum state evolution
        quantum_state = self.quantum_learning_state

        # Decoherence
        coherence_decay = math.exp(-quantum_state["decoherence_rate"])
        quantum_state["coherence_time"] *= coherence_decay

        # Quantum measurement affects optimization parameters
        superposition = quantum_state["superposition_weights"]
        measurement = np.abs(superposition) ** 2
        measurement = measurement / np.sum(measurement)  # Normalize

        # Map quantum measurements to optimization parameters
        param_names = ["learning_rate", "exploration_rate", "batch_size", "timeout_multiplier"]

        for i, param in enumerate(param_names[:len(measurement)]):
            if param in self.optimization_parameters:
                # Quantum-guided parameter adjustment
                quantum_influence = measurement[i] * 0.1  # Small influence
                current_value = self.optimization_parameters[param]

                if param == "learning_rate":
                    self.optimization_parameters[param] = max(0.001,
                        current_value + quantum_influence * 0.01)
                elif param == "exploration_rate":
                    self.optimization_parameters[param] = max(0.01, min(0.3,
                        current_value + quantum_influence * 0.05))

        # Update quantum advantage score
        if self.experiences:
            recent_performance = statistics.mean([exp.reward for exp in self.experiences[-10:]])
            quantum_state["quantum_advantage_score"] = 0.9 * quantum_state["quantum_advantage_score"] + 0.1 * recent_performance

        logger.debug(f"Quantum optimization applied, advantage score: {quantum_state['quantum_advantage_score']:.3f}")

    async def _increase_robustness(self) -> None:
        """Increase system robustness in response to errors"""
        logger.info("Increasing system robustness")

        # Adjust parameters to favor robustness over speed
        if "timeout_multiplier" in self.optimization_parameters:
            self.optimization_parameters["timeout_multiplier"] *= 1.2

        if "retry_count" in self.optimization_parameters:
            self.optimization_parameters["retry_count"] = min(5,
                self.optimization_parameters["retry_count"] + 1)

        # Record robustness improvement in knowledge base
        self.knowledge_base["robustness_level"] = self.knowledge_base.get("robustness_level", 1.0) * 1.1

    async def _optimize_resource_allocation(self) -> None:
        """Optimize resource allocation based on usage patterns"""
        logger.info("Optimizing resource allocation")

        # Analyze resource usage patterns from experiences
        resource_metrics = []
        for exp in self.experiences[-50:]:  # Last 50 experiences
            env_state = exp.environment_state
            if "resource_utilization" in env_state:
                resource_metrics.append(env_state["resource_utilization"])

        if resource_metrics:
            avg_utilization = statistics.mean(resource_metrics)

            # Adjust parallelism based on utilization
            if avg_utilization > 0.8:
                # High utilization - reduce parallelism
                if "parallelism_factor" in self.optimization_parameters:
                    self.optimization_parameters["parallelism_factor"] *= 0.9
            elif avg_utilization < 0.4:
                # Low utilization - increase parallelism
                if "parallelism_factor" in self.optimization_parameters:
                    self.optimization_parameters["parallelism_factor"] *= 1.1

    async def _run_learning_cycle(self) -> None:
        """Run a complete learning cycle"""
        logger.info("Running learning cycle")

        # Pattern recognition
        await self._identify_patterns()

        # Meta-learning update
        await self._update_meta_learning()

        # Knowledge consolidation
        await self._consolidate_knowledge()

    async def _identify_patterns(self) -> None:
        """Identify patterns in experiences"""

        if len(self.experiences) < 20:
            return

        # Group experiences by context similarity
        context_groups = {}

        for exp in self.experiences[-100:]:  # Analyze last 100 experiences
            # Simple context grouping by action and key context features
            context_key = f"{exp.action}_{exp.context.get('task_type', 'unknown')}"

            if context_key not in context_groups:
                context_groups[context_key] = []

            context_groups[context_key].append(exp)

        # Identify patterns in each group
        for context_key, group_experiences in context_groups.items():
            if len(group_experiences) >= 5:  # Need minimum samples

                rewards = [exp.reward for exp in group_experiences]
                avg_reward = statistics.mean(rewards)
                reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

                # Identify successful patterns
                if avg_reward > 0.7:  # High performance pattern
                    pattern = {
                        "context_key": context_key,
                        "avg_reward": avg_reward,
                        "stability": 1.0 / (1.0 + reward_std),
                        "sample_count": len(group_experiences),
                        "success_factors": self._extract_success_factors(group_experiences)
                    }

                    self.learned_patterns[context_key] = pattern
                    logger.debug(f"Identified successful pattern: {context_key} (reward: {avg_reward:.3f})")

    def _extract_success_factors(self, experiences: list[LearningExperience]) -> dict[str, Any]:
        """Extract factors that contribute to success"""

        # Simple analysis of common factors in successful experiences
        high_reward_exps = [exp for exp in experiences if exp.reward > 0.5]

        if not high_reward_exps:
            return {}

        success_factors = {}

        # Analyze environment state factors
        env_factors = {}
        for exp in high_reward_exps:
            for key, value in exp.environment_state.items():
                if key not in env_factors:
                    env_factors[key] = []
                env_factors[key].append(value)

        # Find stable factors (low variance)
        for key, values in env_factors.items():
            if all(isinstance(v, (int, float)) for v in values):
                if len(values) > 1:
                    variance = statistics.variance(values)
                    mean_val = statistics.mean(values)

                    if variance < mean_val * 0.1:  # Low relative variance
                        success_factors[f"stable_{key}"] = mean_val

        return success_factors

    async def _update_meta_learning(self) -> None:
        """Update meta-learning state based on learning strategy performance"""

        # Evaluate performance of different learning strategies
        strategy_rewards = {strategy.value: [] for strategy in self.config.learning_strategies}

        # This is a simplified evaluation - in practice, you'd track which
        # strategies were used for which adaptations and their outcomes

        for exp in self.experiences[-50:]:  # Recent experiences
            for strategy in self.config.learning_strategies:
                if strategy.value in exp.metadata.get("strategies_used", []):
                    strategy_rewards[strategy.value].append(exp.reward)

        # Update strategy performance scores
        for strategy, rewards in strategy_rewards.items():
            if rewards:
                avg_reward = statistics.mean(rewards)
                current_score = self.meta_learning_state["strategy_performance"][strategy]

                # Exponential moving average update
                self.meta_learning_state["strategy_performance"][strategy] = (
                    0.9 * current_score + 0.1 * avg_reward
                )

        # Update learning efficiency
        if self.experiences:
            recent_rewards = [exp.reward for exp in self.experiences[-20:]]
            avg_recent_reward = statistics.mean(recent_rewards) if recent_rewards else 0.0

            self.meta_learning_state["learning_efficiency"] = (
                0.8 * self.meta_learning_state["learning_efficiency"] + 0.2 * avg_recent_reward
            )

    async def _consolidate_knowledge(self) -> None:
        """Consolidate learned knowledge and remove outdated information"""

        # Remove old experiences that are no longer relevant
        current_time = time.time()
        retention_period = 7 * 24 * 3600  # 1 week

        self.experiences = [
            exp for exp in self.experiences
            if current_time - exp.timestamp < retention_period
        ]

        # Consolidate patterns by merging similar ones
        consolidated_patterns = {}

        for pattern_key, pattern in self.learned_patterns.items():
            # Simple consolidation - in practice, use more sophisticated clustering
            base_key = pattern_key.split("_")[0]  # First part of the key

            if base_key not in consolidated_patterns:
                consolidated_patterns[base_key] = pattern
            else:
                # Merge patterns
                existing = consolidated_patterns[base_key]
                if pattern["avg_reward"] > existing["avg_reward"]:
                    consolidated_patterns[base_key] = pattern

        self.learned_patterns = consolidated_patterns

        logger.info(f"Knowledge consolidation completed. Patterns: {len(self.learned_patterns)}, "
                   f"Experiences: {len(self.experiences)}")

    # Utility methods for adaptation condition evaluation
    def _get_current_performance(self) -> float:
        """Get current system performance score"""
        if not self.experiences:
            return 0.5

        recent_rewards = [exp.reward for exp in self.experiences[-10:]]
        return statistics.mean(recent_rewards) if recent_rewards else 0.5

    def _get_baseline_performance(self) -> float:
        """Get baseline performance score"""
        if "baseline_performance" in self.baseline_metrics:
            return self.baseline_metrics["baseline_performance"]

        # Calculate baseline from early experiences
        if len(self.experiences) > 20:
            early_rewards = [exp.reward for exp in self.experiences[:20]]
            baseline = statistics.mean(early_rewards)
            self.baseline_metrics["baseline_performance"] = baseline
            return baseline

        return 0.5  # Default baseline

    def _get_recent_error_rate(self) -> float:
        """Get recent error rate"""
        recent_experiences = self.experiences[-20:] if self.experiences else []

        if not recent_experiences:
            return 0.0

        error_count = sum(1 for exp in recent_experiences
                         if exp.outcome.get("error_rate", 0) > 0 or exp.reward < 0)

        return error_count / len(recent_experiences)

    def _get_resource_utilization(self) -> float:
        """Get current resource utilization"""
        if self.experiences:
            latest_exp = self.experiences[-1]
            return latest_exp.environment_state.get("resource_utilization", 0.5)

        return 0.5

    def _get_time_since_last_optimization(self) -> float:
        """Get time since last optimization"""
        last_optimization = 0.0

        for adaptation in self.meta_learning_state["adaptation_history"]:
            if adaptation["action"] == "run_learning_cycle":
                last_optimization = max(last_optimization, adaptation["timestamp"])

        return time.time() - last_optimization

    async def get_learning_insights(self) -> dict[str, Any]:
        """Get comprehensive learning insights and recommendations"""

        insights = {
            "learning_summary": {
                "total_experiences": len(self.experiences),
                "identified_patterns": len(self.learned_patterns),
                "current_performance": self._get_current_performance(),
                "baseline_performance": self._get_baseline_performance(),
                "learning_efficiency": self.meta_learning_state["learning_efficiency"]
            },
            "strategy_performance": self.meta_learning_state["strategy_performance"].copy(),
            "optimization_parameters": self.optimization_parameters.copy(),
            "successful_patterns": [],
            "recommendations": []
        }

        # Add successful patterns
        for pattern_key, pattern in self.learned_patterns.items():
            if pattern["avg_reward"] > 0.7:
                insights["successful_patterns"].append({
                    "pattern": pattern_key,
                    "performance": pattern["avg_reward"],
                    "stability": pattern["stability"],
                    "usage_count": pattern["sample_count"]
                })

        # Generate recommendations
        current_perf = self._get_current_performance()
        baseline_perf = self._get_baseline_performance()

        if current_perf < baseline_perf * 0.9:
            insights["recommendations"].append({
                "type": "performance",
                "message": "System performance has degraded. Consider running optimization.",
                "priority": "high"
            })

        if self._get_recent_error_rate() > 0.1:
            insights["recommendations"].append({
                "type": "reliability",
                "message": "High error rate detected. Increase robustness measures.",
                "priority": "high"
            })

        if len(self.experiences) < 50:
            insights["recommendations"].append({
                "type": "data",
                "message": "Limited learning data. More experiences needed for better adaptation.",
                "priority": "medium"
            })

        # Quantum learning insights
        if self.config.enable_quantum_learning:
            quantum_advantage = self.quantum_learning_state["quantum_advantage_score"]
            insights["quantum_learning"] = {
                "advantage_score": quantum_advantage,
                "coherence_time": self.quantum_learning_state["coherence_time"],
                "effectiveness": "high" if quantum_advantage > 0.6 else "medium" if quantum_advantage > 0.3 else "low"
            }

        return insights

    async def _save_state(self) -> None:
        """Save learning state to persistence"""
        try:
            state = {
                "experiences": self.experiences[-1000:],  # Save last 1000 experiences
                "learned_patterns": self.learned_patterns,
                "optimization_parameters": self.optimization_parameters,
                "baseline_metrics": self.baseline_metrics,
                "meta_learning_state": self.meta_learning_state,
                "knowledge_base": self.knowledge_base,
                "generation": self.generation
            }

            with open(self.persistence_path, 'wb') as f:
                pickle.dump(state, f)

            logger.debug(f"Learning state saved to {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    async def _load_state(self) -> None:
        """Load learning state from persistence"""
        try:
            if self.persistence_path.exists():
                with open(self.persistence_path, 'rb') as f:
                    state = pickle.load(f)

                self.experiences = state.get("experiences", [])
                self.learned_patterns = state.get("learned_patterns", {})
                self.optimization_parameters = state.get("optimization_parameters", {})
                self.baseline_metrics = state.get("baseline_metrics", {})
                self.meta_learning_state.update(state.get("meta_learning_state", {}))
                self.knowledge_base = state.get("knowledge_base", {})
                self.generation = state.get("generation", 0)

                logger.info(f"Learning state loaded from {self.persistence_path}")

        except Exception as e:
            logger.warning(f"Failed to load learning state: {e}")

    async def start_auto_save(self) -> None:
        """Start automatic state saving"""
        if self.auto_save_task is None:
            self.auto_save_task = asyncio.create_task(self._auto_save_loop())
            logger.info("Auto-save started")

    async def stop_auto_save(self) -> None:
        """Stop automatic state saving"""
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
            self.auto_save_task = None
            logger.info("Auto-save stopped")

    async def _auto_save_loop(self) -> None:
        """Auto-save loop"""
        while True:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save error: {e}")

    async def shutdown(self) -> None:
        """Shutdown the learning system"""
        logger.info("Shutting down AutonomousLearningSystem")

        await self.stop_auto_save()
        await self._save_state()

        logger.info("AutonomousLearningSystem shutdown complete")
