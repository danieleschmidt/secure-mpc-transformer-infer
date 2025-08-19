"""
Experimental Research Framework for Novel Algorithm Validation

Comprehensive experimental framework for rigorous scientific validation of
novel quantum-enhanced MPC algorithms. Designed for academic publication
and peer review with full reproducibility guarantees.

Key Features:
1. Controlled experimental design with proper baselines
2. Statistical significance testing and confidence intervals
3. Reproducible experiment management with versioning
4. Automated data collection and analysis pipelines
5. Publication-ready result generation and visualization

Research Standards:
- Follows FAIR data principles (Findable, Accessible, Interoperable, Reusable)
- Implements proper experimental controls and randomization
- Provides statistical power analysis and effect size calculations
- Generates comprehensive experiment documentation
- Supports multi-dataset validation and cross-validation
"""

import asyncio
import hashlib
import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of research experiments"""
    ALGORITHM_COMPARISON = "algorithm_comparison"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SECURITY_VALIDATION = "security_validation"
    SCALABILITY_STUDY = "scalability_study"
    ABLATION_STUDY = "ablation_study"
    CROSS_VALIDATION = "cross_validation"


class ExperimentStatus(Enum):
    """Experiment execution status"""
    DESIGNED = "designed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZED = "analyzed"
    PUBLISHED = "published"


@dataclass
class ExperimentConfig:
    """Configuration for research experiment"""
    experiment_name: str
    experiment_type: ExperimentType
    description: str
    hypothesis: str

    # Experimental design
    independent_variables: list[str]
    dependent_variables: list[str]
    control_variables: list[str]

    # Statistical parameters
    significance_level: float = 0.05
    statistical_power: float = 0.8
    effect_size: float = 0.5
    min_sample_size: int = 30

    # Reproducibility
    random_seed: int = 42
    num_repetitions: int = 10
    cross_validation_folds: int = 5

    # Resource limits
    max_runtime_hours: float = 24.0
    max_memory_gb: float = 32.0

    # Output configuration
    save_intermediate_results: bool = True
    generate_visualizations: bool = True
    create_publication_figures: bool = True


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    experiment_id: str
    run_id: str
    algorithm_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    execution_time: float
    memory_usage: float
    success: bool
    error_message: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results"""
    sample_size: int
    mean: float
    std_dev: float
    confidence_interval: tuple[float, float]
    p_value: float
    effect_size: float
    statistical_power: float
    significant: bool


class ResearchExperimentRunner:
    """
    Comprehensive research experiment runner for novel algorithm validation.
    
    Provides rigorous experimental methodology for academic research with
    proper statistical analysis and reproducibility guarantees.
    """

    def __init__(self,
                 experiment_config: ExperimentConfig,
                 output_directory: str = "./experiment_results",
                 enable_logging: bool = True):
        self.config = experiment_config
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment state
        self.experiment_id = self._generate_experiment_id()
        self.status = ExperimentStatus.DESIGNED
        self.results: list[ExperimentResult] = []
        self.statistical_analyses: dict[str, StatisticalAnalysis] = {}

        # Reproducibility setup
        np.random.seed(self.config.random_seed)

        # Logging setup
        if enable_logging:
            self._setup_experiment_logging()

        logger.info(f"Initialized experiment: {self.config.experiment_name}")
        logger.info(f"Experiment ID: {self.experiment_id}")

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(asdict(self.config), sort_keys=True).encode()
        ).hexdigest()[:8]

        return f"{self.config.experiment_name}_{timestamp}_{config_hash}"

    def _setup_experiment_logging(self) -> None:
        """Setup dedicated logging for experiment"""
        log_file = self.output_dir / f"{self.experiment_id}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        logger.info(f"Experiment logging initialized: {log_file}")

    async def run_experiment(self,
                           algorithms: dict[str, Callable],
                           datasets: list[dict[str, Any]],
                           parameter_grids: dict[str, list[Any]] | None = None) -> dict[str, Any]:
        """
        Run comprehensive research experiment with multiple algorithms and datasets.
        
        Args:
            algorithms: Dictionary of algorithm implementations {name: function}
            datasets: List of dataset configurations for testing
            parameter_grids: Parameter grids for hyperparameter exploration
            
        Returns:
            Comprehensive experiment results with statistical analysis
        """

        logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.status = ExperimentStatus.RUNNING
        experiment_start = datetime.now()

        try:
            # Validate experimental setup
            self._validate_experimental_setup(algorithms, datasets)

            # Generate experimental conditions
            experimental_conditions = self._generate_experimental_conditions(
                algorithms, datasets, parameter_grids
            )

            logger.info(f"Generated {len(experimental_conditions)} experimental conditions")

            # Execute experimental runs
            await self._execute_experimental_runs(experimental_conditions)

            # Perform statistical analysis
            statistical_results = self._perform_statistical_analysis()

            # Generate experiment report
            experiment_report = self._generate_experiment_report(statistical_results)

            # Save results
            await self._save_experiment_results(experiment_report)

            # Generate visualizations
            if self.config.generate_visualizations:
                await self._generate_visualizations()

            self.status = ExperimentStatus.COMPLETED
            experiment_time = (datetime.now() - experiment_start).total_seconds()

            logger.info(f"Experiment completed successfully in {experiment_time:.2f} seconds")

            return experiment_report

        except Exception as e:
            self.status = ExperimentStatus.FAILED
            logger.error(f"Experiment failed: {str(e)}")
            raise

    def _validate_experimental_setup(self,
                                   algorithms: dict[str, Callable],
                                   datasets: list[dict[str, Any]]) -> None:
        """Validate experimental setup for scientific rigor"""

        # Check minimum requirements
        if len(algorithms) < 2:
            raise ValueError("Need at least 2 algorithms for comparison")

        if len(datasets) < 1:
            raise ValueError("Need at least 1 dataset for validation")

        # Validate dependent variables are measurable
        required_metrics = self.config.dependent_variables
        if not required_metrics:
            raise ValueError("Must specify dependent variables (metrics) to measure")

        # Check statistical power requirements
        if self.config.min_sample_size < 10:
            logger.warning("Sample size < 10 may have insufficient statistical power")

        # Validate reproducibility settings
        if self.config.random_seed is None:
            logger.warning("No random seed specified - results may not be reproducible")

        logger.info("Experimental setup validation passed")

    def _generate_experimental_conditions(self,
                                        algorithms: dict[str, Callable],
                                        datasets: list[dict[str, Any]],
                                        parameter_grids: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
        """Generate all experimental conditions for systematic testing"""

        conditions = []

        for dataset in datasets:
            for algo_name, algo_func in algorithms.items():

                # Base condition
                base_condition = {
                    "algorithm_name": algo_name,
                    "algorithm_function": algo_func,
                    "dataset": dataset,
                    "parameters": {}
                }

                # Parameter grid exploration
                if parameter_grids and algo_name in parameter_grids:
                    param_grid = parameter_grids[algo_name]
                    param_combinations = self._generate_parameter_combinations(param_grid)

                    for params in param_combinations:
                        condition = base_condition.copy()
                        condition["parameters"] = params
                        conditions.append(condition)
                else:
                    conditions.append(base_condition)

        # Ensure minimum sample size through repetitions
        if len(conditions) * self.config.num_repetitions < self.config.min_sample_size:
            logger.warning(f"Total runs ({len(conditions) * self.config.num_repetitions}) " +
                         f"< minimum sample size ({self.config.min_sample_size})")

        return conditions

    def _generate_parameter_combinations(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Generate all combinations of parameters from grid"""

        if not param_grid:
            return [{}]

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = []

        def generate_recursive(current_params: dict[str, Any], param_index: int) -> None:
            if param_index >= len(param_names):
                combinations.append(current_params.copy())
                return

            param_name = param_names[param_index]
            for value in param_values[param_index]:
                current_params[param_name] = value
                generate_recursive(current_params, param_index + 1)
                del current_params[param_name]

        generate_recursive({}, 0)
        return combinations

    async def _execute_experimental_runs(self, conditions: list[dict[str, Any]]) -> None:
        """Execute all experimental runs with proper randomization"""

        total_runs = len(conditions) * self.config.num_repetitions
        completed_runs = 0

        logger.info(f"Executing {total_runs} experimental runs")

        # Randomize execution order for unbiased results
        execution_order = []
        for condition in conditions:
            for rep in range(self.config.num_repetitions):
                execution_order.append((condition, rep))

        np.random.shuffle(execution_order)

        # Execute runs
        for condition, repetition in execution_order:
            try:
                result = await self._execute_single_run(condition, repetition)
                self.results.append(result)

                completed_runs += 1
                if completed_runs % 10 == 0 or completed_runs == total_runs:
                    logger.info(f"Completed {completed_runs}/{total_runs} runs " +
                              f"({completed_runs/total_runs*100:.1f}%)")

                # Save intermediate results
                if self.config.save_intermediate_results and completed_runs % 50 == 0:
                    await self._save_intermediate_results()

            except Exception as e:
                logger.error(f"Run failed for {condition['algorithm_name']}: {str(e)}")

                # Record failed run
                failed_result = ExperimentResult(
                    experiment_id=self.experiment_id,
                    run_id=f"{condition['algorithm_name']}_{repetition}",
                    algorithm_name=condition['algorithm_name'],
                    parameters=condition['parameters'],
                    metrics={},
                    execution_time=0.0,
                    memory_usage=0.0,
                    success=False,
                    error_message=str(e),
                    timestamp=datetime.now()
                )
                self.results.append(failed_result)

        logger.info(f"Experimental runs completed: {completed_runs} total runs")

    async def _execute_single_run(self,
                                condition: dict[str, Any],
                                repetition: int) -> ExperimentResult:
        """Execute a single experimental run"""

        start_time = datetime.now()
        run_id = f"{condition['algorithm_name']}_rep{repetition}_{start_time.strftime('%H%M%S')}"

        # Set random seed for reproducibility
        run_seed = self.config.random_seed + repetition
        np.random.seed(run_seed)

        try:
            # Execute algorithm
            algorithm_func = condition['algorithm_function']
            dataset = condition['dataset']
            parameters = condition['parameters']

            # Measure resource usage
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run algorithm
            result_data = await self._run_algorithm_with_timeout(
                algorithm_func, dataset, parameters
            )

            # Measure final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory

            execution_time = (datetime.now() - start_time).total_seconds()

            # Extract metrics
            metrics = self._extract_metrics(result_data, dataset)

            return ExperimentResult(
                experiment_id=self.experiment_id,
                run_id=run_id,
                algorithm_name=condition['algorithm_name'],
                parameters=parameters,
                metrics=metrics,
                execution_time=execution_time,
                memory_usage=memory_usage,
                success=True,
                timestamp=start_time,
                metadata={
                    "repetition": repetition,
                    "random_seed": run_seed,
                    "dataset_info": dataset.get("metadata", {})
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            return ExperimentResult(
                experiment_id=self.experiment_id,
                run_id=run_id,
                algorithm_name=condition['algorithm_name'],
                parameters=condition.get('parameters', {}),
                metrics={},
                execution_time=execution_time,
                memory_usage=0.0,
                success=False,
                error_message=str(e),
                timestamp=start_time
            )

    async def _run_algorithm_with_timeout(self,
                                        algorithm_func: Callable,
                                        dataset: dict[str, Any],
                                        parameters: dict[str, Any]) -> Any:
        """Run algorithm with timeout protection"""

        timeout_seconds = self.config.max_runtime_hours * 3600

        try:
            result = await asyncio.wait_for(
                algorithm_func(dataset, **parameters),
                timeout=timeout_seconds
            )
            return result

        except asyncio.TimeoutError:
            raise TimeoutError(f"Algorithm execution exceeded {timeout_seconds} seconds")

    def _extract_metrics(self, result_data: Any, dataset: dict[str, Any]) -> dict[str, float]:
        """Extract metrics from algorithm result"""

        metrics = {}

        # Standard performance metrics
        if hasattr(result_data, 'accuracy'):
            metrics['accuracy'] = float(result_data.accuracy)

        if hasattr(result_data, 'execution_time'):
            metrics['algorithm_time'] = float(result_data.execution_time)

        if hasattr(result_data, 'memory_usage'):
            metrics['algorithm_memory'] = float(result_data.memory_usage)

        # Research-specific metrics
        if hasattr(result_data, 'quantum_advantage'):
            metrics['quantum_advantage'] = 1.0 if result_data.quantum_advantage else 0.0

        if hasattr(result_data, 'convergence_rate'):
            metrics['convergence_rate'] = float(result_data.convergence_rate)

        if hasattr(result_data, 'security_score'):
            metrics['security_score'] = float(result_data.security_score)

        # Handle dictionary results
        if isinstance(result_data, dict):
            for key in self.config.dependent_variables:
                if key in result_data:
                    metrics[key] = float(result_data[key])

        return metrics

    def _perform_statistical_analysis(self) -> dict[str, Any]:
        """Perform comprehensive statistical analysis of experimental results"""

        logger.info("Performing statistical analysis")

        # Convert results to DataFrame for analysis
        df = self._results_to_dataframe()

        if df.empty:
            raise ValueError("No successful experimental results to analyze")

        statistical_results = {
            "descriptive_statistics": self._compute_descriptive_statistics(df),
            "significance_tests": self._perform_significance_tests(df),
            "effect_size_analysis": self._compute_effect_sizes(df),
            "power_analysis": self._compute_statistical_power(df),
            "confidence_intervals": self._compute_confidence_intervals(df)
        }

        # Store detailed statistical analyses
        for metric in self.config.dependent_variables:
            if metric in df.columns:
                self.statistical_analyses[metric] = self._analyze_metric(df, metric)

        return statistical_results

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert experiment results to pandas DataFrame"""

        rows = []

        for result in self.results:
            if not result.success:
                continue

            row = {
                "algorithm": result.algorithm_name,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage,
                "repetition": result.metadata.get("repetition", 0) if result.metadata else 0
            }

            # Add parameters
            for param_name, param_value in result.parameters.items():
                row[f"param_{param_name}"] = param_value

            # Add metrics
            row.update(result.metrics)

            rows.append(row)

        return pd.DataFrame(rows)

    def _compute_descriptive_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute descriptive statistics for all metrics"""

        stats_dict = {}

        for metric in self.config.dependent_variables:
            if metric in df.columns:
                metric_data = df[metric].dropna()

                if len(metric_data) > 0:
                    stats_dict[metric] = {
                        "count": len(metric_data),
                        "mean": float(metric_data.mean()),
                        "std": float(metric_data.std()),
                        "min": float(metric_data.min()),
                        "max": float(metric_data.max()),
                        "median": float(metric_data.median()),
                        "q25": float(metric_data.quantile(0.25)),
                        "q75": float(metric_data.quantile(0.75))
                    }

        return stats_dict

    def _perform_significance_tests(self, df: pd.DataFrame) -> dict[str, Any]:
        """Perform statistical significance tests between algorithms"""

        significance_results = {}
        algorithms = df['algorithm'].unique()

        if len(algorithms) < 2:
            return {"error": "Need at least 2 algorithms for significance testing"}

        for metric in self.config.dependent_variables:
            if metric not in df.columns:
                continue

            metric_results = {}

            # Pairwise comparisons between algorithms
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms[i+1:], i+1):

                    data1 = df[df['algorithm'] == alg1][metric].dropna()
                    data2 = df[df['algorithm'] == alg2][metric].dropna()

                    if len(data1) > 1 and len(data2) > 1:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(data1, data2)

                        # Check normality assumption
                        _, p_norm1 = stats.shapiro(data1) if len(data1) < 5000 else (0, 0)
                        _, p_norm2 = stats.shapiro(data2) if len(data2) < 5000 else (0, 0)

                        # Use Mann-Whitney U test if normality violated
                        if p_norm1 < 0.05 or p_norm2 < 0.05:
                            u_stat, p_value_nonparam = stats.mannwhitneyu(data1, data2)
                            test_used = "Mann-Whitney U"
                            test_statistic = u_stat
                        else:
                            test_used = "t-test"
                            test_statistic = t_stat

                        comparison_key = f"{alg1}_vs_{alg2}"
                        metric_results[comparison_key] = {
                            "test_used": test_used,
                            "test_statistic": float(test_statistic),
                            "p_value": float(p_value),
                            "significant": p_value < self.config.significance_level,
                            "sample_size_1": len(data1),
                            "sample_size_2": len(data2),
                            "mean_1": float(data1.mean()),
                            "mean_2": float(data2.mean())
                        }

            significance_results[metric] = metric_results

        return significance_results

    def _compute_effect_sizes(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute effect sizes (Cohen's d) between algorithms"""

        effect_sizes = {}
        algorithms = df['algorithm'].unique()

        for metric in self.config.dependent_variables:
            if metric not in df.columns:
                continue

            metric_effects = {}

            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms[i+1:], i+1):

                    data1 = df[df['algorithm'] == alg1][metric].dropna()
                    data2 = df[df['algorithm'] == alg2][metric].dropna()

                    if len(data1) > 1 and len(data2) > 1:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() +
                                            (len(data2) - 1) * data2.var()) /
                                           (len(data1) + len(data2) - 2))

                        if pooled_std > 0:
                            cohens_d = (data1.mean() - data2.mean()) / pooled_std
                        else:
                            cohens_d = 0.0

                        # Effect size interpretation
                        if abs(cohens_d) < 0.2:
                            interpretation = "negligible"
                        elif abs(cohens_d) < 0.5:
                            interpretation = "small"
                        elif abs(cohens_d) < 0.8:
                            interpretation = "medium"
                        else:
                            interpretation = "large"

                        comparison_key = f"{alg1}_vs_{alg2}"
                        metric_effects[comparison_key] = {
                            "cohens_d": float(cohens_d),
                            "interpretation": interpretation,
                            "abs_effect_size": float(abs(cohens_d))
                        }

            effect_sizes[metric] = metric_effects

        return effect_sizes

    def _compute_statistical_power(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute statistical power of the experiment"""

        power_analysis = {}

        for metric in self.config.dependent_variables:
            if metric not in df.columns:
                continue

            metric_data = df[metric].dropna()

            if len(metric_data) > 10:
                # Estimate power based on observed effect size and sample size
                observed_std = metric_data.std()
                sample_size = len(metric_data)

                # Simplified power calculation
                # In practice, would use more sophisticated power analysis
                effect_size = self.config.effect_size
                alpha = self.config.significance_level

                # Approximate power calculation
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = (effect_size * np.sqrt(sample_size/2) - z_alpha)
                power = stats.norm.cdf(z_beta)

                power_analysis[metric] = {
                    "statistical_power": float(max(0.0, min(1.0, power))),
                    "sample_size": sample_size,
                    "effect_size_target": effect_size,
                    "significance_level": alpha,
                    "power_adequate": power >= self.config.statistical_power
                }

        return power_analysis

    def _compute_confidence_intervals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute confidence intervals for metrics"""

        confidence_intervals = {}
        confidence_level = 1 - self.config.significance_level

        for metric in self.config.dependent_variables:
            if metric not in df.columns:
                continue

            # Group by algorithm
            algorithm_cis = {}

            for algorithm in df['algorithm'].unique():
                metric_data = df[df['algorithm'] == algorithm][metric].dropna()

                if len(metric_data) > 1:
                    mean = metric_data.mean()
                    sem = stats.sem(metric_data)  # Standard error of mean

                    # t-distribution for small samples
                    t_value = stats.t.ppf((1 + confidence_level) / 2, len(metric_data) - 1)
                    margin_of_error = t_value * sem

                    ci_lower = mean - margin_of_error
                    ci_upper = mean + margin_of_error

                    algorithm_cis[algorithm] = {
                        "mean": float(mean),
                        "ci_lower": float(ci_lower),
                        "ci_upper": float(ci_upper),
                        "margin_of_error": float(margin_of_error),
                        "confidence_level": confidence_level,
                        "sample_size": len(metric_data)
                    }

            confidence_intervals[metric] = algorithm_cis

        return confidence_intervals

    def _analyze_metric(self, df: pd.DataFrame, metric: str) -> StatisticalAnalysis:
        """Perform detailed analysis for a specific metric"""

        metric_data = df[metric].dropna()

        if len(metric_data) == 0:
            return StatisticalAnalysis(
                sample_size=0, mean=0.0, std_dev=0.0,
                confidence_interval=(0.0, 0.0), p_value=1.0,
                effect_size=0.0, statistical_power=0.0, significant=False
            )

        mean = metric_data.mean()
        std_dev = metric_data.std()

        # Confidence interval
        sem = stats.sem(metric_data)
        t_value = stats.t.ppf((1 + (1 - self.config.significance_level)) / 2, len(metric_data) - 1)
        margin_of_error = t_value * sem
        ci = (mean - margin_of_error, mean + margin_of_error)

        # One-sample t-test against null hypothesis of zero effect
        t_stat, p_value = stats.ttest_1samp(metric_data, 0)

        # Effect size (Cohen's d against zero)
        effect_size = mean / std_dev if std_dev > 0 else 0.0

        # Power calculation (simplified)
        z_alpha = stats.norm.ppf(1 - self.config.significance_level/2)
        z_beta = (abs(effect_size) * np.sqrt(len(metric_data)) - z_alpha)
        power = stats.norm.cdf(z_beta)

        return StatisticalAnalysis(
            sample_size=len(metric_data),
            mean=float(mean),
            std_dev=float(std_dev),
            confidence_interval=(float(ci[0]), float(ci[1])),
            p_value=float(p_value),
            effect_size=float(effect_size),
            statistical_power=float(max(0.0, min(1.0, power))),
            significant=p_value < self.config.significance_level
        )

    def _generate_experiment_report(self, statistical_results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive experiment report"""

        successful_runs = len([r for r in self.results if r.success])
        total_runs = len(self.results)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0

        # Algorithm performance summary
        algorithm_summary = self._generate_algorithm_summary()

        # Research conclusions
        conclusions = self._generate_research_conclusions(statistical_results)

        experiment_report = {
            "experiment_metadata": {
                "experiment_id": self.experiment_id,
                "experiment_name": self.config.experiment_name,
                "experiment_type": self.config.experiment_type.value,
                "description": self.config.description,
                "hypothesis": self.config.hypothesis,
                "status": self.status.value,
                "execution_date": datetime.now().isoformat(),
                "total_runtime": sum(r.execution_time for r in self.results if r.success)
            },

            "experimental_design": {
                "independent_variables": self.config.independent_variables,
                "dependent_variables": self.config.dependent_variables,
                "control_variables": self.config.control_variables,
                "significance_level": self.config.significance_level,
                "statistical_power_target": self.config.statistical_power,
                "num_repetitions": self.config.num_repetitions,
                "random_seed": self.config.random_seed
            },

            "execution_summary": {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": total_runs - successful_runs,
                "success_rate": success_rate,
                "algorithms_tested": len(set(r.algorithm_name for r in self.results)),
                "unique_conditions": len(set((r.algorithm_name, str(r.parameters)) for r in self.results))
            },

            "statistical_analysis": statistical_results,
            "algorithm_performance": algorithm_summary,
            "research_conclusions": conclusions,
            "reproducibility_info": {
                "random_seed": self.config.random_seed,
                "software_versions": self._get_software_versions(),
                "experiment_config_hash": hashlib.md5(
                    json.dumps(asdict(self.config), sort_keys=True).encode()
                ).hexdigest()
            }
        }

        return experiment_report

    def _generate_algorithm_summary(self) -> dict[str, Any]:
        """Generate performance summary for each algorithm"""

        df = self._results_to_dataframe()

        if df.empty:
            return {}

        algorithm_summary = {}

        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]

            summary = {
                "total_runs": len(alg_data),
                "avg_execution_time": float(alg_data['execution_time'].mean()),
                "avg_memory_usage": float(alg_data['memory_usage'].mean()),
                "metrics": {}
            }

            # Performance metrics
            for metric in self.config.dependent_variables:
                if metric in alg_data.columns:
                    metric_values = alg_data[metric].dropna()
                    if len(metric_values) > 0:
                        summary["metrics"][metric] = {
                            "mean": float(metric_values.mean()),
                            "std": float(metric_values.std()),
                            "best": float(metric_values.max()),
                            "worst": float(metric_values.min())
                        }

            algorithm_summary[algorithm] = summary

        return algorithm_summary

    def _generate_research_conclusions(self, statistical_results: dict[str, Any]) -> dict[str, Any]:
        """Generate research conclusions based on statistical analysis"""

        conclusions = {
            "hypothesis_supported": False,
            "key_findings": [],
            "significant_differences": [],
            "effect_sizes": [],
            "limitations": [],
            "future_work": []
        }

        # Analyze significance tests
        if "significance_tests" in statistical_results:
            sig_tests = statistical_results["significance_tests"]

            for metric, comparisons in sig_tests.items():
                for comparison, results in comparisons.items():
                    if results["significant"]:
                        conclusions["significant_differences"].append({
                            "metric": metric,
                            "comparison": comparison,
                            "p_value": results["p_value"],
                            "effect": "higher" if results["mean_1"] > results["mean_2"] else "lower"
                        })

        # Analyze effect sizes
        if "effect_size_analysis" in statistical_results:
            effect_sizes = statistical_results["effect_size_analysis"]

            for metric, comparisons in effect_sizes.items():
                for comparison, results in comparisons.items():
                    if results["interpretation"] in ["medium", "large"]:
                        conclusions["effect_sizes"].append({
                            "metric": metric,
                            "comparison": comparison,
                            "cohens_d": results["cohens_d"],
                            "interpretation": results["interpretation"]
                        })

        # Power analysis limitations
        if "power_analysis" in statistical_results:
            power_results = statistical_results["power_analysis"]

            for metric, power_info in power_results.items():
                if not power_info.get("power_adequate", False):
                    conclusions["limitations"].append(
                        f"Insufficient statistical power for {metric} "
                        f"(power: {power_info['statistical_power']:.3f})"
                    )

        # Generate key findings
        if conclusions["significant_differences"]:
            conclusions["key_findings"].append(
                f"Found {len(conclusions['significant_differences'])} statistically significant differences"
            )

        if conclusions["effect_sizes"]:
            large_effects = [e for e in conclusions["effect_sizes"] if e["interpretation"] == "large"]
            if large_effects:
                conclusions["key_findings"].append(
                    f"Identified {len(large_effects)} large effect sizes indicating practical significance"
                )

        # Hypothesis evaluation
        if len(conclusions["significant_differences"]) > 0:
            conclusions["hypothesis_supported"] = True
            conclusions["key_findings"].append("Research hypothesis supported by statistical evidence")

        return conclusions

    def _get_software_versions(self) -> dict[str, str]:
        """Get software version information for reproducibility"""

        import platform
        import sys

        try:
            import numpy
            numpy_version = numpy.__version__
        except:
            numpy_version = "unknown"

        try:
            import pandas
            pandas_version = pandas.__version__
        except:
            pandas_version = "unknown"

        try:
            import scipy
            scipy_version = scipy.__version__
        except:
            scipy_version = "unknown"

        return {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy_version,
            "pandas": pandas_version,
            "scipy": scipy_version,
            "experiment_framework": "1.0.0"
        }

    async def _save_experiment_results(self, experiment_report: dict[str, Any]) -> None:
        """Save experiment results in multiple formats"""

        logger.info("Saving experiment results")

        # Create experiment directory
        exp_dir = self.output_dir / self.experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save experiment report (JSON)
        report_file = exp_dir / "experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(experiment_report, f, indent=2, default=str)

        # Save experiment configuration
        config_file = exp_dir / "experiment_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)

        # Save raw results (pickle for Python objects)
        results_file = exp_dir / "raw_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)

        # Save results as CSV
        df = self._results_to_dataframe()
        if not df.empty:
            csv_file = exp_dir / "results.csv"
            df.to_csv(csv_file, index=False)

        # Save statistical analyses
        stats_file = exp_dir / "statistical_analyses.json"
        stats_dict = {k: asdict(v) for k, v in self.statistical_analyses.items()}
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)

        logger.info(f"Experiment results saved to: {exp_dir}")

    async def _save_intermediate_results(self) -> None:
        """Save intermediate results during long experiments"""

        intermediate_file = self.output_dir / f"{self.experiment_id}_intermediate.pkl"

        with open(intermediate_file, 'wb') as f:
            pickle.dump({
                "results": self.results,
                "config": self.config,
                "timestamp": datetime.now()
            }, f)

        logger.debug(f"Saved intermediate results: {len(self.results)} runs completed")

    async def _generate_visualizations(self) -> None:
        """Generate research-quality visualizations"""

        logger.info("Generating experiment visualizations")

        df = self._results_to_dataframe()
        if df.empty:
            logger.warning("No data available for visualization")
            return

        # Create visualization directory
        viz_dir = self.output_dir / self.experiment_id / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Generate different types of plots
        await self._plot_algorithm_comparison(df, viz_dir)
        await self._plot_performance_distributions(df, viz_dir)
        await self._plot_correlation_analysis(df, viz_dir)
        await self._plot_statistical_significance(df, viz_dir)

        if self.config.create_publication_figures:
            await self._create_publication_figures(df, viz_dir)

        logger.info(f"Visualizations saved to: {viz_dir}")

    async def _plot_algorithm_comparison(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create algorithm comparison plots"""

        for metric in self.config.dependent_variables:
            if metric not in df.columns:
                continue

            plt.figure(figsize=(12, 8))

            # Box plot comparison
            sns.boxplot(data=df, x='algorithm', y=metric)
            plt.title(f'{metric.replace("_", " ").title()} by Algorithm')
            plt.xlabel('Algorithm')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45)

            # Add statistical annotations
            algorithms = df['algorithm'].unique()
            if len(algorithms) > 1:
                # Add significance stars (simplified)
                from scipy.stats import ttest_ind

                y_max = df[metric].max()
                y_range = df[metric].max() - df[metric].min()

                for i, alg1 in enumerate(algorithms[:-1]):
                    for j, alg2 in enumerate(algorithms[i+1:], i+1):
                        data1 = df[df['algorithm'] == alg1][metric].dropna()
                        data2 = df[df['algorithm'] == alg2][metric].dropna()

                        if len(data1) > 1 and len(data2) > 1:
                            _, p_val = ttest_ind(data1, data2)

                            if p_val < 0.001:
                                sig_symbol = "***"
                            elif p_val < 0.01:
                                sig_symbol = "**"
                            elif p_val < 0.05:
                                sig_symbol = "*"
                            else:
                                sig_symbol = "ns"

                            # Add significance annotation
                            y_pos = y_max + (j - i) * 0.1 * y_range
                            plt.annotate(sig_symbol,
                                       xy=((i + j) / 2, y_pos),
                                       ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_dir / f'algorithm_comparison_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()

    async def _plot_performance_distributions(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create performance distribution plots"""

        for metric in self.config.dependent_variables:
            if metric not in df.columns:
                continue

            plt.figure(figsize=(12, 8))

            # Create subplots for each algorithm
            algorithms = df['algorithm'].unique()
            n_algorithms = len(algorithms)

            fig, axes = plt.subplots(1, n_algorithms, figsize=(4 * n_algorithms, 6),
                                   sharey=True, sharex=True)

            if n_algorithms == 1:
                axes = [axes]

            for i, algorithm in enumerate(algorithms):
                alg_data = df[df['algorithm'] == algorithm][metric].dropna()

                # Histogram with KDE
                axes[i].hist(alg_data, bins=20, density=True, alpha=0.7,
                           color=f'C{i}', label=f'{algorithm} (n={len(alg_data)})')

                # Add KDE curve
                if len(alg_data) > 1:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(alg_data)
                    x_range = np.linspace(alg_data.min(), alg_data.max(), 100)
                    axes[i].plot(x_range, kde(x_range), color='black', linewidth=2)

                axes[i].set_title(f'{algorithm}')
                axes[i].set_xlabel(metric.replace("_", " ").title())
                if i == 0:
                    axes[i].set_ylabel('Density')

                # Add statistics text
                mean_val = alg_data.mean()
                std_val = alg_data.std()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8)
                axes[i].text(0.05, 0.95, f'μ={mean_val:.3f}\\nσ={std_val:.3f}',
                           transform=axes[i].transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f'Distribution of {metric.replace("_", " ").title()}')
            plt.tight_layout()
            plt.savefig(output_dir / f'distribution_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()

    async def _plot_correlation_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create correlation analysis plots"""

        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return

        # Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()

        mask = np.triu(np.ones_like(correlation_matrix))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.3f')
        plt.title('Correlation Matrix of Performance Metrics')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Pairwise scatter plots for key metrics
        if len(self.config.dependent_variables) >= 2:
            for i, metric1 in enumerate(self.config.dependent_variables[:-1]):
                for metric2 in self.config.dependent_variables[i+1:]:
                    if metric1 in df.columns and metric2 in df.columns:

                        plt.figure(figsize=(10, 8))

                        # Scatter plot colored by algorithm
                        for algorithm in df['algorithm'].unique():
                            alg_data = df[df['algorithm'] == algorithm]
                            plt.scatter(alg_data[metric1], alg_data[metric2],
                                      label=algorithm, alpha=0.7, s=50)

                        plt.xlabel(metric1.replace("_", " ").title())
                        plt.ylabel(metric2.replace("_", " ").title())
                        plt.title(f'{metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}')
                        plt.legend()

                        # Add correlation coefficient
                        corr_coef = df[metric1].corr(df[metric2])
                        plt.text(0.05, 0.95, f'r = {corr_coef:.3f}',
                               transform=plt.gca().transAxes, va='top', ha='left',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                        plt.tight_layout()
                        plt.savefig(output_dir / f'scatter_{metric1}_vs_{metric2}.png',
                                  dpi=300, bbox_inches='tight')
                        plt.close()

    async def _plot_statistical_significance(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create statistical significance visualization"""

        # Get significance test results
        statistical_results = self._perform_significance_tests(df)

        for metric, comparisons in statistical_results.items():
            if not comparisons:
                continue

            # Create significance matrix
            algorithms = df['algorithm'].unique()
            n_algs = len(algorithms)

            if n_algs < 2:
                continue

            sig_matrix = np.ones((n_algs, n_algs))
            p_value_matrix = np.ones((n_algs, n_algs))

            alg_to_idx = {alg: i for i, alg in enumerate(algorithms)}

            for comparison, results in comparisons.items():
                alg1, alg2 = comparison.split('_vs_')
                if alg1 in alg_to_idx and alg2 in alg_to_idx:
                    i, j = alg_to_idx[alg1], alg_to_idx[alg2]

                    sig_val = 0 if results['significant'] else 1
                    sig_matrix[i, j] = sig_val
                    sig_matrix[j, i] = sig_val

                    p_val = results['p_value']
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val

            # Plot significance matrix
            plt.figure(figsize=(10, 8))

            # Create custom colormap
            colors = ['red', 'lightgray']  # red = significant, gray = not significant
            n_bins = 2
            cmap = plt.cm.colors.ListedColormap(colors)

            sns.heatmap(sig_matrix,
                       xticklabels=algorithms,
                       yticklabels=algorithms,
                       annot=p_value_matrix,
                       fmt='.3f',
                       cmap=cmap,
                       cbar_kws={'label': 'Significant (0) / Not Significant (1)'})

            plt.title(f'Statistical Significance Matrix for {metric.replace("_", " ").title()}')
            plt.xlabel('Algorithm')
            plt.ylabel('Algorithm')

            plt.tight_layout()
            plt.savefig(output_dir / f'significance_matrix_{metric}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

    async def _create_publication_figures(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create publication-ready figures"""

        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'legend.frameon': False
        })

        # Main results figure
        if len(self.config.dependent_variables) >= 1:
            main_metric = self.config.dependent_variables[0]

            if main_metric in df.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Left panel: Bar plot with error bars
                algorithm_means = df.groupby('algorithm')[main_metric].agg(['mean', 'std', 'count'])

                bars = ax1.bar(range(len(algorithm_means)), algorithm_means['mean'],
                             yerr=algorithm_means['std'] / np.sqrt(algorithm_means['count']),
                             capsize=5, alpha=0.8, color=['C0', 'C1', 'C2', 'C3'][:len(algorithm_means)])

                ax1.set_xlabel('Algorithm')
                ax1.set_ylabel(main_metric.replace("_", " ").title())
                ax1.set_title(f'Algorithm Performance Comparison\\n({main_metric.replace("_", " ").title()})')
                ax1.set_xticks(range(len(algorithm_means)))
                ax1.set_xticklabels(algorithm_means.index, rotation=0)

                # Add value labels on bars
                for i, (bar, mean_val) in enumerate(zip(bars, algorithm_means['mean'], strict=False)):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

                # Right panel: Violin plot
                sns.violinplot(data=df, x='algorithm', y=main_metric, ax=ax2)
                ax2.set_xlabel('Algorithm')
                ax2.set_ylabel(main_metric.replace("_", " ").title())
                ax2.set_title(f'Performance Distribution\\n({main_metric.replace("_", " ").title()})')

                plt.tight_layout()
                plt.savefig(output_dir / 'publication_main_results.png',
                           dpi=300, bbox_inches='tight')
                plt.close()

        # Reset matplotlib parameters
        plt.rcParams.update(plt.rcParamsDefault)


class ComparativeStudyFramework:
    """
    Framework for conducting comparative studies between novel and baseline algorithms.
    
    Provides systematic comparison methodology with proper experimental controls
    and statistical validation for academic research.
    """

    def __init__(self,
                 baseline_algorithms: dict[str, Callable],
                 novel_algorithms: dict[str, Callable],
                 study_name: str = "Comparative Study"):
        self.baseline_algorithms = baseline_algorithms
        self.novel_algorithms = novel_algorithms
        self.study_name = study_name

        # Combine all algorithms for comparison
        self.all_algorithms = {**baseline_algorithms, **novel_algorithms}

        logger.info(f"Initialized comparative study: {study_name}")
        logger.info(f"Baseline algorithms: {list(baseline_algorithms.keys())}")
        logger.info(f"Novel algorithms: {list(novel_algorithms.keys())}")

    async def run_comparative_study(self,
                                  datasets: list[dict[str, Any]],
                                  metrics: list[str],
                                  output_dir: str = "./comparative_study_results") -> dict[str, Any]:
        """
        Run comprehensive comparative study between baseline and novel algorithms.
        
        Returns detailed comparison results with statistical analysis.
        """

        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name=f"{self.study_name}_comparison",
            experiment_type=ExperimentType.ALGORITHM_COMPARISON,
            description="Comparative study between baseline and novel algorithms",
            hypothesis="Novel algorithms provide significant improvement over baseline methods",
            independent_variables=["algorithm_type", "algorithm_name"],
            dependent_variables=metrics,
            control_variables=["dataset", "random_seed"],
            num_repetitions=20,  # Higher repetitions for robust comparison
            min_sample_size=100,
            generate_visualizations=True,
            create_publication_figures=True
        )

        # Run experiment
        experiment_runner = ResearchExperimentRunner(
            experiment_config=experiment_config,
            output_directory=output_dir
        )

        results = await experiment_runner.run_experiment(
            algorithms=self.all_algorithms,
            datasets=datasets
        )

        # Enhanced comparative analysis
        comparative_analysis = self._perform_comparative_analysis(results)

        # Generate comparative report
        comparative_report = self._generate_comparative_report(results, comparative_analysis)

        return comparative_report

    def _perform_comparative_analysis(self, experiment_results: dict[str, Any]) -> dict[str, Any]:
        """Perform specialized comparative analysis"""

        # Extract performance data
        algorithm_performance = experiment_results.get("algorithm_performance", {})

        comparative_analysis = {
            "baseline_vs_novel_comparison": {},
            "improvement_analysis": {},
            "statistical_superiority": {},
            "practical_significance": {}
        }

        # Compare novel vs baseline algorithms
        for novel_alg in self.novel_algorithms.keys():
            if novel_alg in algorithm_performance:
                novel_performance = algorithm_performance[novel_alg]

                # Find best baseline for comparison
                best_baseline = self._find_best_baseline(algorithm_performance)

                if best_baseline:
                    baseline_performance = algorithm_performance[best_baseline]

                    # Calculate improvements
                    improvements = {}
                    for metric in novel_performance.get("metrics", {}):
                        novel_val = novel_performance["metrics"][metric]["mean"]
                        baseline_val = baseline_performance["metrics"][metric]["mean"]

                        if baseline_val != 0:
                            improvement = (novel_val - baseline_val) / baseline_val * 100
                            improvements[metric] = improvement

                    comparative_analysis["improvement_analysis"][novel_alg] = {
                        "compared_to": best_baseline,
                        "improvements": improvements,
                        "significant_improvement": any(imp > 5 for imp in improvements.values())
                    }

        return comparative_analysis

    def _find_best_baseline(self, algorithm_performance: dict[str, Any]) -> str | None:
        """Find the best performing baseline algorithm"""

        best_baseline = None
        best_score = -float('inf')

        for alg_name in self.baseline_algorithms.keys():
            if alg_name in algorithm_performance:
                perf = algorithm_performance[alg_name]

                # Calculate overall score (simplified)
                if "metrics" in perf:
                    scores = [metric_data["mean"] for metric_data in perf["metrics"].values()]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_baseline = alg_name

        return best_baseline

    def _generate_comparative_report(self,
                                   experiment_results: dict[str, Any],
                                   comparative_analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive comparative study report"""

        return {
            "study_metadata": {
                "study_name": self.study_name,
                "baseline_algorithms": list(self.baseline_algorithms.keys()),
                "novel_algorithms": list(self.novel_algorithms.keys()),
                "comparison_date": datetime.now().isoformat()
            },
            "experiment_results": experiment_results,
            "comparative_analysis": comparative_analysis,
            "conclusions": self._generate_comparative_conclusions(comparative_analysis),
            "recommendations": self._generate_recommendations(comparative_analysis)
        }

    def _generate_comparative_conclusions(self, comparative_analysis: dict[str, Any]) -> list[str]:
        """Generate conclusions from comparative analysis"""

        conclusions = []

        # Analyze improvements
        improvement_analysis = comparative_analysis.get("improvement_analysis", {})

        significant_improvements = 0
        total_comparisons = len(improvement_analysis)

        for novel_alg, analysis in improvement_analysis.items():
            if analysis.get("significant_improvement", False):
                significant_improvements += 1
                improvements = analysis.get("improvements", {})
                max_improvement = max(improvements.values()) if improvements else 0

                conclusions.append(
                    f"{novel_alg} shows significant improvement over baseline "
                    f"with maximum improvement of {max_improvement:.1f}%"
                )

        if significant_improvements > 0:
            conclusions.append(
                f"{significant_improvements}/{total_comparisons} novel algorithms "
                f"demonstrate significant improvements over baselines"
            )
        else:
            conclusions.append("No significant improvements detected in novel algorithms")

        return conclusions

    def _generate_recommendations(self, comparative_analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations based on comparative analysis"""

        recommendations = []

        improvement_analysis = comparative_analysis.get("improvement_analysis", {})

        # Identify best performing novel algorithm
        best_novel = None
        best_improvement = -float('inf')

        for novel_alg, analysis in improvement_analysis.items():
            improvements = analysis.get("improvements", {})
            if improvements:
                avg_improvement = sum(improvements.values()) / len(improvements)
                if avg_improvement > best_improvement:
                    best_improvement = avg_improvement
                    best_novel = novel_alg

        if best_novel and best_improvement > 5:
            recommendations.append(
                f"Recommend {best_novel} for deployment with average improvement of {best_improvement:.1f}%"
            )

        # Further research recommendations
        for novel_alg, analysis in improvement_analysis.items():
            if not analysis.get("significant_improvement", False):
                recommendations.append(
                    f"Consider further optimization of {novel_alg} algorithm"
                )

        recommendations.append("Conduct additional validation with larger datasets")
        recommendations.append("Perform real-world deployment testing")

        return recommendations
