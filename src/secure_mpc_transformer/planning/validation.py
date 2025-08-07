"""
Quantum State Validation and Error Handling

Robust validation and error handling for quantum-inspired task planning
algorithms, ensuring reliability and correctness of quantum computations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta
import hashlib
import json

from .quantum_planner import Task, TaskType, TaskStatus, QuantumTaskConfig

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base class for validation errors"""
    pass


class QuantumStateError(ValidationError):
    """Error in quantum state validation"""
    pass


class TaskDependencyError(ValidationError):
    """Error in task dependency validation"""
    pass


class ResourceConstraintError(ValidationError):
    """Error in resource constraint validation"""
    pass


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    severity: Optional[ErrorSeverity] = None
    suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QuantumStateMetrics:
    """Metrics for quantum state validation"""
    normalization: float
    entanglement_measure: float
    coherence_score: float
    fidelity: float
    purity: float
    entropy: float


class QuantumStateValidator:
    """
    Validator for quantum states and computations in task planning.
    Ensures quantum states remain valid and computations are correct.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.validation_history: List[Dict[str, Any]] = []
        
    def validate_quantum_state(self, 
                             quantum_state: np.ndarray,
                             state_id: str = "unknown") -> ValidationResult:
        """
        Validate quantum state vector for correctness.
        
        Args:
            quantum_state: Quantum state vector to validate
            state_id: Identifier for the state
            
        Returns:
            ValidationResult with validation status and metrics
        """
        try:
            # Check if state is complex
            if not np.iscomplexobj(quantum_state):
                return ValidationResult(
                    is_valid=False,
                    error_type="InvalidStateType",
                    error_message="Quantum state must be complex-valued",
                    severity=ErrorSeverity.HIGH,
                    suggestions=["Convert state to complex dtype"]
                )
            
            # Check normalization
            norm = np.linalg.norm(quantum_state)
            if abs(norm - 1.0) > self.tolerance:
                return ValidationResult(
                    is_valid=False,
                    error_type="NotNormalized",
                    error_message=f"Quantum state not normalized: norm = {norm:.6f}",
                    severity=ErrorSeverity.MEDIUM,
                    suggestions=[
                        "Normalize state: state = state / np.linalg.norm(state)",
                        "Check for numerical instabilities in quantum operations"
                    ]
                )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(quantum_state)) or np.any(np.isinf(quantum_state)):
                return ValidationResult(
                    is_valid=False,
                    error_type="InvalidValues",
                    error_message="Quantum state contains NaN or infinite values",
                    severity=ErrorSeverity.CRITICAL,
                    suggestions=[
                        "Check quantum operations for numerical stability",
                        "Reinitialize quantum state",
                        "Reduce temperature or adjust algorithm parameters"
                    ]
                )
            
            # Calculate quantum metrics
            metrics = self._calculate_quantum_metrics(quantum_state)
            
            # Check coherence
            if metrics.coherence_score < 0.1:
                return ValidationResult(
                    is_valid=True,  # Valid but concerning
                    error_type="LowCoherence",
                    error_message=f"Low quantum coherence: {metrics.coherence_score:.3f}",
                    severity=ErrorSeverity.LOW,
                    suggestions=[
                        "Consider reducing decoherence effects",
                        "Check quantum gate operations for errors"
                    ],
                    metadata=metrics.__dict__
                )
            
            # Log successful validation
            self.validation_history.append({
                "timestamp": datetime.now(),
                "state_id": state_id,
                "status": "valid",
                "metrics": metrics.__dict__
            })
            
            return ValidationResult(
                is_valid=True,
                metadata=metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Quantum state validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_type="ValidationException",
                error_message=f"Validation failed with exception: {str(e)}",
                severity=ErrorSeverity.CRITICAL
            )
    
    def _calculate_quantum_metrics(self, state: np.ndarray) -> QuantumStateMetrics:
        """Calculate comprehensive quantum state metrics"""
        
        # Normalization
        normalization = np.linalg.norm(state)
        
        # Entanglement measure (simplified)
        amplitudes = np.abs(state) ** 2
        entanglement = 1.0 - np.sum(amplitudes ** 2)  # Purity-based measure
        
        # Coherence score (off-diagonal elements significance)
        n = len(state)
        density_matrix = np.outer(state, np.conj(state))
        off_diagonal = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
        coherence = off_diagonal / (n * n - n) if n > 1 else 0.0
        
        # Fidelity with uniform superposition
        uniform_state = np.ones(n, dtype=complex) / np.sqrt(n)
        fidelity = abs(np.vdot(uniform_state, state)) ** 2
        
        # Purity
        purity = np.sum(amplitudes ** 2)
        
        # Entropy
        entropy = -np.sum(amplitudes * np.log2(amplitudes + 1e-12))  # Add small epsilon
        
        return QuantumStateMetrics(
            normalization=normalization,
            entanglement_measure=entanglement,
            coherence_score=coherence,
            fidelity=fidelity,
            purity=purity,
            entropy=entropy
        )
    
    def validate_quantum_evolution(self, 
                                 initial_state: np.ndarray,
                                 final_state: np.ndarray,
                                 evolution_matrix: Optional[np.ndarray] = None) -> ValidationResult:
        """
        Validate quantum state evolution for unitarity and correctness.
        
        Args:
            initial_state: Initial quantum state
            final_state: Final quantum state after evolution
            evolution_matrix: Optional unitary evolution matrix
            
        Returns:
            ValidationResult for the evolution
        """
        # Validate both states individually
        initial_result = self.validate_quantum_state(initial_state, "initial")
        if not initial_result.is_valid:
            return initial_result
        
        final_result = self.validate_quantum_state(final_state, "final")
        if not final_result.is_valid:
            return final_result
        
        # Check evolution properties
        if evolution_matrix is not None:
            # Check unitarity of evolution matrix
            if not self._is_unitary(evolution_matrix):
                return ValidationResult(
                    is_valid=False,
                    error_type="NonUnitaryEvolution",
                    error_message="Evolution matrix is not unitary",
                    severity=ErrorSeverity.HIGH,
                    suggestions=[
                        "Ensure evolution matrix is unitary: U† U = I",
                        "Check quantum gate construction",
                        "Verify numerical precision"
                    ]
                )
            
            # Verify evolution correctness
            expected_final = evolution_matrix @ initial_state
            evolution_error = np.linalg.norm(final_state - expected_final)
            
            if evolution_error > self.tolerance * 10:  # More lenient for evolution
                return ValidationResult(
                    is_valid=False,
                    error_type="IncorrectEvolution",
                    error_message=f"Evolution error too large: {evolution_error:.6f}",
                    severity=ErrorSeverity.MEDIUM,
                    suggestions=[
                        "Check evolution matrix application",
                        "Verify quantum operations sequence",
                        "Increase numerical precision"
                    ]
                )
        
        # Check conservation of probability
        initial_prob = np.sum(np.abs(initial_state) ** 2)
        final_prob = np.sum(np.abs(final_state) ** 2)
        prob_diff = abs(initial_prob - final_prob)
        
        if prob_diff > self.tolerance:
            return ValidationResult(
                is_valid=False,
                error_type="ProbabilityNotConserved",
                error_message=f"Probability not conserved: Δp = {prob_diff:.6f}",
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Check for non-unitary operations",
                    "Verify state normalization throughout evolution",
                    "Check for numerical errors in computation"
                ]
            )
        
        return ValidationResult(is_valid=True)
    
    def _is_unitary(self, matrix: np.ndarray, tolerance: Optional[float] = None) -> bool:
        """Check if matrix is unitary within tolerance"""
        tol = tolerance or self.tolerance
        
        # Calculate U† U
        conjugate_transpose = np.conj(matrix.T)
        product = conjugate_transpose @ matrix
        
        # Check if it equals identity matrix
        identity = np.eye(matrix.shape[0])
        return np.allclose(product, identity, atol=tol)


class TaskDependencyValidator:
    """
    Validator for task dependencies and workflow correctness.
    Ensures task graphs are acyclic and dependencies are satisfiable.
    """
    
    def __init__(self):
        self.validation_cache: Dict[str, ValidationResult] = {}
        
    def validate_task_dependencies(self, tasks: List[Task]) -> ValidationResult:
        """
        Validate task dependency graph for correctness.
        
        Args:
            tasks: List of tasks to validate
            
        Returns:
            ValidationResult for dependency validation
        """
        # Create task lookup
        task_dict = {task.id: task for task in tasks}
        
        # Check for missing dependencies
        missing_deps = []
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_dict:
                    missing_deps.append((task.id, dep_id))
        
        if missing_deps:
            return ValidationResult(
                is_valid=False,
                error_type="MissingDependencies",
                error_message=f"Missing dependencies: {missing_deps}",
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Add missing dependency tasks",
                    "Remove invalid dependency references",
                    "Check task ID consistency"
                ]
            )
        
        # Check for circular dependencies
        cycle = self._find_dependency_cycle(tasks)
        if cycle:
            return ValidationResult(
                is_valid=False,
                error_type="CircularDependencies",
                error_message=f"Circular dependency detected: {' -> '.join(cycle)}",
                severity=ErrorSeverity.CRITICAL,
                suggestions=[
                    "Remove circular dependencies",
                    "Redesign task workflow",
                    "Split tasks to break cycles"
                ]
            )
        
        # Validate dependency types (MPC-specific)
        type_violations = self._validate_dependency_types(tasks)
        if type_violations:
            return ValidationResult(
                is_valid=False,
                error_type="InvalidDependencyTypes",
                error_message=f"Invalid dependency types: {type_violations}",
                severity=ErrorSeverity.MEDIUM,
                suggestions=[
                    "Check MPC protocol requirements",
                    "Ensure proper task ordering",
                    "Verify layer dependencies"
                ]
            )
        
        return ValidationResult(is_valid=True)
    
    def _find_dependency_cycle(self, tasks: List[Task]) -> Optional[List[str]]:
        """Find circular dependencies using DFS"""
        task_dict = {task.id: task for task in tasks}
        
        # Graph coloring: white (0), gray (1), black (2)
        colors = {task.id: 0 for task in tasks}
        parent = {task.id: None for task in tasks}
        
        def dfs_visit(task_id: str) -> Optional[List[str]]:
            colors[task_id] = 1  # Gray
            
            task = task_dict[task_id]
            for dep_id in task.dependencies:
                if dep_id in colors:
                    if colors[dep_id] == 1:  # Back edge - cycle found
                        # Reconstruct cycle
                        cycle = [dep_id]
                        current = task_id
                        while current != dep_id:
                            cycle.append(current)
                            current = parent.get(current)
                            if current is None:
                                break
                        cycle.append(dep_id)
                        return cycle[::-1]
                    
                    elif colors[dep_id] == 0:
                        parent[dep_id] = task_id
                        cycle = dfs_visit(dep_id)
                        if cycle:
                            return cycle
            
            colors[task_id] = 2  # Black
            return None
        
        for task in tasks:
            if colors[task.id] == 0:
                cycle = dfs_visit(task.id)
                if cycle:
                    return cycle
        
        return None
    
    def _validate_dependency_types(self, tasks: List[Task]) -> List[str]:
        """Validate MPC-specific dependency type constraints"""
        violations = []
        task_dict = {task.id: task for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_dict:
                    dep_task = task_dict[dep_id]
                    
                    # MPC protocol constraints
                    if (task.task_type == TaskType.ATTENTION and 
                        dep_task.task_type not in [TaskType.EMBEDDING, TaskType.ATTENTION, TaskType.PROTOCOL_INIT]):
                        violations.append(f"{task.id} -> {dep_id}: Invalid attention dependency")
                    
                    if (task.task_type == TaskType.RESULT_RECONSTRUCTION and 
                        dep_task.task_type == TaskType.PROTOCOL_INIT):
                        violations.append(f"{task.id} -> {dep_id}: Cannot reconstruct before protocol complete")
        
        return violations


class ResourceValidator:
    """
    Validator for resource constraints and allocation.
    Ensures resource requirements are feasible and constraints are met.
    """
    
    def __init__(self, available_resources: Optional[Dict[str, float]] = None):
        self.available_resources = available_resources or {
            "cpu": 1.0,
            "memory": 1.0,
            "gpu": 1.0
        }
        
    def validate_resource_requirements(self, tasks: List[Task]) -> ValidationResult:
        """
        Validate resource requirements for all tasks.
        
        Args:
            tasks: List of tasks to validate
            
        Returns:
            ValidationResult for resource validation
        """
        # Check individual task requirements
        invalid_tasks = []
        for task in tasks:
            for resource, amount in task.required_resources.items():
                if amount < 0:
                    invalid_tasks.append(f"{task.id}: negative {resource} requirement")
                elif amount > self.available_resources.get(resource, 0):
                    invalid_tasks.append(f"{task.id}: {resource} requirement ({amount}) exceeds available ({self.available_resources.get(resource, 0)})")
        
        if invalid_tasks:
            return ValidationResult(
                is_valid=False,
                error_type="InvalidResourceRequirements",
                error_message=f"Invalid resource requirements: {invalid_tasks}",
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Adjust resource requirements to be positive",
                    "Scale down resource requirements",
                    "Increase available resources"
                ]
            )
        
        # Check total resource constraints
        total_requirements = {}
        for task in tasks:
            for resource, amount in task.required_resources.items():
                total_requirements[resource] = total_requirements.get(resource, 0) + amount
        
        overallocated = []
        for resource, total in total_requirements.items():
            available = self.available_resources.get(resource, 0)
            if total > available * len(tasks):  # Allow some oversubscription
                overallocated.append(f"{resource}: {total:.2f} > {available * len(tasks):.2f}")
        
        if overallocated:
            return ValidationResult(
                is_valid=False,
                error_type="ResourceOverallocation",
                error_message=f"Resource overallocation: {overallocated}",
                severity=ErrorSeverity.MEDIUM,
                suggestions=[
                    "Reduce task resource requirements",
                    "Implement resource sharing",
                    "Scale task execution over time"
                ]
            )
        
        return ValidationResult(is_valid=True)


class QuantumPlanningValidator:
    """
    Comprehensive validator for quantum-inspired task planning.
    Coordinates all validation types and provides unified error handling.
    """
    
    def __init__(self, 
                 quantum_tolerance: float = 1e-10,
                 available_resources: Optional[Dict[str, float]] = None):
        self.quantum_validator = QuantumStateValidator(quantum_tolerance)
        self.dependency_validator = TaskDependencyValidator()
        self.resource_validator = ResourceValidator(available_resources)
        
        self.validation_log: List[Dict[str, Any]] = []
        
    def validate_complete_plan(self, 
                             tasks: List[Task],
                             quantum_state: Optional[np.ndarray] = None,
                             config: Optional[QuantumTaskConfig] = None) -> ValidationResult:
        """
        Perform comprehensive validation of quantum task planning.
        
        Args:
            tasks: List of tasks to validate
            quantum_state: Optional quantum state to validate
            config: Optional configuration to validate
            
        Returns:
            Comprehensive ValidationResult
        """
        validation_start = datetime.now()
        errors = []
        warnings = []
        
        # Validate task dependencies
        dep_result = self.dependency_validator.validate_task_dependencies(tasks)
        if not dep_result.is_valid:
            errors.append(f"Dependencies: {dep_result.error_message}")
        elif dep_result.severity == ErrorSeverity.MEDIUM:
            warnings.append(f"Dependencies: {dep_result.error_message}")
        
        # Validate resources
        resource_result = self.resource_validator.validate_resource_requirements(tasks)
        if not resource_result.is_valid:
            errors.append(f"Resources: {resource_result.error_message}")
        elif resource_result.severity == ErrorSeverity.MEDIUM:
            warnings.append(f"Resources: {resource_result.error_message}")
        
        # Validate quantum state if provided
        quantum_result = None
        if quantum_state is not None:
            quantum_result = self.quantum_validator.validate_quantum_state(quantum_state)
            if not quantum_result.is_valid:
                errors.append(f"Quantum State: {quantum_result.error_message}")
            elif quantum_result.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                warnings.append(f"Quantum State: {quantum_result.error_message}")
        
        # Validate configuration
        config_result = self._validate_config(config)
        if not config_result.is_valid:
            errors.append(f"Configuration: {config_result.error_message}")
        
        # Log validation
        validation_time = (datetime.now() - validation_start).total_seconds()
        self.validation_log.append({
            "timestamp": datetime.now(),
            "tasks_validated": len(tasks),
            "validation_time": validation_time,
            "errors": len(errors),
            "warnings": len(warnings),
            "quantum_state_validated": quantum_state is not None
        })
        
        # Determine overall result
        if errors:
            return ValidationResult(
                is_valid=False,
                error_type="MultipleValidationErrors",
                error_message="; ".join(errors),
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Address all validation errors",
                    "Review task configuration",
                    "Check quantum state initialization"
                ],
                metadata={
                    "errors": errors,
                    "warnings": warnings,
                    "validation_time": validation_time,
                    "quantum_metrics": quantum_result.metadata if quantum_result else None
                }
            )
        
        elif warnings:
            return ValidationResult(
                is_valid=True,
                error_type="ValidationWarnings",
                error_message="; ".join(warnings),
                severity=ErrorSeverity.LOW,
                suggestions=[
                    "Review warnings for potential issues",
                    "Consider optimizing configuration"
                ],
                metadata={
                    "warnings": warnings,
                    "validation_time": validation_time,
                    "quantum_metrics": quantum_result.metadata if quantum_result else None
                }
            )
        
        else:
            return ValidationResult(
                is_valid=True,
                metadata={
                    "validation_time": validation_time,
                    "quantum_metrics": quantum_result.metadata if quantum_result else None
                }
            )
    
    def _validate_config(self, config: Optional[QuantumTaskConfig]) -> ValidationResult:
        """Validate quantum task configuration"""
        if config is None:
            return ValidationResult(is_valid=True)
        
        errors = []
        
        if config.max_parallel_tasks <= 0:
            errors.append("max_parallel_tasks must be positive")
        
        if config.quantum_annealing_steps <= 0:
            errors.append("quantum_annealing_steps must be positive")
        
        if not (0 < config.temperature_decay < 1):
            errors.append("temperature_decay must be between 0 and 1")
        
        if config.optimization_rounds <= 0:
            errors.append("optimization_rounds must be positive")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                error_type="InvalidConfiguration",
                error_message="; ".join(errors),
                severity=ErrorSeverity.HIGH
            )
        
        return ValidationResult(is_valid=True)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations"""
        if not self.validation_log:
            return {"status": "no_validations"}
        
        total_validations = len(self.validation_log)
        total_tasks = sum(log["tasks_validated"] for log in self.validation_log)
        total_time = sum(log["validation_time"] for log in self.validation_log)
        total_errors = sum(log["errors"] for log in self.validation_log)
        total_warnings = sum(log["warnings"] for log in self.validation_log)
        
        return {
            "total_validations": total_validations,
            "total_tasks_validated": total_tasks,
            "total_validation_time": total_time,
            "average_validation_time": total_time / total_validations,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "error_rate": total_errors / total_validations,
            "quantum_state_validations": sum(1 for log in self.validation_log if log["quantum_state_validated"]),
            "recent_validations": self.validation_log[-10:]  # Last 10 validations
        }