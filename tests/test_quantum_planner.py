"""
Tests for Quantum Task Planner

Comprehensive test suite for quantum-inspired task planning algorithms,
including quantum state validation, optimization algorithms, and scheduling.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from src.secure_mpc_transformer.planning.quantum_planner import (
    QuantumTaskPlanner, 
    QuantumTaskConfig,
    Task,
    TaskType,
    TaskStatus
)
from src.secure_mpc_transformer.planning.optimization import (
    QuantumOptimizer,
    OptimizationObjective,
    OptimizationConstraints,
    OptimizationResult
)
from src.secure_mpc_transformer.planning.validation import (
    QuantumPlanningValidator,
    QuantumStateValidator,
    ValidationResult,
    ErrorSeverity
)


class TestQuantumTaskPlanner:
    """Test suite for QuantumTaskPlanner"""
    
    @pytest.fixture
    def planner(self):
        """Create test planner instance"""
        config = QuantumTaskConfig(
            max_parallel_tasks=4,
            quantum_annealing_steps=100,
            optimization_rounds=10
        )
        return QuantumTaskPlanner(config)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing"""
        return [
            Task(
                id="task_1",
                task_type=TaskType.EMBEDDING,
                priority=0.9,
                estimated_duration=1.0,
                required_resources={"gpu": 0.5, "memory": 0.3},
                dependencies=[]
            ),
            Task(
                id="task_2",
                task_type=TaskType.ATTENTION,
                priority=0.8,
                estimated_duration=2.0,
                required_resources={"gpu": 0.7, "memory": 0.4},
                dependencies=["task_1"]
            ),
            Task(
                id="task_3",
                task_type=TaskType.FEEDFORWARD,
                priority=0.7,
                estimated_duration=1.5,
                required_resources={"gpu": 0.4, "memory": 0.2},
                dependencies=["task_2"]
            ),
            Task(
                id="task_4",
                task_type=TaskType.RESULT_RECONSTRUCTION,
                priority=0.9,
                estimated_duration=0.5,
                required_resources={"cpu": 0.3, "memory": 0.1},
                dependencies=["task_3"]
            )
        ]
    
    def test_planner_initialization(self):
        """Test planner initialization"""
        config = QuantumTaskConfig(max_parallel_tasks=8)
        planner = QuantumTaskPlanner(config)
        
        assert planner.config.max_parallel_tasks == 8
        assert planner.tasks == {}
        assert planner.execution_history == []
        assert isinstance(planner._quantum_state_cache, dict)
    
    def test_add_and_get_task(self, planner, sample_tasks):
        """Test task addition and retrieval"""
        task = sample_tasks[0]
        
        # Add task
        planner.add_task(task)
        assert task.id in planner.tasks
        
        # Retrieve task
        retrieved_task = planner.get_task(task.id)
        assert retrieved_task == task
        assert retrieved_task.id == task.id
        assert retrieved_task.task_type == task.task_type
    
    def test_remove_task(self, planner, sample_tasks):
        """Test task removal"""
        task = sample_tasks[0]
        
        # Add and then remove task
        planner.add_task(task)
        assert planner.remove_task(task.id) == True
        assert task.id not in planner.tasks
        
        # Try to remove non-existent task
        assert planner.remove_task("non_existent") == False
    
    def test_get_pending_tasks(self, planner, sample_tasks):
        """Test retrieval of pending tasks"""
        # Add tasks with different statuses
        task1, task2 = sample_tasks[0], sample_tasks[1]
        task1.status = TaskStatus.PENDING
        task2.status = TaskStatus.COMPLETED
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        pending_tasks = planner.get_pending_tasks()
        assert len(pending_tasks) == 1
        assert pending_tasks[0].id == task1.id
    
    def test_get_ready_tasks(self, planner, sample_tasks):
        """Test retrieval of ready tasks (dependencies met)"""
        for task in sample_tasks:
            planner.add_task(task)
        
        ready_tasks = planner.get_ready_tasks()
        
        # Only task_1 should be ready (no dependencies)
        assert len(ready_tasks) == 1
        assert ready_tasks[0].id == "task_1"
        
        # Complete task_1 and check again
        planner.tasks["task_1"].status = TaskStatus.COMPLETED
        ready_tasks = planner.get_ready_tasks()
        
        # Now task_2 should be ready
        assert len(ready_tasks) == 1
        assert ready_tasks[0].id == "task_2"
    
    def test_dependencies_met(self, planner, sample_tasks):
        """Test dependency checking"""
        task1, task2 = sample_tasks[0], sample_tasks[1]
        planner.add_task(task1)
        planner.add_task(task2)
        
        # task_2 depends on task_1
        assert planner._dependencies_met(task1) == True  # No dependencies
        assert planner._dependencies_met(task2) == False  # task_1 not completed
        
        # Complete task_1
        task1.status = TaskStatus.COMPLETED
        assert planner._dependencies_met(task2) == True  # Now dependencies are met
    
    def test_calculate_quantum_priority(self, planner, sample_tasks):
        """Test quantum priority calculation"""
        tasks = sample_tasks[:2]  # Use first two tasks
        
        priorities = planner.calculate_quantum_priority(tasks)
        
        assert len(priorities) == 2
        assert all(isinstance(score, float) for _, score in priorities)
        assert all(score >= 0 for _, score in priorities)  # Scores should be non-negative
        
        # Higher priority task should have higher score
        task_scores = {task.id: score for task, score in priorities}
        assert len(task_scores) == 2
    
    def test_quantum_anneal_schedule(self, planner, sample_tasks):
        """Test quantum annealing scheduling"""
        tasks = [sample_tasks[0]]  # Single task for simplicity
        
        batches = planner.quantum_anneal_schedule(tasks)
        
        assert len(batches) > 0
        assert all(isinstance(batch, list) for batch in batches)
        assert all(isinstance(task, Task) for batch in batches for task in batch)
        
        # Check that all tasks are scheduled
        scheduled_tasks = [task for batch in batches for task in batch]
        assert len(scheduled_tasks) == len(tasks)
    
    def test_resource_efficiency_calculation(self, planner):
        """Test resource efficiency calculation"""
        task_high_resource = Task(
            id="high_resource",
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=2.0,
            required_resources={"gpu": 0.9, "memory": 0.8},
            dependencies=[]
        )
        
        task_low_resource = Task(
            id="low_resource",
            task_type=TaskType.EMBEDDING,
            priority=0.8,
            estimated_duration=2.0,
            required_resources={"gpu": 0.1, "memory": 0.1},
            dependencies=[]
        )
        
        high_efficiency = planner._calculate_resource_efficiency(task_high_resource)
        low_efficiency = planner._calculate_resource_efficiency(task_low_resource)
        
        # Lower resource usage should have higher efficiency
        assert low_efficiency > high_efficiency
        assert 0 < high_efficiency < 1
        assert 0 < low_efficiency <= 1
    
    def test_entanglement_factor(self, planner, sample_tasks):
        """Test quantum entanglement factor calculation"""
        task = sample_tasks[1]  # Has dependencies
        
        entanglement = planner._calculate_entanglement_factor(task, sample_tasks)
        
        assert isinstance(entanglement, float)
        assert entanglement >= 1.0  # Should have boost factor
        assert entanglement <= 2.0  # Should be capped
    
    @pytest.mark.asyncio
    async def test_execute_quantum_plan_no_tasks(self, planner):
        """Test execution with no ready tasks"""
        result = await planner.execute_quantum_plan()
        
        assert result["status"] == "no_ready_tasks"
        assert result["execution_time"] == 0
        assert result["tasks_completed"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_quantum_plan_with_tasks(self, planner, sample_tasks):
        """Test quantum plan execution with tasks"""
        # Add a ready task (no dependencies)
        task = sample_tasks[0]
        planner.add_task(task)
        
        result = await planner.execute_quantum_plan()
        
        assert result["status"] == "completed"
        assert result["execution_time"] > 0
        assert result["tasks_completed"] >= 0
        assert result["batches_executed"] > 0
    
    @pytest.mark.asyncio
    async def test_execute_single_task(self, planner, sample_tasks):
        """Test single task execution"""
        task = sample_tasks[0]
        
        success = await planner._execute_single_task(task)
        
        assert isinstance(success, bool)
        assert task.start_time is not None
        assert task.completion_time is not None
        assert task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
    
    def test_execution_stats(self, planner, sample_tasks):
        """Test execution statistics calculation"""
        # Add tasks with different statuses
        task1, task2, task3 = sample_tasks[:3]
        task1.status = TaskStatus.COMPLETED
        task1.start_time = datetime.now() - timedelta(seconds=5)
        task1.completion_time = datetime.now()
        
        task2.status = TaskStatus.FAILED
        task3.status = TaskStatus.PENDING
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.add_task(task3)
        
        stats = planner.get_execution_stats()
        
        assert stats["total_tasks"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 1
        assert 0 <= stats["success_rate"] <= 1
        assert stats["total_execution_time"] >= 0
    
    def test_can_fit_in_batch(self, planner):
        """Test batch fitting logic"""
        task = Task(
            id="test_task",
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=2.0,
            required_resources={"gpu": 0.3, "memory": 0.2},
            dependencies=[]
        )
        
        batch = []
        available_resources = {"gpu": 1.0, "memory": 1.0}
        
        # Should fit
        assert planner._can_fit_in_batch(task, batch, available_resources) == True
        
        # Reduce available resources
        available_resources = {"gpu": 0.1, "memory": 0.1}
        
        # Should not fit
        assert planner._can_fit_in_batch(task, batch, available_resources) == False
    
    def test_update_available_resources(self, planner):
        """Test resource updating logic"""
        task = Task(
            id="test_task",
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=2.0,
            required_resources={"gpu": 0.3, "memory": 0.2},
            dependencies=[]
        )
        
        available_resources = {"gpu": 1.0, "memory": 1.0}
        planner._update_available_resources(task, available_resources)
        
        assert available_resources["gpu"] == 0.7  # 1.0 - 0.3
        assert available_resources["memory"] == 0.8  # 1.0 - 0.2


class TestQuantumOptimizer:
    """Test suite for QuantumOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create test optimizer"""
        return QuantumOptimizer(
            objective=OptimizationObjective.BALANCE_ALL,
            max_iterations=50,
            convergence_threshold=1e-4
        )
    
    @pytest.fixture
    def sample_optimization_tasks(self):
        """Create sample tasks for optimization"""
        return [
            {
                "id": "opt_task_1",
                "type": "embedding",
                "priority": 0.9,
                "estimated_duration": 1.0,
                "required_resources": {"gpu": 0.5, "memory": 0.3},
                "dependencies": []
            },
            {
                "id": "opt_task_2",
                "type": "attention",
                "priority": 0.8,
                "estimated_duration": 2.0,
                "required_resources": {"gpu": 0.7, "memory": 0.4},
                "dependencies": ["opt_task_1"]
            }
        ]
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = QuantumOptimizer(
            objective=OptimizationObjective.MINIMIZE_LATENCY,
            max_iterations=100
        )
        
        assert optimizer.objective == OptimizationObjective.MINIMIZE_LATENCY
        assert optimizer.max_iterations == 100
        assert optimizer.initial_temperature == 10.0
        assert optimizer.cooling_rate == 0.95
    
    def test_initialize_quantum_state(self, optimizer):
        """Test quantum state initialization"""
        n_tasks = 5
        state = optimizer._initialize_quantum_state(n_tasks)
        
        assert isinstance(state, np.ndarray)
        assert state.dtype == complex
        assert len(state) == n_tasks
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10
    
    def test_quantum_variation(self, optimizer, sample_optimization_tasks):
        """Test quantum variation circuit"""
        n_tasks = len(sample_optimization_tasks)
        quantum_state = optimizer._initialize_quantum_state(n_tasks)
        
        schedule = optimizer._quantum_variation(quantum_state, sample_optimization_tasks)
        
        assert isinstance(schedule, dict)
        assert "task_order" in schedule
        assert "batch_assignments" in schedule
        assert "resource_allocation" in schedule
        assert "execution_timeline" in schedule
        
        # Validate task order
        task_order = schedule["task_order"]
        assert len(task_order) == n_tasks
        assert all(isinstance(idx, (int, np.integer)) for idx in task_order)
    
    def test_evaluate_objective_functions(self, optimizer, sample_optimization_tasks):
        """Test different objective function evaluations"""
        # Create mock schedule
        schedule = {
            "task_order": [0, 1],
            "execution_timeline": {0: (0.0, 1.0), 1: (1.0, 3.0)},
            "resource_allocation": {0: {"gpu": 0.5}, 1: {"gpu": 0.7}}
        }
        
        constraints = OptimizationConstraints()
        resources = {"gpu": 1.0, "memory": 1.0}
        
        # Test different objectives
        latency = optimizer._calculate_latency_objective(schedule, sample_optimization_tasks)
        throughput = optimizer._calculate_throughput_objective(schedule, sample_optimization_tasks)
        resource_cost = optimizer._calculate_resource_objective(schedule, sample_optimization_tasks, resources)
        balanced = optimizer._calculate_balanced_objective(schedule, sample_optimization_tasks, constraints, resources)
        
        assert isinstance(latency, float)
        assert isinstance(throughput, float)
        assert isinstance(resource_cost, float)
        assert isinstance(balanced, float)
        
        assert latency > 0
        assert throughput >= 0
        assert resource_cost >= 0
        assert balanced >= 0
    
    def test_acceptance_probability(self, optimizer):
        """Test quantum annealing acceptance probability"""
        # Better energy should always be accepted
        prob = optimizer._calculate_acceptance_probability(5.0, 10.0, 1.0)
        assert prob == 1.0
        
        # Worse energy at high temperature should have high probability
        prob = optimizer._calculate_acceptance_probability(10.0, 5.0, 10.0)
        assert 0 < prob < 1
        
        # Worse energy at zero temperature should be rejected
        prob = optimizer._calculate_acceptance_probability(10.0, 5.0, 0.0)
        assert prob == 0.0
    
    def test_temperature_update(self, optimizer):
        """Test temperature update schedules"""
        initial_temp = 10.0
        
        # Exponential cooling
        optimizer.temperature_schedule = "exponential"
        new_temp = optimizer._update_temperature(initial_temp, 1)
        assert new_temp == initial_temp * optimizer.cooling_rate
        
        # Linear cooling
        optimizer.temperature_schedule = "linear"
        new_temp = optimizer._update_temperature(initial_temp, 1)
        assert 0 <= new_temp <= initial_temp
        
        # Logarithmic cooling
        optimizer.temperature_schedule = "logarithmic"
        new_temp = optimizer._update_temperature(initial_temp, 1)
        assert 0 < new_temp <= initial_temp
    
    def test_optimize_task_schedule(self, optimizer, sample_optimization_tasks):
        """Test complete task schedule optimization"""
        constraints = OptimizationConstraints(
            max_execution_time=10.0,
            max_memory_usage=1.0
        )
        resources = {"gpu": 1.0, "memory": 1.0}
        
        result = optimizer.optimize_task_schedule(
            sample_optimization_tasks, 
            constraints, 
            resources
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.optimization_time >= 0
        assert result.iterations_completed > 0
        assert isinstance(result.optimal_solution, dict)
        assert isinstance(result.convergence_achieved, bool)
        
        # Check if solution makes sense
        if result.optimal_solution:
            assert "task_order" in result.optimal_solution
            assert "execution_timeline" in result.optimal_solution
    
    def test_optimize_resource_allocation(self, optimizer):
        """Test resource allocation optimization"""
        demands = {"cpu": 0.7, "memory": 0.8, "gpu": 0.5}
        available = {"cpu": 1.0, "memory": 1.0, "gpu": 1.0}
        
        result = optimizer.optimize_resource_allocation(demands, available)
        
        assert isinstance(result, OptimizationResult)
        assert result.optimization_time >= 0
        assert isinstance(result.optimal_solution, dict)
        
        # Check allocation doesn't exceed available resources
        allocation = result.optimal_solution
        for resource, amount in allocation.items():
            assert amount <= available.get(resource, 0)


class TestQuantumStateValidator:
    """Test suite for QuantumStateValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create test validator"""
        return QuantumStateValidator(tolerance=1e-10)
    
    def test_valid_quantum_state(self, validator):
        """Test validation of valid quantum state"""
        # Create normalized complex state
        state = np.array([0.6+0.8j, 0.0+0.0j], dtype=complex)
        state = state / np.linalg.norm(state)  # Normalize
        
        result = validator.validate_quantum_state(state, "test_state")
        
        assert result.is_valid == True
        assert result.error_type is None
        assert result.metadata is not None
        
        # Check metrics
        metrics = result.metadata
        assert "normalization" in metrics
        assert "entropy" in metrics
        assert "coherence_score" in metrics
    
    def test_invalid_quantum_state_not_complex(self, validator):
        """Test validation of non-complex state"""
        state = np.array([1.0, 0.0], dtype=float)  # Real, not complex
        
        result = validator.validate_quantum_state(state)
        
        assert result.is_valid == False
        assert result.error_type == "InvalidStateType"
        assert result.severity == ErrorSeverity.HIGH
    
    def test_invalid_quantum_state_not_normalized(self, validator):
        """Test validation of non-normalized state"""
        state = np.array([2.0+0j, 3.0+0j], dtype=complex)  # Not normalized
        
        result = validator.validate_quantum_state(state)
        
        assert result.is_valid == False
        assert result.error_type == "NotNormalized"
        assert result.severity == ErrorSeverity.MEDIUM
    
    def test_invalid_quantum_state_nan_values(self, validator):
        """Test validation of state with NaN values"""
        state = np.array([np.nan+0j, 0.0+0j], dtype=complex)
        
        result = validator.validate_quantum_state(state)
        
        assert result.is_valid == False
        assert result.error_type == "InvalidValues"
        assert result.severity == ErrorSeverity.CRITICAL
    
    def test_quantum_state_evolution_validation(self, validator):
        """Test quantum state evolution validation"""
        initial_state = np.array([1.0+0j, 0.0+0j], dtype=complex)
        
        # Valid unitary evolution (identity)
        evolution_matrix = np.eye(2, dtype=complex)
        final_state = evolution_matrix @ initial_state
        
        result = validator.validate_quantum_evolution(
            initial_state, final_state, evolution_matrix
        )
        
        assert result.is_valid == True
    
    def test_non_unitary_evolution_validation(self, validator):
        """Test validation of non-unitary evolution"""
        initial_state = np.array([1.0+0j, 0.0+0j], dtype=complex)
        
        # Non-unitary matrix
        non_unitary = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=complex)
        final_state = non_unitary @ initial_state
        
        result = validator.validate_quantum_evolution(
            initial_state, final_state, non_unitary
        )
        
        assert result.is_valid == False
        assert result.error_type == "NonUnitaryEvolution"
    
    def test_is_unitary_check(self, validator):
        """Test unitarity checking"""
        # Identity matrix is unitary
        identity = np.eye(2, dtype=complex)
        assert validator._is_unitary(identity) == True
        
        # Pauli-X matrix is unitary
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        assert validator._is_unitary(pauli_x) == True
        
        # Non-unitary matrix
        non_unitary = np.array([[2, 0], [0, 1]], dtype=complex)
        assert validator._is_unitary(non_unitary) == False
    
    def test_quantum_fidelity_calculation(self, validator):
        """Test quantum fidelity calculation"""
        state1 = np.array([1.0+0j, 0.0+0j], dtype=complex)
        state2 = np.array([0.0+0j, 1.0+0j], dtype=complex)
        state3 = np.array([1.0+0j, 0.0+0j], dtype=complex)
        
        # Orthogonal states should have fidelity 0
        fidelity_orthogonal = validator._quantum_fidelity(state1, state2)
        assert abs(fidelity_orthogonal) < 1e-10
        
        # Identical states should have fidelity 1
        fidelity_identical = validator._quantum_fidelity(state1, state3)
        assert abs(fidelity_identical - 1.0) < 1e-10


@pytest.mark.integration
class TestQuantumPlannerIntegration:
    """Integration tests for quantum planner components"""
    
    @pytest.fixture
    def full_system(self):
        """Create complete quantum planning system"""
        config = QuantumTaskConfig(
            max_parallel_tasks=4,
            quantum_annealing_steps=50,
            optimization_rounds=5
        )
        
        planner = QuantumTaskPlanner(config)
        optimizer = QuantumOptimizer(max_iterations=20)
        validator = QuantumPlanningValidator()
        
        return {
            "planner": planner,
            "optimizer": optimizer,
            "validator": validator,
            "config": config
        }
    
    @pytest.fixture
    def complex_workflow_tasks(self):
        """Create complex workflow for integration testing"""
        tasks = []
        
        # Protocol initialization
        tasks.append(Task(
            id="protocol_init",
            task_type=TaskType.PROTOCOL_INIT,
            priority=1.0,
            estimated_duration=0.5,
            required_resources={"cpu": 0.2, "memory": 0.1},
            dependencies=[]
        ))
        
        # Embedding layers
        for i in range(3):
            tasks.append(Task(
                id=f"embedding_{i}",
                task_type=TaskType.EMBEDDING,
                priority=0.9,
                estimated_duration=1.0,
                required_resources={"gpu": 0.4, "memory": 0.3},
                dependencies=["protocol_init"] if i == 0 else [f"embedding_{i-1}"]
            ))
        
        # Attention layers
        for i in range(6):
            tasks.append(Task(
                id=f"attention_{i}",
                task_type=TaskType.ATTENTION,
                priority=0.8,
                estimated_duration=2.0,
                required_resources={"gpu": 0.6, "memory": 0.4},
                dependencies=[f"embedding_{2}" if i == 0 else f"attention_{i-1}"]
            ))
        
        # Feedforward layers
        for i in range(6):
            tasks.append(Task(
                id=f"feedforward_{i}",
                task_type=TaskType.FEEDFORWARD,
                priority=0.7,
                estimated_duration=1.5,
                required_resources={"gpu": 0.5, "memory": 0.3},
                dependencies=[f"attention_{i}"]
            ))
        
        # Result reconstruction
        tasks.append(Task(
            id="reconstruction",
            task_type=TaskType.RESULT_RECONSTRUCTION,
            priority=0.9,
            estimated_duration=0.5,
            required_resources={"cpu": 0.3, "memory": 0.2},
            dependencies=[f"feedforward_{i}" for i in range(6)]
        ))
        
        return tasks
    
    def test_full_workflow_validation(self, full_system, complex_workflow_tasks):
        """Test validation of complete workflow"""
        validator = full_system["validator"]
        
        result = validator.validate_complete_plan(complex_workflow_tasks)
        
        assert result.is_valid == True
        assert result.error_message is None
    
    def test_quantum_optimization_workflow(self, full_system, complex_workflow_tasks):
        """Test quantum optimization on complex workflow"""
        optimizer = full_system["optimizer"]
        
        # Convert tasks to optimization format
        task_dicts = []
        for task in complex_workflow_tasks:
            task_dict = {
                "id": task.id,
                "type": task.task_type.value,
                "priority": task.priority,
                "estimated_duration": task.estimated_duration,
                "required_resources": task.required_resources,
                "dependencies": task.dependencies
            }
            task_dicts.append(task_dict)
        
        constraints = OptimizationConstraints(
            max_execution_time=30.0,
            max_memory_usage=1.0
        )
        resources = {"cpu": 1.0, "memory": 1.0, "gpu": 1.0}
        
        result = optimizer.optimize_task_schedule(task_dicts, constraints, resources)
        
        assert isinstance(result, OptimizationResult)
        assert result.optimization_time > 0
        assert result.iterations_completed > 0
        assert len(result.optimal_solution) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_execution(self, full_system, complex_workflow_tasks):
        """Test end-to-end quantum planning and execution"""
        planner = full_system["planner"]
        
        # Add all tasks
        for task in complex_workflow_tasks:
            planner.add_task(task)
        
        # Execute quantum plan
        result = await planner.execute_quantum_plan()
        
        assert result["status"] == "completed"
        assert result["tasks_completed"] > 0
        assert result["execution_time"] > 0
        assert result["batches_executed"] > 0
        
        # Verify some tasks were completed
        stats = planner.get_execution_stats()
        assert stats["completed"] > 0
        assert 0 <= stats["success_rate"] <= 1
    
    def test_performance_under_load(self, full_system):
        """Test performance with high task load"""
        planner = full_system["planner"]
        
        # Create many small tasks
        tasks = []
        for i in range(50):
            task = Task(
                id=f"load_test_{i}",
                task_type=TaskType.EMBEDDING,
                priority=0.5,
                estimated_duration=0.1,
                required_resources={"cpu": 0.1, "memory": 0.05},
                dependencies=[]
            )
            tasks.append(task)
        
        # Add all tasks
        start_time = datetime.now()
        for task in tasks:
            planner.add_task(task)
        add_time = (datetime.now() - start_time).total_seconds()
        
        # Check performance
        assert add_time < 1.0  # Should add 50 tasks in under 1 second
        assert len(planner.tasks) == 50
        
        # Test priority calculation performance
        start_time = datetime.now()
        priorities = planner.calculate_quantum_priority(tasks[:10])
        calc_time = (datetime.now() - start_time).total_seconds()
        
        assert calc_time < 0.5  # Should calculate priorities quickly
        assert len(priorities) == 10
    
    def test_error_handling_and_recovery(self, full_system, complex_workflow_tasks):
        """Test error handling and recovery mechanisms"""
        planner = full_system["planner"]
        validator = full_system["validator"]
        
        # Create invalid task (circular dependency)
        invalid_task = Task(
            id="invalid_task",
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=1.0,
            required_resources={"gpu": 0.5},
            dependencies=["invalid_task"]  # Self-dependency (circular)
        )
        
        invalid_tasks = complex_workflow_tasks + [invalid_task]
        
        # Validation should catch the circular dependency
        result = validator.validate_complete_plan(invalid_tasks)
        assert result.is_valid == False
        assert "circular" in result.error_message.lower()
    
    def test_resource_constraint_handling(self, full_system):
        """Test handling of resource constraints"""
        planner = full_system["planner"]
        validator = full_system["validator"]
        
        # Create task requiring more resources than available
        over_resourced_task = Task(
            id="over_resourced",
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=1.0,
            required_resources={"gpu": 2.0, "memory": 3.0},  # More than 100%
            dependencies=[]
        )
        
        result = validator.validate_complete_plan([over_resourced_task])
        
        # Should detect resource constraint violation
        assert result.is_valid == False
        assert "resource" in result.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])