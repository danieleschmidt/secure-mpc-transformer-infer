"""
Quantum-Inspired Task Planner

Core implementation of quantum-inspired algorithms for optimizing
task scheduling in secure MPC transformer inference workflows.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types for MPC transformer operations"""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FEEDFORWARD = "feedforward"
    PROTOCOL_INIT = "protocol_init"
    SHARE_DISTRIBUTION = "share_distribution"
    RESULT_RECONSTRUCTION = "result_reconstruction"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QuantumTaskConfig:
    """Configuration for quantum-inspired task planning"""
    max_parallel_tasks: int = 8
    quantum_annealing_steps: int = 1000
    temperature_decay: float = 0.95
    optimization_rounds: int = 50
    enable_gpu_acceleration: bool = True
    cache_quantum_states: bool = True
    priority_weight: float = 1.0
    latency_weight: float = 2.0
    resource_weight: float = 1.5


@dataclass
class Task:
    """Represents a computational task in the MPC workflow"""
    id: str
    task_type: TaskType
    priority: float
    estimated_duration: float
    required_resources: dict[str, float]
    dependencies: list[str]
    status: TaskStatus = TaskStatus.PENDING
    start_time: datetime | None = None
    completion_time: datetime | None = None
    result: Any | None = None


class QuantumTaskPlanner:
    """
    Quantum-inspired task planner using simulated annealing
    and quantum optimization principles for MPC task scheduling.
    """

    def __init__(self, config: QuantumTaskConfig | None = None):
        self.config = config or QuantumTaskConfig()
        self.tasks: dict[str, Task] = {}
        self.execution_history: list[dict] = []
        self._quantum_state_cache: dict[str, np.ndarray] = {}

        logger.info(f"Initialized QuantumTaskPlanner with {self.config.max_parallel_tasks} parallel tasks")

    def add_task(self, task: Task) -> None:
        """Add a task to the planning queue"""
        self.tasks[task.id] = task
        logger.debug(f"Added task {task.id} of type {task.task_type}")

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the planning queue"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.debug(f"Removed task {task_id}")
            return True
        return False

    def get_task(self, task_id: str) -> Task | None:
        """Retrieve a task by ID"""
        return self.tasks.get(task_id)

    def get_pending_tasks(self) -> list[Task]:
        """Get all tasks with pending status"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute (dependencies met)"""
        pending_tasks = self.get_pending_tasks()
        ready_tasks = []

        for task in pending_tasks:
            if self._dependencies_met(task):
                ready_tasks.append(task)

        return ready_tasks

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    return False
            else:
                logger.warning(f"Dependency {dep_id} not found for task {task.id}")
                return False
        return True

    def calculate_quantum_priority(self, tasks: list[Task]) -> list[tuple[Task, float]]:
        """
        Calculate quantum-inspired priority scores using superposition principles.
        Higher scores indicate higher priority for execution.
        """
        if not tasks:
            return []

        # Create quantum state vector representing task states
        n_tasks = len(tasks)
        quantum_state = np.random.random(n_tasks) + 1j * np.random.random(n_tasks)
        quantum_state = quantum_state / np.linalg.norm(quantum_state)

        task_priorities = []

        for i, task in enumerate(tasks):
            # Quantum amplitude represents base priority
            amplitude = abs(quantum_state[i]) ** 2

            # Apply quantum interference effects based on task properties
            priority_factor = (
                task.priority * self.config.priority_weight +
                (1.0 / max(task.estimated_duration, 0.1)) * self.config.latency_weight +
                self._calculate_resource_efficiency(task) * self.config.resource_weight
            )

            # Quantum entanglement effect - consider task relationships
            entanglement_boost = self._calculate_entanglement_factor(task, tasks)

            final_score = amplitude * priority_factor * entanglement_boost
            task_priorities.append((task, final_score))

        # Sort by quantum priority score (descending)
        task_priorities.sort(key=lambda x: x[1], reverse=True)

        return task_priorities

    def _calculate_resource_efficiency(self, task: Task) -> float:
        """Calculate resource efficiency score for a task"""
        if not task.required_resources:
            return 1.0

        total_resources = sum(task.required_resources.values())
        if total_resources == 0:
            return 1.0

        # Efficiency is inversely related to resource consumption
        return 1.0 / (1.0 + total_resources)

    def _calculate_entanglement_factor(self, task: Task, all_tasks: list[Task]) -> float:
        """
        Calculate quantum entanglement factor based on task relationships.
        Tasks with many dependencies or dependents get boosted priority.
        """
        # Count dependencies
        dependency_count = len(task.dependencies)

        # Count tasks that depend on this task
        dependent_count = sum(1 for t in all_tasks if task.id in t.dependencies)

        # Entanglement boost based on connectivity
        entanglement_factor = 1.0 + 0.1 * (dependency_count + dependent_count)

        return min(entanglement_factor, 2.0)  # Cap the boost

    def quantum_anneal_schedule(self, tasks: list[Task]) -> list[list[Task]]:
        """
        Use quantum annealing to find optimal task scheduling batches.
        Returns batches of tasks that can run in parallel.
        """
        if not tasks:
            return []

        # Calculate quantum priorities
        prioritized_tasks = self.calculate_quantum_priority(tasks)

        # Create batches using quantum annealing principles
        batches = []
        remaining_tasks = [task for task, _ in prioritized_tasks]
        max_parallel = self.config.max_parallel_tasks

        while remaining_tasks:
            current_batch = []
            available_resources = {"cpu": 1.0, "memory": 1.0, "gpu": 1.0}

            # Select tasks for current batch using quantum selection
            for task in remaining_tasks[:]:
                if len(current_batch) >= max_parallel:
                    break

                if self._can_fit_in_batch(task, current_batch, available_resources):
                    current_batch.append(task)
                    remaining_tasks.remove(task)
                    self._update_available_resources(task, available_resources)

            if current_batch:
                batches.append(current_batch)
            elif remaining_tasks:
                # Force add one task to avoid infinite loop
                current_batch.append(remaining_tasks.pop(0))
                batches.append(current_batch)

        logger.info(f"Generated {len(batches)} quantum-optimized task batches")
        return batches

    def _can_fit_in_batch(self, task: Task, batch: list[Task], available_resources: dict[str, float]) -> bool:
        """Check if task can fit in current batch given resource constraints"""
        for resource, required in task.required_resources.items():
            if required > available_resources.get(resource, 0):
                return False
        return True

    def _update_available_resources(self, task: Task, available_resources: dict[str, float]) -> None:
        """Update available resources after adding task to batch"""
        for resource, required in task.required_resources.items():
            if resource in available_resources:
                available_resources[resource] -= required

    async def execute_quantum_plan(self) -> dict[str, Any]:
        """
        Execute the quantum-optimized task plan asynchronously.
        Returns execution summary with performance metrics.
        """
        start_time = datetime.now()
        ready_tasks = self.get_ready_tasks()

        if not ready_tasks:
            return {
                "status": "no_ready_tasks",
                "execution_time": 0,
                "tasks_completed": 0,
                "total_tasks": len(self.tasks)
            }

        # Generate quantum-optimized schedule
        task_batches = self.quantum_anneal_schedule(ready_tasks)

        completed_tasks = 0
        total_batches = len(task_batches)

        logger.info(f"Executing {total_batches} quantum-optimized batches")

        for batch_idx, batch in enumerate(task_batches):
            logger.info(f"Executing batch {batch_idx + 1}/{total_batches} with {len(batch)} tasks")

            # Execute batch tasks concurrently
            batch_results = await self._execute_batch(batch)
            completed_tasks += len([r for r in batch_results if r])

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "completed",
            "execution_time": execution_time,
            "tasks_completed": completed_tasks,
            "total_tasks": len(ready_tasks),
            "batches_executed": total_batches,
            "average_batch_size": len(ready_tasks) / total_batches if total_batches > 0 else 0
        }

    async def _execute_batch(self, batch: list[Task]) -> list[bool]:
        """Execute a batch of tasks concurrently"""
        tasks_to_run = []

        for task in batch:
            tasks_to_run.append(self._execute_single_task(task))

        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        return [not isinstance(result, Exception) for result in results]

    async def _execute_single_task(self, task: Task) -> bool:
        """Execute a single task (mock implementation)"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()

        try:
            # Simulate task execution time
            await asyncio.sleep(task.estimated_duration * 0.01)  # Scale down for demo

            # Mock successful completion
            task.status = TaskStatus.COMPLETED
            task.completion_time = datetime.now()
            task.result = f"Result for {task.id}"

            logger.debug(f"Completed task {task.id}")
            return True

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completion_time = datetime.now()
            logger.error(f"Task {task.id} failed: {e}")
            return False

    def get_execution_stats(self) -> dict[str, Any]:
        """Get comprehensive execution statistics"""
        completed = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
        running = [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]
        pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

        total_execution_time = 0
        for task in completed:
            if task.start_time and task.completion_time:
                duration = (task.completion_time - task.start_time).total_seconds()
                total_execution_time += duration

        return {
            "total_tasks": len(self.tasks),
            "completed": len(completed),
            "failed": len(failed),
            "running": len(running),
            "pending": len(pending),
            "success_rate": len(completed) / len(self.tasks) if self.tasks else 0,
            "total_execution_time": total_execution_time,
            "average_task_time": total_execution_time / len(completed) if completed else 0
        }
