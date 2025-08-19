"""
Quantum Task Scheduler

High-level scheduler that integrates quantum planning with MPC transformer
workflows, providing intelligent task orchestration and execution management.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .optimization import OptimizationObjective, OptimizationResult, QuantumOptimizer
from .quantum_planner import (
    QuantumTaskConfig,
    QuantumTaskPlanner,
    Task,
    TaskStatus,
    TaskType,
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    BACKGROUND = 0.2


class SchedulerStatus(Enum):
    """Scheduler operational status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SchedulerConfig:
    """Configuration for the quantum scheduler"""
    max_concurrent_tasks: int = 8
    task_timeout: float = 300.0  # seconds
    retry_attempts: int = 3
    enable_adaptive_scheduling: bool = True
    quantum_optimization: bool = True
    performance_monitoring: bool = True
    auto_scaling: bool = False
    resource_limits: dict[str, float] | None = None


@dataclass
class SchedulerMetrics:
    """Scheduler performance metrics"""
    tasks_scheduled: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    total_execution_time: float = 0.0
    quantum_optimization_time: float = 0.0
    resource_utilization: dict[str, float] = None

    def __post_init__(self):
        if self.resource_utilization is None:
            self.resource_utilization = {}


class QuantumScheduler:
    """
    Quantum-enhanced task scheduler for MPC transformer workflows.
    
    Integrates quantum planning algorithms with practical scheduling
    capabilities for optimal resource utilization and performance.
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self.planner = QuantumTaskPlanner(QuantumTaskConfig(
            max_parallel_tasks=self.config.max_concurrent_tasks,
            enable_gpu_acceleration=True
        ))
        self.optimizer = QuantumOptimizer(
            objective=OptimizationObjective.BALANCE_ALL,
            max_iterations=500
        )

        # Scheduler state
        self.status = SchedulerStatus.IDLE
        self.metrics = SchedulerMetrics()
        self.active_tasks: dict[str, Task] = {}
        self.task_futures: dict[str, asyncio.Future] = {}
        self.resource_monitors: dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)

        # Event callbacks
        self.task_callbacks: dict[str, list[Callable]] = {
            'on_task_start': [],
            'on_task_complete': [],
            'on_task_fail': [],
            'on_batch_complete': []
        }

        logger.info(f"Initialized QuantumScheduler with {self.config.max_concurrent_tasks} max concurrent tasks")

    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self.task_callbacks:
            self.task_callbacks[event].append(callback)

    def create_mpc_task(self,
                       task_id: str,
                       task_type: TaskType,
                       priority: TaskPriority = TaskPriority.MEDIUM,
                       estimated_duration: float = 1.0,
                       required_resources: dict[str, float] | None = None,
                       dependencies: list[str] | None = None,
                       metadata: dict[str, Any] | None = None) -> Task:
        """
        Create an MPC-specific task with quantum-enhanced properties.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of MPC operation
            priority: Task priority level
            estimated_duration: Expected execution time
            required_resources: Resource requirements
            dependencies: Task dependencies
            metadata: Additional task metadata
            
        Returns:
            Configured Task object
        """
        task = Task(
            id=task_id,
            task_type=task_type,
            priority=priority.value,
            estimated_duration=estimated_duration,
            required_resources=required_resources or {},
            dependencies=dependencies or []
        )

        # Add metadata
        if metadata:
            task.metadata = metadata

        # Add to planner
        self.planner.add_task(task)

        logger.debug(f"Created MPC task {task_id} of type {task_type}")
        return task

    def create_inference_workflow(self,
                                model_name: str,
                                input_data: Any,
                                workflow_id: str | None = None,
                                priority: TaskPriority = TaskPriority.MEDIUM) -> list[Task]:
        """
        Create a complete inference workflow with quantum-optimized task dependencies.
        
        Args:
            model_name: Name of the transformer model
            input_data: Input data for inference
            workflow_id: Optional workflow identifier
            priority: Priority for all workflow tasks
            
        Returns:
            List of tasks representing the complete workflow
        """
        workflow_id = workflow_id or f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tasks = []

        # 1. Protocol initialization
        protocol_task = self.create_mpc_task(
            task_id=f"{workflow_id}_protocol_init",
            task_type=TaskType.PROTOCOL_INIT,
            priority=priority,
            estimated_duration=2.0,
            required_resources={"cpu": 0.3, "memory": 0.2},
            metadata={"model": model_name, "workflow": workflow_id}
        )
        tasks.append(protocol_task)

        # 2. Input embedding
        embedding_task = self.create_mpc_task(
            task_id=f"{workflow_id}_embedding",
            task_type=TaskType.EMBEDDING,
            priority=priority,
            estimated_duration=1.5,
            required_resources={"gpu": 0.4, "memory": 0.3},
            dependencies=[protocol_task.id],
            metadata={"model": model_name, "workflow": workflow_id}
        )
        tasks.append(embedding_task)

        # 3. Attention layers (assume 12 layers for BERT-like model)
        attention_tasks = []
        for layer_idx in range(12):
            attention_task = self.create_mpc_task(
                task_id=f"{workflow_id}_attention_layer_{layer_idx}",
                task_type=TaskType.ATTENTION,
                priority=priority,
                estimated_duration=3.0,
                required_resources={"gpu": 0.6, "memory": 0.4},
                dependencies=[embedding_task.id if layer_idx == 0 else attention_tasks[-1].id],
                metadata={
                    "model": model_name,
                    "workflow": workflow_id,
                    "layer": layer_idx
                }
            )
            tasks.append(attention_task)
            attention_tasks.append(attention_task)

        # 4. Feed-forward layers
        ff_tasks = []
        for layer_idx in range(12):
            ff_task = self.create_mpc_task(
                task_id=f"{workflow_id}_feedforward_layer_{layer_idx}",
                task_type=TaskType.FEEDFORWARD,
                priority=priority,
                estimated_duration=2.0,
                required_resources={"gpu": 0.5, "memory": 0.3},
                dependencies=[attention_tasks[layer_idx].id],
                metadata={
                    "model": model_name,
                    "workflow": workflow_id,
                    "layer": layer_idx
                }
            )
            tasks.append(ff_task)
            ff_tasks.append(ff_task)

        # 5. Result reconstruction
        reconstruction_task = self.create_mpc_task(
            task_id=f"{workflow_id}_reconstruction",
            task_type=TaskType.RESULT_RECONSTRUCTION,
            priority=priority,
            estimated_duration=1.0,
            required_resources={"cpu": 0.4, "memory": 0.2},
            dependencies=[task.id for task in ff_tasks],
            metadata={"model": model_name, "workflow": workflow_id}
        )
        tasks.append(reconstruction_task)

        logger.info(f"Created inference workflow {workflow_id} with {len(tasks)} tasks")
        return tasks

    async def schedule_and_execute(self,
                                 task_filter: Callable[[Task], bool] | None = None) -> dict[str, Any]:
        """
        Execute quantum-optimized scheduling and task execution.
        
        Args:
            task_filter: Optional filter to select specific tasks
            
        Returns:
            Execution summary with performance metrics
        """
        if self.status != SchedulerStatus.IDLE:
            raise RuntimeError(f"Scheduler is not idle (current status: {self.status})")

        self.status = SchedulerStatus.RUNNING
        start_time = datetime.now()

        try:
            # Get ready tasks
            ready_tasks = self.planner.get_ready_tasks()
            if task_filter:
                ready_tasks = [task for task in ready_tasks if task_filter(task)]

            if not ready_tasks:
                logger.info("No ready tasks to schedule")
                return {
                    "status": "no_tasks",
                    "execution_time": 0,
                    "tasks_processed": 0
                }

            # Quantum optimization phase
            optimization_start = datetime.now()

            if self.config.quantum_optimization:
                optimization_result = await self._optimize_task_schedule(ready_tasks)
                logger.info(f"Quantum optimization completed in {optimization_result.optimization_time:.2f}s")
            else:
                optimization_result = None

            optimization_time = (datetime.now() - optimization_start).total_seconds()
            self.metrics.quantum_optimization_time += optimization_time

            # Execute optimized plan
            execution_result = await self.planner.execute_quantum_plan()

            # Update metrics
            self.metrics.tasks_scheduled += len(ready_tasks)
            self.metrics.tasks_completed += execution_result.get("tasks_completed", 0)
            self.metrics.total_execution_time += execution_result.get("execution_time", 0)

            if self.metrics.tasks_completed > 0:
                self.metrics.average_execution_time = (
                    self.metrics.total_execution_time / self.metrics.tasks_completed
                )

            # Fire completion callbacks
            for callback in self.task_callbacks['on_batch_complete']:
                try:
                    callback(execution_result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            total_time = (datetime.now() - start_time).total_seconds()

            return {
                "status": "completed",
                "total_execution_time": total_time,
                "quantum_optimization_time": optimization_time,
                "tasks_processed": len(ready_tasks),
                "tasks_completed": execution_result.get("tasks_completed", 0),
                "batches_executed": execution_result.get("batches_executed", 0),
                "optimization_result": optimization_result.optimal_solution if optimization_result else None
            }

        except Exception as e:
            self.status = SchedulerStatus.ERROR
            logger.error(f"Scheduler execution failed: {e}")
            raise
        finally:
            if self.status != SchedulerStatus.ERROR:
                self.status = SchedulerStatus.IDLE

    async def _optimize_task_schedule(self, tasks: list[Task]) -> OptimizationResult:
        """Run quantum optimization on task schedule"""
        # Convert tasks to optimization format
        task_dicts = []
        for task in tasks:
            task_dict = {
                "id": task.id,
                "type": task.task_type.value,
                "priority": task.priority,
                "estimated_duration": task.estimated_duration,
                "required_resources": task.required_resources,
                "dependencies": task.dependencies
            }
            task_dicts.append(task_dict)

        # Define available resources
        resources = self.config.resource_limits or {
            "cpu": 1.0,
            "memory": 1.0,
            "gpu": 1.0
        }

        # Run optimization
        from .optimization import OptimizationConstraints
        constraints = OptimizationConstraints(
            max_execution_time=self.config.task_timeout * len(tasks),
            max_memory_usage=1.0,
            max_gpu_usage=1.0
        )

        return self.optimizer.optimize_task_schedule(task_dicts, constraints, resources)

    def get_task_status(self, task_id: str) -> TaskStatus | None:
        """Get current status of a task"""
        task = self.planner.get_task(task_id)
        return task.status if task else None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        with self._lock:
            task = self.planner.get_task(task_id)
            if not task:
                return False

            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED

                # Cancel future if exists
                if task_id in self.task_futures:
                    future = self.task_futures[task_id]
                    future.cancel()
                    del self.task_futures[task_id]

                logger.info(f"Cancelled task {task_id}")
                return True

            return False

    def pause_scheduler(self) -> bool:
        """Pause the scheduler"""
        if self.status == SchedulerStatus.RUNNING:
            self.status = SchedulerStatus.PAUSED
            logger.info("Scheduler paused")
            return True
        return False

    def resume_scheduler(self) -> bool:
        """Resume the scheduler"""
        if self.status == SchedulerStatus.PAUSED:
            self.status = SchedulerStatus.RUNNING
            logger.info("Scheduler resumed")
            return True
        return False

    def get_scheduler_metrics(self) -> SchedulerMetrics:
        """Get current scheduler metrics"""
        return self.metrics

    def get_active_tasks(self) -> list[Task]:
        """Get currently active (running) tasks"""
        return [task for task in self.planner.tasks.values() if task.status == TaskStatus.RUNNING]

    def get_pending_tasks(self) -> list[Task]:
        """Get pending tasks"""
        return self.planner.get_pending_tasks()

    def estimate_completion_time(self, task_ids: list[str] | None = None) -> float:
        """
        Estimate completion time for specified tasks or all pending tasks.
        
        Args:
            task_ids: Optional list of task IDs to estimate for
            
        Returns:
            Estimated completion time in seconds
        """
        if task_ids:
            tasks = [self.planner.get_task(tid) for tid in task_ids if self.planner.get_task(tid)]
        else:
            tasks = self.get_pending_tasks()

        if not tasks:
            return 0.0

        # Simple estimation based on average execution time and parallelism
        total_duration = sum(task.estimated_duration for task in tasks)
        parallel_capacity = self.config.max_concurrent_tasks

        # Account for dependencies (simplified)
        dependency_penalty = 1.2  # 20% overhead for dependencies

        estimated_time = (total_duration / parallel_capacity) * dependency_penalty
        return estimated_time

    def optimize_resource_allocation(self) -> dict[str, float]:
        """
        Optimize resource allocation for current workload.
        
        Returns:
            Optimized resource allocation
        """
        pending_tasks = self.get_pending_tasks()
        if not pending_tasks:
            return {}

        # Aggregate resource demands
        demands = {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}
        for task in pending_tasks:
            for resource, amount in task.required_resources.items():
                if resource in demands:
                    demands[resource] += amount

        # Available resources
        available = self.config.resource_limits or {"cpu": 1.0, "memory": 1.0, "gpu": 1.0}

        # Use quantum optimizer for allocation
        result = self.optimizer.optimize_resource_allocation(demands, available)

        return result.optimal_solution

    def generate_schedule_report(self) -> dict[str, Any]:
        """Generate comprehensive scheduling report"""
        stats = self.planner.get_execution_stats()

        return {
            "scheduler_status": self.status.value,
            "metrics": {
                "tasks_scheduled": self.metrics.tasks_scheduled,
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": stats["success_rate"],
                "average_execution_time": self.metrics.average_execution_time,
                "total_execution_time": self.metrics.total_execution_time,
                "quantum_optimization_time": self.metrics.quantum_optimization_time
            },
            "task_statistics": stats,
            "resource_utilization": self.metrics.resource_utilization,
            "active_tasks": len(self.get_active_tasks()),
            "pending_tasks": len(self.get_pending_tasks()),
            "estimated_completion": self.estimate_completion_time(),
            "configuration": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "quantum_optimization_enabled": self.config.quantum_optimization,
                "adaptive_scheduling": self.config.enable_adaptive_scheduling
            }
        }

    def cleanup(self):
        """Clean up scheduler resources"""
        self.status = SchedulerStatus.STOPPING

        # Cancel all active futures
        for future in self.task_futures.values():
            future.cancel()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        self.status = SchedulerStatus.IDLE
        logger.info("Scheduler cleanup completed")
