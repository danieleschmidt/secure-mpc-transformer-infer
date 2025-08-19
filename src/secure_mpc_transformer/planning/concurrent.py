"""
Concurrent Quantum Task Execution

High-performance concurrent execution engine for quantum-inspired task planning
with advanced parallelization, load balancing, and resource management.
"""

import asyncio
import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import psutil

from .monitoring import MetricsCollector, QuantumPerformanceMonitor
from .quantum_planner import Task, TaskStatus, TaskType
from .validation import QuantumPlanningValidator

logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Types of task executors"""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"
    GPU = "gpu"


class LoadBalanceStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    QUANTUM_AWARE = "quantum_aware"
    RESOURCE_BASED = "resource_based"


@dataclass
class WorkerStats:
    """Statistics for individual worker"""
    worker_id: str
    executor_type: ExecutorType
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    current_load: float = 0.0
    resource_usage: dict[str, float] = field(default_factory=dict)
    last_activity: datetime | None = None

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0

    @property
    def avg_execution_time(self) -> float:
        return self.total_execution_time / self.tasks_completed if self.tasks_completed > 0 else 0.0


@dataclass
class ExecutionContext:
    """Context for task execution"""
    task: Task
    worker_id: str
    start_time: datetime
    quantum_state: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantumTaskWorker:
    """
    Individual worker for executing quantum-related tasks.
    Handles both CPU and GPU acceleration capabilities.
    """

    def __init__(self,
                 worker_id: str,
                 executor_type: ExecutorType = ExecutorType.THREAD,
                 gpu_device: int | None = None,
                 max_concurrent_tasks: int = 1):
        self.worker_id = worker_id
        self.executor_type = executor_type
        self.gpu_device = gpu_device
        self.max_concurrent_tasks = max_concurrent_tasks

        # Worker state
        self.active_tasks: dict[str, ExecutionContext] = {}
        self.stats = WorkerStats(worker_id, executor_type)
        self.is_available = True
        self.shutdown_requested = False

        # Synchronization
        self._lock = threading.RLock()
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # Performance monitoring
        self.metrics = MetricsCollector()

        logger.info(f"Initialized QuantumTaskWorker {worker_id} ({executor_type.value})")

    async def execute_task(self, task: Task) -> dict[str, Any]:
        """
        Execute a quantum task with appropriate acceleration.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result with performance metrics
        """
        execution_start = datetime.now()

        with self._lock:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                raise RuntimeError(f"Worker {self.worker_id} at capacity")

            context = ExecutionContext(
                task=task,
                worker_id=self.worker_id,
                start_time=execution_start
            )
            self.active_tasks[task.id] = context

        try:
            task.status = TaskStatus.RUNNING
            task.start_time = execution_start

            # Execute based on task type
            if task.task_type == TaskType.EMBEDDING:
                result = await self._execute_embedding_task(task, context)
            elif task.task_type == TaskType.ATTENTION:
                result = await self._execute_attention_task(task, context)
            elif task.task_type == TaskType.FEEDFORWARD:
                result = await self._execute_feedforward_task(task, context)
            elif task.task_type == TaskType.PROTOCOL_INIT:
                result = await self._execute_protocol_init(task, context)
            elif task.task_type == TaskType.RESULT_RECONSTRUCTION:
                result = await self._execute_reconstruction_task(task, context)
            else:
                result = await self._execute_generic_task(task, context)

            # Mark successful completion
            task.status = TaskStatus.COMPLETED
            task.completion_time = datetime.now()
            task.result = result

            # Update statistics
            execution_time = (datetime.now() - execution_start).total_seconds()
            with self._lock:
                self.stats.tasks_completed += 1
                self.stats.total_execution_time += execution_time
                self.stats.last_activity = datetime.now()

            self.metrics.record_timer("task_execution_time", execution_time, {
                "task_type": task.task_type.value,
                "worker_id": self.worker_id
            })

            return {
                "status": "completed",
                "result": result,
                "execution_time": execution_time,
                "worker_id": self.worker_id
            }

        except Exception as e:
            # Mark failed
            task.status = TaskStatus.FAILED
            task.completion_time = datetime.now()

            with self._lock:
                self.stats.tasks_failed += 1
                self.stats.last_activity = datetime.now()

            self.metrics.record_counter("task_execution_errors", 1.0, {
                "task_type": task.task_type.value,
                "worker_id": self.worker_id,
                "error": str(e)
            })

            logger.error(f"Task execution failed in worker {self.worker_id}: {e}")

            return {
                "status": "failed",
                "error": str(e),
                "execution_time": (datetime.now() - execution_start).total_seconds(),
                "worker_id": self.worker_id
            }

        finally:
            with self._lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]

    async def _execute_embedding_task(self, task: Task, context: ExecutionContext) -> dict[str, Any]:
        """Execute embedding layer computation"""
        # Simulate embedding computation with quantum enhancement
        await asyncio.sleep(task.estimated_duration * 0.1)  # Scaled for demo

        # Mock quantum-enhanced embedding
        input_dim = 768
        output_dim = 768

        if self.gpu_device is not None:
            # GPU-accelerated computation
            computation_time = task.estimated_duration * 0.6  # GPU speedup
        else:
            computation_time = task.estimated_duration

        return {
            "layer_type": "embedding",
            "input_shape": [1, input_dim],
            "output_shape": [1, output_dim],
            "computation_time": computation_time,
            "quantum_enhanced": True,
            "gpu_accelerated": self.gpu_device is not None
        }

    async def _execute_attention_task(self, task: Task, context: ExecutionContext) -> dict[str, Any]:
        """Execute attention mechanism with quantum optimization"""
        await asyncio.sleep(task.estimated_duration * 0.1)  # Scaled for demo

        # Simulate quantum-optimized attention
        seq_length = 512
        num_heads = 12
        head_dim = 64

        # Quantum superposition for attention weight calculation
        quantum_attention_weights = np.random.random((num_heads, seq_length, seq_length))
        quantum_attention_weights = quantum_attention_weights / np.sum(quantum_attention_weights, axis=-1, keepdims=True)

        context.quantum_state = quantum_attention_weights.flatten()[:256]  # Truncate for demo

        return {
            "layer_type": "attention",
            "num_heads": num_heads,
            "sequence_length": seq_length,
            "attention_weights_shape": quantum_attention_weights.shape,
            "quantum_superposition_applied": True,
            "coherence_score": np.random.uniform(0.7, 0.95)
        }

    async def _execute_feedforward_task(self, task: Task, context: ExecutionContext) -> dict[str, Any]:
        """Execute feedforward layer with MPC integration"""
        await asyncio.sleep(task.estimated_duration * 0.1)  # Scaled for demo

        # Simulate secure multi-party computation
        input_dim = 768
        hidden_dim = 3072

        # Mock secret sharing computation
        shares = {
            "party_1": np.random.random(hidden_dim),
            "party_2": np.random.random(hidden_dim),
            "party_3": np.random.random(hidden_dim)
        }

        return {
            "layer_type": "feedforward",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "mpc_computation": True,
            "secret_shares": len(shares),
            "security_level": 128
        }

    async def _execute_protocol_init(self, task: Task, context: ExecutionContext) -> dict[str, Any]:
        """Execute MPC protocol initialization"""
        await asyncio.sleep(task.estimated_duration * 0.05)  # Fast initialization

        return {
            "protocol": "aby3",
            "parties": 3,
            "security_level": 128,
            "initialization_time": task.estimated_duration * 0.05,
            "status": "initialized"
        }

    async def _execute_reconstruction_task(self, task: Task, context: ExecutionContext) -> dict[str, Any]:
        """Execute result reconstruction from secret shares"""
        await asyncio.sleep(task.estimated_duration * 0.1)  # Scaled for demo

        # Mock reconstruction from shares
        result_shape = [1, 768]
        reconstructed_result = np.random.random(result_shape)

        return {
            "reconstructed_shape": result_shape,
            "reconstruction_time": task.estimated_duration * 0.1,
            "verification_passed": True,
            "final_result": "encrypted_output_tensor"
        }

    async def _execute_generic_task(self, task: Task, context: ExecutionContext) -> dict[str, Any]:
        """Execute generic task"""
        await asyncio.sleep(task.estimated_duration * 0.1)  # Scaled for demo

        return {
            "task_type": task.task_type.value,
            "generic_execution": True,
            "estimated_duration": task.estimated_duration
        }

    def get_current_load(self) -> float:
        """Get current worker load (0.0 to 1.0)"""
        with self._lock:
            return len(self.active_tasks) / self.max_concurrent_tasks

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage"""
        try:
            process = psutil.Process()
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024
            }
        except:
            return {"cpu_percent": 0.0, "memory_percent": 0.0, "memory_mb": 0.0}

    def is_healthy(self) -> bool:
        """Check if worker is healthy and responsive"""
        with self._lock:
            # Check if worker has been inactive for too long
            if self.stats.last_activity:
                inactive_time = datetime.now() - self.stats.last_activity
                if inactive_time.total_seconds() > 300:  # 5 minutes
                    return False

            # Check success rate
            if self.stats.success_rate < 0.8 and (self.stats.tasks_completed + self.stats.tasks_failed) > 10:
                return False

            return not self.shutdown_requested


class ConcurrentQuantumExecutor:
    """
    High-performance concurrent executor for quantum task planning.
    Manages multiple workers with intelligent load balancing and scaling.
    """

    def __init__(self,
                 max_workers: int = None,
                 load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.QUANTUM_AWARE,
                 enable_gpu_acceleration: bool = True,
                 auto_scaling: bool = True):

        # Determine optimal worker count
        if max_workers is None:
            cpu_count = mp.cpu_count()
            max_workers = min(cpu_count * 2, 16)  # Cap at 16 workers

        self.max_workers = max_workers
        self.load_balance_strategy = load_balance_strategy
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.auto_scaling = auto_scaling

        # Worker management
        self.workers: dict[str, QuantumTaskWorker] = {}
        self.worker_futures: dict[str, asyncio.Future] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # Load balancing
        self.round_robin_index = 0
        self.worker_loads: dict[str, float] = {}

        # Performance monitoring
        self.monitor = QuantumPerformanceMonitor(MetricsCollector())
        self.validator = QuantumPlanningValidator()

        # Synchronization
        self._lock = threading.RLock()
        self.shutdown_event = asyncio.Event()

        # Auto-scaling parameters
        self.scale_up_threshold = 0.8    # Scale up when avg load > 80%
        self.scale_down_threshold = 0.3  # Scale down when avg load < 30%
        self.min_workers = max(1, max_workers // 4)

        logger.info(f"Initialized ConcurrentQuantumExecutor with {max_workers} max workers")

    async def start(self):
        """Start the concurrent executor"""
        # Initialize minimum workers
        for i in range(self.min_workers):
            await self._create_worker(f"worker_{i}", ExecutorType.THREAD)

        # Start auto-scaling if enabled
        if self.auto_scaling:
            asyncio.create_task(self._auto_scaling_loop())

        # Start load monitoring
        asyncio.create_task(self._monitoring_loop())

        logger.info(f"ConcurrentQuantumExecutor started with {len(self.workers)} workers")

    async def execute_tasks(self, tasks: list[Task]) -> list[dict[str, Any]]:
        """
        Execute multiple tasks concurrently with optimal load balancing.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            List of execution results
        """
        if not tasks:
            return []

        # Validate tasks before execution
        validation_result = self.validator.validate_complete_plan(tasks)
        if not validation_result.is_valid:
            logger.error(f"Task validation failed: {validation_result.error_message}")
            raise ValueError(f"Invalid task configuration: {validation_result.error_message}")

        # Start monitoring session
        session_id = f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.monitor.start_quantum_session(session_id, {"task_count": len(tasks)})

        try:
            # Create execution futures
            execution_futures = []

            for task in tasks:
                # Select optimal worker
                worker = await self._select_worker_for_task(task)
                if worker is None:
                    raise RuntimeError("No available workers for task execution")

                # Submit task for execution
                future = asyncio.create_task(worker.execute_task(task))
                execution_futures.append(future)

                # Update load tracking
                self._update_worker_load(worker.worker_id)

            # Wait for all tasks to complete
            results = await asyncio.gather(*execution_futures, return_exceptions=True)

            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "status": "failed",
                        "error": str(result),
                        "task_id": tasks[i].id
                    })

                    self.monitor.record_error(session_id, "ExecutionException", str(result))
                else:
                    processed_results.append(result)

            # Record final metrics
            successful_tasks = sum(1 for r in processed_results if r.get("status") == "completed")
            total_time = max(r.get("execution_time", 0) for r in processed_results if isinstance(r, dict))

            self.monitor.record_optimization_step(
                session_id,
                objective_value=successful_tasks / len(tasks),
                convergence_rate=1.0,  # Assume convergence for execution
                step_duration=total_time
            )

            return processed_results

        finally:
            # End monitoring session
            self.monitor.end_quantum_session(session_id)

    async def _select_worker_for_task(self, task: Task) -> QuantumTaskWorker | None:
        """Select optimal worker for task execution"""
        if not self.workers:
            return None

        if self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._select_round_robin()
        elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_LOADED:
            return self._select_least_loaded()
        elif self.load_balance_strategy == LoadBalanceStrategy.QUANTUM_AWARE:
            return self._select_quantum_aware(task)
        elif self.load_balance_strategy == LoadBalanceStrategy.RESOURCE_BASED:
            return self._select_resource_based(task)
        else:
            return self._select_round_robin()

    def _select_round_robin(self) -> QuantumTaskWorker | None:
        """Select worker using round-robin strategy"""
        available_workers = [w for w in self.workers.values() if w.is_available and w.is_healthy()]
        if not available_workers:
            return None

        worker = available_workers[self.round_robin_index % len(available_workers)]
        self.round_robin_index += 1
        return worker

    def _select_least_loaded(self) -> QuantumTaskWorker | None:
        """Select least loaded worker"""
        available_workers = [w for w in self.workers.values() if w.is_available and w.is_healthy()]
        if not available_workers:
            return None

        return min(available_workers, key=lambda w: w.get_current_load())

    def _select_quantum_aware(self, task: Task) -> QuantumTaskWorker | None:
        """Select worker based on quantum task characteristics"""
        available_workers = [w for w in self.workers.values() if w.is_available and w.is_healthy()]
        if not available_workers:
            return None

        # Prefer GPU workers for compute-intensive quantum tasks
        gpu_intensive_types = {TaskType.ATTENTION, TaskType.FEEDFORWARD, TaskType.EMBEDDING}

        if task.task_type in gpu_intensive_types and self.enable_gpu_acceleration:
            gpu_workers = [w for w in available_workers if w.gpu_device is not None]
            if gpu_workers:
                return min(gpu_workers, key=lambda w: w.get_current_load())

        # For other tasks, use least loaded
        return min(available_workers, key=lambda w: w.get_current_load())

    def _select_resource_based(self, task: Task) -> QuantumTaskWorker | None:
        """Select worker based on resource requirements"""
        available_workers = [w for w in self.workers.values() if w.is_available and w.is_healthy()]
        if not available_workers:
            return None

        # Score workers based on resource fit
        worker_scores = []

        for worker in available_workers:
            score = 0.0

            # Resource availability score
            resource_usage = worker.get_resource_usage()
            cpu_available = 100 - resource_usage.get("cpu_percent", 0)
            memory_available = 100 - resource_usage.get("memory_percent", 0)

            # Required resources
            required_cpu = task.required_resources.get("cpu", 0.5) * 100
            required_memory = task.required_resources.get("memory", 0.3) * 100

            # Check if resources are available
            if cpu_available >= required_cpu and memory_available >= required_memory:
                # Higher score for better fit
                score = (cpu_available - required_cpu) + (memory_available - required_memory)
                score += (100 - worker.get_current_load() * 100)  # Load bonus
            else:
                score = -1  # Cannot handle task

            worker_scores.append((worker, score))

        # Select worker with highest score
        valid_workers = [(w, s) for w, s in worker_scores if s >= 0]
        if not valid_workers:
            return min(available_workers, key=lambda w: w.get_current_load())  # Fallback

        return max(valid_workers, key=lambda x: x[1])[0]

    async def _create_worker(self, worker_id: str, executor_type: ExecutorType, gpu_device: int | None = None) -> QuantumTaskWorker:
        """Create new worker instance"""
        worker = QuantumTaskWorker(
            worker_id=worker_id,
            executor_type=executor_type,
            gpu_device=gpu_device,
            max_concurrent_tasks=2 if executor_type == ExecutorType.THREAD else 1
        )

        self.workers[worker_id] = worker
        self.worker_loads[worker_id] = 0.0

        logger.info(f"Created worker {worker_id} ({executor_type.value})")
        return worker

    def _update_worker_load(self, worker_id: str):
        """Update worker load tracking"""
        if worker_id in self.workers:
            self.worker_loads[worker_id] = self.workers[worker_id].get_current_load()

    async def _auto_scaling_loop(self):
        """Auto-scaling loop to adjust worker count based on load"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.workers:
                    continue

                # Calculate average load
                loads = [self.workers[wid].get_current_load() for wid in self.workers.keys()]
                avg_load = sum(loads) / len(loads)

                current_worker_count = len(self.workers)

                # Scale up if overloaded
                if (avg_load > self.scale_up_threshold and
                    current_worker_count < self.max_workers):

                    new_worker_id = f"worker_{current_worker_count}"
                    await self._create_worker(new_worker_id, ExecutorType.THREAD)
                    logger.info(f"Scaled up: created worker {new_worker_id} (avg load: {avg_load:.2f})")

                # Scale down if underloaded
                elif (avg_load < self.scale_down_threshold and
                      current_worker_count > self.min_workers):

                    # Find least active worker to remove
                    idle_workers = [(wid, w) for wid, w in self.workers.items()
                                   if w.get_current_load() == 0.0]

                    if idle_workers:
                        worker_id, worker = min(idle_workers,
                                              key=lambda x: x[1].stats.last_activity or datetime.min)

                        await self._remove_worker(worker_id)
                        logger.info(f"Scaled down: removed worker {worker_id} (avg load: {avg_load:.2f})")

            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")

    async def _monitoring_loop(self):
        """Performance monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute

                # Collect worker statistics
                total_tasks = sum(w.stats.tasks_completed for w in self.workers.values())
                total_errors = sum(w.stats.tasks_failed for w in self.workers.values())
                avg_success_rate = sum(w.stats.success_rate for w in self.workers.values()) / len(self.workers) if self.workers else 0

                # Log performance summary
                logger.info(f"Performance Summary: {len(self.workers)} workers, "
                          f"{total_tasks} tasks completed, {total_errors} errors, "
                          f"avg success rate: {avg_success_rate:.1%}")

                # Check for unhealthy workers
                unhealthy_workers = [wid for wid, w in self.workers.items() if not w.is_healthy()]
                for worker_id in unhealthy_workers:
                    logger.warning(f"Removing unhealthy worker: {worker_id}")
                    await self._remove_worker(worker_id)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    async def _remove_worker(self, worker_id: str):
        """Remove worker from pool"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.shutdown_requested = True

            # Wait for active tasks to complete (with timeout)
            timeout = 30  # 30 seconds
            start_time = time.time()

            while worker.active_tasks and (time.time() - start_time) < timeout:
                await asyncio.sleep(1)

            # Clean up
            del self.workers[worker_id]
            if worker_id in self.worker_loads:
                del self.worker_loads[worker_id]
            if worker_id in self.worker_futures:
                future = self.worker_futures[worker_id]
                if not future.done():
                    future.cancel()
                del self.worker_futures[worker_id]

    async def shutdown(self):
        """Shutdown the executor and all workers"""
        self.shutdown_event.set()

        # Request shutdown for all workers
        for worker in self.workers.values():
            worker.shutdown_requested = True

        # Wait for workers to finish current tasks
        max_wait = 60  # 1 minute maximum wait
        start_time = time.time()

        while self.workers and (time.time() - start_time) < max_wait:
            active_workers = [w for w in self.workers.values() if w.active_tasks]
            if not active_workers:
                break
            await asyncio.sleep(1)

        # Force cleanup remaining workers
        self.workers.clear()
        self.worker_loads.clear()

        logger.info("ConcurrentQuantumExecutor shutdown completed")

    def get_executor_stats(self) -> dict[str, Any]:
        """Get comprehensive executor statistics"""
        if not self.workers:
            return {"status": "no_workers"}

        worker_stats = [w.stats for w in self.workers.values()]

        return {
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.workers.values() if w.active_tasks]),
            "total_tasks_completed": sum(s.tasks_completed for s in worker_stats),
            "total_tasks_failed": sum(s.tasks_failed for s in worker_stats),
            "average_success_rate": sum(s.success_rate for s in worker_stats) / len(worker_stats),
            "average_load": sum(self.worker_loads.values()) / len(self.worker_loads),
            "total_execution_time": sum(s.total_execution_time for s in worker_stats),
            "load_balance_strategy": self.load_balance_strategy.value,
            "auto_scaling_enabled": self.auto_scaling,
            "gpu_acceleration_enabled": self.enable_gpu_acceleration
        }
