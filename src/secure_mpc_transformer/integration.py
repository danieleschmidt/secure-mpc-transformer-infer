"""
MPC Transformer + Quantum Planning Integration

Seamless integration layer between the quantum task planner
and existing secure MPC transformer infrastructure.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .config import SecurityConfig
from .models.secure_transformer import SecureTransformer
from .planning.scheduler import QuantumScheduler, SchedulerConfig, TaskPriority
from .services.inference_service import InferenceService

logger = logging.getLogger(__name__)


class QuantumMPCIntegrator:
    """
    Integration layer that combines quantum-inspired task planning
    with secure MPC transformer inference workflows.
    """

    def __init__(self,
                 security_config: SecurityConfig,
                 scheduler_config: SchedulerConfig | None = None):
        self.security_config = security_config
        self.scheduler = QuantumScheduler(scheduler_config or SchedulerConfig())

        # Initialize MPC components
        self.transformer: SecureTransformer | None = None
        self.inference_service: InferenceService | None = None

        # Integration state
        self.active_workflows: dict[str, dict[str, Any]] = {}
        self.performance_history: list[dict[str, Any]] = []

        logger.info("Initialized QuantumMPCIntegrator")

    def initialize_transformer(self, model_name: str, **kwargs) -> SecureTransformer:
        """
        Initialize secure transformer with quantum scheduling integration.
        
        Args:
            model_name: Name/path of the transformer model
            **kwargs: Additional transformer configuration
            
        Returns:
            Configured SecureTransformer instance
        """
        self.transformer = SecureTransformer.from_pretrained(
            model_name,
            security_config=self.security_config,
            **kwargs
        )

        # Initialize inference service
        self.inference_service = InferenceService(
            transformer=self.transformer,
            config=self.security_config
        )

        logger.info(f"Initialized transformer model: {model_name}")
        return self.transformer

    async def quantum_inference(self,
                              text_inputs: list[str],
                              priority: TaskPriority = TaskPriority.MEDIUM,
                              workflow_id: str | None = None,
                              optimize_schedule: bool = True) -> dict[str, Any]:
        """
        Perform quantum-optimized secure MPC transformer inference.
        
        Args:
            text_inputs: List of input texts for inference
            priority: Task priority level
            workflow_id: Optional workflow identifier
            optimize_schedule: Enable quantum optimization
            
        Returns:
            Inference results with performance metrics
        """
        if not self.transformer or not self.inference_service:
            raise RuntimeError("Transformer not initialized. Call initialize_transformer() first.")

        workflow_id = workflow_id or f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now()

        logger.info(f"Starting quantum inference workflow {workflow_id} with {len(text_inputs)} inputs")

        try:
            # Create quantum-optimized workflow tasks for each input
            all_tasks = []
            input_results = {}

            for i, text_input in enumerate(text_inputs):
                input_workflow_id = f"{workflow_id}_input_{i}"

                # Create inference workflow tasks
                workflow_tasks = self.scheduler.create_inference_workflow(
                    model_name=self.transformer.model_name if hasattr(self.transformer, 'model_name') else "transformer",
                    input_data=text_input,
                    workflow_id=input_workflow_id,
                    priority=priority
                )

                all_tasks.extend(workflow_tasks)
                input_results[input_workflow_id] = {
                    "input_text": text_input,
                    "tasks": [task.id for task in workflow_tasks],
                    "status": "pending"
                }

            # Store workflow state
            self.active_workflows[workflow_id] = {
                "start_time": start_time,
                "inputs": text_inputs,
                "priority": priority,
                "tasks": [task.id for task in all_tasks],
                "input_results": input_results,
                "status": "running"
            }

            # Execute quantum-optimized scheduling
            scheduler_config = self.scheduler.config
            scheduler_config.quantum_optimization = optimize_schedule

            execution_result = await self.scheduler.schedule_and_execute()

            # Process results for each input
            final_results = []
            for i, text_input in enumerate(text_inputs):
                input_workflow_id = f"{workflow_id}_input_{i}"

                # Simulate inference result (in real implementation, would extract from task results)
                inference_result = await self._simulate_inference_result(text_input)

                final_results.append({
                    "input": text_input,
                    "output": inference_result,
                    "workflow_id": input_workflow_id,
                    "tasks_completed": len(input_results[input_workflow_id]["tasks"])
                })

                input_results[input_workflow_id]["status"] = "completed"
                input_results[input_workflow_id]["result"] = inference_result

            # Update workflow status
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now()

            # Record performance metrics
            total_time = (datetime.now() - start_time).total_seconds()
            self.performance_history.append({
                "workflow_id": workflow_id,
                "inputs_processed": len(text_inputs),
                "total_tasks": len(all_tasks),
                "execution_time": total_time,
                "quantum_optimization_time": execution_result.get("quantum_optimization_time", 0),
                "tasks_completed": execution_result.get("tasks_completed", 0),
                "success_rate": execution_result.get("tasks_completed", 0) / len(all_tasks) if all_tasks else 0,
                "timestamp": datetime.now()
            })

            return {
                "workflow_id": workflow_id,
                "results": final_results,
                "performance": {
                    "total_execution_time": total_time,
                    "quantum_optimization_time": execution_result.get("quantum_optimization_time", 0),
                    "tasks_created": len(all_tasks),
                    "tasks_completed": execution_result.get("tasks_completed", 0),
                    "batches_executed": execution_result.get("batches_executed", 0),
                    "scheduler_metrics": self.scheduler.get_scheduler_metrics().__dict__
                },
                "status": "completed"
            }

        except Exception as e:
            # Update workflow status on error
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "failed"
                self.active_workflows[workflow_id]["error"] = str(e)

            logger.error(f"Quantum inference failed for workflow {workflow_id}: {e}")
            raise

    async def _simulate_inference_result(self, text_input: str) -> dict[str, Any]:
        """
        Perform actual secure inference using the transformer and MPC protocols.
        """
        if not self.transformer:
            raise RuntimeError("Transformer not initialized")

        start_time = datetime.now()

        try:
            # Use the actual secure transformer for inference
            result = self.transformer.predict_secure(text_input)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract relevant information from secure computation
            return {
                "processed_text": f"Securely processed: {len(text_input)} characters",
                "embeddings_shape": list(result["input_shape"]),
                "output_shape": list(result["output_shape"]),
                "secure_computation_info": result["protocol_info"],
                "latency_ms": result["latency_ms"],
                "processing_time": processing_time,
                "security_level": self.security_config.security_level,
                "protocol_used": self.security_config.protocol_name
            }

        except Exception as e:
            # Fall back to simulation if secure inference fails
            logger.warning(f"Secure inference failed, using simulation: {e}")

            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "processed_text": f"Simulated processing: {text_input[:50]}...",
                "embeddings_shape": [1, min(len(text_input.split()), 512), 768],
                "output_shape": [1, min(len(text_input.split()), 512), 768],
                "simulation_mode": True,
                "processing_time": processing_time,
                "error": str(e)
            }

    def batch_inference(self,
                       text_batches: list[list[str]],
                       priority: TaskPriority = TaskPriority.MEDIUM,
                       concurrent_batches: int = 2) -> list[asyncio.Task]:
        """
        Process multiple batches of text inputs concurrently.
        
        Args:
            text_batches: List of text input batches
            priority: Task priority level
            concurrent_batches: Number of batches to process concurrently
            
        Returns:
            List of asyncio tasks for batch processing
        """
        tasks = []

        for i, batch in enumerate(text_batches):
            workflow_id = f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            task = asyncio.create_task(
                self.quantum_inference(
                    text_inputs=batch,
                    priority=priority,
                    workflow_id=workflow_id,
                    optimize_schedule=True
                )
            )
            tasks.append(task)

            # Limit concurrent batches
            if len(tasks) >= concurrent_batches:
                break

        logger.info(f"Created {len(tasks)} batch inference tasks")
        return tasks

    def get_workflow_status(self, workflow_id: str) -> dict[str, Any] | None:
        """Get status of a specific workflow"""
        return self.active_workflows.get(workflow_id)

    def get_active_workflows(self) -> dict[str, dict[str, Any]]:
        """Get all active workflows"""
        return {wid: info for wid, info in self.active_workflows.items()
                if info["status"] == "running"}

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary across all workflows"""
        if not self.performance_history:
            return {"status": "no_data"}

        total_workflows = len(self.performance_history)
        total_inputs = sum(p["inputs_processed"] for p in self.performance_history)
        total_execution_time = sum(p["execution_time"] for p in self.performance_history)
        total_optimization_time = sum(p["quantum_optimization_time"] for p in self.performance_history)

        avg_execution_time = total_execution_time / total_workflows
        avg_optimization_time = total_optimization_time / total_workflows
        avg_success_rate = sum(p["success_rate"] for p in self.performance_history) / total_workflows

        return {
            "total_workflows": total_workflows,
            "total_inputs_processed": total_inputs,
            "average_execution_time": avg_execution_time,
            "average_optimization_time": avg_optimization_time,
            "average_success_rate": avg_success_rate,
            "optimization_overhead": (avg_optimization_time / avg_execution_time * 100) if avg_execution_time > 0 else 0,
            "throughput": total_inputs / total_execution_time if total_execution_time > 0 else 0,
            "scheduler_metrics": self.scheduler.get_scheduler_metrics().__dict__
        }

    def optimize_for_workload(self, expected_workload: dict[str, Any]) -> SchedulerConfig:
        """
        Optimize scheduler configuration for expected workload.
        
        Args:
            expected_workload: Dict with 'inputs_per_hour', 'avg_input_length', etc.
            
        Returns:
            Optimized SchedulerConfig
        """
        inputs_per_hour = expected_workload.get("inputs_per_hour", 100)
        avg_input_length = expected_workload.get("avg_input_length", 100)
        peak_concurrency = expected_workload.get("peak_concurrency", 8)

        # Calculate optimal configuration
        optimal_concurrent_tasks = min(peak_concurrency, 16)  # Cap at 16
        task_timeout = max(300, avg_input_length * 2)  # Scale with input length

        optimized_config = SchedulerConfig(
            max_concurrent_tasks=optimal_concurrent_tasks,
            task_timeout=task_timeout,
            enable_adaptive_scheduling=True,
            quantum_optimization=inputs_per_hour > 50,  # Enable for high throughput
            performance_monitoring=True,
            auto_scaling=inputs_per_hour > 200,  # Enable for very high throughput
            resource_limits={
                "cpu": 1.0,
                "memory": min(1.0, avg_input_length / 1000),  # Scale memory with input size
                "gpu": 0.8  # Reserve some GPU capacity
            }
        )

        logger.info(f"Optimized scheduler config for {inputs_per_hour} inputs/hour")
        return optimized_config

    def create_inference_pipeline(self,
                                 preprocessing: Callable | None = None,
                                 postprocessing: Callable | None = None) -> Callable:
        """
        Create a complete inference pipeline with quantum optimization.
        
        Args:
            preprocessing: Optional preprocessing function
            postprocessing: Optional postprocessing function
            
        Returns:
            Pipeline function for end-to-end inference
        """
        async def pipeline(inputs: list[str], **kwargs) -> list[Any]:
            # Preprocessing
            if preprocessing:
                inputs = [preprocessing(inp) for inp in inputs]

            # Quantum inference
            result = await self.quantum_inference(inputs, **kwargs)
            outputs = [r["output"] for r in result["results"]]

            # Postprocessing
            if postprocessing:
                outputs = [postprocessing(out) for out in outputs]

            return outputs

        return pipeline

    def cleanup(self):
        """Clean up integrator resources"""
        self.scheduler.cleanup()
        self.active_workflows.clear()
        logger.info("QuantumMPCIntegrator cleanup completed")
