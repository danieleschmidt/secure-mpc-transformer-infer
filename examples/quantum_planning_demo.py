#!/usr/bin/env python3
"""
Quantum-Inspired Task Planning Demo

Demonstrates the quantum task planner integrated with secure MPC
transformer inference for optimal performance and resource utilization.
"""

import asyncio
import logging
import time
from typing import List
from pathlib import Path
import sys

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from secure_mpc_transformer import (
    SecurityConfig, 
    QuantumTaskPlanner, 
    QuantumScheduler, 
    TaskPriority
)
from secure_mpc_transformer.integration import QuantumMPCIntegrator
from secure_mpc_transformer.planning.scheduler import SchedulerConfig
from secure_mpc_transformer.planning.quantum_planner import TaskType, Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_quantum_planning():
    """Demo basic quantum task planning capabilities"""
    print("\n🔬 QUANTUM TASK PLANNER DEMO")
    print("=" * 50)
    
    # Initialize quantum planner
    planner = QuantumTaskPlanner()
    
    # Create sample MPC tasks
    tasks = [
        Task(
            id="embedding_layer",
            task_type=TaskType.EMBEDDING,
            priority=0.9,
            estimated_duration=2.0,
            required_resources={"gpu": 0.4, "memory": 0.3},
            dependencies=[]
        ),
        Task(
            id="attention_layer_1",
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=3.0,
            required_resources={"gpu": 0.6, "memory": 0.4},
            dependencies=["embedding_layer"]
        ),
        Task(
            id="attention_layer_2", 
            task_type=TaskType.ATTENTION,
            priority=0.8,
            estimated_duration=3.0,
            required_resources={"gpu": 0.6, "memory": 0.4},
            dependencies=["attention_layer_1"]
        ),
        Task(
            id="feedforward_layer_1",
            task_type=TaskType.FEEDFORWARD,
            priority=0.7,
            estimated_duration=2.0,
            required_resources={"gpu": 0.5, "memory": 0.3},
            dependencies=["attention_layer_1"]
        ),
        Task(
            id="result_reconstruction",
            task_type=TaskType.RESULT_RECONSTRUCTION,
            priority=0.9,
            estimated_duration=1.0,
            required_resources={"cpu": 0.4, "memory": 0.2},
            dependencies=["attention_layer_2", "feedforward_layer_1"]
        )
    ]
    
    # Add tasks to planner
    for task in tasks:
        planner.add_task(task)
    
    print(f"📋 Added {len(tasks)} MPC transformer tasks")
    
    # Test quantum priority calculation
    ready_tasks = planner.get_ready_tasks()
    print(f"🚀 Ready tasks: {len(ready_tasks)}")
    
    quantum_priorities = planner.calculate_quantum_priority(ready_tasks)
    print("⚛️ Quantum Priority Rankings:")
    for i, (task, score) in enumerate(quantum_priorities[:3]):
        print(f"   {i+1}. {task.id}: {score:.3f}")
    
    # Test quantum annealing schedule
    task_batches = planner.quantum_anneal_schedule(ready_tasks)
    print(f"📦 Generated {len(task_batches)} quantum-optimized batches")
    for i, batch in enumerate(task_batches):
        batch_tasks = [task.id for task in batch]
        print(f"   Batch {i+1}: {batch_tasks}")
    
    # Execute quantum plan
    print("\n⚡ Executing quantum-optimized plan...")
    execution_result = await planner.execute_quantum_plan()
    
    print("📊 Execution Results:")
    print(f"   Status: {execution_result['status']}")
    print(f"   Tasks completed: {execution_result['tasks_completed']}/{execution_result['total_tasks']}")
    print(f"   Execution time: {execution_result['execution_time']:.2f}s")
    print(f"   Batches executed: {execution_result['batches_executed']}")
    
    # Get final statistics
    stats = planner.get_execution_stats()
    print(f"   Success rate: {stats['success_rate']:.1%}")
    

async def demo_quantum_scheduler():
    """Demo quantum scheduler with MPC workflows"""
    print("\n🎯 QUANTUM SCHEDULER DEMO")
    print("=" * 50)
    
    # Initialize scheduler with optimized config
    config = SchedulerConfig(
        max_concurrent_tasks=4,
        quantum_optimization=True,
        performance_monitoring=True
    )
    
    scheduler = QuantumScheduler(config)
    
    # Create complete inference workflow
    print("🏗️ Creating inference workflow...")
    workflow_tasks = scheduler.create_inference_workflow(
        model_name="bert-base-uncased",
        input_data="The quick brown fox jumps over the lazy dog",
        priority=TaskPriority.HIGH
    )
    
    print(f"📝 Created workflow with {len(workflow_tasks)} tasks")
    print("🔗 Task dependency chain:")
    for task in workflow_tasks[:5]:  # Show first 5
        deps = ", ".join(task.dependencies) if task.dependencies else "None"
        print(f"   {task.id}: {task.task_type.value} -> deps: {deps}")
    
    # Execute with quantum optimization
    print("\n⚡ Executing with quantum optimization...")
    start_time = time.time()
    
    result = await scheduler.schedule_and_execute()
    
    execution_time = time.time() - start_time
    
    print("📊 Scheduler Results:")
    print(f"   Status: {result['status']}")
    print(f"   Total execution time: {execution_time:.2f}s")
    print(f"   Quantum optimization time: {result.get('quantum_optimization_time', 0):.2f}s")
    print(f"   Tasks processed: {result['tasks_processed']}")
    print(f"   Tasks completed: {result['tasks_completed']}")
    print(f"   Batches executed: {result['batches_executed']}")
    
    # Get scheduler metrics
    metrics = scheduler.get_scheduler_metrics()
    print(f"   Average task time: {metrics.average_execution_time:.3f}s")
    print(f"   Total scheduled: {metrics.tasks_scheduled}")
    
    # Generate schedule report
    report = scheduler.generate_schedule_report()
    print(f"   Success rate: {report['metrics']['success_rate']:.1%}")


async def demo_mpc_integration():
    """Demo full MPC + Quantum Planning integration"""
    print("\n🔐 MPC + QUANTUM INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize security config
    security_config = SecurityConfig(
        protocol="3pc",
        security_level=128,
        gpu_acceleration=True
    )
    
    # Initialize integrator
    integrator = QuantumMPCIntegrator(
        security_config=security_config,
        scheduler_config=SchedulerConfig(
            max_concurrent_tasks=6,
            quantum_optimization=True
        )
    )
    
    print("🔧 Initialized MPC-Quantum integrator")
    
    # Simulate transformer initialization
    print("🤖 Initializing secure transformer...")
    # In real implementation: transformer = integrator.initialize_transformer("bert-base-uncased")
    
    # Prepare test inputs
    test_inputs = [
        "Secure multi-party computation enables privacy-preserving machine learning",
        "Quantum-inspired optimization can improve task scheduling performance",
        "GPU acceleration makes homomorphic encryption practical for transformers",
        "The intersection of quantum computing and secure computation is fascinating"
    ]
    
    print(f"📝 Processing {len(test_inputs)} text inputs with quantum optimization")
    
    # Execute quantum-optimized inference
    start_time = time.time()
    
    try:
        # Note: This will use simulated results since we don't have a real model loaded
        result = await integrator.quantum_inference(
            text_inputs=test_inputs,
            priority=TaskPriority.HIGH,
            optimize_schedule=True
        )
        
        execution_time = time.time() - start_time
        
        print("✅ Inference completed successfully!")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Total execution time: {execution_time:.2f}s")
        print(f"   Quantum optimization time: {result['performance']['quantum_optimization_time']:.2f}s")
        print(f"   Tasks created: {result['performance']['tasks_created']}")
        print(f"   Tasks completed: {result['performance']['tasks_completed']}")
        print(f"   Batches executed: {result['performance']['batches_executed']}")
        
        # Show sample results
        print("\n📋 Sample Results:")
        for i, res in enumerate(result['results'][:2]):
            print(f"   Input {i+1}: {res['input'][:50]}...")
            print(f"   Output: {res['output']['final_output']}")
            print(f"   Confidence: {res['output']['confidence']}")
        
        # Performance summary
        perf_summary = integrator.get_performance_summary()
        print("\n📊 Performance Summary:")
        print(f"   Total workflows: {perf_summary['total_workflows']}")
        print(f"   Average execution time: {perf_summary['average_execution_time']:.3f}s")
        print(f"   Average success rate: {perf_summary['average_success_rate']:.1%}")
        print(f"   Throughput: {perf_summary['throughput']:.1f} inputs/second")
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        print("   (This is expected without a real transformer model)")
    
    finally:
        integrator.cleanup()


async def demo_workload_optimization():
    """Demo automatic workload optimization"""
    print("\n⚙️ WORKLOAD OPTIMIZATION DEMO")
    print("=" * 50)
    
    integrator = QuantumMPCIntegrator(
        security_config=SecurityConfig(),
        scheduler_config=SchedulerConfig()
    )
    
    # Define different workload scenarios
    workloads = [
        {
            "name": "Light Workload",
            "inputs_per_hour": 50,
            "avg_input_length": 100,
            "peak_concurrency": 2
        },
        {
            "name": "Medium Workload", 
            "inputs_per_hour": 200,
            "avg_input_length": 250,
            "peak_concurrency": 6
        },
        {
            "name": "Heavy Workload",
            "inputs_per_hour": 1000,
            "avg_input_length": 500,
            "peak_concurrency": 12
        }
    ]
    
    print("🎛️ Optimizing configurations for different workloads:")
    
    for workload in workloads:
        print(f"\n📈 {workload['name']}:")
        print(f"   {workload['inputs_per_hour']} inputs/hour, avg length {workload['avg_input_length']}")
        
        # Get optimized config
        optimal_config = integrator.optimize_for_workload(workload)
        
        print(f"   ⚡ Optimal concurrent tasks: {optimal_config.max_concurrent_tasks}")
        print(f"   ⏱️ Task timeout: {optimal_config.task_timeout}s")
        print(f"   🧠 Quantum optimization: {optimal_config.quantum_optimization}")
        print(f"   📊 Auto-scaling: {optimal_config.auto_scaling}")
        print(f"   💾 Memory limit: {optimal_config.resource_limits.get('memory', 'N/A')}")
    
    integrator.cleanup()


async def main():
    """Run all demonstrations"""
    print("🚀 QUANTUM-INSPIRED TASK PLANNER DEMONSTRATIONS")
    print("=" * 60)
    print("Showcasing quantum algorithms for MPC transformer optimization")
    
    try:
        await demo_basic_quantum_planning()
        await demo_quantum_scheduler()
        await demo_mpc_integration()
        await demo_workload_optimization()
        
        print("\n✨ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  🔬 Quantum superposition-based task prioritization")
        print("  ⚛️ Quantum annealing for optimal scheduling")
        print("  🎯 Intelligent workflow orchestration")
        print("  🔐 Integration with secure MPC transformers")
        print("  📊 Performance monitoring and optimization")
        print("  ⚙️ Adaptive workload configuration")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demonstrations interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstrations: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    asyncio.run(main())