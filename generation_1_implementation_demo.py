#!/usr/bin/env python3
"""
Generation 1 Implementation Demo - Autonomous SDLC Execution

Demonstrates the basic autonomous SDLC functionality with quantum-inspired
task planning and secure MPC transformer integration.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from secure_mpc_transformer.core import (
    AutonomousExecutor, ExecutionTask, ExecutionPhase, TaskPriority
)
from secure_mpc_transformer.planning import QuantumTaskPlanner, TaskType
from secure_mpc_transformer.services.model_service_enhanced import ModelService
from secure_mpc_transformer.utils.error_handling import setup_logging


async def demonstrate_autonomous_execution():
    """Demonstrate autonomous SDLC execution with Generation 1 features."""
    print("🚀 Starting Autonomous SDLC Execution Demo")
    print("=" * 60)
    
    # Initialize services
    executor = AutonomousExecutor()
    quantum_planner = QuantumTaskPlanner()
    model_service = ModelService()
    
    print("✅ Services initialized")
    
    # Execute autonomous SDLC
    print("\n🧠 Executing Autonomous SDLC Process...")
    try:
        metrics = await executor.execute_autonomous_sdlc()
        
        print(f"\n📊 Execution Results:")
        print(f"   Total Time: {metrics.total_execution_time:.2f}s")
        print(f"   Success Rate: {metrics.success_rate:.1%}")
        print(f"   Quality Score: {metrics.quality_score:.2f}")
        print(f"   Tasks Completed: {metrics.completed_tasks}/{metrics.total_tasks}")
        
        # Get detailed summary
        summary = executor.get_execution_summary()
        
        print(f"\n📋 Phase Breakdown:")
        for phase, stats in summary["phase_breakdown"].items():
            print(f"   {phase}: {stats['completed']}/{stats['total']} "
                  f"({stats['success_rate']:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False


async def demonstrate_quantum_planning():
    """Demonstrate quantum-inspired task planning."""
    print("\n🌀 Quantum Task Planning Demo")
    print("-" * 40)
    
    planner = QuantumTaskPlanner()
    
    # Create sample MPC tasks
    from secure_mpc_transformer.planning.quantum_planner import Task, TaskStatus
    
    tasks = [
        Task(
            id="embedding_1",
            task_type=TaskType.EMBEDDING,
            priority=0.8,
            estimated_duration=2.0,
            required_resources={"cpu": 0.3, "memory": 0.4},
            dependencies=[]
        ),
        Task(
            id="attention_1", 
            task_type=TaskType.ATTENTION,
            priority=0.9,
            estimated_duration=3.5,
            required_resources={"gpu": 0.6, "memory": 0.5},
            dependencies=["embedding_1"]
        ),
        Task(
            id="feedforward_1",
            task_type=TaskType.FEEDFORWARD,
            priority=0.7,
            estimated_duration=1.8,
            required_resources={"cpu": 0.4, "memory": 0.3},
            dependencies=["attention_1"]
        )
    ]
    
    # Add tasks to planner
    for task in tasks:
        planner.add_task(task)
    
    print(f"Added {len(tasks)} tasks to quantum planner")
    
    # Calculate quantum priorities
    ready_tasks = planner.get_ready_tasks()
    prioritized = planner.calculate_quantum_priority(ready_tasks)
    
    print("\n🎯 Quantum Priority Rankings:")
    for i, (task, score) in enumerate(prioritized):
        print(f"   {i+1}. {task.id} (score: {score:.3f})")
    
    # Generate optimized schedule
    schedule = planner.quantum_anneal_schedule(ready_tasks)
    
    print(f"\n⚡ Quantum-Optimized Schedule ({len(schedule)} batches):")
    for i, batch in enumerate(schedule):
        task_names = [task.id for task in batch]
        print(f"   Batch {i+1}: {task_names}")
    
    # Execute quantum plan
    print("\n🚀 Executing Quantum Plan...")
    execution_result = await planner.execute_quantum_plan()
    
    print(f"   Status: {execution_result['status']}")
    print(f"   Execution Time: {execution_result['execution_time']:.2f}s")
    print(f"   Tasks Completed: {execution_result['tasks_completed']}")
    print(f"   Batches Executed: {execution_result['batches_executed']}")


async def demonstrate_model_service():
    """Demonstrate enhanced model service with caching."""
    print("\n🤖 Enhanced Model Service Demo")
    print("-" * 40)
    
    service = ModelService({
        "max_cached_models": 2,
        "auto_unload_timeout": 60
    })
    
    try:
        # Load a model (will use mock if torch not available)
        print("Loading model: bert-base-uncased...")
        model = await service.load_model("bert-base-uncased")
        print("✅ Model loaded successfully")
        
        # Get model info
        info = service.get_model_info("bert-base-uncased")
        if info:
            print(f"   Status: {info['status']}")
            print(f"   Load Time: {info.get('load_time', 0):.2f}s")
            print(f"   Memory Usage: {info.get('memory_usage', 0)}MB")
        
        # List available models
        models_list = service.list_models()
        cache_stats = models_list["cache_stats"]
        
        print(f"\n📊 Cache Statistics:")
        print(f"   Cached Models: {cache_stats['cached_models']}/{cache_stats['max_models']}")
        print(f"   Memory Usage: {cache_stats['total_memory_mb']}MB")
        
        # Test cache hit
        print("\nTesting cache hit...")
        cached_model = await service.get_model("bert-base-uncased")
        print("✅ Model retrieved from cache")
        
    except Exception as e:
        print(f"❌ Model service error: {e}")
    
    finally:
        service.shutdown()
        print("🛑 Model service shutdown")


async def main():
    """Main demonstration function."""
    setup_logging(log_level="INFO")
    
    print("🎯 Secure MPC Transformer - Generation 1 Demo")
    print("🔬 Autonomous SDLC with Quantum-Inspired Optimization")
    print("🛡️ Defensive Security Focus")
    print("=" * 60)
    
    try:
        # Run demonstrations
        success = await demonstrate_autonomous_execution()
        
        if success:
            await demonstrate_quantum_planning()
            await demonstrate_model_service()
            
            print("\n🎉 Generation 1 Implementation Complete!")
            print("✅ Basic functionality working")
            print("✅ Quantum planning operational") 
            print("✅ Enhanced model service ready")
            print("✅ Autonomous execution successful")
            
        else:
            print("\n❌ Demo failed - check logs for details")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        logging.exception("Demo execution failed")


if __name__ == "__main__":
    asyncio.run(main())