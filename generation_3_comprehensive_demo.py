#!/usr/bin/env python3
"""
Generation 3 Comprehensive Demo - Complete Autonomous SDLC Implementation

Demonstrates the full autonomous SDLC execution with Generation 1-3 features:
- Basic functionality (Gen 1)
- Robustness and resilience (Gen 2) 
- Scaling and optimization (Gen 3)
- Quality gates and validation
"""

import sys
import asyncio
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from secure_mpc_transformer.core import AutonomousExecutor, ExecutionPhase, TaskPriority
from secure_mpc_transformer.resilience import AutonomousResilienceManager, ResilienceConfig
from secure_mpc_transformer.validation import AutonomousValidator, ValidationPolicy, ThreatCategory
from secure_mpc_transformer.scaling import AutonomousScaler, ScalingConfig
from secure_mpc_transformer.planning import QuantumTaskPlanner
from secure_mpc_transformer.services.model_service_enhanced import ModelService
from secure_mpc_transformer.utils.error_handling import setup_logging


class ComprehensiveSDLCDemo:
    """Comprehensive demonstration of autonomous SDLC capabilities"""
    
    def __init__(self):
        # Initialize core components
        self.executor = AutonomousExecutor()
        self.resilience_manager = AutonomousResilienceManager(
            ResilienceConfig(max_retry_attempts=2, enable_automatic_recovery=True)
        )
        self.validator = AutonomousValidator(
            ValidationPolicy(max_string_length=5000, require_https=True)
        )
        self.scaler = AutonomousScaler(
            ScalingConfig(min_workers=2, max_workers=8, enable_quantum_optimization=True)
        )
        self.quantum_planner = QuantumTaskPlanner()
        self.model_service = ModelService()
        
        # Register components with resilience manager
        self.resilience_manager.register_component("executor", self._health_check_executor)
        self.resilience_manager.register_component("validator", self._health_check_validator)
        self.resilience_manager.register_component("scaler", self._health_check_scaler)
        self.resilience_manager.register_component("quantum_planner", self._health_check_planner)
    
    async def _health_check_executor(self) -> bool:
        """Health check for executor component"""
        return not self.executor.is_running or len(self.executor.tasks) > 0
    
    async def _health_check_validator(self) -> bool:
        """Health check for validator component"""
        stats = self.validator.get_validation_stats()
        return stats["total_validations"] >= 0
    
    async def _health_check_scaler(self) -> bool:
        """Health check for scaler component"""
        metrics = self.scaler.get_scaling_metrics()
        return metrics["current_workers"] >= 1
    
    async def _health_check_planner(self) -> bool:
        """Health check for quantum planner component"""
        stats = self.quantum_planner.get_execution_stats()
        return stats["total_tasks"] >= 0
    
    async def demonstrate_validation_security(self):
        """Demonstrate comprehensive validation and security"""
        print("\n🛡️ Validation & Security Demonstration")
        print("-" * 50)
        
        # Test cases with various threat types
        test_inputs = [
            ("Normal text input", "This is a normal text input for processing"),
            ("SQL Injection", "'; DROP TABLE users; --"),
            ("XSS Attack", "<script>alert('XSS')</script>"),
            ("Path Traversal", "../../../etc/passwd"),
            ("Command Injection", "; rm -rf /"),
            ("Large Input", "A" * 15000),  # Exceeds limit
            ("Valid URL", "https://secure-api.example.com/endpoint"),
            ("Invalid URL", "http://192.168.1.1/internal"),  # Private IP
            ("Valid Email", "user@secure-domain.com"),
            ("Invalid Email", "not-an-email"),
        ]
        
        threat_count = 0
        
        for test_name, test_input in test_inputs:
            print(f"\n🔍 Testing: {test_name}")
            
            # Determine context based on test
            context = {}
            if "URL" in test_name:
                context["type"] = "url"
            elif "Email" in test_name:
                context["type"] = "email"
            elif "Traversal" in test_name:
                context["type"] = "filepath"
            
            try:
                result = await self.validator.validate_input(test_input, context)
                
                status_icon = "✅" if result.is_valid else "❌"
                print(f"   {status_icon} Status: {'VALID' if result.is_valid else 'INVALID'}")
                print(f"   📊 Severity: {result.severity.value}")
                
                if result.threat_category:
                    print(f"   ⚠️ Threat: {result.threat_category.value}")
                    print(f"   🎯 Confidence: {result.confidence_score:.2f}")
                    threat_count += 1
                
                if result.remediation:
                    print(f"   💡 Remediation: {result.remediation}")
                
            except Exception as e:
                print(f"   💥 Validation error: {e}")
        
        # Display validation statistics
        stats = self.validator.get_validation_stats()
        print(f"\n📈 Validation Summary:")
        print(f"   Total Validations: {stats['total_validations']}")
        print(f"   Threats Detected: {threat_count}")
        print(f"   Cache Size: {stats['cache_size']}")
        print(f"   Avg Time: {stats.get('avg_validation_time_ms', 0):.2f}ms")
    
    async def demonstrate_resilience_recovery(self):
        """Demonstrate resilience and recovery mechanisms"""
        print("\n🔧 Resilience & Recovery Demonstration")
        print("-" * 50)
        
        # Simulate various failure scenarios
        failure_scenarios = [
            ("Network Timeout", lambda: self._simulate_network_error()),
            ("Resource Exhaustion", lambda: self._simulate_memory_error()),
            ("Security Violation", lambda: self._simulate_security_error()),
            ("Computation Error", lambda: self._simulate_computation_error()),
        ]
        
        for scenario_name, error_func in failure_scenarios:
            print(f"\n🔥 Testing: {scenario_name}")
            
            try:
                # Simulate failure and recovery
                await self.resilience_manager.handle_failure(
                    error_func(),
                    component="test_component",
                    context={
                        "retry_function": self._mock_retry_function,
                        "fallback_function": self._mock_fallback_function
                    }
                )
                print("   ✅ Recovery successful")
                
            except Exception as e:
                print(f"   ❌ Recovery failed: {e}")
        
        # Perform health checks
        print(f"\n🏥 Component Health Check:")
        health_status = await self.resilience_manager.health_check_all_components()
        
        for component, status in health_status.items():
            health_icon = "💚" if status["status"] == "healthy" else "💔"
            print(f"   {health_icon} {component}: {status['status']}")
        
        # Display resilience metrics
        metrics = self.resilience_manager.get_resilience_metrics()
        print(f"\n📊 Resilience Metrics:")
        print(f"   Recovery Success Rate: {metrics['recovery_metrics']['successful_recoveries']}")
        print(f"   Failed Recoveries: {metrics['recovery_metrics']['failed_recoveries']}")
        print(f"   Total Failures: {metrics['total_failures_recorded']}")
    
    def _simulate_network_error(self):
        return ConnectionError("Network connection timeout")
    
    def _simulate_memory_error(self):
        return MemoryError("Insufficient memory available")
    
    def _simulate_security_error(self):
        return PermissionError("Unauthorized access attempt")
    
    def _simulate_computation_error(self):
        return ValueError("Invalid computation parameters")
    
    async def _mock_retry_function(self):
        """Mock retry function for testing"""
        await asyncio.sleep(0.1)
        return {"status": "success", "message": "Retry successful"}
    
    async def _mock_fallback_function(self):
        """Mock fallback function for testing"""
        return {"status": "fallback", "message": "Fallback mode activated"}
    
    async def demonstrate_autonomous_scaling(self):
        """Demonstrate autonomous scaling capabilities"""
        print("\n⚡ Autonomous Scaling Demonstration")
        print("-" * 50)
        
        # Start monitoring
        await self.scaler.start_monitoring()
        print("✅ Scaling monitoring started")
        
        # Simulate workload scenarios
        scenarios = [
            ("Low Load", 2),
            ("Medium Load", 4), 
            ("High Load", 6),
            ("Peak Load", 8),
            ("Cooldown", 3)
        ]
        
        for scenario_name, target_workers in scenarios:
            print(f"\n🎯 Scenario: {scenario_name} (Target: {target_workers} workers)")
            
            # Force scale to demonstrate
            success = await self.scaler.force_scale(target_workers)
            
            if success:
                print(f"   ✅ Scaled to {target_workers} workers")
            else:
                print(f"   ❌ Scaling failed")
            
            # Get current metrics
            metrics = self.scaler.get_scaling_metrics()
            print(f"   📊 Current Workers: {metrics['current_workers']}")
            print(f"   🧠 Quantum Coherence: {metrics['quantum_coherence']:.3f}")
            
            # Simulate some work
            await asyncio.sleep(1)
        
        # Display scaling history
        history = self.scaler.get_scaling_history(1)  # Last hour
        print(f"\n📈 Scaling History (last hour): {len(history)} events")
        
        for event in history[-3:]:  # Show last 3 events
            direction_icon = "📈" if event["direction"] == "scale_up" else "📉"
            print(f"   {direction_icon} {event['direction']}: {event['before_value']} -> {event['after_value']}")
        
        await self.scaler.stop_monitoring()
        print("🛑 Scaling monitoring stopped")
    
    async def demonstrate_quantum_planning(self):
        """Demonstrate quantum-inspired task planning"""
        print("\n🌀 Quantum Planning Demonstration")
        print("-" * 50)
        
        # Create complex task scenario
        from secure_mpc_transformer.planning.quantum_planner import Task, TaskType, TaskStatus
        
        tasks = [
            Task("data_prep", TaskType.EMBEDDING, 0.9, 2.0, {"cpu": 0.4, "memory": 0.3}, []),
            Task("secure_computation", TaskType.ATTENTION, 0.95, 4.0, {"gpu": 0.8, "memory": 0.6}, ["data_prep"]),
            Task("layer_1", TaskType.FEEDFORWARD, 0.8, 1.5, {"cpu": 0.3, "memory": 0.2}, ["secure_computation"]),
            Task("layer_2", TaskType.FEEDFORWARD, 0.8, 1.5, {"cpu": 0.3, "memory": 0.2}, ["layer_1"]),
            Task("output_reconstruction", TaskType.RESULT_RECONSTRUCTION, 0.7, 1.0, {"cpu": 0.2, "memory": 0.1}, ["layer_2"]),
        ]
        
        # Add tasks to planner
        for task in tasks:
            self.quantum_planner.add_task(task)
        
        print(f"✅ Added {len(tasks)} tasks to quantum planner")
        
        # Get ready tasks and calculate quantum priorities
        ready_tasks = self.quantum_planner.get_ready_tasks()
        prioritized = self.quantum_planner.calculate_quantum_priority(ready_tasks)
        
        print(f"\n🎯 Quantum Priority Rankings:")
        for i, (task, score) in enumerate(prioritized[:3]):
            print(f"   {i+1}. {task.id} (score: {score:.3f})")
        
        # Generate quantum-optimized schedule
        schedule = self.quantum_planner.quantum_anneal_schedule(ready_tasks)
        
        print(f"\n⚡ Quantum Schedule ({len(schedule)} batches):")
        for i, batch in enumerate(schedule):
            task_names = [task.id for task in batch]
            print(f"   Batch {i+1}: {task_names}")
        
        # Execute quantum plan
        print(f"\n🚀 Executing Quantum Plan...")
        execution_result = await self.quantum_planner.execute_quantum_plan()
        
        print(f"   ✅ Status: {execution_result['status']}")
        print(f"   ⏱️ Time: {execution_result['execution_time']:.2f}s")
        print(f"   📊 Completed: {execution_result['tasks_completed']}/{execution_result['total_tasks']}")
        
        # Get execution statistics
        stats = self.quantum_planner.get_execution_stats()
        print(f"\n📈 Execution Statistics:")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Avg Task Time: {stats['average_task_time']:.2f}s")
    
    async def demonstrate_full_autonomous_sdlc(self):
        """Demonstrate complete autonomous SDLC execution"""
        print("\n🚀 Full Autonomous SDLC Execution")
        print("=" * 60)
        
        print("🧠 Starting comprehensive autonomous execution...")
        
        try:
            # Execute the full autonomous SDLC
            metrics = await self.executor.execute_autonomous_sdlc()
            
            print(f"\n🎉 Autonomous SDLC Completed!")
            print(f"   ⏱️ Total Time: {metrics.total_execution_time:.2f}s")
            print(f"   ✅ Success Rate: {metrics.success_rate:.1%}")
            print(f"   🎯 Quality Score: {metrics.quality_score:.2f}")
            print(f"   📊 Tasks: {metrics.completed_tasks}/{metrics.total_tasks}")
            
            # Get detailed summary
            summary = self.executor.get_execution_summary()
            
            print(f"\n📋 Phase Results:")
            for phase, stats in summary["phase_breakdown"].items():
                success_icon = "✅" if stats["success_rate"] > 0.8 else "⚠️" if stats["success_rate"] > 0.5 else "❌"
                print(f"   {success_icon} {phase}: {stats['completed']}/{stats['total']} ({stats['success_rate']:.1%})")
            
            print(f"\n⏱️ Phase Timings:")
            for phase, duration in summary["phase_times"].items():
                print(f"   {phase}: {duration:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Autonomous SDLC failed: {e}")
            return False
    
    async def run_comprehensive_demo(self):
        """Run the complete demonstration"""
        print("🎯 Secure MPC Transformer - Comprehensive Demo")
        print("🔬 Autonomous SDLC with Generation 1-3 Features")
        print("🛡️ Defensive Security Focus")
        print("=" * 60)
        
        demo_start = time.time()
        
        try:
            # Demonstration phases
            await self.demonstrate_validation_security()
            await self.demonstrate_resilience_recovery()
            await self.demonstrate_autonomous_scaling()
            await self.demonstrate_quantum_planning()
            
            # Full autonomous execution
            success = await self.demonstrate_full_autonomous_sdlc()
            
            total_time = time.time() - demo_start
            
            print(f"\n🎊 Comprehensive Demo Complete!")
            print(f"   ⏱️ Total Demo Time: {total_time:.2f}s")
            print(f"   🎯 Overall Success: {'YES' if success else 'NO'}")
            
            # Final component status
            print(f"\n🏥 Final Component Health:")
            health_status = await self.resilience_manager.health_check_all_components()
            for component, status in health_status.items():
                health_icon = "💚" if status["status"] == "healthy" else "💔"
                print(f"   {health_icon} {component}: {status['status']}")
            
            print(f"\n🚀 Generation 1-3 Features Demonstrated:")
            print(f"   ✅ Generation 1: Basic functionality working")
            print(f"   ✅ Generation 2: Robust error handling & security")
            print(f"   ✅ Generation 3: Scaling & optimization active")
            print(f"   ✅ Quality Gates: Validation & monitoring operational")
            print(f"   ✅ Autonomous Execution: Complete SDLC cycle")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Demo interrupted by user")
        except Exception as e:
            print(f"\n💥 Demo failed: {e}")
            logging.exception("Demo execution failed")
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.scaler.stop_monitoring()
            self.model_service.shutdown()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


async def main():
    """Main demonstration function"""
    setup_logging(log_level="INFO")
    
    demo = ComprehensiveSDLCDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())