"""
Integration Tests for Quantum-Inspired Task Planner

End-to-end integration tests for the complete quantum planning system
integrated with secure MPC transformer infrastructure.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil
import logging

from src.secure_mpc_transformer.config import SecurityConfig
from src.secure_mpc_transformer.integration import QuantumMPCIntegrator
from src.secure_mpc_transformer.planning import (
    QuantumTaskPlanner,
    QuantumScheduler,
    TaskPriority,
    QuantumPerformanceMonitor,
    MetricsCollector,
    create_quantum_cache_system
)
from src.secure_mpc_transformer.planning.scheduler import SchedulerConfig
from src.secure_mpc_transformer.planning.concurrent import ConcurrentQuantumExecutor
from src.secure_mpc_transformer.planning.quantum_planner import TaskType, TaskStatus


@pytest.fixture
def security_config():
    """Create test security configuration"""
    return SecurityConfig(
        protocol="3pc",
        security_level=128,
        gpu_acceleration=True
    )


@pytest.fixture
def scheduler_config():
    """Create test scheduler configuration"""
    return SchedulerConfig(
        max_concurrent_tasks=4,
        quantum_optimization=True,
        performance_monitoring=True,
        auto_scaling=False  # Disable for predictable testing
    )


@pytest.fixture
def integrator(security_config, scheduler_config):
    """Create test integrator"""
    return QuantumMPCIntegrator(
        security_config=security_config,
        scheduler_config=scheduler_config
    )


@pytest.fixture
def sample_text_inputs():
    """Create sample text inputs for testing"""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Secure multi-party computation enables privacy-preserving machine learning",
        "Quantum algorithms can provide computational advantages for optimization problems",
        "GPU acceleration makes homomorphic encryption practical for real-world applications"
    ]


class TestQuantumMPCIntegrator:
    """Test suite for QuantumMPCIntegrator"""
    
    def test_integrator_initialization(self, security_config, scheduler_config):
        """Test integrator initialization"""
        integrator = QuantumMPCIntegrator(
            security_config=security_config,
            scheduler_config=scheduler_config
        )
        
        assert integrator.security_config == security_config
        assert isinstance(integrator.scheduler, QuantumScheduler)
        assert integrator.transformer is None  # Not initialized yet
        assert integrator.inference_service is None
        assert isinstance(integrator.active_workflows, dict)
        assert isinstance(integrator.performance_history, list)
    
    @patch('src.secure_mpc_transformer.models.secure_transformer.SecureTransformer')
    @patch('src.secure_mpc_transformer.services.inference_service.InferenceService')
    def test_transformer_initialization(self, mock_inference_service, mock_transformer, integrator):
        """Test transformer initialization"""
        mock_transformer_instance = Mock()
        mock_transformer.from_pretrained.return_value = mock_transformer_instance
        
        mock_service_instance = Mock()
        mock_inference_service.return_value = mock_service_instance
        
        result = integrator.initialize_transformer("bert-base-uncased", some_param="test")
        
        assert result == mock_transformer_instance
        assert integrator.transformer == mock_transformer_instance
        assert integrator.inference_service == mock_service_instance
        
        # Verify calls
        mock_transformer.from_pretrained.assert_called_once_with(
            "bert-base-uncased",
            security_config=integrator.security_config,
            some_param="test"
        )
        mock_inference_service.assert_called_once_with(
            transformer=mock_transformer_instance,
            config=integrator.security_config
        )
    
    @pytest.mark.asyncio
    async def test_quantum_inference_no_transformer(self, integrator, sample_text_inputs):
        """Test quantum inference without initialized transformer"""
        with pytest.raises(RuntimeError, match="Transformer not initialized"):
            await integrator.quantum_inference(sample_text_inputs[:2])
    
    @pytest.mark.asyncio
    async def test_quantum_inference_with_mocked_transformer(self, integrator, sample_text_inputs):
        """Test quantum inference with mocked transformer"""
        # Mock transformer and inference service
        integrator.transformer = Mock()
        integrator.transformer.model_name = "test-model"
        integrator.inference_service = Mock()
        
        result = await integrator.quantum_inference(
            text_inputs=sample_text_inputs[:2],
            priority=TaskPriority.HIGH,
            optimize_schedule=True
        )
        
        assert isinstance(result, dict)
        assert "workflow_id" in result
        assert "results" in result
        assert "performance" in result
        assert "status" in result
        
        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        
        # Check results structure
        for res in result["results"]:
            assert "input" in res
            assert "output" in res
            assert "workflow_id" in res
            assert "tasks_completed" in res
        
        # Check performance metrics
        perf = result["performance"]
        assert "total_execution_time" in perf
        assert "quantum_optimization_time" in perf
        assert "tasks_created" in perf
        assert "scheduler_metrics" in perf


class TestEndToEndQuantumPlanning:
    """End-to-end tests for quantum planning system"""
    
    @pytest.fixture
    def complete_system(self, security_config):
        """Create complete quantum planning system"""
        scheduler_config = SchedulerConfig(
            max_concurrent_tasks=3,
            quantum_optimization=True,
            performance_monitoring=True,
            auto_scaling=False
        )
        
        integrator = QuantumMPCIntegrator(security_config, scheduler_config)
        
        # Create cache system
        cache_config = {
            "quantum_cache_size": 100,
            "quantum_cache_memory_mb": 50,
            "optimization_cache_size": 50
        }
        cache_system = create_quantum_cache_system(cache_config)
        
        # Create concurrent executor
        executor = ConcurrentQuantumExecutor(
            max_workers=3,
            auto_scaling=False
        )
        
        return {
            "integrator": integrator,
            "cache_system": cache_system,
            "executor": executor,
            "scheduler": integrator.scheduler
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, complete_system, sample_text_inputs):
        """Test complete workflow from input to output"""
        integrator = complete_system["integrator"]
        executor = complete_system["executor"]
        scheduler = complete_system["scheduler"]
        
        # Mock transformer
        integrator.transformer = Mock()
        integrator.transformer.model_name = "test-bert"
        integrator.inference_service = Mock()
        
        # Start executor
        await executor.start()
        
        try:
            # Create workflow tasks
            workflow_tasks = scheduler.create_inference_workflow(
                model_name="test-bert",
                input_data=sample_text_inputs[0],
                priority=TaskPriority.HIGH
            )
            
            assert len(workflow_tasks) > 0
            
            # Execute with scheduler
            scheduler_result = await scheduler.schedule_and_execute()
            
            assert scheduler_result["status"] == "completed"
            assert scheduler_result["tasks_processed"] > 0
            
            # Execute quantum inference
            inference_result = await integrator.quantum_inference(
                text_inputs=sample_text_inputs[:2],
                priority=TaskPriority.HIGH
            )
            
            assert inference_result["status"] == "completed"
            assert len(inference_result["results"]) == 2
            
        finally:
            await executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])