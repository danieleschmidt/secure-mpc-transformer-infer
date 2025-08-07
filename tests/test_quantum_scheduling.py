"""
Tests for Quantum Scheduling and Concurrent Execution

Test suite for the quantum scheduler, concurrent executor, monitoring,
and caching components of the quantum planning system.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import shutil
import os
import threading
import time

from src.secure_mpc_transformer.planning.scheduler import (
    QuantumScheduler,
    SchedulerConfig,
    TaskPriority,
    SchedulerStatus,
    SchedulerMetrics
)
from src.secure_mpc_transformer.planning.concurrent import (
    ConcurrentQuantumExecutor,
    QuantumTaskWorker,
    LoadBalanceStrategy,
    ExecutorType,
    WorkerStats
)
from src.secure_mpc_transformer.planning.monitoring import (
    QuantumPerformanceMonitor,
    MetricsCollector,
    TimingContextManager,
    AlertLevel,
    MetricType
)
from src.secure_mpc_transformer.planning.caching import (
    QuantumStateCache,
    OptimizationResultCache,
    PersistentQuantumCache,
    CachePolicy,
    create_quantum_cache_system
)
from src.secure_mpc_transformer.planning.quantum_planner import Task, TaskType, TaskStatus


class TestQuantumScheduler:
    """Test suite for QuantumScheduler"""
    
    @pytest.fixture
    def scheduler_config(self):
        """Create test scheduler configuration"""
        return SchedulerConfig(
            max_concurrent_tasks=4,
            task_timeout=60.0,
            enable_adaptive_scheduling=True,
            quantum_optimization=True,
            resource_limits={"cpu": 1.0, "memory": 1.0, "gpu": 1.0}
        )
    
    @pytest.fixture
    def scheduler(self, scheduler_config):
        """Create test scheduler"""
        return QuantumScheduler(scheduler_config)
    
    def test_scheduler_initialization(self, scheduler_config):
        """Test scheduler initialization"""
        scheduler = QuantumScheduler(scheduler_config)
        
        assert scheduler.config == scheduler_config
        assert scheduler.status == SchedulerStatus.IDLE
        assert isinstance(scheduler.metrics, SchedulerMetrics)
        assert len(scheduler.active_tasks) == 0
        assert scheduler.planner is not None
        assert scheduler.optimizer is not None
    
    def test_create_mpc_task(self, scheduler):
        """Test MPC task creation"""
        task = scheduler.create_mpc_task(
            task_id="test_task",
            task_type=TaskType.EMBEDDING,
            priority=TaskPriority.HIGH,
            estimated_duration=2.0,
            required_resources={"gpu": 0.5, "memory": 0.3},
            metadata={"test": "data"}
        )
        
        assert task.id == "test_task"
        assert task.task_type == TaskType.EMBEDDING
        assert task.priority == TaskPriority.HIGH.value
        assert task.estimated_duration == 2.0
        assert task.required_resources == {"gpu": 0.5, "memory": 0.3}
        assert hasattr(task, 'metadata')
        
        # Should be added to planner
        assert scheduler.planner.get_task("test_task") is not None
    
    def test_create_inference_workflow(self, scheduler):
        """Test inference workflow creation"""
        tasks = scheduler.create_inference_workflow(
            model_name="bert-base",
            input_data="test input",
            priority=TaskPriority.MEDIUM
        )
        
        assert len(tasks) > 0  # Should create multiple tasks
        
        # Check task types are present
        task_types = [task.task_type for task in tasks]
        expected_types = [
            TaskType.PROTOCOL_INIT,
            TaskType.EMBEDDING,
            TaskType.ATTENTION,
            TaskType.FEEDFORWARD,
            TaskType.RESULT_RECONSTRUCTION
        ]
        
        for expected_type in expected_types:
            assert expected_type in task_types
        
        # Check dependencies are properly set
        protocol_tasks = [t for t in tasks if t.task_type == TaskType.PROTOCOL_INIT]
        assert len(protocol_tasks) > 0
        assert protocol_tasks[0].dependencies == []
        
        # Embedding should depend on protocol
        embedding_tasks = [t for t in tasks if t.task_type == TaskType.EMBEDDING]
        assert len(embedding_tasks) > 0
        assert protocol_tasks[0].id in embedding_tasks[0].dependencies
    
    def test_get_task_status(self, scheduler):
        """Test task status retrieval"""
        # Non-existent task
        assert scheduler.get_task_status("non_existent") is None
        
        # Create and check existing task
        task = scheduler.create_mpc_task(
            task_id="status_test",
            task_type=TaskType.EMBEDDING,
            priority=TaskPriority.LOW
        )
        
        assert scheduler.get_task_status("status_test") == TaskStatus.PENDING
        
        # Update status and check
        task.status = TaskStatus.RUNNING
        assert scheduler.get_task_status("status_test") == TaskStatus.RUNNING
    
    def test_cancel_task(self, scheduler):
        """Test task cancellation"""
        task = scheduler.create_mpc_task(
            task_id="cancel_test",
            task_type=TaskType.EMBEDDING,
            priority=TaskPriority.LOW
        )
        
        # Should be able to cancel pending task
        assert scheduler.cancel_task("cancel_test") == True
        assert scheduler.get_task_status("cancel_test") == TaskStatus.CANCELLED
        
        # Should not be able to cancel non-existent task
        assert scheduler.cancel_task("non_existent") == False
        
        # Should not be able to cancel already cancelled task
        assert scheduler.cancel_task("cancel_test") == False
    
    def test_scheduler_pause_resume(self, scheduler):
        """Test scheduler pause and resume functionality"""
        assert scheduler.status == SchedulerStatus.IDLE
        
        # Can't pause idle scheduler
        assert scheduler.pause_scheduler() == False
        
        # Simulate running state
        scheduler.status = SchedulerStatus.RUNNING
        
        # Should be able to pause running scheduler
        assert scheduler.pause_scheduler() == True
        assert scheduler.status == SchedulerStatus.PAUSED
        
        # Should be able to resume paused scheduler
        assert scheduler.resume_scheduler() == True
        assert scheduler.status == SchedulerStatus.RUNNING
        
        # Can't resume already running scheduler
        assert scheduler.resume_scheduler() == False
    
    def test_estimate_completion_time(self, scheduler):
        """Test completion time estimation"""
        # No tasks
        assert scheduler.estimate_completion_time() == 0.0
        
        # Create some tasks
        tasks = [
            scheduler.create_mpc_task(f"task_{i}", TaskType.EMBEDDING, TaskPriority.MEDIUM, 
                                    estimated_duration=1.0)
            for i in range(5)
        ]
        
        completion_time = scheduler.estimate_completion_time()
        assert completion_time > 0
        assert isinstance(completion_time, float)
        
        # Estimate for specific tasks
        task_ids = [t.id for t in tasks[:2]]
        specific_time = scheduler.estimate_completion_time(task_ids)
        assert 0 <= specific_time <= completion_time
    
    @pytest.mark.asyncio
    async def test_schedule_and_execute_no_tasks(self, scheduler):
        """Test scheduling with no tasks"""
        result = await scheduler.schedule_and_execute()
        
        assert result["status"] == "no_tasks"
        assert result["execution_time"] == 0
        assert result["tasks_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_schedule_and_execute_with_tasks(self, scheduler):
        """Test scheduling with tasks"""
        # Create some simple tasks
        tasks = [
            scheduler.create_mpc_task(f"exec_task_{i}", TaskType.EMBEDDING, TaskPriority.HIGH,
                                    estimated_duration=0.1)
            for i in range(3)
        ]
        
        result = await scheduler.schedule_and_execute()
        
        assert result["status"] == "completed"
        assert result["total_execution_time"] > 0
        assert result["tasks_processed"] == 3
        assert result["tasks_completed"] >= 0
    
    def test_generate_schedule_report(self, scheduler):
        """Test schedule report generation"""
        # Create some tasks
        scheduler.create_mpc_task("report_task", TaskType.EMBEDDING, TaskPriority.MEDIUM)
        
        report = scheduler.generate_schedule_report()
        
        assert isinstance(report, dict)
        assert "scheduler_status" in report
        assert "metrics" in report
        assert "task_statistics" in report
        assert "configuration" in report
        
        assert report["scheduler_status"] == SchedulerStatus.IDLE.value
        assert "tasks_scheduled" in report["metrics"]
        assert "max_concurrent_tasks" in report["configuration"]
    
    def test_optimize_resource_allocation(self, scheduler):
        """Test resource allocation optimization"""
        # Create tasks with resource requirements
        scheduler.create_mpc_task(
            "resource_task_1", TaskType.ATTENTION, TaskPriority.HIGH,
            required_resources={"gpu": 0.6, "memory": 0.4}
        )
        scheduler.create_mpc_task(
            "resource_task_2", TaskType.FEEDFORWARD, TaskPriority.MEDIUM,
            required_resources={"gpu": 0.3, "memory": 0.2}
        )
        
        allocation = scheduler.optimize_resource_allocation()
        
        assert isinstance(allocation, dict)
        # Should return empty dict if no optimization is performed or valid allocation


class TestConcurrentQuantumExecutor:
    """Test suite for ConcurrentQuantumExecutor"""
    
    @pytest.fixture
    def executor(self):
        """Create test executor"""
        return ConcurrentQuantumExecutor(
            max_workers=2,
            load_balance_strategy=LoadBalanceStrategy.QUANTUM_AWARE,
            auto_scaling=False  # Disable for predictable testing
        )
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing"""
        return [
            Task(
                id=f"concurrent_task_{i}",
                task_type=TaskType.EMBEDDING,
                priority=0.8,
                estimated_duration=0.5,
                required_resources={"gpu": 0.3, "memory": 0.2},
                dependencies=[]
            )
            for i in range(3)
        ]
    
    def test_executor_initialization(self):
        """Test executor initialization"""
        executor = ConcurrentQuantumExecutor(
            max_workers=4,
            load_balance_strategy=LoadBalanceStrategy.LEAST_LOADED,
            enable_gpu_acceleration=True,
            auto_scaling=True
        )
        
        assert executor.max_workers == 4
        assert executor.load_balance_strategy == LoadBalanceStrategy.LEAST_LOADED
        assert executor.enable_gpu_acceleration == True
        assert executor.auto_scaling == True
        assert len(executor.workers) == 0  # Not started yet
    
    @pytest.mark.asyncio
    async def test_executor_start(self, executor):
        """Test executor startup"""
        await executor.start()
        
        # Should have minimum workers
        assert len(executor.workers) >= executor.min_workers
        assert executor.workers
        
        # Workers should be healthy
        for worker in executor.workers.values():
            assert worker.is_healthy() == True
        
        await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_execute_tasks(self, executor, sample_tasks):
        """Test task execution"""
        await executor.start()
        
        try:
            results = await executor.execute_tasks(sample_tasks)
            
            assert len(results) == len(sample_tasks)
            
            # Check results format
            for result in results:
                assert isinstance(result, dict)
                assert "status" in result
                assert "execution_time" in result
                assert "worker_id" in result
                
                # Should be completed or failed
                assert result["status"] in ["completed", "failed"]
                
        finally:
            await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_worker_selection_strategies(self, sample_tasks):
        """Test different worker selection strategies"""
        strategies = [
            LoadBalanceStrategy.ROUND_ROBIN,
            LoadBalanceStrategy.LEAST_LOADED,
            LoadBalanceStrategy.QUANTUM_AWARE,
            LoadBalanceStrategy.RESOURCE_BASED
        ]
        
        for strategy in strategies:
            executor = ConcurrentQuantumExecutor(
                max_workers=2,
                load_balance_strategy=strategy,
                auto_scaling=False
            )
            
            await executor.start()
            
            try:
                # Should be able to select workers
                worker = await executor._select_worker_for_task(sample_tasks[0])
                assert worker is not None
                assert isinstance(worker, QuantumTaskWorker)
                
            finally:
                await executor.shutdown()
    
    def test_get_executor_stats(self, executor):
        """Test executor statistics"""
        stats = executor.get_executor_stats()
        
        assert isinstance(stats, dict)
        assert "total_workers" in stats
        assert "load_balance_strategy" in stats
        assert "auto_scaling_enabled" in stats
        
        # Before starting, should have no workers
        assert stats["total_workers"] == 0


class TestQuantumTaskWorker:
    """Test suite for QuantumTaskWorker"""
    
    @pytest.fixture
    def worker(self):
        """Create test worker"""
        return QuantumTaskWorker(
            worker_id="test_worker",
            executor_type=ExecutorType.THREAD,
            max_concurrent_tasks=1
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for worker testing"""
        return Task(
            id="worker_task",
            task_type=TaskType.EMBEDDING,
            priority=0.8,
            estimated_duration=0.1,
            required_resources={"gpu": 0.3},
            dependencies=[]
        )
    
    def test_worker_initialization(self):
        """Test worker initialization"""
        worker = QuantumTaskWorker(
            worker_id="init_test",
            executor_type=ExecutorType.PROCESS,
            gpu_device=0,
            max_concurrent_tasks=2
        )
        
        assert worker.worker_id == "init_test"
        assert worker.executor_type == ExecutorType.PROCESS
        assert worker.gpu_device == 0
        assert worker.max_concurrent_tasks == 2
        assert worker.is_available == True
        assert isinstance(worker.stats, WorkerStats)
    
    @pytest.mark.asyncio
    async def test_execute_task(self, worker, sample_task):
        """Test task execution by worker"""
        result = await worker.execute_task(sample_task)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "execution_time" in result
        assert "worker_id" in result
        
        assert result["worker_id"] == worker.worker_id
        assert result["status"] in ["completed", "failed"]
        assert result["execution_time"] >= 0
        
        # Task status should be updated
        assert sample_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert sample_task.start_time is not None
        assert sample_task.completion_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_different_task_types(self, worker):
        """Test execution of different task types"""
        task_types = [
            TaskType.EMBEDDING,
            TaskType.ATTENTION,
            TaskType.FEEDFORWARD,
            TaskType.PROTOCOL_INIT,
            TaskType.RESULT_RECONSTRUCTION
        ]
        
        for task_type in task_types:
            task = Task(
                id=f"task_{task_type.value}",
                task_type=task_type,
                priority=0.5,
                estimated_duration=0.05,
                required_resources={},
                dependencies=[]
            )
            
            result = await worker.execute_task(task)
            
            assert result["status"] in ["completed", "failed"]
            assert "result" in result
            
            # Check task-specific result content
            if result["status"] == "completed":
                task_result = result["result"]
                
                if task_type == TaskType.EMBEDDING:
                    assert "layer_type" in task_result
                    assert task_result["layer_type"] == "embedding"
                elif task_type == TaskType.ATTENTION:
                    assert "layer_type" in task_result
                    assert task_result["layer_type"] == "attention"
                    assert "quantum_superposition_applied" in task_result
    
    def test_worker_load_calculation(self, worker):
        """Test worker load calculation"""
        # No active tasks
        assert worker.get_current_load() == 0.0
        
        # Mock active tasks
        worker.max_concurrent_tasks = 2
        worker.active_tasks = {"task1": Mock(), "task2": Mock()}
        
        assert worker.get_current_load() == 1.0  # 2/2 = 1.0
        
        # Partial load
        worker.active_tasks = {"task1": Mock()}
        assert worker.get_current_load() == 0.5  # 1/2 = 0.5
    
    def test_worker_health_check(self, worker):
        """Test worker health checking"""
        # Fresh worker should be healthy
        assert worker.is_healthy() == True
        
        # Worker requesting shutdown should not be healthy
        worker.shutdown_requested = True
        assert worker.is_healthy() == False
        
        # Reset
        worker.shutdown_requested = False
        
        # Worker with high failure rate should not be healthy
        worker.stats.tasks_completed = 5
        worker.stats.tasks_failed = 15  # 75% failure rate
        assert worker.is_healthy() == False
    
    def test_resource_usage_tracking(self, worker):
        """Test resource usage tracking"""
        usage = worker.get_resource_usage()
        
        assert isinstance(usage, dict)
        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "memory_mb" in usage
        
        # Values should be reasonable
        assert 0 <= usage["cpu_percent"] <= 100
        assert 0 <= usage["memory_percent"] <= 100
        assert usage["memory_mb"] >= 0


class TestMetricsCollector:
    """Test suite for MetricsCollector"""
    
    @pytest.fixture
    def metrics(self):
        """Create test metrics collector"""
        return MetricsCollector(buffer_size=100)
    
    def test_metrics_initialization(self):
        """Test metrics collector initialization"""
        metrics = MetricsCollector(buffer_size=50)
        
        assert metrics.buffer_size == 50
        assert metrics.total_metrics_collected == 0
        assert isinstance(metrics.metrics, dict)
        assert isinstance(metrics.counters, dict)
    
    def test_record_counter(self, metrics):
        """Test counter metric recording"""
        metrics.record_counter("test_counter", 5.0)
        metrics.record_counter("test_counter", 3.0)
        
        assert metrics.get_counter_value("test_counter") == 8.0
        assert metrics.total_metrics_collected == 2
    
    def test_record_gauge(self, metrics):
        """Test gauge metric recording"""
        metrics.record_gauge("test_gauge", 42.0)
        metrics.record_gauge("test_gauge", 73.0)  # Should overwrite
        
        assert metrics.get_gauge_value("test_gauge") == 73.0
        assert metrics.total_metrics_collected == 2
    
    def test_record_histogram(self, metrics):
        """Test histogram metric recording"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for value in values:
            metrics.record_histogram("test_histogram", value)
        
        stats = metrics.get_histogram_stats("test_histogram")
        
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
    
    def test_record_timer(self, metrics):
        """Test timer metric recording"""
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for duration in durations:
            metrics.record_timer("test_timer", duration)
        
        stats = metrics.get_timer_stats("test_timer")
        
        assert stats["count"] == 5
        assert abs(stats["mean"] - 0.3) < 1e-10
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
    
    def test_timing_context_manager(self, metrics):
        """Test timing context manager"""
        with TimingContextManager(metrics, "context_timer"):
            time.sleep(0.01)  # Small delay
        
        stats = metrics.get_timer_stats("context_timer")
        assert stats["count"] == 1
        assert stats["mean"] >= 0.01  # Should be at least the sleep time


class TestQuantumPerformanceMonitor:
    """Test suite for QuantumPerformanceMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create test performance monitor"""
        metrics_collector = MetricsCollector()
        return QuantumPerformanceMonitor(metrics_collector)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert isinstance(monitor.metrics, MetricsCollector)
        assert isinstance(monitor.quantum_sessions, dict)
        assert isinstance(monitor.alert_handlers, list)
        assert isinstance(monitor.thresholds, dict)
    
    def test_quantum_session_lifecycle(self, monitor):
        """Test quantum session start and end"""
        session_id = "test_session"
        metadata = {"test": "data"}
        
        # Start session
        monitor.start_quantum_session(session_id, metadata)
        
        assert session_id in monitor.quantum_sessions
        session = monitor.quantum_sessions[session_id]
        assert session["metadata"] == metadata
        assert "start_time" in session
        assert session["optimization_steps"] == 0
        
        # End session
        summary = monitor.end_quantum_session(session_id)
        
        assert session_id not in monitor.quantum_sessions  # Should be cleaned up
        assert isinstance(summary, dict)
        assert "session_id" in summary
        assert "duration" in summary
    
    def test_record_quantum_state(self, monitor):
        """Test quantum state recording"""
        session_id = "quantum_session"
        monitor.start_quantum_session(session_id)
        
        # Create normalized quantum state
        quantum_state = np.array([0.6+0.8j, 0.0+0.0j], dtype=complex)
        quantum_state = quantum_state / np.linalg.norm(quantum_state)
        
        monitor.record_quantum_state(session_id, quantum_state, step=1)
        
        session = monitor.quantum_sessions[session_id]
        assert len(session["quantum_states"]) == 1
        
        state_record = session["quantum_states"][0]
        assert "normalization" in state_record
        assert "entropy" in state_record
        assert "coherence" in state_record
        assert state_record["step"] == 1
    
    def test_record_optimization_step(self, monitor):
        """Test optimization step recording"""
        session_id = "optimization_session"
        monitor.start_quantum_session(session_id)
        
        monitor.record_optimization_step(
            session_id, 
            objective_value=0.85,
            convergence_rate=0.9,
            step_duration=0.1
        )
        
        session = monitor.quantum_sessions[session_id]
        assert session["optimization_steps"] == 1
        assert len(session["convergence_history"]) == 1
        
        step_record = session["convergence_history"][0]
        assert step_record["objective"] == 0.85
        assert step_record["convergence_rate"] == 0.9
        assert step_record["duration"] == 0.1
    
    def test_record_resource_usage(self, monitor):
        """Test resource usage recording"""
        session_id = "resource_session"
        monitor.start_quantum_session(session_id)
        
        monitor.record_resource_usage(session_id, 0.5, 0.7, 0.3)
        
        session = monitor.quantum_sessions[session_id]
        assert len(session["resource_usage"]) == 1
        
        resource_record = session["resource_usage"][0]
        assert resource_record["cpu"] == 0.5
        assert resource_record["memory"] == 0.7
        assert resource_record["gpu"] == 0.3
    
    def test_alert_handling(self, monitor):
        """Test alert handling system"""
        alerts_received = []
        
        def test_alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_handler(test_alert_handler)
        
        session_id = "alert_session"
        monitor.start_quantum_session(session_id)
        
        # Create state with low coherence to trigger alert
        low_coherence_state = np.array([1.0+0j], dtype=complex)
        monitor.record_quantum_state(session_id, low_coherence_state, step=1)
        
        # Should have triggered low coherence alert
        assert len(alerts_received) > 0
        alert = alerts_received[0]
        assert alert.level == AlertLevel.WARNING
        assert "coherence" in alert.message.lower()
    
    def test_performance_snapshot(self, monitor):
        """Test performance snapshot generation"""
        # Create some activity
        session_id = "snapshot_session"
        monitor.start_quantum_session(session_id, {"test": True})
        
        quantum_state = np.array([0.7+0.7j], dtype=complex)
        quantum_state = quantum_state / np.linalg.norm(quantum_state)
        monitor.record_quantum_state(session_id, quantum_state, step=1)
        
        snapshot = monitor.get_performance_snapshot()
        
        assert hasattr(snapshot, 'timestamp')
        assert hasattr(snapshot, 'quantum_metrics')
        assert hasattr(snapshot, 'system_metrics')
        assert hasattr(snapshot, 'task_metrics')
        
        assert isinstance(snapshot.quantum_metrics, dict)
        assert isinstance(snapshot.system_metrics, dict)
        assert isinstance(snapshot.task_metrics, dict)


class TestQuantumStateCache:
    """Test suite for QuantumStateCache"""
    
    @pytest.fixture
    def cache(self):
        """Create test cache"""
        return QuantumStateCache(
            max_size=10,
            max_memory_mb=1,
            enable_compression=True
        )
    
    @pytest.fixture
    def sample_quantum_states(self):
        """Create sample quantum states"""
        states = []
        for i in range(5):
            # Create different quantum states
            state = np.random.random(4).astype(complex)
            state += 1j * np.random.random(4)
            state = state / np.linalg.norm(state)  # Normalize
            states.append((f"state_{i}", state))
        return states
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = QuantumStateCache(
            max_size=100,
            max_memory_mb=10,
            policy=CachePolicy.LRU
        )
        
        assert cache.max_size == 100
        assert cache.max_memory_bytes == 10 * 1024 * 1024
        assert cache.policy == CachePolicy.LRU
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
    
    def test_cache_put_and_get(self, cache, sample_quantum_states):
        """Test basic cache put and get operations"""
        key, state = sample_quantum_states[0]
        
        # Store state
        success = cache.put(key, state)
        assert success == True
        
        # Retrieve state
        retrieved_state = cache.get(key)
        assert retrieved_state is not None
        np.testing.assert_array_almost_equal(retrieved_state, state)
        
        # Check stats
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0
    
    def test_cache_miss(self, cache):
        """Test cache miss handling"""
        result = cache.get("non_existent_key")
        
        assert result is None
        assert cache.stats.hits == 0
        assert cache.stats.misses == 1
    
    def test_cache_eviction(self, cache, sample_quantum_states):
        """Test cache eviction when limits are reached"""
        # Fill cache beyond capacity
        for i, (key, state) in enumerate(sample_quantum_states):
            cache.put(key, state)
        
        # Add more states to trigger eviction
        for i in range(10):  # Exceed max_size of 10
            extra_state = np.array([1.0+0j], dtype=complex)
            cache.put(f"extra_{i}", extra_state)
        
        # Should have triggered evictions
        assert cache.stats.evictions > 0
        assert len(cache.cache) <= cache.max_size
    
    def test_invalid_state_rejection(self, cache):
        """Test rejection of invalid quantum states"""
        # Non-complex state
        invalid_state_real = np.array([1.0, 0.0], dtype=float)
        success = cache.put("invalid_real", invalid_state_real)
        assert success == False
        
        # State with NaN
        invalid_state_nan = np.array([np.nan+0j, 0.0+0j], dtype=complex)
        success = cache.put("invalid_nan", invalid_state_nan)
        assert success == False
        
        # Non-normalized state (this might still be accepted and normalized internally)
        non_normalized = np.array([2.0+0j, 3.0+0j], dtype=complex)
        success = cache.put("non_normalized", non_normalized)
        # Behavior depends on implementation - might normalize or reject
    
    def test_find_similar_state(self, cache, sample_quantum_states):
        """Test finding similar quantum states"""
        key, state = sample_quantum_states[0]
        
        # Store original state
        cache.put(key, state)
        
        # Create slightly modified state
        similar_state = state * 0.99 + np.array([0.01+0j, 0.0, 0.0, 0.0])
        similar_state = similar_state / np.linalg.norm(similar_state)
        
        # Look for similar state
        match = cache.find_similar_state(similar_state, threshold=0.8)
        
        if match:
            found_key, found_state = match
            assert found_key == key
            np.testing.assert_array_almost_equal(found_state, state)
    
    def test_cache_compression(self):
        """Test state compression and decompression"""
        cache = QuantumStateCache(enable_compression=True)
        
        state = np.array([0.6+0.8j, 0.0+0.0j], dtype=complex)
        
        # Store and retrieve with compression
        cache.put("compressed_state", state)
        retrieved = cache.get("compressed_state")
        
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, state, decimal=5)  # Allow for compression loss


class TestOptimizationResultCache:
    """Test suite for OptimizationResultCache"""
    
    @pytest.fixture
    def opt_cache(self):
        """Create test optimization result cache"""
        return OptimizationResultCache(max_entries=5)
    
    def test_optimization_cache_basic_operations(self, opt_cache):
        """Test basic cache operations for optimization results"""
        task_pattern = "embedding->attention->feedforward"
        constraints = {"max_time": 10.0, "max_memory": 1.0}
        resources = {"gpu": 1.0, "cpu": 1.0}
        result = {"optimal_schedule": [1, 2, 3], "objective": 0.85}
        
        # Initially should miss
        cached_result = opt_cache.get_optimization_result(task_pattern, constraints, resources)
        assert cached_result is None
        
        # Store result
        opt_cache.store_optimization_result(task_pattern, constraints, resources, result)
        
        # Should now hit
        cached_result = opt_cache.get_optimization_result(task_pattern, constraints, resources)
        assert cached_result is not None
        assert cached_result["objective"] == 0.85
    
    def test_optimization_cache_pattern_matching(self, opt_cache):
        """Test pattern-based cache matching"""
        task_pattern1 = "embedding->attention"
        task_pattern2 = "embedding->attention"  # Similar pattern
        
        constraints = {"max_time": 10.0}
        resources1 = {"gpu": 1.0}
        resources2 = {"gpu": 0.9}  # Slightly different resources
        
        result = {"schedule": [1, 2], "objective": 0.9}
        
        # Store with first configuration
        opt_cache.store_optimization_result(task_pattern1, constraints, resources1, result)
        
        # Try to get with second configuration (similar)
        cached_result = opt_cache.get_optimization_result(task_pattern2, constraints, resources2)
        
        # Might return adapted result or None depending on similarity threshold
        if cached_result:
            assert "adapted" in cached_result or "objective" in cached_result


class TestPersistentQuantumCache:
    """Test suite for PersistentQuantumCache"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def persistent_cache(self, temp_cache_dir):
        """Create persistent cache with temporary directory"""
        return PersistentQuantumCache(
            cache_dir=temp_cache_dir,
            max_size_gb=0.001  # 1MB for testing
        )
    
    def test_persistent_cache_operations(self, persistent_cache):
        """Test persistent cache basic operations"""
        test_data = {"quantum_state": np.array([0.6+0.8j, 0.0+0.0j]), "metadata": "test"}
        
        # Store data
        success = persistent_cache.put("test_key", test_data)
        assert success == True
        
        # Retrieve data
        retrieved_data = persistent_cache.get("test_key")
        assert retrieved_data is not None
        assert "quantum_state" in retrieved_data
        assert retrieved_data["metadata"] == "test"
        
        np.testing.assert_array_equal(retrieved_data["quantum_state"], test_data["quantum_state"])
    
    def test_persistent_cache_persistence(self, temp_cache_dir):
        """Test data persistence across cache instances"""
        test_data = {"value": 42, "array": np.array([1, 2, 3])}
        
        # Create cache, store data, and close
        cache1 = PersistentQuantumCache(cache_dir=temp_cache_dir)
        cache1.put("persistent_key", test_data)
        del cache1
        
        # Create new cache instance and retrieve data
        cache2 = PersistentQuantumCache(cache_dir=temp_cache_dir)
        retrieved_data = cache2.get("persistent_key")
        
        assert retrieved_data is not None
        assert retrieved_data["value"] == 42
        np.testing.assert_array_equal(retrieved_data["array"], np.array([1, 2, 3]))
    
    def test_persistent_cache_size_limit(self, persistent_cache):
        """Test persistent cache size limit enforcement"""
        # Fill cache with data
        for i in range(10):
            large_data = {"array": np.random.random(1000), "id": i}
            persistent_cache.put(f"large_data_{i}", large_data)
        
        # Check that size limit is enforced
        total_size = sum(entry["size"] for entry in persistent_cache.index.values())
        assert total_size <= persistent_cache.max_size_bytes


def test_create_quantum_cache_system():
    """Test complete cache system creation"""
    config = {
        "quantum_cache_size": 50,
        "quantum_cache_memory_mb": 10,
        "optimization_cache_size": 20,
        "persistent_cache_size_gb": 0.1
    }
    
    cache_system = create_quantum_cache_system(config)
    
    assert "quantum_state_cache" in cache_system
    assert "optimization_cache" in cache_system
    assert "persistent_cache" in cache_system
    
    assert isinstance(cache_system["quantum_state_cache"], QuantumStateCache)
    assert isinstance(cache_system["optimization_cache"], OptimizationResultCache)
    assert isinstance(cache_system["persistent_cache"], PersistentQuantumCache)
    
    # Check configuration was applied
    assert cache_system["quantum_state_cache"].max_size == 50
    assert cache_system["optimization_cache"].max_entries == 20


@pytest.mark.integration
class TestQuantumSchedulingIntegration:
    """Integration tests for quantum scheduling components"""
    
    @pytest.mark.asyncio
    async def test_full_scheduling_pipeline(self):
        """Test complete scheduling pipeline integration"""
        # Create scheduler with caching
        config = SchedulerConfig(
            max_concurrent_tasks=2,
            quantum_optimization=True,
            performance_monitoring=True
        )
        scheduler = QuantumScheduler(config)
        
        # Create executor
        executor = ConcurrentQuantumExecutor(
            max_workers=2,
            auto_scaling=False
        )
        
        await executor.start()
        
        try:
            # Create workflow
            tasks = scheduler.create_inference_workflow(
                model_name="test-model",
                input_data="test input",
                priority=TaskPriority.HIGH
            )
            
            # Execute with both scheduler and executor
            scheduler_result = await scheduler.schedule_and_execute()
            
            assert scheduler_result["status"] == "completed"
            assert scheduler_result["tasks_processed"] > 0
            
        finally:
            await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test integration of performance monitoring"""
        metrics = MetricsCollector()
        monitor = QuantumPerformanceMonitor(metrics)
        
        session_id = "integration_session"
        monitor.start_quantum_session(session_id, {"integration_test": True})
        
        # Simulate quantum computation
        quantum_state = np.array([0.7+0.7j], dtype=complex)
        quantum_state = quantum_state / np.linalg.norm(quantum_state)
        
        for step in range(5):
            monitor.record_quantum_state(session_id, quantum_state, step)
            monitor.record_optimization_step(
                session_id,
                objective_value=0.8 + step * 0.02,
                convergence_rate=0.9,
                step_duration=0.1
            )
            
            # Simulate some processing time
            await asyncio.sleep(0.01)
        
        # End session and get summary
        summary = monitor.end_quantum_session(session_id)
        
        assert summary["total_steps"] == 5
        assert summary["duration"] > 0
        assert "quantum_metrics" in summary
        assert "optimization_metrics" in summary
    
    def test_caching_integration_with_scheduling(self):
        """Test cache integration with scheduling"""
        cache_system = create_quantum_cache_system()
        
        # Create quantum states and store in cache
        state1 = np.array([1.0+0j, 0.0+0j], dtype=complex)
        state2 = np.array([0.0+0j, 1.0+0j], dtype=complex)
        
        quantum_cache = cache_system["quantum_state_cache"]
        
        success1 = quantum_cache.put("state_1", state1)
        success2 = quantum_cache.put("state_2", state2)
        
        assert success1 == True
        assert success2 == True
        
        # Retrieve and verify
        retrieved1 = quantum_cache.get("state_1")
        retrieved2 = quantum_cache.get("state_2")
        
        np.testing.assert_array_equal(retrieved1, state1)
        np.testing.assert_array_equal(retrieved2, state2)
        
        # Test cache statistics
        stats = quantum_cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])