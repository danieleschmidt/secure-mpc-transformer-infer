"""
Test suite for new SDLC implementations
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any

# Test imports that don't require external dependencies
def test_imports_without_dependencies():
    """Test that modules can be parsed and have correct structure"""
    
    # Test advanced circuit breaker
    import ast
    import importlib.util
    
    files_to_test = [
        'src/secure_mpc_transformer/resilience/advanced_circuit_breaker.py',
        'src/secure_mpc_transformer/monitoring/real_time_alerting.py', 
        'src/secure_mpc_transformer/validation/comprehensive_input_validator.py',
        'src/secure_mpc_transformer/scaling/horizontal_scaling_manager.py',
        'src/secure_mpc_transformer/optimization/performance_optimizer.py'
    ]
    
    for file_path in files_to_test:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Test that file can be parsed
        tree = ast.parse(source)
        
        # Test that file has classes
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert len(classes) > 0, f"File {file_path} should have at least one class"
        
        # Test that file has async functions
        async_funcs = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
        assert len(async_funcs) > 0, f"File {file_path} should have at least one async function"

class TestAdvancedCircuitBreaker:
    """Test advanced circuit breaker functionality"""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        # Mock the sklearn dependency
        with patch('secure_mpc_transformer.resilience.advanced_circuit_breaker.IsolationForest'):
            with patch('secure_mpc_transformer.resilience.advanced_circuit_breaker.StandardScaler'):
                from secure_mpc_transformer.resilience.advanced_circuit_breaker import (
                    AdvancedCircuitBreaker, CircuitBreakerState
                )
                
                cb = AdvancedCircuitBreaker("test_cb", enable_ml_prediction=False)
                
                # Test initial state
                assert cb.state == CircuitBreakerState.CLOSED
                assert cb.consecutive_failures == 0
                
                # Test failure recording
                cb._record_failure(Exception("test error"), 1.0)
                assert cb.consecutive_failures == 1
                
                # Test success recording
                cb._record_success(0.5)
                assert cb.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_call(self):
        """Test circuit breaker call functionality"""
        with patch('secure_mpc_transformer.resilience.advanced_circuit_breaker.IsolationForest'):
            with patch('secure_mpc_transformer.resilience.advanced_circuit_breaker.StandardScaler'):
                from secure_mpc_transformer.resilience.advanced_circuit_breaker import (
                    AdvancedCircuitBreaker, CircuitBreakerOpenException
                )
                
                cb = AdvancedCircuitBreaker("test_cb", enable_ml_prediction=False)
                
                # Test successful call
                async def success_func():
                    return "success"
                
                result = await cb.call(success_func)
                assert result == "success"
                
                # Test failure handling
                async def failing_func():
                    raise Exception("test failure")
                
                # Multiple failures should eventually open circuit
                for _ in range(10):
                    try:
                        await cb.call(failing_func)
                    except Exception:
                        pass
                
                # Circuit should be open now
                assert cb.state.value in ["open", "half_open"]

class TestRealTimeAlerting:
    """Test real-time alerting system"""
    
    def test_alert_creation(self):
        """Test alert creation and management"""
        from secure_mpc_transformer.monitoring.real_time_alerting import (
            Alert, AlertSeverity, AlertStatus, AlertRule, ScalingTrigger
        )
        
        # Test alert creation
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            rule_name="Test Rule",
            description="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.OPEN,
            metric_name="cpu_usage",
            metric_value=85.0,
            threshold=80.0,
            timestamp=time.time()
        )
        
        assert alert.id == "test_alert"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == AlertStatus.OPEN
    
    def test_alert_rule_creation(self):
        """Test alert rule creation"""
        from secure_mpc_transformer.monitoring.real_time_alerting import (
            AlertRule, AlertSeverity, AlertChannel, ScalingTrigger
        )
        
        rule = AlertRule(
            id="cpu_rule",
            name="CPU High Usage",
            description="CPU usage too high",
            condition="cpu_utilization",
            severity=AlertSeverity.MEDIUM,
            threshold=75.0,
            comparison="gt",
            channels=[AlertChannel.EMAIL]
        )
        
        assert rule.id == "cpu_rule"
        assert rule.threshold == 75.0
        assert rule.comparison == "gt"

class TestInputValidator:
    """Test comprehensive input validator"""
    
    @pytest.mark.asyncio
    async def test_basic_validation(self):
        """Test basic input validation"""
        from secure_mpc_transformer.validation.comprehensive_input_validator import (
            ComprehensiveInputValidator, ValidationLevel
        )
        
        validator = ComprehensiveInputValidator({
            "validation_level": "standard"
        })
        
        # Test safe input
        result = await validator.validate_input("hello world", "text")
        assert result.is_valid
        assert result.risk_score == 0.0
        
        # Test potentially dangerous input
        result = await validator.validate_input("<script>alert('xss')</script>", "html")
        assert not result.is_valid or result.risk_score > 0.5
    
    def test_threat_pattern_initialization(self):
        """Test threat pattern initialization"""
        from secure_mpc_transformer.validation.comprehensive_input_validator import (
            ComprehensiveInputValidator, ThreatType
        )
        
        validator = ComprehensiveInputValidator()
        
        # Test that threat patterns are loaded
        assert len(validator.threat_patterns) > 0
        assert ThreatType.XSS in validator.threat_patterns
        assert ThreatType.SQL_INJECTION in validator.threat_patterns
        
        # Test pattern structure
        xss_patterns = validator.threat_patterns[ThreatType.XSS]
        assert len(xss_patterns) > 0
        
        for pattern in xss_patterns:
            assert hasattr(pattern, 'pattern')
            assert hasattr(pattern, 'severity')
            assert hasattr(pattern, 'confidence')

class TestHorizontalScaling:
    """Test horizontal scaling manager"""
    
    def test_scaling_metrics(self):
        """Test scaling metrics structure"""
        from secure_mpc_transformer.scaling.horizontal_scaling_manager import (
            ScalingMetrics, ScalingDecision, ScalingDirection
        )
        
        metrics = ScalingMetrics(
            cpu_utilization=75.0,
            memory_utilization=60.0,
            request_rate=500.0
        )
        
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 60.0
        assert metrics.request_rate == 500.0
        
        # Test scaling decision
        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            target_count=5,
            current_count=3,
            reasoning=["High CPU utilization"],
            confidence=0.8,
            triggered_by=[],
            timestamp=time.time()
        )
        
        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.target_count == 5
        assert decision.confidence == 0.8
    
    def test_scaling_rule_creation(self):
        """Test scaling rule creation"""
        from secure_mpc_transformer.scaling.horizontal_scaling_manager import (
            ScalingRule, ScalingTrigger
        )
        
        rule = ScalingRule(
            id="cpu_scaling",
            name="CPU Scaling Rule",
            trigger=ScalingTrigger.CPU_UTILIZATION,
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_step=2
        )
        
        assert rule.id == "cpu_scaling"
        assert rule.trigger == ScalingTrigger.CPU_UTILIZATION
        assert rule.threshold_up == 80.0
        assert rule.scale_up_step == 2

class TestPerformanceOptimizer:
    """Test performance optimizer"""
    
    def test_optimization_parameter(self):
        """Test optimization parameter structure"""
        from secure_mpc_transformer.optimization.performance_optimizer import (
            OptimizationParameter, PerformanceMetrics
        )
        
        param = OptimizationParameter(
            name="max_connections",
            current_value=100,
            min_value=50,
            max_value=500,
            step_size=25,
            parameter_type="int",
            description="Maximum connections"
        )
        
        assert param.name == "max_connections"
        assert param.current_value == 100
        assert param.parameter_type == "int"
    
    def test_performance_metrics(self):
        """Test performance metrics structure"""
        from secure_mpc_transformer.optimization.performance_optimizer import (
            PerformanceMetrics, OptimizationTarget
        )
        
        metrics = PerformanceMetrics(
            latency_p95=150.0,
            throughput_rps=1000.0,
            cpu_utilization=65.0,
            cache_hit_rate=0.85
        )
        
        assert metrics.latency_p95 == 150.0
        assert metrics.throughput_rps == 1000.0
        assert metrics.cache_hit_rate == 0.85
        
        # Test optimization target enum
        assert OptimizationTarget.LATENCY.value == "latency"
        assert OptimizationTarget.THROUGHPUT.value == "throughput"

class TestIntegration:
    """Integration tests for components working together"""
    
    @pytest.mark.asyncio 
    async def test_component_interfaces(self):
        """Test that components have compatible interfaces"""
        # Test that alerting system can work with metrics from scaling manager
        from secure_mpc_transformer.scaling.horizontal_scaling_manager import ScalingMetrics
        from secure_mpc_transformer.monitoring.real_time_alerting import RealTimeAlertingSystem
        
        # Create metrics
        metrics = ScalingMetrics(
            cpu_utilization=85.0,
            memory_utilization=70.0,
            request_rate=1500.0,
            threat_level=0.3
        )
        
        # Test that alerting system can process these metrics
        alerting_config = {"anomaly_detection": {}, "ab_testing_enabled": False}
        
        # Mock sklearn for alerting system
        with patch('secure_mpc_transformer.monitoring.real_time_alerting.TfidfVectorizer'):
            with patch('secure_mpc_transformer.monitoring.real_time_alerting.RandomForestClassifier'):
                alerting_system = RealTimeAlertingSystem(alerting_config)
                
                # Test metric addition (should not fail)
                await alerting_system.add_metric_point("cpu_utilization", metrics.cpu_utilization)
                await alerting_system.add_metric_point("request_rate", metrics.request_rate)
        
    def test_data_compatibility(self):
        """Test data structure compatibility between components"""
        # Test that metrics can be shared between components
        from secure_mpc_transformer.scaling.horizontal_scaling_manager import ScalingMetrics
        from secure_mpc_transformer.optimization.performance_optimizer import PerformanceMetrics
        
        # Create scaling metrics
        scaling_metrics = ScalingMetrics(
            cpu_utilization=75.0,
            memory_utilization=60.0,
            request_rate=800.0
        )
        
        # Convert to performance metrics (test compatibility)
        perf_metrics = PerformanceMetrics(
            throughput_rps=scaling_metrics.request_rate,
            cpu_utilization=scaling_metrics.cpu_utilization,
            memory_utilization=scaling_metrics.memory_utilization
        )
        
        assert perf_metrics.throughput_rps == 800.0
        assert perf_metrics.cpu_utilization == 75.0

def test_code_quality_metrics():
    """Test code quality metrics"""
    import ast
    
    files_to_analyze = [
        'src/secure_mpc_transformer/resilience/advanced_circuit_breaker.py',
        'src/secure_mpc_transformer/monitoring/real_time_alerting.py',
        'src/secure_mpc_transformer/validation/comprehensive_input_validator.py',
        'src/secure_mpc_transformer/scaling/horizontal_scaling_manager.py',
        'src/secure_mpc_transformer/optimization/performance_optimizer.py'
    ]
    
    total_lines = 0
    total_functions = 0
    total_classes = 0
    total_async_functions = 0
    
    for file_path in files_to_analyze:
        with open(file_path, 'r') as f:
            source = f.read()
            lines = len(source.splitlines())
            total_lines += lines
        
        tree = ast.parse(source)
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        async_functions = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        total_functions += len(functions)
        total_async_functions += len(async_functions) 
        total_classes += len(classes)
    
    # Quality assertions
    assert total_lines > 5000, f"Expected substantial implementation, got {total_lines} lines"
    assert total_classes > 10, f"Expected multiple classes, got {total_classes}"
    assert total_functions + total_async_functions > 50, f"Expected many functions, got {total_functions + total_async_functions}"
    assert total_async_functions > 10, f"Expected async functions, got {total_async_functions}"
    
    print(f"\nCode Quality Metrics:")
    print(f"  Total lines: {total_lines}")
    print(f"  Total classes: {total_classes}")
    print(f"  Total functions: {total_functions}")
    print(f"  Total async functions: {total_async_functions}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])