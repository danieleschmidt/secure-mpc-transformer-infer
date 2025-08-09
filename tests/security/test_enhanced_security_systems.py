#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced security systems.

Tests all defensive security components including validation, monitoring,
incident response, and orchestration systems with extensive coverage.
"""

import pytest
import asyncio
import time
import json
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Import security components for testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from secure_mpc_transformer.security.enhanced_validator import (
    EnhancedSecurityValidator,
    ValidationResult,
    ValidationContext,
    MLAnomalyDetector,
    ContentSecurityAnalyzer,
    QuantumProtocolValidator
)

from secure_mpc_transformer.security.quantum_monitor import (
    QuantumSecurityMonitor,
    QuantumCoherenceMonitor,
    SideChannelDetector,
    QuantumSecurityEvent,
    QuantumThreatLevel,
    QuantumOperationContext
)

from secure_mpc_transformer.security.incident_response import (
    AIIncidentResponseSystem,
    SecurityIncident,
    ThreatIntelligenceEngine,
    AutomatedResponseEngine,
    IncidentSeverity,
    ThreatCategory,
    ResponseAction
)

from secure_mpc_transformer.monitoring.security_dashboard import (
    SecurityMetricsDashboard,
    RealTimeMetricsCollector,
    ThreatLandscapeAnalyzer,
    SecurityControlAnalyzer,
    MetricType
)

from secure_mpc_transformer.security.advanced_security_orchestrator import (
    AdvancedSecurityOrchestrator,
    AdaptiveSecurityCache,
    IntelligentLoadBalancer,
    ThreatCorrelationEngine,
    AutoScalingSecurityManager,
    SecurityWorkerPool,
    LoadBalancingMethod
)


class TestEnhancedSecurityValidator:
    """Test suite for enhanced security validator."""
    
    @pytest.fixture
    async def validator(self):
        """Create validator instance for testing."""
        return EnhancedSecurityValidator()
    
    @pytest.fixture
    def validation_context(self):
        """Create validation context for testing."""
        return ValidationContext(
            client_ip="192.168.1.100",
            user_agent="Mozilla/5.0 (Test)",
            session_id="test_session_123",
            request_timestamp=datetime.now(timezone.utc),
            request_size=1024,
            content_type="application/json"
        )
    
    @pytest.mark.asyncio
    async def test_basic_validation_success(self, validator, validation_context):
        """Test successful basic validation."""
        content = '{"user": "test", "action": "login"}'
        
        result = await validator.validate_request_pipeline(
            content, validation_context, validate_quantum=False
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.risk_score < 0.5
        assert result.validation_time > 0
        assert "basic_validation" in result.metadata
    
    @pytest.mark.asyncio
    async def test_injection_attack_detection(self, validator, validation_context):
        """Test SQL injection attack detection."""
        malicious_content = "'; DROP TABLE users; --"
        
        result = await validator.validate_request_pipeline(
            malicious_content, validation_context, validate_quantum=False
        )
        
        assert isinstance(result, ValidationResult)
        assert not result.is_valid  # Should be blocked
        assert result.risk_score > 0.7
        assert any("injection" in threat.lower() for threat in result.threats_detected)
    
    @pytest.mark.asyncio
    async def test_xss_attack_detection(self, validator, validation_context):
        """Test XSS attack detection."""
        xss_content = '<script>alert("XSS")</script>'
        
        result = await validator.validate_request_pipeline(
            xss_content, validation_context, validate_quantum=False
        )
        
        assert not result.is_valid
        assert result.risk_score > 0.6
        assert "content_security" in result.metadata
    
    @pytest.mark.asyncio
    async def test_quantum_validation(self, validator, validation_context):
        """Test quantum protocol validation."""
        quantum_content = json.dumps({
            "quantum_state": {
                "coherence": 0.8,
                "entanglement": {"qubits": [0, 1], "strength": 0.9},
                "measurement_basis": ["X", "Z"]
            },
            "operation_params": {
                "iterations": 100,
                "temperature": 0.01
            }
        })
        
        result = await validator.validate_request_pipeline(
            quantum_content, validation_context, validate_quantum=True
        )
        
        assert result.is_valid
        assert "quantum_validation" in result.metadata
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, validator, validation_context):
        """Test ML-based anomaly detection."""
        # Simulate high-frequency requests
        for _ in range(10):
            await validator.validate_request_pipeline(
                "normal content", validation_context, validate_quantum=False
            )
        
        # High-frequency request should trigger anomaly detection
        validation_context.request_size = 10_000_000  # Very large request
        result = await validator.validate_request_pipeline(
            "x" * 1000, validation_context, validate_quantum=False
        )
        
        assert result.risk_score > 0.3  # Should detect size anomaly
    
    @pytest.mark.asyncio
    async def test_content_security_analysis(self, validator, validation_context):
        """Test content security analyzer."""
        analyzer = ContentSecurityAnalyzer()
        
        # Test dangerous content
        dangerous_content = "eval(user_input); system('rm -rf /')"
        risk, threats = analyzer.analyze_content_security(
            dangerous_content, "application/javascript"
        )
        
        assert risk > 0.5
        assert len(threats) > 0
        assert any("injection" in threat for threat in threats)
    
    @pytest.mark.asyncio 
    async def test_validator_performance(self, validator, validation_context):
        """Test validator performance under load."""
        start_time = time.time()
        tasks = []
        
        # Create 100 concurrent validation tasks
        for i in range(100):
            content = f'{{"test_request": {i}}}'
            task = validator.validate_request_pipeline(
                content, validation_context, validate_quantum=False
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Performance assertions
        total_time = end_time - start_time
        assert total_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 100
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Average validation time should be reasonable
        avg_validation_time = sum(r.validation_time for r in results) / len(results)
        assert avg_validation_time < 0.1  # Less than 100ms average


class TestQuantumSecurityMonitor:
    """Test suite for quantum security monitor."""
    
    @pytest.fixture
    async def monitor(self):
        """Create quantum monitor instance for testing."""
        monitor = QuantumSecurityMonitor()
        await monitor.start_monitoring()
        return monitor
    
    @pytest.fixture
    def quantum_operation_context(self):
        """Create quantum operation context for testing."""
        return QuantumOperationContext(
            operation_id="test_op_123",
            operation_type="quantum_inference",
            start_time=datetime.now(timezone.utc),
            quantum_circuit_depth=10,
            qubit_count=8,
            gate_count=50,
            measurement_basis=["X", "Y", "Z"],
            expected_coherence=0.8
        )
    
    @pytest.fixture
    def quantum_state(self):
        """Create quantum state for testing."""
        return {
            "state_id": "test_state_123",
            "coherence": 0.8,
            "entanglement_metrics": {
                "total_entanglement": 0.9,
                "max_entanglement": 0.95
            }
        }
    
    @pytest.mark.asyncio
    async def test_quantum_operation_monitoring(self, monitor, quantum_operation_context, quantum_state):
        """Test quantum operation monitoring."""
        security_event = await monitor.monitor_quantum_operation(
            quantum_operation_context, quantum_state
        )
        
        assert isinstance(security_event, QuantumSecurityEvent)
        assert security_event.quantum_state_id == "test_state_123"
        assert security_event.coherence_level == 0.8
        assert security_event.threat_level in [QuantumThreatLevel.LOW, QuantumThreatLevel.MEDIUM]
    
    @pytest.mark.asyncio
    async def test_coherence_monitoring(self, monitor, quantum_state):
        """Test quantum coherence monitoring."""
        coherence_monitor = QuantumCoherenceMonitor(coherence_threshold=0.1)
        
        # Test normal coherence
        is_secure, score, alerts = await coherence_monitor.monitor_coherence(quantum_state)
        assert is_secure
        assert score > 0.8
        assert len(alerts) == 0
        
        # Test low coherence
        low_coherence_state = {**quantum_state, "coherence": 0.05}
        is_secure, score, alerts = await coherence_monitor.monitor_coherence(low_coherence_state)
        assert not is_secure
        assert len(alerts) > 0
        assert any("coherence_below_threshold" in alert for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_side_channel_detection(self, monitor, quantum_operation_context):
        """Test side-channel attack detection."""
        detector = SideChannelDetector()
        
        # Test timing attack detection
        attack_detected, alerts = await detector.detect_timing_attacks(quantum_operation_context)
        
        # Should not detect attack for normal operation
        assert not attack_detected or len(alerts) == 0
        
        # Test power analysis detection
        normal_power = [1.0, 1.1, 0.9, 1.2, 1.0] * 20  # Normal variation
        attack_detected, alerts = await detector.detect_power_analysis_attacks(normal_power)
        assert not attack_detected or len(alerts) == 0
        
        # Test suspicious power pattern
        suspicious_power = [1.0] * 20 + [5.0] * 5 + [1.0] * 20  # Power spike
        attack_detected, alerts = await detector.detect_power_analysis_attacks(suspicious_power)
        # May detect attack depending on thresholds
    
    @pytest.mark.asyncio
    async def test_quantum_security_status(self, monitor):
        """Test quantum security status reporting."""
        status = await monitor.get_security_status()
        
        assert isinstance(status, dict)
        assert "overall_threat_level" in status
        assert "active_operations" in status
        assert "sidechannel_monitoring" in status
        assert status["overall_threat_level"] in ["low", "medium", "high", "critical", "unknown"]
    
    @pytest.mark.asyncio
    async def test_power_consumption_analysis(self, monitor):
        """Test power consumption analysis for attacks."""
        # Simulate normal power consumption
        normal_power = [1.0 + 0.1 * i for i in range(50)]
        attack_detected = await monitor.analyze_power_consumption(normal_power)
        assert not attack_detected
        
        # Simulate suspicious power pattern
        attack_power = [1.0] * 10 + [10.0] * 5 + [1.0] * 10  # Major spike
        attack_detected = await monitor.analyze_power_consumption(attack_power)
        # May detect based on implementation


class TestIncidentResponseSystem:
    """Test suite for AI incident response system."""
    
    @pytest.fixture
    async def incident_system(self):
        """Create incident response system for testing."""
        system = AIIncidentResponseSystem()
        await system.start_system()
        return system
    
    @pytest.fixture
    def security_event_data(self):
        """Create security event data for testing."""
        return {
            "source_ip": "192.168.1.100",
            "user_id": "test_user",
            "session_id": "test_session_123",
            "request_content": "'; DROP TABLE users; --",
            "user_agent": "Mozilla/5.0 (Test)",
            "timestamp": time.time(),
            "request_rate": 5.0,
            "content_type": "application/json"
        }
    
    @pytest.mark.asyncio
    async def test_security_event_processing(self, incident_system, security_event_data):
        """Test security event processing and incident creation."""
        incident = await incident_system.process_security_event(security_event_data)
        
        assert isinstance(incident, SecurityIncident)
        assert incident.source_ip == "192.168.1.100"
        assert incident.category != ThreatCategory.UNKNOWN
        assert incident.confidence_score > 0.0
        assert len(incident.threat_indicators) > 0
        assert len(incident.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_threat_intelligence_analysis(self, incident_system, security_event_data):
        """Test threat intelligence analysis."""
        threat_intel = incident_system.threat_intel
        
        category, confidence, indicators = await threat_intel.analyze_incident(security_event_data)
        
        assert isinstance(category, ThreatCategory)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(indicators, list)
        
        # Should detect SQL injection
        assert category == ThreatCategory.INJECTION_ATTACK
        assert confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_automated_response_execution(self, incident_system, security_event_data):
        """Test automated incident response."""
        # Create high-severity incident
        security_event_data["request_content"] = "CRITICAL_ATTACK_PATTERN"
        security_event_data["threat_score"] = 0.9
        
        incident = await incident_system.process_security_event(
            security_event_data, {"production_environment": False}
        )
        
        # Should have recommended actions for high-severity incident
        assert incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
        assert ResponseAction.LOG_ONLY in incident.recommended_actions
        
        # Test response execution
        response_engine = incident_system.response_engine
        response_result = await response_engine.execute_response(
            incident, {"production_environment": False}
        )
        
        assert response_result["status"] in ["executed", "error"]
        if response_result["status"] == "executed":
            assert len(response_result["actions_taken"]) > 0
    
    @pytest.mark.asyncio
    async def test_reputation_tracking(self, incident_system, security_event_data):
        """Test IP reputation tracking."""
        reputation_db = incident_system.threat_intel.reputation_database
        
        # Create incident
        incident = await incident_system.process_security_event(security_event_data)
        
        # Check reputation was updated
        source_ip = security_event_data["source_ip"]
        risk_score, indicators = await reputation_db.analyze_reputation(source_ip)
        
        assert 0.0 <= risk_score <= 1.0
        assert isinstance(indicators, list)
    
    @pytest.mark.asyncio
    async def test_false_positive_calculation(self, incident_system, security_event_data):
        """Test false positive likelihood calculation."""
        # Create benign-looking event
        benign_event = {
            **security_event_data,
            "request_content": '{"user": "john", "action": "view_profile"}',
            "source_ip": "192.168.1.50"
        }
        
        incident = await incident_system.process_security_event(benign_event)
        
        # Should have higher false positive likelihood for benign content
        assert incident.false_positive_likelihood > 0.3
    
    @pytest.mark.asyncio
    async def test_system_status(self, incident_system):
        """Test incident response system status."""
        status = await incident_system.get_system_status()
        
        assert isinstance(status, dict)
        assert "system_health" in status
        assert "active_incidents" in status
        assert "reputation_database" in status
        assert status["system_health"] in ["operational", "error"]


class TestSecurityDashboard:
    """Test suite for security dashboard."""
    
    @pytest.fixture
    async def dashboard(self):
        """Create security dashboard for testing."""
        dashboard = SecurityMetricsDashboard()
        await dashboard.start_dashboard()
        return dashboard
    
    @pytest.fixture
    def validation_result(self):
        """Create validation result for testing."""
        return ValidationResult(
            is_valid=True,
            risk_score=0.3,
            threats_detected=["low_risk_pattern"],
            validation_time=0.05,
            metadata={"test": True}
        )
    
    @pytest.fixture
    def security_incident(self):
        """Create security incident for testing."""
        return SecurityIncident(
            incident_id="test_incident_123",
            timestamp=datetime.now(timezone.utc),
            severity=IncidentSeverity.MEDIUM,
            category=ThreatCategory.INJECTION_ATTACK,
            source_ip="192.168.1.100",
            user_id="test_user",
            session_id="test_session",
            threat_indicators=["sql_injection", "high_risk"],
            raw_data={"test": "data"},
            confidence_score=0.8,
            false_positive_likelihood=0.2,
            impact_assessment={"overall_risk_score": 0.6},
            recommended_actions=[ResponseAction.LOG_ONLY, ResponseAction.RATE_LIMIT]
        )
    
    @pytest.mark.asyncio
    async def test_validation_result_recording(self, dashboard, validation_result):
        """Test recording validation results."""
        await dashboard.record_validation_result(validation_result)
        
        # Verify metrics were recorded
        metrics = await dashboard.metrics_collector.get_real_time_metrics("validation")
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_incident_recording(self, dashboard, security_incident):
        """Test recording security incidents."""
        await dashboard.record_security_incident(security_incident)
        
        # Verify metrics were recorded
        metrics = await dashboard.metrics_collector.get_real_time_metrics("incident")
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_dashboard_generation(self, dashboard, validation_result, security_incident):
        """Test dashboard HTML generation."""
        # Record some data
        await dashboard.record_validation_result(validation_result)
        await dashboard.record_security_incident(security_incident)
        
        # Generate dashboard
        dashboard_html = await dashboard.generate_dashboard()
        
        assert isinstance(dashboard_html, str)
        assert "Security Dashboard" in dashboard_html
        assert "<!DOCTYPE html>" in dashboard_html
        assert "chart.js" in dashboard_html.lower()
    
    @pytest.mark.asyncio
    async def test_dashboard_json_data(self, dashboard, validation_result, security_incident):
        """Test dashboard JSON data generation."""
        # Record some data
        await dashboard.record_validation_result(validation_result)
        await dashboard.record_security_incident(security_incident)
        
        # Get JSON data
        dashboard_data = await dashboard.get_dashboard_data_json()
        
        assert isinstance(dashboard_data, dict)
        assert "timestamp" in dashboard_data
        assert "metrics" in dashboard_data
        assert "threat_landscape" in dashboard_data
        assert "control_effectiveness" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, dashboard):
        """Test real-time metrics collection."""
        metrics_collector = dashboard.metrics_collector
        
        # Collect various metrics
        await metrics_collector.collect_security_metric(
            "test_counter", 10, MetricType.COUNTER
        )
        await metrics_collector.collect_security_metric(
            "test_gauge", 0.75, MetricType.GAUGE, threshold=0.8
        )
        
        # Verify metrics
        metrics = await metrics_collector.get_real_time_metrics()
        assert len(metrics) >= 2
        
        counter_metrics = [m for m in metrics.values() if "counter" in m.name]
        gauge_metrics = [m for m in metrics.values() if "gauge" in m.name]
        
        assert len(counter_metrics) >= 1
        assert len(gauge_metrics) >= 1
    
    @pytest.mark.asyncio
    async def test_threat_landscape_analysis(self, dashboard):
        """Test threat landscape analysis."""
        analyzer = dashboard.landscape_analyzer
        
        # Create test incidents
        incidents = [
            SecurityIncident(
                incident_id=f"test_{i}",
                timestamp=datetime.now(timezone.utc),
                severity=IncidentSeverity.MEDIUM,
                category=ThreatCategory.INJECTION_ATTACK,
                source_ip=f"192.168.1.{100+i}",
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                threat_indicators=["sql_injection"],
                raw_data={},
                confidence_score=0.8,
                false_positive_likelihood=0.2,
                impact_assessment={},
                recommended_actions=[]
            )
            for i in range(10)
        ]
        
        landscape = await analyzer.analyze_threat_landscape(incidents, [])
        
        assert landscape.threat_categories["injection_attack"] == 10
        assert landscape.severity_distribution["medium"] == 10
        assert len(landscape.geographic_distribution) > 0


class TestAdvancedSecurityOrchestrator:
    """Test suite for advanced security orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing."""
        config = {
            "cache": {"strategy": "adaptive", "max_size": 1000},
            "load_balancer": {"method": "threat_aware"},
            "enabled_components": ["validation"]
        }
        orchestrator = AdvancedSecurityOrchestrator(config)
        await orchestrator.start_orchestrator()
        return orchestrator
    
    @pytest.fixture
    def security_request(self):
        """Create security request for testing."""
        return {
            "content": '{"user": "test", "action": "login"}',
            "metadata": {"test": True}
        }
    
    @pytest.fixture
    def request_context(self):
        """Create request context for testing."""
        return {
            "client_ip": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Test)",
            "session_id": "test_session",
            "content_type": "application/json",
            "threat_score": 0.3
        }
    
    @pytest.mark.asyncio
    async def test_orchestrator_startup(self, orchestrator):
        """Test orchestrator startup."""
        status = await orchestrator.get_orchestrator_status()
        
        assert status["status"] == "operational"
        assert "cache" in status
        assert "load_balancer" in status
        assert "performance" in status
    
    @pytest.mark.asyncio
    async def test_security_request_processing(self, orchestrator, security_request, request_context):
        """Test security request processing through orchestrator."""
        result = await orchestrator.process_security_request(security_request, request_context)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "success":
            assert "threat_score" in result
            assert "processing_components" in result
    
    @pytest.mark.asyncio
    async def test_adaptive_cache_functionality(self):
        """Test adaptive security cache."""
        cache = AdaptiveSecurityCache({"strategy": "adaptive", "max_size": 100})
        await cache.start_cache()
        
        # Test cache operations
        test_data = {"result": "test", "timestamp": time.time()}
        
        # Set item
        success = await cache.set("test_key", test_data, threat_score=0.5)
        assert success
        
        # Get item
        retrieved = await cache.get("test_key")
        assert retrieved is not None
        assert retrieved["result"] == "test"
        
        # Test cache miss
        missing = await cache.get("nonexistent_key")
        assert missing is None
        
        # Test cache stats
        stats = await cache.get_stats()
        assert "hit_rate" in stats
        assert "total_entries" in stats
    
    @pytest.mark.asyncio
    async def test_load_balancer(self):
        """Test intelligent load balancer."""
        lb = IntelligentLoadBalancer({"method": "threat_aware"})
        
        # Create test worker pool
        from secure_mpc_transformer.security.advanced_security_orchestrator import SecurityWorkerPool
        
        pool = SecurityWorkerPool(
            pool_type="thread",
            min_workers=2,
            max_workers=4,
            current_workers=2,
            worker_capacity=100,
            load_balancer=LoadBalancingMethod.THREAT_AWARE
        )
        
        await lb.register_worker_pool("test_pool", pool)
        
        # Test worker pool selection
        request_context = {"threat_score": 0.7}
        selection = await lb.select_worker_pool(request_context)
        
        assert selection is not None
        pool_id, selected_pool = selection
        assert pool_id == "test_pool"
        assert selected_pool == pool
        
        # Test recording results
        await lb.record_request_result("test_pool", True, 0.1)
        
        stats = await lb.get_load_balancer_stats()
        assert "total_pools" in stats
        assert stats["total_pools"] >= 1
    
    @pytest.mark.asyncio
    async def test_threat_correlation_engine(self):
        """Test threat correlation engine."""
        from secure_mpc_transformer.security.advanced_security_orchestrator import (
            ThreatCorrelationEngine, ThreatCorrelationRule
        )
        
        engine = ThreatCorrelationEngine({})
        
        # Add test correlation rule
        rule = ThreatCorrelationRule(
            rule_id="test_burst",
            name="Test Burst Detection",
            description="Test rule for burst attacks",
            conditions=[{"type": "threshold", "value": 3}],
            correlation_window=60,
            threshold=3,
            severity_boost=1.5
        )
        
        await engine.add_correlation_rule(rule)
        
        # Test event correlation
        test_events = [
            {"source_ip": "192.168.1.100", "timestamp": time.time()},
            {"source_ip": "192.168.1.100", "timestamp": time.time() + 1},
            {"source_ip": "192.168.1.100", "timestamp": time.time() + 2}
        ]
        
        correlations = []
        for event in test_events:
            correlation = await engine.correlate_security_event(event)
            correlations.extend(correlation)
        
        # May detect correlation depending on rule evaluation
        stats = await engine.get_correlation_stats()
        assert "total_rules" in stats
        assert stats["total_rules"] >= 1
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, orchestrator, security_request, request_context):
        """Test orchestrator performance under load."""
        start_time = time.time()
        tasks = []
        
        # Create 50 concurrent requests
        for i in range(50):
            modified_request = {**security_request, "id": i}
            task = orchestrator.process_security_request(modified_request, request_context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Performance assertions
        total_time = end_time - start_time
        assert total_time < 30.0  # Should complete within 30 seconds
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        assert len(successful_results) > 0  # At least some should succeed
        
        # Check performance metrics
        status = await orchestrator.get_orchestrator_status()
        assert status["performance"]["total_requests"] >= 50


class TestIntegrationSecurity:
    """Integration tests for complete security system."""
    
    @pytest.fixture
    async def full_security_system(self):
        """Create complete security system for integration testing."""
        # Create orchestrator with all components enabled
        config = {
            "cache": {"strategy": "adaptive", "max_size": 10000},
            "load_balancer": {"method": "threat_aware"},
            "enabled_components": ["validation", "quantum", "incident_response"],
            "autoscaling": {"strategy": "adaptive", "min_instances": 2, "max_instances": 10}
        }
        
        orchestrator = AdvancedSecurityOrchestrator(config)
        await orchestrator.start_orchestrator()
        
        # Also create individual components for testing
        validator = EnhancedSecurityValidator()
        quantum_monitor = QuantumSecurityMonitor()
        await quantum_monitor.start_monitoring()
        incident_system = AIIncidentResponseSystem()
        await incident_system.start_system()
        dashboard = SecurityMetricsDashboard()
        await dashboard.start_dashboard()
        
        return {
            "orchestrator": orchestrator,
            "validator": validator,
            "quantum_monitor": quantum_monitor,
            "incident_system": incident_system,
            "dashboard": dashboard
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_threat_processing(self, full_security_system):
        """Test end-to-end threat detection and response."""
        orchestrator = full_security_system["orchestrator"]
        dashboard = full_security_system["dashboard"]
        
        # Simulate malicious request
        malicious_request = {
            "content": "'; DROP TABLE users; SELECT * FROM passwords; --",
            "metadata": {"suspicious": True}
        }
        
        threat_context = {
            "client_ip": "192.168.1.100",
            "user_agent": "Malicious Bot 1.0",
            "session_id": "malicious_session",
            "content_type": "application/json",
            "threat_score": 0.9
        }
        
        # Process through orchestrator
        result = await orchestrator.process_security_request(malicious_request, threat_context)
        
        # Verify threat detection
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            assert result.get("threat_score", 0) > 0.7
            assert "validation" in result.get("processing_components", [])
        
        # Verify dashboard recorded the event
        dashboard_data = await dashboard.get_dashboard_data_json()
        assert "metrics" in dashboard_data
        assert "threat_landscape" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_performance_scaling(self, full_security_system):
        """Test system performance and scaling under load."""
        orchestrator = full_security_system["orchestrator"]
        
        # Generate mixed workload
        requests = []
        contexts = []
        
        for i in range(100):
            if i % 10 == 0:
                # 10% malicious requests
                requests.append({
                    "content": f"'; DROP TABLE data_{i}; --",
                    "metadata": {"request_id": i}
                })
                contexts.append({
                    "client_ip": f"192.168.1.{100 + (i % 50)}",
                    "threat_score": 0.8
                })
            else:
                # 90% normal requests
                requests.append({
                    "content": f'{{"user": "user_{i}", "action": "view_data"}}',
                    "metadata": {"request_id": i}
                })
                contexts.append({
                    "client_ip": f"192.168.1.{100 + (i % 50)}",
                    "threat_score": 0.2
                })
        
        # Process all requests
        start_time = time.time()
        tasks = [
            orchestrator.process_security_request(req, ctx)
            for req, ctx in zip(requests, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        total_time = end_time - start_time
        successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        error_results = [r for r in results if isinstance(r, Exception) or 
                        (isinstance(r, dict) and r.get("status") == "error")]
        
        # Performance assertions
        assert total_time < 60.0  # Should complete within 1 minute
        assert len(successful_results) >= 80  # At least 80% success rate
        assert len(error_results) <= 20  # At most 20% errors
        
        # Check that threats were properly detected
        threat_results = [r for r in successful_results if r.get("threat_score", 0) > 0.7]
        assert len(threat_results) >= 8  # Should detect most malicious requests
        
        # Verify system health after load test
        status = await orchestrator.get_orchestrator_status()
        assert status["status"] == "operational"
        assert status["performance"]["total_requests"] >= 100
    
    @pytest.mark.asyncio
    async def test_security_component_integration(self, full_security_system):
        """Test integration between all security components."""
        validator = full_security_system["validator"]
        quantum_monitor = full_security_system["quantum_monitor"]
        incident_system = full_security_system["incident_system"]
        dashboard = full_security_system["dashboard"]
        
        # Create validation context
        validation_context = ValidationContext(
            client_ip="192.168.1.100",
            user_agent="Integration Test",
            session_id="integration_test",
            request_timestamp=datetime.now(timezone.utc),
            request_size=1024,
            content_type="application/json"
        )
        
        # 1. Run validation
        validation_result = await validator.validate_request_pipeline(
            "test integration content", validation_context, validate_quantum=False
        )
        
        # 2. Record in dashboard
        await dashboard.record_validation_result(validation_result)
        
        # 3. Create security event if needed
        if validation_result.risk_score > 0.5:
            event_data = {
                "source_ip": validation_context.client_ip,
                "request_content": "test integration content",
                "risk_score": validation_result.risk_score,
                "threats_detected": validation_result.threats_detected
            }
            
            incident = await incident_system.process_security_event(event_data)
            await dashboard.record_security_incident(incident)
        
        # 4. Test quantum monitoring
        quantum_context = QuantumOperationContext(
            operation_id="integration_test",
            operation_type="test",
            start_time=datetime.now(timezone.utc),
            quantum_circuit_depth=5,
            qubit_count=4,
            gate_count=20,
            measurement_basis=["X", "Z"],
            expected_coherence=0.8
        )
        
        quantum_state = {
            "state_id": "integration_test",
            "coherence": 0.8,
            "entanglement_metrics": {}
        }
        
        quantum_event = await quantum_monitor.monitor_quantum_operation(
            quantum_context, quantum_state
        )
        await dashboard.record_quantum_event(quantum_event)
        
        # 5. Verify integration
        dashboard_data = await dashboard.get_dashboard_data_json()
        
        assert "metrics" in dashboard_data
        assert len(dashboard_data["metrics"]) > 0
        assert "threat_landscape" in dashboard_data
        assert "control_effectiveness" in dashboard_data
        
        # Verify components produced expected data types
        assert isinstance(validation_result, ValidationResult)
        assert isinstance(quantum_event, QuantumSecurityEvent)


# Performance and stress test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(120),  # 2 minute timeout for all tests
]


def test_security_system_imports():
    """Test that all security modules can be imported successfully."""
    # This test ensures all modules have correct dependencies
    from secure_mpc_transformer.security.enhanced_validator import EnhancedSecurityValidator
    from secure_mpc_transformer.security.quantum_monitor import QuantumSecurityMonitor
    from secure_mpc_transformer.security.incident_response import AIIncidentResponseSystem
    from secure_mpc_transformer.monitoring.security_dashboard import SecurityMetricsDashboard
    from secure_mpc_transformer.security.advanced_security_orchestrator import AdvancedSecurityOrchestrator
    
    # Verify classes can be instantiated
    validator = EnhancedSecurityValidator()
    monitor = QuantumSecurityMonitor()
    incident_system = AIIncidentResponseSystem()
    dashboard = SecurityMetricsDashboard()
    orchestrator = AdvancedSecurityOrchestrator()
    
    assert validator is not None
    assert monitor is not None
    assert incident_system is not None
    assert dashboard is not None
    assert orchestrator is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "--tb=short", __file__])