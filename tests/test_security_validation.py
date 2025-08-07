"""
Security Validation Tests

Comprehensive test suite for quantum planning security analysis,
threat modeling, and vulnerability assessment.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.secure_mpc_transformer.planning.security import (
    QuantumSecurityAnalyzer,
    SecurityThreat,
    ThreatLevel,
    AttackVector,
    SecurityProperty,
    SecurityAuditResult,
    QuantumSecurityMetrics
)


class TestQuantumSecurityAnalyzer:
    """Test suite for QuantumSecurityAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create test security analyzer"""
        return QuantumSecurityAnalyzer(security_level=128)
    
    @pytest.fixture
    def sample_quantum_state(self):
        """Create sample quantum state for testing"""
        state = np.array([0.6+0.8j, 0.0+0.0j], dtype=complex)
        return state / np.linalg.norm(state)
    
    def test_analyzer_initialization(self):
        """Test security analyzer initialization"""
        analyzer = QuantumSecurityAnalyzer(security_level=256)
        
        assert analyzer.security_level == 256
        assert len(analyzer.threat_database) > 0
        assert isinstance(analyzer.audit_log, list)
        assert isinstance(analyzer.timing_measurements, dict)
        
        # Check that threat database is populated
        threat_ids = [threat.threat_id for threat in analyzer.threat_database]
        expected_threats = ["QP-001", "QP-002", "QP-003", "QP-004", "QP-005", "QP-006"]
        for expected_id in expected_threats:
            assert expected_id in threat_ids
    
    def test_threat_database_structure(self, analyzer):
        """Test structure of threat database"""
        for threat in analyzer.threat_database:
            assert isinstance(threat, SecurityThreat)
            assert threat.threat_id.startswith("QP-")
            assert isinstance(threat.name, str)
            assert isinstance(threat.description, str)
            assert isinstance(threat.attack_vector, AttackVector)
            assert isinstance(threat.threat_level, ThreatLevel)
            assert isinstance(threat.affected_components, list)
            assert isinstance(threat.mitigation_strategies, list)
            
            # CVSS scores should be reasonable if present
            if threat.cvss_score is not None:
                assert 0.0 <= threat.cvss_score <= 10.0
    
    def test_analyze_quantum_state_security(self, analyzer, sample_quantum_state):
        """Test quantum state security analysis"""
        metrics = analyzer.analyze_quantum_state_security(
            sample_quantum_state, 
            "test_operation"
        )
        
        assert isinstance(metrics, QuantumSecurityMetrics)
        assert isinstance(metrics.state_entropy, float)
        assert isinstance(metrics.information_leakage, float)
        assert isinstance(metrics.timing_variance, float)
        assert isinstance(metrics.resource_utilization, dict)
        assert isinstance(metrics.quantum_coherence_stability, float)
        
        # Entropy should be reasonable for a 2-state system
        assert 0.0 <= metrics.state_entropy <= 1.0  # Max entropy for 2-state system is 1
        
        # Information leakage should be 0-1 range
        assert 0.0 <= metrics.information_leakage <= 1.0
        
        # Coherence stability should be 0-1 range
        assert 0.0 <= metrics.quantum_coherence_stability <= 1.0
        
        # Resource utilization should have expected keys
        expected_resources = ["cpu", "memory", "gpu"]
        for resource in expected_resources:
            assert resource in metrics.resource_utilization
            assert 0.0 <= metrics.resource_utilization[resource] <= 1.0
    
    def test_information_leakage_calculation(self, analyzer):
        """Test information leakage calculation"""
        # Test different quantum states for leakage assessment
        
        # Uniform superposition (low leakage)
        uniform_state = np.array([0.5+0.5j, 0.5+0.5j], dtype=complex)
        uniform_state = uniform_state / np.linalg.norm(uniform_state)
        
        # Concentrated state (potentially high leakage)
        concentrated_state = np.array([0.95+0.31j, 0.0+0.0j], dtype=complex)
        concentrated_state = concentrated_state / np.linalg.norm(concentrated_state)
        
        uniform_leakage = analyzer._calculate_information_leakage(uniform_state)
        concentrated_leakage = analyzer._calculate_information_leakage(concentrated_state)
        
        # Concentrated state should have higher leakage potential
        assert concentrated_leakage >= uniform_leakage
        assert 0.0 <= uniform_leakage <= 1.0
        assert 0.0 <= concentrated_leakage <= 1.0
    
    def test_coherence_stability_assessment(self, analyzer):
        """Test quantum coherence stability assessment"""
        # Test different coherence levels
        
        # Pure state (maximum coherence)
        pure_state = np.array([1.0+0.0j, 0.0+0.0j], dtype=complex)
        
        # Mixed state (balanced coherence)
        mixed_state = np.array([0.707+0.0j, 0.707+0.0j], dtype=complex)
        
        pure_stability = analyzer._assess_coherence_stability(pure_state)
        mixed_stability = analyzer._assess_coherence_stability(mixed_state)
        
        assert 0.0 <= pure_stability <= 1.0
        assert 0.0 <= mixed_stability <= 1.0
        
        # Mixed state should have better stability (closer to optimal coherence)
        assert mixed_stability >= pure_stability * 0.5  # Allow some tolerance
    
    def test_component_security_audit(self, analyzer):
        """Test security audit of components"""
        component_data = {
            "encrypted": True,
            "access_control": True,
            "integrity_checks": False,
            "audit_logging": True,
            "rate_limiting": False,
            "authentication": True,
            "authorization": False
        }
        
        audit = analyzer.audit_component_security("quantum_planner", component_data)
        
        assert isinstance(audit, SecurityAuditResult)
        assert audit.component == "quantum_planner"
        assert isinstance(audit.security_properties, dict)
        assert isinstance(audit.threats_identified, list)
        assert isinstance(audit.vulnerabilities, list)
        assert isinstance(audit.recommendations, list)
        assert isinstance(audit.risk_score, float)
        
        # Risk score should be reasonable
        assert 0.0 <= audit.risk_score <= 10.0
        
        # Should have identified some threats for quantum_planner
        assert len(audit.threats_identified) > 0
        
        # Check security properties assessment
        assert SecurityProperty.CONFIDENTIALITY in audit.security_properties
        assert audit.security_properties[SecurityProperty.CONFIDENTIALITY] == True  # Both encrypted and access_control are True
    
    def test_security_properties_assessment(self, analyzer):
        """Test assessment of security properties"""
        # Test different configurations
        configurations = [
            {
                "encrypted": True,
                "access_control": True,
                "integrity_checks": True,
                "audit_logging": True,
                "rate_limiting": True,
                "authentication": True,
                "authorization": True,
                "key_rotation": True,
                "ephemeral_keys": True,
                "post_quantum_cryptography": True,
                "quantum_safe": True,
                "digital_signatures": True,
                "timestamping": True,
                "redundancy": True
            },
            {
                "encrypted": False,
                "access_control": False,
                "integrity_checks": False,
                "audit_logging": False,
                "rate_limiting": False
            }
        ]
        
        for i, config in enumerate(configurations):
            properties = analyzer._assess_security_properties(f"test_component_{i}", config)
            
            assert isinstance(properties, dict)
            assert len(properties) == len(SecurityProperty)
            
            for prop, value in properties.items():
                assert isinstance(prop, SecurityProperty)
                assert isinstance(value, bool)
            
            if i == 0:  # Secure configuration
                # Should have most properties satisfied
                satisfied_count = sum(properties.values())
                assert satisfied_count >= len(SecurityProperty) * 0.5
            
            if i == 1:  # Insecure configuration
                # Should have fewer properties satisfied
                satisfied_count = sum(properties.values())
                assert satisfied_count < len(SecurityProperty) * 0.5
    
    def test_threat_identification(self, analyzer):
        """Test threat identification for components"""
        # Test different component types
        components = [
            "quantum_planner",
            "quantum_optimizer", 
            "scheduler",
            "cache_manager",
            "concurrent_executor"
        ]
        
        for component in components:
            threats = analyzer._identify_component_threats(component)
            
            assert isinstance(threats, list)
            assert all(isinstance(threat, SecurityThreat) for threat in threats)
            
            # Should identify relevant threats
            if "quantum" in component:
                quantum_threats = [t for t in threats if "quantum" in t.name.lower() or "quantum" in t.description.lower()]
                assert len(quantum_threats) > 0
            
            if "cache" in component:
                cache_threats = [t for t in threats if "cache" in t.name.lower() or "cache" in t.description.lower()]
                assert len(cache_threats) > 0
    
    def test_vulnerability_scanning(self, analyzer):
        """Test vulnerability scanning"""
        test_cases = [
            {
                "component": "quantum_cache",
                "data": {"cache_isolation": False, "cache_encryption": False},
                "expected_vulnerabilities": 2
            },
            {
                "component": "quantum_state_manager",
                "data": {"state_verification": False, "decoherence_protection": False},
                "expected_vulnerabilities": 2
            },
            {
                "component": "scheduler",
                "data": {"resource_limits": False, "schedule_randomization": False},
                "expected_vulnerabilities": 2
            },
            {
                "component": "optimizer",
                "data": {"constant_time": False, "convergence_obfuscation": False},
                "expected_vulnerabilities": 2
            }
        ]
        
        for test_case in test_cases:
            vulnerabilities = analyzer._scan_vulnerabilities(
                test_case["component"], 
                test_case["data"]
            )
            
            assert isinstance(vulnerabilities, list)
            assert len(vulnerabilities) >= test_case["expected_vulnerabilities"]
            assert all(isinstance(vuln, str) for vuln in vulnerabilities)
    
    def test_risk_score_calculation(self, analyzer):
        """Test risk score calculation"""
        # Create test threats with different levels
        threats = [
            SecurityThreat(
                threat_id="TEST-001",
                name="Test Critical",
                description="Test critical threat",
                attack_vector=AttackVector.QUANTUM_STATE_MANIPULATION,
                threat_level=ThreatLevel.CRITICAL,
                affected_components=["test"],
                cvss_score=9.5
            ),
            SecurityThreat(
                threat_id="TEST-002", 
                name="Test High",
                description="Test high threat",
                attack_vector=AttackVector.SIDE_CHANNEL,
                threat_level=ThreatLevel.HIGH,
                affected_components=["test"],
                cvss_score=7.2
            )
        ]
        
        vulnerabilities = ["Test vulnerability 1", "Test vulnerability 2"]
        
        risk_score = analyzer._calculate_risk_score(threats, vulnerabilities)
        
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 10.0
        assert risk_score > 0  # Should have some risk with threats and vulnerabilities
    
    def test_timing_attack_detection(self, analyzer):
        """Test timing attack detection"""
        operation_name = "test_operation"
        
        # Simulate normal timing measurements
        normal_timings = np.random.normal(0.1, 0.01, 20)  # Mean 0.1s, std 0.01s
        for timing in normal_timings:
            analyzer.timing_measurements[operation_name].append(abs(timing))
        
        # Add some outliers (potential attacks)
        analyzer.timing_measurements[operation_name].extend([0.5, 0.6])  # Significant outliers
        
        result = analyzer.detect_timing_attacks(operation_name, threshold_factor=2.0)
        
        assert isinstance(result, dict)
        assert result["status"] == "analyzed"
        assert result["operation"] == operation_name
        assert result["sample_count"] > 0
        assert "mean_time" in result
        assert "std_time" in result
        assert "outliers" in result
        assert "risk_level" in result
        assert "recommendations" in result
        
        # Should detect outliers
        assert len(result["outliers"]) > 0
        assert result["outlier_rate"] > 0
    
    def test_timing_pattern_analysis(self, analyzer):
        """Test timing pattern analysis"""
        # Test periodic pattern
        periodic_timings = [0.1 + 0.05 * np.sin(2 * np.pi * i / 10) for i in range(50)]
        patterns = analyzer._analyze_timing_patterns(periodic_timings)
        
        assert isinstance(patterns, dict)
        assert "periodicity_detected" in patterns
        assert "trend_detected" in patterns
        assert "clustering_detected" in patterns
        
        # Test trending pattern
        trending_timings = [0.1 + 0.01 * i for i in range(50)]  # Linear trend
        patterns = analyzer._analyze_timing_patterns(trending_timings)
        
        # Should detect trend
        assert patterns["trend_detected"] == True
    
    def test_security_report_generation(self, analyzer):
        """Test comprehensive security report generation"""
        components = [
            "quantum_planner",
            "quantum_optimizer",
            "scheduler",
            "cache_manager"
        ]
        
        report = analyzer.generate_security_report(components)
        
        assert isinstance(report, dict)
        assert "report_timestamp" in report
        assert "components_analyzed" in report
        assert "overall_risk_score" in report
        assert "executive_summary" in report
        assert "component_audits" in report
        assert "threat_summary" in report
        assert "vulnerability_summary" in report
        assert "recommendations" in report
        assert "compliance_status" in report
        assert "next_audit_date" in report
        
        # Validate structure
        assert report["components_analyzed"] == len(components)
        assert isinstance(report["overall_risk_score"], float)
        assert 0.0 <= report["overall_risk_score"] <= 10.0
        
        # Threat summary should have counts
        threat_summary = report["threat_summary"]
        assert "total_threats" in threat_summary
        assert "critical_threats" in threat_summary
        assert "high_threats" in threat_summary
        
        # Recommendations should be prioritized
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        if recommendations:
            assert all("recommendation" in rec for rec in recommendations)
            assert all("priority" in rec for rec in recommendations)
    
    def test_executive_summary_generation(self, analyzer):
        """Test executive summary generation"""
        # Create mock threats
        threats = [
            SecurityThreat(
                threat_id="EXEC-001",
                name="Critical Test Threat",
                description="Test critical threat description",
                attack_vector=AttackVector.QUANTUM_STATE_MANIPULATION,
                threat_level=ThreatLevel.CRITICAL,
                affected_components=["test"]
            ),
            SecurityThreat(
                threat_id="EXEC-002",
                name="High Test Threat", 
                description="Test high threat description",
                attack_vector=AttackVector.TIMING_ATTACK,
                threat_level=ThreatLevel.HIGH,
                affected_components=["test"]
            )
        ]
        
        vulnerabilities = ["Test vulnerability 1", "Test vulnerability 2"]
        
        summary = analyzer._generate_executive_summary(7.5, threats, vulnerabilities)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "HIGH" in summary  # Should indicate risk level
        assert "CRITICAL" in summary  # Should mention critical threats
        assert str(len(threats)) in summary  # Should mention threat count
        assert str(len(vulnerabilities)) in summary  # Should mention vulnerability count
    
    def test_recommendation_prioritization(self, analyzer):
        """Test security recommendation prioritization"""
        # Create mock audit results
        audits = [
            SecurityAuditResult(
                component="high_risk_component",
                security_properties={},
                threats_identified=[],
                vulnerabilities=[],
                recommendations=[
                    "Critical fix needed immediately",
                    "Implement enhanced security",
                    "Regular monitoring recommended"
                ],
                risk_score=8.5
            ),
            SecurityAuditResult(
                component="low_risk_component",
                security_properties={},
                threats_identified=[],
                vulnerabilities=[],
                recommendations=[
                    "Consider minor improvements",
                    "Add logging capability"
                ],
                risk_score=2.1
            )
        ]
        
        prioritized = analyzer._prioritize_recommendations(audits)
        
        assert isinstance(prioritized, list)
        assert all(isinstance(rec, dict) for rec in prioritized)
        assert all("recommendation" in rec for rec in prioritized)
        assert all("priority" in rec for rec in prioritized)
        assert all("component" in rec for rec in prioritized)
        
        # Should be sorted by priority (descending)
        if len(prioritized) > 1:
            for i in range(len(prioritized) - 1):
                assert prioritized[i]["priority"] >= prioritized[i + 1]["priority"]
    
    def test_compliance_assessment(self, analyzer):
        """Test compliance status assessment"""
        mock_audits = []  # Empty for this test
        
        compliance = analyzer._assess_compliance_status(mock_audits)
        
        assert isinstance(compliance, dict)
        assert "frameworks" in compliance
        assert "average_compliance" in compliance
        assert "compliance_gaps" in compliance
        
        # Should have common compliance frameworks
        frameworks = compliance["frameworks"]
        expected_frameworks = ["ISO 27001", "NIST Cybersecurity Framework", "SOC 2", "GDPR"]
        for framework in expected_frameworks:
            assert framework in frameworks
            assert 0.0 <= frameworks[framework] <= 1.0
        
        # Average should be reasonable
        assert 0.0 <= compliance["average_compliance"] <= 1.0
    
    def test_security_metrics(self, analyzer, sample_quantum_state):
        """Test security metrics collection"""
        # Generate some activity
        analyzer.analyze_quantum_state_security(sample_quantum_state, "test_op_1")
        analyzer.analyze_quantum_state_security(sample_quantum_state, "test_op_2")
        
        component_data = {"encrypted": True, "access_control": False}
        analyzer.audit_component_security("test_component", component_data)
        
        metrics = analyzer.get_security_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_audits_performed" in metrics
        assert "average_risk_score" in metrics
        assert "timing_measurements_collected" in metrics
        assert "unique_operations_monitored" in metrics
        assert "security_events_logged" in metrics
        
        # Should reflect the activity we generated
        assert metrics["total_audits_performed"] > 0
        assert metrics["timing_measurements_collected"] > 0
        assert metrics["unique_operations_monitored"] >= 2
    
    def test_insufficient_timing_data(self, analyzer):
        """Test timing attack detection with insufficient data"""
        operation_name = "sparse_operation"
        
        # Add very few measurements
        analyzer.timing_measurements[operation_name] = [0.1, 0.2]
        
        result = analyzer.detect_timing_attacks(operation_name)
        
        assert result["status"] == "insufficient_data"
        assert result["sample_count"] < 10
    
    def test_no_timing_data(self, analyzer):
        """Test timing attack detection with no data"""
        result = analyzer.detect_timing_attacks("non_existent_operation")
        
        assert result["status"] == "no_data"
    
    def test_security_event_logging(self, analyzer):
        """Test security event logging"""
        initial_log_count = len(analyzer.audit_log)
        
        # Trigger events that should be logged
        sample_state = np.array([1.0+0j], dtype=complex)
        analyzer.analyze_quantum_state_security(sample_state, "logging_test")
        
        component_data = {"encrypted": False}
        analyzer.audit_component_security("test_logging", component_data)
        
        # Should have logged events
        final_log_count = len(analyzer.audit_log)
        assert final_log_count > initial_log_count


class TestSecurityThreatModeling:
    """Test security threat modeling components"""
    
    def test_security_threat_creation(self):
        """Test SecurityThreat creation and validation"""
        threat = SecurityThreat(
            threat_id="TEST-001",
            name="Test Threat",
            description="A test threat for validation",
            attack_vector=AttackVector.SIDE_CHANNEL,
            threat_level=ThreatLevel.HIGH,
            affected_components=["component1", "component2"],
            mitigation_strategies=["Strategy 1", "Strategy 2"],
            cvss_score=7.5,
            references=["https://example.com/threat-info"]
        )
        
        assert threat.threat_id == "TEST-001"
        assert threat.name == "Test Threat"
        assert threat.attack_vector == AttackVector.SIDE_CHANNEL
        assert threat.threat_level == ThreatLevel.HIGH
        assert len(threat.affected_components) == 2
        assert len(threat.mitigation_strategies) == 2
        assert threat.cvss_score == 7.5
    
    def test_attack_vector_enum(self):
        """Test AttackVector enum completeness"""
        expected_vectors = [
            "SIDE_CHANNEL",
            "TIMING_ATTACK", 
            "QUANTUM_STATE_MANIPULATION",
            "RESOURCE_EXHAUSTION",
            "INFORMATION_LEAKAGE",
            "PROTOCOL_BYPASS",
            "CACHE_POISONING",
            "METADATA_INFERENCE"
        ]
        
        for vector_name in expected_vectors:
            assert hasattr(AttackVector, vector_name)
            vector = getattr(AttackVector, vector_name)
            assert isinstance(vector, AttackVector)
    
    def test_threat_level_enum(self):
        """Test ThreatLevel enum completeness"""
        expected_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        for level_name in expected_levels:
            assert hasattr(ThreatLevel, level_name)
            level = getattr(ThreatLevel, level_name)
            assert isinstance(level, ThreatLevel)
    
    def test_security_property_enum(self):
        """Test SecurityProperty enum completeness"""
        expected_properties = [
            "CONFIDENTIALITY",
            "INTEGRITY", 
            "AVAILABILITY",
            "AUTHENTICITY",
            "NON_REPUDIATION",
            "FORWARD_SECRECY",
            "QUANTUM_RESISTANCE"
        ]
        
        for prop_name in expected_properties:
            assert hasattr(SecurityProperty, prop_name)
            prop = getattr(SecurityProperty, prop_name)
            assert isinstance(prop, SecurityProperty)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])