"""
Security Analysis and Validation for Quantum Planning

Comprehensive security analysis, threat modeling, and validation
for quantum-inspired task planning in secure MPC environments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import hashlib
import hmac
import secrets
import time
import logging
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackVector(Enum):
    """Potential attack vectors"""
    SIDE_CHANNEL = "side_channel"
    TIMING_ATTACK = "timing_attack"
    QUANTUM_STATE_MANIPULATION = "quantum_state_manipulation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INFORMATION_LEAKAGE = "information_leakage"
    PROTOCOL_BYPASS = "protocol_bypass"
    CACHE_POISONING = "cache_poisoning"
    METADATA_INFERENCE = "metadata_inference"


class SecurityProperty(Enum):
    """Security properties to validate"""
    CONFIDENTIALITY = "confidentiality"
    INTEGRITY = "integrity"
    AVAILABILITY = "availability"
    AUTHENTICITY = "authenticity"
    NON_REPUDIATION = "non_repudiation"
    FORWARD_SECRECY = "forward_secrecy"
    QUANTUM_RESISTANCE = "quantum_resistance"


@dataclass
class SecurityThreat:
    """Security threat model"""
    threat_id: str
    name: str
    description: str
    attack_vector: AttackVector
    threat_level: ThreatLevel
    affected_components: List[str]
    mitigation_strategies: List[str] = field(default_factory=list)
    cvss_score: Optional[float] = None
    references: List[str] = field(default_factory=list)


@dataclass
class SecurityAuditResult:
    """Result of security audit"""
    component: str
    security_properties: Dict[SecurityProperty, bool]
    threats_identified: List[SecurityThreat]
    vulnerabilities: List[str]
    recommendations: List[str]
    audit_timestamp: datetime = field(default_factory=datetime.now)
    risk_score: float = 0.0


@dataclass
class QuantumSecurityMetrics:
    """Security metrics for quantum operations"""
    state_entropy: float
    information_leakage: float
    timing_variance: float
    resource_utilization: Dict[str, float]
    cache_hit_pattern: List[float]
    error_rate: float
    quantum_coherence_stability: float


class QuantumSecurityAnalyzer:
    """
    Security analyzer for quantum-inspired task planning operations.
    Detects potential security vulnerabilities and information leakage.
    """
    
    def __init__(self, security_level: int = 128):
        self.security_level = security_level
        self.threat_database = self._initialize_threat_database()
        self.audit_log: List[Dict[str, Any]] = []
        
        # Security monitoring
        self.timing_measurements: Dict[str, List[float]] = defaultdict(list)
        self.resource_usage_history: List[Dict[str, float]] = []
        self.cache_access_patterns: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)
        
        logger.info(f"Initialized QuantumSecurityAnalyzer with {security_level}-bit security level")
    
    def _initialize_threat_database(self) -> List[SecurityThreat]:
        """Initialize database of known security threats"""
        return [
            SecurityThreat(
                threat_id="QP-001",
                name="Quantum State Side-Channel Attack",
                description="Adversary infers information from quantum state measurements or transitions",
                attack_vector=AttackVector.SIDE_CHANNEL,
                threat_level=ThreatLevel.HIGH,
                affected_components=["quantum_planner", "quantum_optimizer"],
                mitigation_strategies=[
                    "Implement state blinding techniques",
                    "Add random noise to quantum operations",
                    "Use constant-time quantum operations"
                ],
                cvss_score=7.5
            ),
            SecurityThreat(
                threat_id="QP-002", 
                name="Timing Attack on Optimization",
                description="Adversary infers task structure from optimization timing patterns",
                attack_vector=AttackVector.TIMING_ATTACK,
                threat_level=ThreatLevel.MEDIUM,
                affected_components=["quantum_optimizer", "scheduler"],
                mitigation_strategies=[
                    "Implement constant-time optimization",
                    "Add random delays",
                    "Use timing obfuscation techniques"
                ],
                cvss_score=5.3
            ),
            SecurityThreat(
                threat_id="QP-003",
                name="Cache-Based Information Leakage",
                description="Adversary infers sensitive information from cache access patterns",
                attack_vector=AttackVector.INFORMATION_LEAKAGE,
                threat_level=ThreatLevel.HIGH,
                affected_components=["quantum_state_cache", "optimization_cache"],
                mitigation_strategies=[
                    "Implement oblivious cache access patterns",
                    "Use cache partitioning",
                    "Add dummy cache operations"
                ],
                cvss_score=6.8
            ),
            SecurityThreat(
                threat_id="QP-004",
                name="Resource Exhaustion Attack",
                description="Adversary causes denial of service through resource exhaustion",
                attack_vector=AttackVector.RESOURCE_EXHAUSTION,
                threat_level=ThreatLevel.MEDIUM,
                affected_components=["concurrent_executor", "scheduler"],
                mitigation_strategies=[
                    "Implement resource quotas and limits",
                    "Add rate limiting",
                    "Monitor resource usage patterns"
                ],
                cvss_score=4.9
            ),
            SecurityThreat(
                threat_id="QP-005",
                name="Quantum State Manipulation",
                description="Adversary manipulates quantum states to influence computation results",
                attack_vector=AttackVector.QUANTUM_STATE_MANIPULATION,
                threat_level=ThreatLevel.CRITICAL,
                affected_components=["quantum_planner", "quantum_validator"],
                mitigation_strategies=[
                    "Implement quantum state verification",
                    "Use cryptographic commitments to quantum states",
                    "Add integrity checks for quantum operations"
                ],
                cvss_score=9.1
            ),
            SecurityThreat(
                threat_id="QP-006",
                name="Metadata Inference Attack", 
                description="Adversary infers sensitive metadata from task scheduling patterns",
                attack_vector=AttackVector.METADATA_INFERENCE,
                threat_level=ThreatLevel.MEDIUM,
                affected_components=["scheduler", "task_planner"],
                mitigation_strategies=[
                    "Implement schedule randomization",
                    "Use padding tasks to obscure patterns",
                    "Add decoy scheduling operations"
                ],
                cvss_score=5.7
            )
        ]
    
    def analyze_quantum_state_security(self, quantum_state: np.ndarray, operation_context: str) -> QuantumSecurityMetrics:
        """
        Analyze security properties of quantum state operations.
        
        Args:
            quantum_state: Quantum state to analyze
            operation_context: Context of the operation
            
        Returns:
            Security metrics for the quantum state
        """
        start_time = time.time()
        
        # Calculate state entropy
        amplitudes = np.abs(quantum_state) ** 2
        state_entropy = -np.sum(amplitudes * np.log2(amplitudes + 1e-12))
        
        # Assess information leakage potential
        information_leakage = self._calculate_information_leakage(quantum_state)
        
        # Measure timing variance
        operation_time = time.time() - start_time
        self.timing_measurements[operation_context].append(operation_time)
        timing_variance = np.var(self.timing_measurements[operation_context]) if len(self.timing_measurements[operation_context]) > 1 else 0.0
        
        # Analyze coherence stability
        coherence_stability = self._assess_coherence_stability(quantum_state)
        
        # Mock resource utilization (in real implementation, would measure actual resources)
        resource_utilization = {
            "cpu": np.random.uniform(0.1, 0.8),
            "memory": np.random.uniform(0.2, 0.6),
            "gpu": np.random.uniform(0.0, 0.9)
        }
        
        metrics = QuantumSecurityMetrics(
            state_entropy=state_entropy,
            information_leakage=information_leakage,
            timing_variance=timing_variance,
            resource_utilization=resource_utilization,
            cache_hit_pattern=[],  # Would be populated from actual cache metrics
            error_rate=0.0,  # Would be calculated from operation success rates
            quantum_coherence_stability=coherence_stability
        )
        
        # Log security-relevant information
        self._log_security_event({
            "event_type": "quantum_state_analysis",
            "operation_context": operation_context,
            "state_entropy": state_entropy,
            "information_leakage": information_leakage,
            "timing_variance": timing_variance,
            "timestamp": datetime.now()
        })
        
        return metrics
    
    def _calculate_information_leakage(self, quantum_state: np.ndarray) -> float:
        """Calculate potential information leakage from quantum state"""
        # Simplified information leakage assessment
        # In practice, this would involve more sophisticated analysis
        
        # Check for states that might leak information
        amplitudes = np.abs(quantum_state) ** 2
        
        # High concentration in few states might indicate leakage
        max_amplitude = np.max(amplitudes)
        amplitude_concentration = max_amplitude / np.mean(amplitudes)
        
        # Normalize to 0-1 range
        leakage_score = min(amplitude_concentration / 10.0, 1.0)
        
        return leakage_score
    
    def _assess_coherence_stability(self, quantum_state: np.ndarray) -> float:
        """Assess quantum coherence stability"""
        # Create density matrix
        density_matrix = np.outer(quantum_state, np.conj(quantum_state))
        
        # Calculate off-diagonal coherence
        n = len(quantum_state)
        if n <= 1:
            return 1.0
        
        diagonal_sum = np.sum(np.abs(np.diag(density_matrix)))
        total_sum = np.sum(np.abs(density_matrix))
        off_diagonal_sum = total_sum - diagonal_sum
        
        # Coherence stability score
        coherence_ratio = off_diagonal_sum / (total_sum + 1e-12)
        stability_score = 1.0 - abs(coherence_ratio - 0.5)  # Optimal at 50% coherence
        
        return stability_score
    
    def audit_component_security(self, component_name: str, component_data: Dict[str, Any]) -> SecurityAuditResult:
        """
        Perform comprehensive security audit of a system component.
        
        Args:
            component_name: Name of component to audit
            component_data: Component data and configuration
            
        Returns:
            Security audit result
        """
        audit_start = datetime.now()
        
        # Analyze security properties
        security_properties = self._assess_security_properties(component_name, component_data)
        
        # Identify applicable threats
        threats_identified = self._identify_component_threats(component_name)
        
        # Scan for vulnerabilities
        vulnerabilities = self._scan_vulnerabilities(component_name, component_data)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(
            component_name, threats_identified, vulnerabilities
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(threats_identified, vulnerabilities)
        
        audit_result = SecurityAuditResult(
            component=component_name,
            security_properties=security_properties,
            threats_identified=threats_identified,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            audit_timestamp=audit_start,
            risk_score=risk_score
        )
        
        # Log audit
        self.audit_log.append({
            "component": component_name,
            "audit_timestamp": audit_start,
            "risk_score": risk_score,
            "threats_count": len(threats_identified),
            "vulnerabilities_count": len(vulnerabilities)
        })
        
        logger.info(f"Completed security audit for {component_name} (Risk Score: {risk_score:.2f})")
        
        return audit_result
    
    def _assess_security_properties(self, component_name: str, component_data: Dict[str, Any]) -> Dict[SecurityProperty, bool]:
        """Assess security properties of a component"""
        properties = {}
        
        # Confidentiality assessment
        has_encryption = component_data.get("encrypted", False)
        has_access_control = component_data.get("access_control", False)
        properties[SecurityProperty.CONFIDENTIALITY] = has_encryption and has_access_control
        
        # Integrity assessment
        has_checksums = component_data.get("integrity_checks", False)
        has_signatures = component_data.get("digital_signatures", False)
        properties[SecurityProperty.INTEGRITY] = has_checksums or has_signatures
        
        # Availability assessment
        has_redundancy = component_data.get("redundancy", False)
        has_rate_limiting = component_data.get("rate_limiting", False)
        properties[SecurityProperty.AVAILABILITY] = has_redundancy and has_rate_limiting
        
        # Authenticity assessment
        has_authentication = component_data.get("authentication", False)
        has_authorization = component_data.get("authorization", False)
        properties[SecurityProperty.AUTHENTICITY] = has_authentication and has_authorization
        
        # Non-repudiation assessment
        has_audit_log = component_data.get("audit_logging", False)
        has_timestamping = component_data.get("timestamping", False)
        properties[SecurityProperty.NON_REPUDIATION] = has_audit_log and has_timestamping
        
        # Forward secrecy assessment
        has_key_rotation = component_data.get("key_rotation", False)
        has_ephemeral_keys = component_data.get("ephemeral_keys", False)
        properties[SecurityProperty.FORWARD_SECRECY] = has_key_rotation and has_ephemeral_keys
        
        # Quantum resistance assessment
        uses_post_quantum_crypto = component_data.get("post_quantum_cryptography", False)
        quantum_safe_protocols = component_data.get("quantum_safe", False)
        properties[SecurityProperty.QUANTUM_RESISTANCE] = uses_post_quantum_crypto and quantum_safe_protocols
        
        return properties
    
    def _identify_component_threats(self, component_name: str) -> List[SecurityThreat]:
        """Identify threats applicable to a specific component"""
        applicable_threats = []
        
        for threat in self.threat_database:
            if any(component in component_name.lower() for component in threat.affected_components):
                applicable_threats.append(threat)
        
        return applicable_threats
    
    def _scan_vulnerabilities(self, component_name: str, component_data: Dict[str, Any]) -> List[str]:
        """Scan for known vulnerabilities in component"""
        vulnerabilities = []
        
        # Check for common vulnerability patterns
        if "cache" in component_name.lower():
            if not component_data.get("cache_isolation", False):
                vulnerabilities.append("Cache lacks proper isolation mechanisms")
            if not component_data.get("cache_encryption", False):
                vulnerabilities.append("Cache contents not encrypted")
        
        if "quantum" in component_name.lower():
            if not component_data.get("state_verification", False):
                vulnerabilities.append("Quantum states lack integrity verification")
            if not component_data.get("decoherence_protection", False):
                vulnerabilities.append("Insufficient protection against decoherence attacks")
        
        if "scheduler" in component_name.lower() or "executor" in component_name.lower():
            if not component_data.get("resource_limits", False):
                vulnerabilities.append("Missing resource exhaustion protection")
            if not component_data.get("schedule_randomization", False):
                vulnerabilities.append("Predictable scheduling patterns may leak information")
        
        if "optimization" in component_name.lower():
            if not component_data.get("constant_time", False):
                vulnerabilities.append("Optimization timing may leak sensitive information")
            if not component_data.get("convergence_obfuscation", False):
                vulnerabilities.append("Convergence patterns may reveal problem structure")
        
        return vulnerabilities
    
    def _generate_security_recommendations(self, 
                                         component_name: str, 
                                         threats: List[SecurityThreat], 
                                         vulnerabilities: List[str]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Add threat-based recommendations
        for threat in threats:
            if threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                recommendations.extend(threat.mitigation_strategies)
        
        # Add vulnerability-specific recommendations
        if vulnerabilities:
            if "cache" in " ".join(vulnerabilities).lower():
                recommendations.extend([
                    "Implement cache partitioning and isolation",
                    "Add cache encryption and authentication",
                    "Use oblivious cache access patterns"
                ])
            
            if "quantum" in " ".join(vulnerabilities).lower():
                recommendations.extend([
                    "Add quantum state integrity checks",
                    "Implement quantum error correction",
                    "Use quantum state commitment schemes"
                ])
            
            if "timing" in " ".join(vulnerabilities).lower():
                recommendations.extend([
                    "Implement constant-time operations",
                    "Add random timing delays",
                    "Use timing attack mitigations"
                ])
        
        # Generic security recommendations
        recommendations.extend([
            f"Implement comprehensive logging and monitoring for {component_name}",
            f"Regularly update and patch {component_name} dependencies",
            f"Conduct periodic security assessments of {component_name}",
            f"Implement defense-in-depth for {component_name}"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_risk_score(self, threats: List[SecurityThreat], vulnerabilities: List[str]) -> float:
        """Calculate overall risk score"""
        risk_score = 0.0
        
        # Add threat-based risk
        for threat in threats:
            if threat.cvss_score:
                risk_score += threat.cvss_score * 0.1  # Scale down CVSS scores
            else:
                # Default scores based on threat level
                level_scores = {
                    ThreatLevel.LOW: 1.0,
                    ThreatLevel.MEDIUM: 3.0,
                    ThreatLevel.HIGH: 6.0,
                    ThreatLevel.CRITICAL: 9.0
                }
                risk_score += level_scores.get(threat.threat_level, 0.0)
        
        # Add vulnerability-based risk
        risk_score += len(vulnerabilities) * 2.0
        
        # Normalize to 0-10 scale
        return min(risk_score, 10.0)
    
    def detect_timing_attacks(self, operation_name: str, threshold_factor: float = 3.0) -> Dict[str, Any]:
        """
        Detect potential timing attacks based on operation timing patterns.
        
        Args:
            operation_name: Name of operation to analyze
            threshold_factor: Threshold for anomaly detection
            
        Returns:
            Timing attack analysis results
        """
        if operation_name not in self.timing_measurements:
            return {"status": "no_data", "operation": operation_name}
        
        timings = self.timing_measurements[operation_name]
        
        if len(timings) < 10:
            return {"status": "insufficient_data", "operation": operation_name, "sample_count": len(timings)}
        
        # Statistical analysis
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        
        # Detect outliers (potential attacks)
        outliers = []
        for i, timing in enumerate(timings):
            if abs(timing - mean_time) > threshold_factor * std_time:
                outliers.append((i, timing))
        
        # Analyze timing patterns
        patterns = self._analyze_timing_patterns(timings)
        
        # Risk assessment
        risk_level = ThreatLevel.LOW
        if len(outliers) > len(timings) * 0.1:  # >10% outliers
            risk_level = ThreatLevel.HIGH
        elif patterns["periodicity_detected"]:
            risk_level = ThreatLevel.MEDIUM
        
        return {
            "status": "analyzed",
            "operation": operation_name,
            "sample_count": len(timings),
            "mean_time": mean_time,
            "std_time": std_time,
            "outliers": outliers,
            "outlier_rate": len(outliers) / len(timings),
            "patterns": patterns,
            "risk_level": risk_level.value,
            "recommendations": self._get_timing_attack_recommendations(risk_level)
        }
    
    def _analyze_timing_patterns(self, timings: List[float]) -> Dict[str, Any]:
        """Analyze timing patterns for anomalies"""
        # Simple pattern analysis (in practice, would use more sophisticated methods)
        patterns = {
            "periodicity_detected": False,
            "trend_detected": False,
            "clustering_detected": False
        }
        
        if len(timings) > 20:
            # Check for periodicity using autocorrelation
            timings_array = np.array(timings)
            autocorr = np.correlate(timings_array, timings_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks in autocorrelation (indicating periodicity)
            if len(autocorr) > 1 and np.max(autocorr[1:]) > 0.7 * autocorr[0]:
                patterns["periodicity_detected"] = True
            
            # Check for trends
            if abs(np.corrcoef(range(len(timings)), timings)[0, 1]) > 0.5:
                patterns["trend_detected"] = True
        
        return patterns
    
    def _get_timing_attack_recommendations(self, risk_level: ThreatLevel) -> List[str]:
        """Get recommendations for timing attack mitigation"""
        base_recommendations = [
            "Implement constant-time operations where possible",
            "Add random delays to obscure timing patterns",
            "Monitor timing patterns for anomalies"
        ]
        
        if risk_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            base_recommendations.extend([
                "Implement comprehensive timing attack countermeasures",
                "Use blinding techniques to hide operation timing",
                "Consider using secure multi-party computation for timing-sensitive operations",
                "Implement dummy operations to normalize timing patterns"
            ])
        
        return base_recommendations
    
    def generate_security_report(self, components: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive security report for multiple components.
        
        Args:
            components: List of component names to analyze
            
        Returns:
            Comprehensive security report
        """
        report_timestamp = datetime.now()
        component_audits = []
        
        # Audit each component
        for component_name in components:
            # Mock component data (in practice, would retrieve actual configuration)
            component_data = {
                "encrypted": False,
                "access_control": False,
                "integrity_checks": False,
                "audit_logging": True,
                "rate_limiting": False
            }
            
            audit = self.audit_component_security(component_name, component_data)
            component_audits.append(audit)
        
        # Calculate overall risk
        overall_risk = np.mean([audit.risk_score for audit in component_audits])
        
        # Aggregate threats and vulnerabilities
        all_threats = []
        all_vulnerabilities = []
        
        for audit in component_audits:
            all_threats.extend(audit.threats_identified)
            all_vulnerabilities.extend(audit.vulnerabilities)
        
        # Remove duplicates
        unique_threats = list({threat.threat_id: threat for threat in all_threats}.values())
        unique_vulnerabilities = list(set(all_vulnerabilities))
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            overall_risk, unique_threats, unique_vulnerabilities
        )
        
        return {
            "report_timestamp": report_timestamp,
            "components_analyzed": len(components),
            "overall_risk_score": overall_risk,
            "executive_summary": executive_summary,
            "component_audits": [
                {
                    "component": audit.component,
                    "risk_score": audit.risk_score,
                    "threats_count": len(audit.threats_identified),
                    "vulnerabilities_count": len(audit.vulnerabilities),
                    "recommendations_count": len(audit.recommendations)
                }
                for audit in component_audits
            ],
            "threat_summary": {
                "total_threats": len(unique_threats),
                "critical_threats": len([t for t in unique_threats if t.threat_level == ThreatLevel.CRITICAL]),
                "high_threats": len([t for t in unique_threats if t.threat_level == ThreatLevel.HIGH]),
                "medium_threats": len([t for t in unique_threats if t.threat_level == ThreatLevel.MEDIUM]),
                "low_threats": len([t for t in unique_threats if t.threat_level == ThreatLevel.LOW])
            },
            "vulnerability_summary": {
                "total_vulnerabilities": len(unique_vulnerabilities),
                "top_vulnerabilities": unique_vulnerabilities[:10]
            },
            "recommendations": self._prioritize_recommendations(component_audits),
            "compliance_status": self._assess_compliance_status(component_audits),
            "next_audit_date": report_timestamp + timedelta(days=90)
        }
    
    def _generate_executive_summary(self, 
                                  overall_risk: float, 
                                  threats: List[SecurityThreat], 
                                  vulnerabilities: List[str]) -> str:
        """Generate executive summary of security assessment"""
        risk_level_desc = {
            (0, 3): "LOW",
            (3, 6): "MEDIUM", 
            (6, 8): "HIGH",
            (8, 10): "CRITICAL"
        }
        
        risk_desc = "LOW"
        for (low, high), desc in risk_level_desc.items():
            if low <= overall_risk < high:
                risk_desc = desc
                break
        
        critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
        high_threats = [t for t in threats if t.threat_level == ThreatLevel.HIGH]
        
        summary = f"""
        QUANTUM PLANNING SECURITY ASSESSMENT SUMMARY
        
        Overall Risk Level: {risk_desc} ({overall_risk:.1f}/10.0)
        
        Key Findings:
        - {len(threats)} security threats identified
        - {len(critical_threats)} CRITICAL threats requiring immediate attention
        - {len(high_threats)} HIGH priority threats
        - {len(vulnerabilities)} vulnerabilities discovered
        
        Critical Issues:
        """
        
        for threat in critical_threats[:3]:  # Top 3 critical threats
            summary += f"- {threat.name}: {threat.description}\n        "
        
        summary += f"""
        
        Immediate Actions Required:
        - Address all CRITICAL and HIGH priority threats
        - Implement recommended security controls
        - Establish continuous security monitoring
        - Schedule follow-up assessment in 90 days
        """
        
        return summary.strip()
    
    def _prioritize_recommendations(self, audits: List[SecurityAuditResult]) -> List[Dict[str, Any]]:
        """Prioritize security recommendations across all components"""
        all_recommendations = []
        
        for audit in audits:
            for rec in audit.recommendations:
                all_recommendations.append({
                    "recommendation": rec,
                    "component": audit.component,
                    "risk_score": audit.risk_score,
                    "priority": self._calculate_recommendation_priority(rec, audit.risk_score)
                })
        
        # Sort by priority (higher is more important)
        all_recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return all_recommendations[:20]  # Top 20 recommendations
    
    def _calculate_recommendation_priority(self, recommendation: str, component_risk: float) -> float:
        """Calculate priority score for a recommendation"""
        base_priority = component_risk
        
        # Boost priority for certain keywords
        high_priority_keywords = ["critical", "immediate", "urgent", "fix", "patch"]
        medium_priority_keywords = ["implement", "add", "enhance", "improve"]
        
        rec_lower = recommendation.lower()
        
        if any(keyword in rec_lower for keyword in high_priority_keywords):
            base_priority += 3.0
        elif any(keyword in rec_lower for keyword in medium_priority_keywords):
            base_priority += 1.0
        
        return base_priority
    
    def _assess_compliance_status(self, audits: List[SecurityAuditResult]) -> Dict[str, Any]:
        """Assess compliance with security standards"""
        # Mock compliance assessment
        compliance_frameworks = {
            "ISO 27001": 0.75,
            "NIST Cybersecurity Framework": 0.68,
            "SOC 2": 0.82,
            "GDPR": 0.71
        }
        
        return {
            "frameworks": compliance_frameworks,
            "average_compliance": np.mean(list(compliance_frameworks.values())),
            "compliance_gaps": [
                "Missing encryption for data at rest",
                "Insufficient access controls",
                "Limited audit logging coverage",
                "Incomplete incident response procedures"
            ]
        }
    
    def _log_security_event(self, event: Dict[str, Any]):
        """Log security-relevant events"""
        event_entry = {
            "timestamp": datetime.now(),
            "event_id": secrets.token_hex(8),
            **event
        }
        
        # In production, this would write to a secure audit log
        logger.info(f"Security Event: {event['event_type']} - {event_entry['event_id']}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        return {
            "total_audits_performed": len(self.audit_log),
            "average_risk_score": np.mean([audit["risk_score"] for audit in self.audit_log]) if self.audit_log else 0.0,
            "timing_measurements_collected": sum(len(timings) for timings in self.timing_measurements.values()),
            "unique_operations_monitored": len(self.timing_measurements),
            "security_events_logged": len(self.audit_log),
            "last_audit_timestamp": self.audit_log[-1]["audit_timestamp"] if self.audit_log else None
        }