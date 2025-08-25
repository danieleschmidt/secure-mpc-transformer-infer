"""
Defensive Security System for Bio-Enhanced MPC Transformer

Implements comprehensive defensive security measures with bio-inspired
adaptive threat detection and response mechanisms.
"""

import asyncio
import logging
import hashlib
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from contextlib import asynccontextmanager


class ThreatLevel(Enum):
    """Security threat levels for adaptive response."""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class SecurityEventType(Enum):
    """Types of security events detected."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTEMPT = "injection_attempt"
    BRUTE_FORCE = "brute_force"
    TIMING_ATTACK = "timing_attack"
    RESOURCE_ABUSE = "resource_abuse"
    PROTOCOL_VIOLATION = "protocol_violation"
    QUANTUM_INTERFERENCE = "quantum_interference"


@dataclass
class SecurityEvent:
    """Represents a security event or threat detected."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    details: Dict[str, Any]
    affected_components: List[str]
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    bio_signature: Optional[str] = None


@dataclass
class ThreatPattern:
    """Bio-inspired threat pattern for adaptive detection."""
    pattern_id: str
    pattern_name: str
    signature: str
    confidence_threshold: float
    detection_count: int = 0
    last_detected: Optional[datetime] = None
    evolutionary_adaptations: int = 0
    false_positive_rate: float = 0.0


class DefensiveSecuritySystem:
    """
    Bio-enhanced defensive security system with adaptive threat detection
    and autonomous response capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Security state tracking
        self.active_threats: Dict[str, SecurityEvent] = {}
        self.resolved_threats: List[SecurityEvent] = []
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        
        # Adaptive defense parameters
        self.base_security_level = ThreatLevel.MEDIUM
        self.current_security_level = ThreatLevel.MEDIUM
        self.adaptive_threshold = 0.75
        self.learning_rate = 0.1
        
        # Rate limiting and monitoring
        self.request_history: Dict[str, List[datetime]] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = {}
        
        # Bio-inspired security genetics
        self.security_genes: Dict[str, float] = {
            "threat_detection_sensitivity": 0.8,
            "false_positive_tolerance": 0.15,
            "adaptive_learning_rate": 0.12,
            "response_aggressiveness": 0.7,
            "pattern_recognition_depth": 0.85
        }
        
        # Initialize threat patterns
        self._initialize_threat_patterns()
        
        self.logger.info("Defensive Security System initialized with bio-enhancement")
        
    def _initialize_threat_patterns(self) -> None:
        """Initialize known threat patterns with bio-inspired signatures."""
        
        patterns = [
            ThreatPattern(
                pattern_id="sql_injection",
                pattern_name="SQL Injection Attack",
                signature="(?i)(union|select|insert|delete|drop|create|alter|exec|script)",
                confidence_threshold=0.85
            ),
            ThreatPattern(
                pattern_id="xss_attempt",
                pattern_name="Cross-Site Scripting",
                signature=r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)",
                confidence_threshold=0.80
            ),
            ThreatPattern(
                pattern_id="path_traversal",
                pattern_name="Directory Traversal",
                signature=r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                confidence_threshold=0.90
            ),
            ThreatPattern(
                pattern_id="command_injection",
                pattern_name="Command Injection",
                signature=r"(;|\||&&|`|\$\(|>\s*&)",
                confidence_threshold=0.88
            ),
            ThreatPattern(
                pattern_id="timing_anomaly",
                pattern_name="Timing Attack Pattern",
                signature="timing_variance_threshold_exceeded",
                confidence_threshold=0.75
            ),
            ThreatPattern(
                pattern_id="quantum_tampering",
                pattern_name="Quantum State Tampering",
                signature="quantum_coherence_anomaly",
                confidence_threshold=0.92
            ),
            ThreatPattern(
                pattern_id="mpc_protocol_violation",
                pattern_name="MPC Protocol Violation",
                signature="secret_sharing_integrity_failure",
                confidence_threshold=0.95
            ),
            ThreatPattern(
                pattern_id="resource_exhaustion",
                pattern_name="Resource Exhaustion Attack",
                signature="resource_usage_spike_detected",
                confidence_threshold=0.82
            )
        ]
        
        for pattern in patterns:
            self.threat_patterns[pattern.pattern_id] = pattern
            
        self.logger.info(f"Initialized {len(patterns)} threat detection patterns")
        
    async def analyze_request_security(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incoming request for security threats using bio-enhanced detection."""
        
        analysis_start = time.time()
        source_ip = request_data.get("source_ip", "unknown")
        request_content = json.dumps(request_data, default=str)
        
        # Multi-layered security analysis
        security_analysis = {
            "timestamp": datetime.now().isoformat(),
            "source_ip": source_ip,
            "threat_level": ThreatLevel.MINIMAL,
            "detected_patterns": [],
            "security_score": 1.0,
            "mitigation_actions": [],
            "bio_adaptations_applied": [],
            "analysis_time_ms": 0
        }
        
        # Rate limiting check
        rate_limit_result = await self._check_rate_limits(source_ip)
        if rate_limit_result["violated"]:
            security_analysis["threat_level"] = ThreatLevel.HIGH
            security_analysis["detected_patterns"].append("rate_limit_violation")
            security_analysis["mitigation_actions"].append("rate_limit_enforcement")
            
        # IP reputation check
        if source_ip in self.blocked_ips:
            security_analysis["threat_level"] = ThreatLevel.CRITICAL
            security_analysis["detected_patterns"].append("blocked_ip_access")
            security_analysis["mitigation_actions"].append("block_request")
            
        # Pattern-based threat detection
        pattern_results = await self._detect_threat_patterns(request_content)
        for pattern_match in pattern_results:
            security_analysis["detected_patterns"].append(pattern_match["pattern_id"])
            
            # Update threat level based on pattern severity
            pattern_threat_level = self._calculate_pattern_threat_level(pattern_match)
            if pattern_threat_level.value > security_analysis["threat_level"].value:
                security_analysis["threat_level"] = pattern_threat_level
                
        # Bio-enhanced adaptive analysis
        bio_analysis = await self._bio_enhanced_threat_analysis(request_data)
        security_analysis["bio_adaptations_applied"] = bio_analysis["adaptations"]
        security_analysis["security_score"] = bio_analysis["security_score"]
        
        # Calculate final security score
        base_score = 1.0
        pattern_penalty = len(security_analysis["detected_patterns"]) * 0.2
        ip_penalty = 0.5 if source_ip in self.blocked_ips else 0.0
        rate_penalty = 0.3 if rate_limit_result["violated"] else 0.0
        
        security_analysis["security_score"] = max(0.0, base_score - pattern_penalty - ip_penalty - rate_penalty)
        
        # Generate mitigation actions
        if security_analysis["threat_level"].value >= ThreatLevel.HIGH.value:
            security_analysis["mitigation_actions"].extend([
                "enhanced_monitoring",
                "request_throttling",
                "detailed_logging"
            ])
            
        if security_analysis["threat_level"] == ThreatLevel.CRITICAL:
            security_analysis["mitigation_actions"].extend([
                "block_source_ip",
                "alert_security_team",
                "forensic_analysis"
            ])
            
        # Record analysis time
        security_analysis["analysis_time_ms"] = (time.time() - analysis_start) * 1000
        
        # Log security event if threats detected
        if security_analysis["threat_level"].value > ThreatLevel.LOW.value:
            await self._log_security_event(security_analysis)
            
        return security_analysis
        
    async def _check_rate_limits(self, source_ip: str) -> Dict[str, Any]:
        """Check if source IP is violating rate limits."""
        
        current_time = datetime.now()
        time_window = timedelta(minutes=5)  # 5-minute window
        max_requests = 100  # Bio-adaptive threshold
        
        # Clean old entries
        if source_ip in self.request_history:
            self.request_history[source_ip] = [
                req_time for req_time in self.request_history[source_ip]
                if current_time - req_time <= time_window
            ]
        else:
            self.request_history[source_ip] = []
            
        # Add current request
        self.request_history[source_ip].append(current_time)
        
        # Check violation
        request_count = len(self.request_history[source_ip])
        violated = request_count > max_requests
        
        # Bio-adaptive rate limiting adjustment
        if violated and source_ip not in self.blocked_ips:
            # Gradually increase sensitivity for repeat offenders
            repeat_violations = self.suspicious_patterns.get(source_ip, 0) + 1
            self.suspicious_patterns[source_ip] = repeat_violations
            
            # Auto-block after multiple violations
            if repeat_violations >= 3:
                self.blocked_ips.add(source_ip)
                self.logger.warning(f"Auto-blocked IP {source_ip} after {repeat_violations} violations")
                
        return {
            "violated": violated,
            "request_count": request_count,
            "max_requests": max_requests,
            "time_window_minutes": 5,
            "repeat_violations": self.suspicious_patterns.get(source_ip, 0)
        }
        
    async def _detect_threat_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Detect threat patterns in request content using bio-enhanced matching."""
        
        detected_patterns = []
        
        for pattern_id, pattern in self.threat_patterns.items():
            confidence = await self._calculate_pattern_confidence(content, pattern)
            
            if confidence >= pattern.confidence_threshold:
                # Bio-enhanced pattern adaptation
                pattern.detection_count += 1
                pattern.last_detected = datetime.now()
                
                # Evolutionary adaptation - adjust threshold based on detection history
                if pattern.detection_count > 10:
                    adaptation_factor = self.security_genes["adaptive_learning_rate"]
                    if pattern.false_positive_rate < 0.1:
                        # Lower threshold for accurate patterns
                        pattern.confidence_threshold = max(0.5, pattern.confidence_threshold - adaptation_factor)
                    else:
                        # Raise threshold for noisy patterns
                        pattern.confidence_threshold = min(0.98, pattern.confidence_threshold + adaptation_factor)
                        
                    pattern.evolutionary_adaptations += 1
                
                detected_patterns.append({
                    "pattern_id": pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "confidence": confidence,
                    "detection_count": pattern.detection_count,
                    "adaptations": pattern.evolutionary_adaptations
                })
                
        return detected_patterns
        
    async def _calculate_pattern_confidence(self, content: str, pattern: ThreatPattern) -> float:
        """Calculate confidence score for pattern match with bio-enhancement."""
        
        import re
        
        # Basic regex matching
        if pattern.signature.startswith("(?i)"):
            # Case-insensitive regex
            matches = re.findall(pattern.signature, content, re.IGNORECASE)
        else:
            # Special pattern types
            if pattern.pattern_id == "timing_anomaly":
                return await self._check_timing_anomaly()
            elif pattern.pattern_id == "quantum_tampering":
                return await self._check_quantum_tampering()
            elif pattern.pattern_id == "mpc_protocol_violation":
                return await self._check_mpc_protocol()
            elif pattern.pattern_id == "resource_exhaustion":
                return await self._check_resource_exhaustion()
            else:
                matches = re.findall(pattern.signature, content)
                
        if not matches:
            return 0.0
            
        # Bio-enhanced confidence calculation
        base_confidence = min(1.0, len(matches) * 0.3)  # Multiple matches increase confidence
        
        # Apply bio-genetic factors
        sensitivity = self.security_genes["threat_detection_sensitivity"]
        pattern_depth = self.security_genes["pattern_recognition_depth"]
        
        enhanced_confidence = base_confidence * sensitivity * pattern_depth
        
        # Context-aware adjustments
        if pattern.detection_count > 5:  # Pattern has history
            historical_accuracy = 1.0 - pattern.false_positive_rate
            enhanced_confidence *= historical_accuracy
            
        return min(1.0, enhanced_confidence)
        
    async def _check_timing_anomaly(self) -> float:
        """Check for timing attack patterns using statistical analysis."""
        
        # Simulated timing analysis - in production would analyze actual request times
        import random
        baseline_time = 0.1  # Expected processing time
        current_time = random.uniform(0.05, 0.3)  # Simulated current time
        
        variance = abs(current_time - baseline_time) / baseline_time
        
        # High variance indicates potential timing attack
        return min(1.0, variance * 2.0) if variance > 0.2 else 0.0
        
    async def _check_quantum_tampering(self) -> float:
        """Check for quantum state tampering or interference."""
        
        # Simulated quantum coherence check
        import random
        expected_coherence = 0.95
        measured_coherence = random.uniform(0.85, 0.98)
        
        coherence_deviation = abs(expected_coherence - measured_coherence)
        
        # Significant deviation indicates tampering
        return min(1.0, coherence_deviation * 10.0) if coherence_deviation > 0.05 else 0.0
        
    async def _check_mpc_protocol(self) -> float:
        """Check for MPC protocol violations or integrity failures."""
        
        # Simulated secret sharing integrity check
        import random
        integrity_score = random.uniform(0.92, 1.0)
        
        # Low integrity indicates protocol violation
        return max(0.0, 1.0 - integrity_score) * 2.0 if integrity_score < 0.98 else 0.0
        
    async def _check_resource_exhaustion(self) -> float:
        """Check for resource exhaustion attack patterns."""
        
        # Simulated resource usage monitoring
        import random
        cpu_usage = random.uniform(0.3, 0.95)
        memory_usage = random.uniform(0.4, 0.9)
        
        resource_pressure = (cpu_usage + memory_usage) / 2.0
        
        # High resource pressure indicates potential attack
        return max(0.0, resource_pressure - 0.7) * 3.0 if resource_pressure > 0.8 else 0.0
        
    async def _bio_enhanced_threat_analysis(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bio-enhanced threat analysis using evolutionary algorithms."""
        
        analysis = {
            "security_score": 1.0,
            "adaptations": [],
            "confidence": 0.8
        }
        
        # Analyze request patterns using bio-inspired heuristics
        request_size = len(json.dumps(request_data, default=str))
        request_complexity = len(request_data.keys()) if isinstance(request_data, dict) else 1
        
        # Bio-genetic analysis factors
        size_factor = min(1.0, request_size / 10000)  # Normalize request size
        complexity_factor = min(1.0, request_complexity / 20)  # Normalize complexity
        
        # Apply bio-genetic weights
        aggressiveness = self.security_genes["response_aggressiveness"]
        sensitivity = self.security_genes["threat_detection_sensitivity"]
        
        # Calculate bio-enhanced security score
        bio_score = 1.0 - ((size_factor * 0.3) + (complexity_factor * 0.2)) * aggressiveness
        analysis["security_score"] = max(0.0, bio_score)
        
        # Adaptive responses based on bio-genetics
        if size_factor > 0.7:
            analysis["adaptations"].append("large_payload_analysis")
        if complexity_factor > 0.6:
            analysis["adaptations"].append("complex_structure_validation")
            
        # Evolutionary learning
        if len(self.active_threats) > 5:
            analysis["adaptations"].append("heightened_alert_mode")
            analysis["security_score"] *= 0.9  # More cautious when under attack
            
        return analysis
        
    def _calculate_pattern_threat_level(self, pattern_match: Dict[str, Any]) -> ThreatLevel:
        """Calculate threat level based on pattern match confidence and severity."""
        
        confidence = pattern_match["confidence"]
        pattern_id = pattern_match["pattern_id"]
        
        # Pattern-specific severity mapping
        critical_patterns = ["mpc_protocol_violation", "quantum_tampering"]
        high_patterns = ["command_injection", "sql_injection"]
        medium_patterns = ["xss_attempt", "path_traversal"]
        
        if pattern_id in critical_patterns:
            base_level = ThreatLevel.CRITICAL if confidence > 0.8 else ThreatLevel.HIGH
        elif pattern_id in high_patterns:
            base_level = ThreatLevel.HIGH if confidence > 0.8 else ThreatLevel.MEDIUM
        elif pattern_id in medium_patterns:
            base_level = ThreatLevel.MEDIUM if confidence > 0.7 else ThreatLevel.LOW
        else:
            base_level = ThreatLevel.LOW if confidence > 0.6 else ThreatLevel.MINIMAL
            
        return base_level
        
    async def _log_security_event(self, analysis: Dict[str, Any]) -> None:
        """Log security event with bio-enhanced context."""
        
        event_id = hashlib.md5(
            f"{analysis['timestamp']}{analysis['source_ip']}{analysis['threat_level']}".encode()
        ).hexdigest()[:16]
        
        security_event = SecurityEvent(
            event_id=event_id,
            event_type=SecurityEventType.SUSPICIOUS_PATTERN,  # Would be determined by analysis
            threat_level=analysis["threat_level"],
            timestamp=datetime.fromisoformat(analysis["timestamp"]),
            source_ip=analysis["source_ip"],
            details=analysis,
            affected_components=["bio_security_system"],
            bio_signature=self._generate_bio_signature(analysis)
        )
        
        self.active_threats[event_id] = security_event
        
        self.logger.warning(
            f"Security threat detected: {security_event.threat_level.name} from {security_event.source_ip} "
            f"(Event ID: {event_id})"
        )
        
    def _generate_bio_signature(self, analysis: Dict[str, Any]) -> str:
        """Generate bio-inspired signature for threat pattern evolution."""
        
        signature_components = [
            str(analysis["threat_level"].value),
            str(len(analysis["detected_patterns"])),
            str(int(analysis["security_score"] * 100)),
            analysis["source_ip"].split(".")[-1] if "." in analysis["source_ip"] else "0"
        ]
        
        bio_signature = hashlib.sha256(
            "_".join(signature_components).encode()
        ).hexdigest()[:12]
        
        return f"bio_{bio_signature}"
        
    async def evolve_security_genetics(self) -> Dict[str, Any]:
        """Evolve security genetics based on threat landscape and effectiveness."""
        
        evolution_start = time.time()
        
        # Analyze recent security performance
        recent_threats = [
            threat for threat in self.active_threats.values()
            if (datetime.now() - threat.timestamp).days < 7
        ]
        
        false_positive_rate = self._calculate_false_positive_rate()
        detection_accuracy = self._calculate_detection_accuracy()
        response_effectiveness = self._calculate_response_effectiveness()
        
        # Evolutionary adaptations
        adaptations = []
        
        # Adapt threat detection sensitivity
        if false_positive_rate > 0.2:  # Too many false positives
            self.security_genes["threat_detection_sensitivity"] *= 0.95
            adaptations.append("reduced_detection_sensitivity")
        elif false_positive_rate < 0.05 and detection_accuracy > 0.9:  # Very accurate
            self.security_genes["threat_detection_sensitivity"] *= 1.05
            adaptations.append("increased_detection_sensitivity")
            
        # Adapt response aggressiveness
        if response_effectiveness < 0.7:  # Responses not effective
            self.security_genes["response_aggressiveness"] *= 1.1
            adaptations.append("increased_response_aggressiveness")
        elif len(recent_threats) == 0:  # No recent threats, can be less aggressive
            self.security_genes["response_aggressiveness"] *= 0.98
            adaptations.append("reduced_response_aggressiveness")
            
        # Adapt learning rate based on threat dynamics
        threat_variety = len(set(threat.event_type for threat in recent_threats))
        if threat_variety > 3:  # High variety requires faster learning
            self.security_genes["adaptive_learning_rate"] *= 1.1
            adaptations.append("increased_learning_rate")
            
        # Ensure genetics stay within bounds
        for gene_name in self.security_genes:
            self.security_genes[gene_name] = max(0.1, min(1.0, self.security_genes[gene_name]))
            
        evolution_results = {
            "evolution_time_ms": (time.time() - evolution_start) * 1000,
            "adaptations_made": adaptations,
            "current_genetics": self.security_genes.copy(),
            "performance_metrics": {
                "false_positive_rate": false_positive_rate,
                "detection_accuracy": detection_accuracy,
                "response_effectiveness": response_effectiveness,
                "recent_threats": len(recent_threats)
            }
        }
        
        if adaptations:
            self.logger.info(f"Security genetics evolved: {', '.join(adaptations)}")
            
        return evolution_results
        
    def _calculate_false_positive_rate(self) -> float:
        """Calculate false positive rate from resolved threats."""
        if not self.resolved_threats:
            return 0.1  # Default assumption
            
        false_positives = len([
            threat for threat in self.resolved_threats 
            if threat.details.get("false_positive", False)
        ])
        
        return false_positives / len(self.resolved_threats)
        
    def _calculate_detection_accuracy(self) -> float:
        """Calculate overall detection accuracy."""
        if not self.resolved_threats:
            return 0.8  # Default assumption
            
        accurate_detections = len([
            threat for threat in self.resolved_threats
            if not threat.details.get("false_positive", False)
        ])
        
        return accurate_detections / len(self.resolved_threats)
        
    def _calculate_response_effectiveness(self) -> float:
        """Calculate response effectiveness based on threat resolution."""
        if not self.resolved_threats:
            return 0.7  # Default assumption
            
        effective_responses = len([
            threat for threat in self.resolved_threats
            if len(threat.mitigation_actions) > 0 and threat.resolved
        ])
        
        return effective_responses / len(self.resolved_threats)
        
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security system status."""
        
        return {
            "system_status": "active",
            "current_security_level": self.current_security_level.name,
            "active_threats": len(self.active_threats),
            "resolved_threats": len(self.resolved_threats),
            "blocked_ips": len(self.blocked_ips),
            "threat_patterns": len(self.threat_patterns),
            "security_genetics": self.security_genes,
            "recent_adaptations": await self.evolve_security_genetics(),
            "performance_metrics": {
                "detection_patterns_evolved": sum(
                    pattern.evolutionary_adaptations for pattern in self.threat_patterns.values()
                ),
                "total_detections": sum(
                    pattern.detection_count for pattern in self.threat_patterns.values()
                ),
                "system_uptime_hours": 24  # Placeholder
            }
        }


async def main():
    """Demonstrate the defensive security system."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize defensive security system
    security_system = DefensiveSecuritySystem({
        "bio_enhancement": True,
        "adaptive_learning": True,
        "quantum_monitoring": True
    })
    
    logger = logging.getLogger(__name__)
    logger.info("🛡️ Testing Defensive Security System")
    
    # Simulate various security scenarios
    test_scenarios = [
        {
            "name": "Normal Request",
            "request": {"source_ip": "192.168.1.100", "data": "normal request data", "action": "query"}
        },
        {
            "name": "SQL Injection Attempt", 
            "request": {"source_ip": "10.0.0.1", "query": "SELECT * FROM users WHERE id = 1 UNION SELECT * FROM admin", "action": "search"}
        },
        {
            "name": "XSS Attempt",
            "request": {"source_ip": "10.0.0.2", "content": "<script>alert('xss')</script>", "action": "submit"}
        },
        {
            "name": "Rate Limit Test",
            "request": {"source_ip": "10.0.0.3", "data": "rapid request", "action": "query"}
        },
        {
            "name": "Command Injection",
            "request": {"source_ip": "10.0.0.4", "input": "test; rm -rf /", "action": "execute"}
        }
    ]
    
    print("\n🛡️ DEFENSIVE SECURITY SYSTEM DEMONSTRATION")
    print("="*60)
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        # Simulate multiple requests for rate limiting test
        if scenario['name'] == "Rate Limit Test":
            for i in range(5):
                analysis = await security_system.analyze_request_security(scenario['request'])
                
        analysis = await security_system.analyze_request_security(scenario['request'])
        
        print(f"  Threat Level: {analysis['threat_level'].name}")
        print(f"  Security Score: {analysis['security_score']:.3f}")
        print(f"  Detected Patterns: {len(analysis['detected_patterns'])}")
        if analysis['detected_patterns']:
            print(f"    Patterns: {', '.join(analysis['detected_patterns'])}")
        print(f"  Mitigation Actions: {len(analysis['mitigation_actions'])}")
        if analysis['mitigation_actions']:
            print(f"    Actions: {', '.join(analysis['mitigation_actions'])}")
        print(f"  Bio Adaptations: {len(analysis['bio_adaptations_applied'])}")
        print(f"  Analysis Time: {analysis['analysis_time_ms']:.2f}ms")
        
    # Test bio-genetic evolution
    print(f"\n🧬 Bio-Genetic Security Evolution:")
    evolution_results = await security_system.evolve_security_genetics()
    
    print(f"  Adaptations Made: {len(evolution_results['adaptations_made'])}")
    if evolution_results['adaptations_made']:
        for adaptation in evolution_results['adaptations_made']:
            print(f"    - {adaptation}")
            
    print(f"  Current Security Genetics:")
    for gene, value in evolution_results['current_genetics'].items():
        print(f"    {gene}: {value:.3f}")
        
    # System status
    print(f"\n📊 Security System Status:")
    status = await security_system.get_security_status()
    
    print(f"  System Status: {status['system_status']}")
    print(f"  Security Level: {status['current_security_level']}")
    print(f"  Active Threats: {status['active_threats']}")
    print(f"  Blocked IPs: {status['blocked_ips']}")
    print(f"  Total Pattern Detections: {status['performance_metrics']['total_detections']}")
    print(f"  Pattern Evolutionary Adaptations: {status['performance_metrics']['detection_patterns_evolved']}")
    
    print(f"\n🎯 Defensive Security System: Bio-Enhanced Generation 2 COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main())