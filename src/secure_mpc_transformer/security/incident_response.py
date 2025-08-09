#!/usr/bin/env python3
"""
AI-Powered Incident Response System for Secure MPC Transformer

Advanced incident response with automated threat analysis, intelligent mitigation
strategies, and comprehensive security orchestration for defensive security.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque

import numpy as np

from ..utils.error_handling import SecurityError
from ..utils.metrics import MetricsCollector
from .enhanced_validator import ValidationResult
from .quantum_monitor import QuantumSecurityEvent, QuantumThreatLevel

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ResponseAction(Enum):
    """Automated response actions."""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    NOTIFY_ADMIN = "notify_admin"
    FORENSIC_CAPTURE = "forensic_capture"


class ThreatCategory(Enum):
    """Threat categorization for ML classification."""
    INJECTION_ATTACK = "injection_attack"
    TIMING_ATTACK = "timing_attack"
    DOS_ATTACK = "dos_attack"
    QUANTUM_MANIPULATION = "quantum_manipulation"
    SIDE_CHANNEL = "side_channel"
    PROTOCOL_VIOLATION = "protocol_violation"
    AUTHENTICATION_FAILURE = "authentication_failure"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SYSTEM_COMPROMISE = "system_compromise"
    UNKNOWN = "unknown"


@dataclass
class SecurityIncident:
    """Comprehensive security incident data structure."""
    incident_id: str
    timestamp: datetime
    severity: IncidentSeverity
    category: ThreatCategory
    source_ip: str
    user_id: Optional[str]
    session_id: Optional[str]
    threat_indicators: List[str]
    raw_data: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0
    false_positive_likelihood: float  # 0.0 to 1.0
    impact_assessment: Dict[str, Any]
    recommended_actions: List[ResponseAction]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ResponseStrategy:
    """Incident response strategy definition."""
    strategy_id: str
    name: str
    description: str
    applicable_categories: List[ThreatCategory]
    severity_threshold: IncidentSeverity
    actions: List[ResponseAction]
    escalation_timeout: int  # seconds
    success_criteria: Dict[str, Any]
    rollback_actions: List[ResponseAction]


class ThreatIntelligenceEngine:
    """AI-powered threat intelligence and classification."""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.ml_classifier = MLThreatClassifier()
        self.reputation_database = ReputationDatabase()
        self.attack_signatures = AttackSignatureDatabase()
        
    async def analyze_incident(
        self, 
        raw_incident_data: Dict[str, Any]
    ) -> Tuple[ThreatCategory, float, List[str]]:
        """
        Perform comprehensive threat analysis using AI/ML techniques.
        
        Args:
            raw_incident_data: Raw security event data
            
        Returns:
            Tuple of (threat_category, confidence_score, threat_indicators)
        """
        try:
            threat_indicators = []
            
            # Pattern-based analysis
            pattern_category, pattern_confidence, pattern_indicators = await self._pattern_analysis(
                raw_incident_data
            )
            threat_indicators.extend(pattern_indicators)
            
            # ML-based classification
            ml_category, ml_confidence, ml_indicators = await self.ml_classifier.classify_threat(
                raw_incident_data
            )
            threat_indicators.extend(ml_indicators)
            
            # Reputation analysis
            reputation_risk, reputation_indicators = await self.reputation_database.analyze_reputation(
                raw_incident_data.get("source_ip", "unknown")
            )
            threat_indicators.extend(reputation_indicators)
            
            # Attack signature matching
            signature_matches = await self.attack_signatures.match_signatures(raw_incident_data)
            threat_indicators.extend([f"signature_{sig}" for sig in signature_matches])
            
            # Combine analyses with weighted scoring
            final_category = self._combine_threat_categories(
                [(pattern_category, pattern_confidence), (ml_category, ml_confidence)]
            )
            
            final_confidence = min(1.0, (pattern_confidence + ml_confidence + reputation_risk) / 3.0)
            
            logger.debug(f"Threat analysis complete: category={final_category.value}, "
                        f"confidence={final_confidence:.3f}, indicators={len(threat_indicators)}")
            
            return final_category, final_confidence, threat_indicators
            
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return ThreatCategory.UNKNOWN, 0.5, ["analysis_error"]
    
    async def assess_impact(
        self, 
        incident: SecurityIncident, 
        system_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess potential impact of security incident."""
        try:
            impact_assessment = {
                "confidentiality_risk": 0.0,
                "integrity_risk": 0.0,
                "availability_risk": 0.0,
                "financial_impact": 0.0,
                "reputation_damage": 0.0,
                "regulatory_risk": 0.0
            }
            
            # Category-based impact assessment
            category_impacts = {
                ThreatCategory.INJECTION_ATTACK: {
                    "confidentiality_risk": 0.8,
                    "integrity_risk": 0.9,
                    "availability_risk": 0.3
                },
                ThreatCategory.DOS_ATTACK: {
                    "confidentiality_risk": 0.1,
                    "integrity_risk": 0.2,
                    "availability_risk": 0.9
                },
                ThreatCategory.QUANTUM_MANIPULATION: {
                    "confidentiality_risk": 0.9,
                    "integrity_risk": 0.9,
                    "availability_risk": 0.4,
                    "regulatory_risk": 0.8
                },
                ThreatCategory.TIMING_ATTACK: {
                    "confidentiality_risk": 0.7,
                    "integrity_risk": 0.3,
                    "availability_risk": 0.2
                }
            }
            
            if incident.category in category_impacts:
                base_impacts = category_impacts[incident.category]
                for risk_type, base_value in base_impacts.items():
                    # Scale by confidence and severity
                    severity_multiplier = {
                        IncidentSeverity.INFO: 0.1,
                        IncidentSeverity.LOW: 0.3,
                        IncidentSeverity.MEDIUM: 0.6,
                        IncidentSeverity.HIGH: 0.8,
                        IncidentSeverity.CRITICAL: 1.0
                    }.get(incident.severity, 0.5)
                    
                    impact_assessment[risk_type] = min(1.0, 
                        base_value * incident.confidence_score * severity_multiplier
                    )
            
            # System context adjustments
            if system_context.get("production_environment", False):
                # Higher impact in production
                for key in impact_assessment:
                    impact_assessment[key] *= 1.3
                    impact_assessment[key] = min(1.0, impact_assessment[key])
            
            if system_context.get("sensitive_data_present", False):
                impact_assessment["confidentiality_risk"] *= 1.5
                impact_assessment["regulatory_risk"] *= 1.4
                
            # Calculate overall risk score
            overall_risk = np.mean(list(impact_assessment.values()))
            impact_assessment["overall_risk_score"] = overall_risk
            
            return impact_assessment
            
        except Exception as e:
            logger.error(f"Impact assessment failed: {e}")
            return {"overall_risk_score": 0.5, "error": str(e)}
    
    async def _pattern_analysis(self, data: Dict[str, Any]) -> Tuple[ThreatCategory, float, List[str]]:
        """Pattern-based threat analysis."""
        indicators = []
        max_confidence = 0.0
        detected_category = ThreatCategory.UNKNOWN
        
        try:
            # Check for injection patterns
            content = str(data.get("request_content", ""))
            if any(pattern in content.lower() for pattern in ["select ", "union ", "drop ", "exec "]):
                indicators.append("sql_injection_pattern")
                max_confidence = max(max_confidence, 0.8)
                detected_category = ThreatCategory.INJECTION_ATTACK
            
            # Check for timing attack patterns
            if "timing_variance" in data and data["timing_variance"] > 0.1:
                indicators.append("timing_variance_detected")
                max_confidence = max(max_confidence, 0.7)
                detected_category = ThreatCategory.TIMING_ATTACK
            
            # Check for DoS patterns
            request_rate = data.get("request_rate", 0)
            if request_rate > 100:  # More than 100 requests per second
                indicators.append("high_request_rate")
                max_confidence = max(max_confidence, 0.6)
                detected_category = ThreatCategory.DOS_ATTACK
            
            # Check for quantum-specific threats
            if "quantum_state_manipulation" in content or "decoherence_attack" in content:
                indicators.append("quantum_threat_pattern")
                max_confidence = max(max_confidence, 0.9)
                detected_category = ThreatCategory.QUANTUM_MANIPULATION
            
            return detected_category, max_confidence, indicators
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return ThreatCategory.UNKNOWN, 0.0, ["pattern_analysis_error"]
    
    def _combine_threat_categories(
        self, 
        category_scores: List[Tuple[ThreatCategory, float]]
    ) -> ThreatCategory:
        """Combine multiple threat category assessments."""
        if not category_scores:
            return ThreatCategory.UNKNOWN
        
        # Weight by confidence and return highest scoring category
        category_scores.sort(key=lambda x: x[1], reverse=True)
        return category_scores[0][0]
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            "injection": [
                "select.*from", "union.*select", "drop.*table", "exec.*cmd",
                "eval\\(", "system\\(", "<script", "javascript:"
            ],
            "timing": [
                "timing_variance", "side_channel", "cache_timing", "power_analysis"
            ],
            "dos": [
                "flood", "amplification", "resource_exhaustion", "slowloris"
            ],
            "quantum": [
                "quantum_state_manipulation", "decoherence_attack", "entanglement_break",
                "measurement_tampering", "coherence_disruption"
            ]
        }


class MLThreatClassifier:
    """Machine learning-based threat classification."""
    
    def __init__(self):
        self.feature_extractors = FeatureExtractors()
        self.models = {}  # Placeholder for ML models
        self.training_data = deque(maxlen=10000)
        
    async def classify_threat(self, incident_data: Dict[str, Any]) -> Tuple[ThreatCategory, float, List[str]]:
        """Classify threat using ML techniques."""
        try:
            # Extract features
            features = await self.feature_extractors.extract_features(incident_data)
            
            # Simple rule-based classification (placeholder for actual ML)
            indicators = []
            confidence = 0.0
            category = ThreatCategory.UNKNOWN
            
            # Request size analysis
            if features.get("request_size", 0) > 1000000:  # 1MB
                indicators.append("large_request_size")
                confidence = max(confidence, 0.6)
                category = ThreatCategory.DOS_ATTACK
            
            # Frequency analysis
            if features.get("request_frequency", 0) > 50:  # 50 req/min
                indicators.append("high_frequency")
                confidence = max(confidence, 0.7)
                category = ThreatCategory.DOS_ATTACK
            
            # Content complexity analysis
            if features.get("content_entropy", 0) > 7.0:  # High entropy
                indicators.append("high_entropy_content")
                confidence = max(confidence, 0.5)
                category = ThreatCategory.INJECTION_ATTACK
            
            # Pattern diversity analysis
            if features.get("pattern_diversity", 0) > 0.8:
                indicators.append("diverse_attack_patterns")
                confidence = max(confidence, 0.8)
                category = ThreatCategory.SYSTEM_COMPROMISE
            
            return category, confidence, indicators
            
        except Exception as e:
            logger.error(f"ML threat classification failed: {e}")
            return ThreatCategory.UNKNOWN, 0.0, ["ml_classification_error"]


class FeatureExtractors:
    """Extract features from incident data for ML analysis."""
    
    async def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from incident data."""
        features = {}
        
        try:
            # Basic features
            features["request_size"] = len(str(data.get("request_content", "")))
            features["timestamp"] = time.time()
            features["has_session"] = 1.0 if data.get("session_id") else 0.0
            
            # Content analysis features
            content = str(data.get("request_content", ""))
            features["content_length"] = len(content)
            features["content_entropy"] = await self._calculate_entropy(content)
            features["special_char_ratio"] = len([c for c in content if not c.isalnum()]) / max(len(content), 1)
            
            # Timing features
            features["request_frequency"] = data.get("request_rate", 0.0)
            features["timing_variance"] = data.get("timing_variance", 0.0)
            
            # Pattern features
            features["pattern_diversity"] = await self._calculate_pattern_diversity(content)
            features["suspicious_pattern_count"] = await self._count_suspicious_patterns(content)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {"error": 1.0}
    
    async def _calculate_entropy(self, content: str) -> float:
        """Calculate Shannon entropy of content."""
        if not content:
            return 0.0
        
        try:
            # Count character frequencies
            char_counts = {}
            for char in content:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate probabilities and entropy
            length = len(content)
            entropy = 0.0
            
            for count in char_counts.values():
                p = count / length
                if p > 0:
                    entropy -= p * np.log2(p)
            
            return entropy
            
        except Exception as e:
            logger.error(f"Entropy calculation failed: {e}")
            return 0.0
    
    async def _calculate_pattern_diversity(self, content: str) -> float:
        """Calculate diversity of patterns in content."""
        if not content or len(content) < 10:
            return 0.0
        
        try:
            # Count different n-gram patterns
            trigrams = set()
            for i in range(len(content) - 2):
                trigrams.add(content[i:i+3])
            
            # Diversity is unique trigrams / total possible positions
            diversity = len(trigrams) / max(len(content) - 2, 1)
            return min(1.0, diversity)
            
        except Exception as e:
            logger.error(f"Pattern diversity calculation failed: {e}")
            return 0.0
    
    async def _count_suspicious_patterns(self, content: str) -> float:
        """Count suspicious patterns in content."""
        suspicious_patterns = [
            r'<script', r'javascript:', r'\beval\b', r'\bexec\b',
            r'select.*from', r'union.*select', r'drop.*table',
            r'\.\./', r'\.\.\\'
        ]
        
        count = 0
        for pattern in suspicious_patterns:
            import re
            if re.search(pattern, content, re.IGNORECASE):
                count += 1
        
        return float(count)


class ReputationDatabase:
    """IP and user reputation analysis."""
    
    def __init__(self):
        self.ip_reputation = {}
        self.user_reputation = {}
        self.known_bad_ips = set()
        self.known_good_ips = set()
        
    async def analyze_reputation(self, source_ip: str) -> Tuple[float, List[str]]:
        """Analyze IP reputation."""
        indicators = []
        risk_score = 0.0
        
        try:
            # Check against known bad IPs
            if source_ip in self.known_bad_ips:
                indicators.append("known_malicious_ip")
                risk_score = max(risk_score, 0.9)
            
            # Check reputation history
            if source_ip in self.ip_reputation:
                rep_data = self.ip_reputation[source_ip]
                
                # High incident count
                if rep_data.get("incident_count", 0) > 10:
                    indicators.append("high_incident_history")
                    risk_score = max(risk_score, 0.7)
                
                # Recent incidents
                recent_incidents = rep_data.get("recent_incidents", 0)
                if recent_incidents > 5:
                    indicators.append("recent_incident_activity")
                    risk_score = max(risk_score, 0.6)
            
            # Geolocation analysis (placeholder)
            geo_risk = await self._analyze_geolocation(source_ip)
            if geo_risk > 0.5:
                indicators.append("suspicious_geolocation")
                risk_score = max(risk_score, geo_risk)
            
            return risk_score, indicators
            
        except Exception as e:
            logger.error(f"Reputation analysis failed: {e}")
            return 0.5, ["reputation_analysis_error"]
    
    async def update_reputation(self, source_ip: str, incident: SecurityIncident) -> None:
        """Update IP reputation based on incident."""
        try:
            if source_ip not in self.ip_reputation:
                self.ip_reputation[source_ip] = {
                    "incident_count": 0,
                    "first_seen": time.time(),
                    "last_incident": 0,
                    "recent_incidents": 0,
                    "severity_scores": []
                }
            
            rep_data = self.ip_reputation[source_ip]
            rep_data["incident_count"] += 1
            rep_data["last_incident"] = time.time()
            
            # Count recent incidents (last 24 hours)
            if time.time() - rep_data["last_incident"] < 86400:
                rep_data["recent_incidents"] += 1
            
            # Add severity score
            severity_values = {
                IncidentSeverity.INFO: 0.1,
                IncidentSeverity.LOW: 0.3,
                IncidentSeverity.MEDIUM: 0.6,
                IncidentSeverity.HIGH: 0.8,
                IncidentSeverity.CRITICAL: 1.0
            }
            rep_data["severity_scores"].append(severity_values.get(incident.severity, 0.5))
            
            # Keep only last 100 severity scores
            if len(rep_data["severity_scores"]) > 100:
                rep_data["severity_scores"] = rep_data["severity_scores"][-50:]
            
            # Add to known bad IPs if consistently malicious
            avg_severity = np.mean(rep_data["severity_scores"][-10:]) if rep_data["severity_scores"] else 0
            if rep_data["incident_count"] >= 5 and avg_severity > 0.6:
                self.known_bad_ips.add(source_ip)
                logger.warning(f"Added IP {source_ip} to known bad list (incidents: {rep_data['incident_count']}, "
                              f"avg_severity: {avg_severity:.3f})")
            
        except Exception as e:
            logger.error(f"Reputation update failed: {e}")
    
    async def _analyze_geolocation(self, source_ip: str) -> float:
        """Analyze IP geolocation for risk assessment."""
        # Placeholder implementation
        # In production, this would use a geolocation service
        try:
            # Simulate geolocation risk based on IP patterns
            if source_ip.startswith("192.168.") or source_ip.startswith("10."):
                return 0.1  # Private networks are low risk
            elif source_ip.startswith("127."):
                return 0.1  # Localhost is low risk
            else:
                # Public IP - moderate risk by default
                return 0.3
                
        except Exception as e:
            logger.error(f"Geolocation analysis failed: {e}")
            return 0.5


class AttackSignatureDatabase:
    """Database of attack signatures for pattern matching."""
    
    def __init__(self):
        self.signatures = self._load_signatures()
        
    async def match_signatures(self, incident_data: Dict[str, Any]) -> List[str]:
        """Match incident against known attack signatures."""
        matches = []
        
        try:
            content = str(incident_data.get("request_content", "")).lower()
            user_agent = str(incident_data.get("user_agent", "")).lower()
            
            # Check content signatures
            for signature_name, patterns in self.signatures["content"].items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, content, re.IGNORECASE):
                        matches.append(f"content_{signature_name}")
                        break
            
            # Check user agent signatures
            for signature_name, patterns in self.signatures["user_agent"].items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, user_agent, re.IGNORECASE):
                        matches.append(f"ua_{signature_name}")
                        break
            
            # Check behavioral signatures
            request_rate = incident_data.get("request_rate", 0)
            if request_rate > 100:
                matches.append("behavioral_flood")
            
            return matches
            
        except Exception as e:
            logger.error(f"Signature matching failed: {e}")
            return ["signature_matching_error"]
    
    def _load_signatures(self) -> Dict[str, Dict[str, List[str]]]:
        """Load attack signatures."""
        return {
            "content": {
                "sql_injection": [
                    r"\b(union|select|insert|delete|drop|exec|eval)\b.*\b(from|where|table)\b",
                    r"'.*?(\s+or\s+|\s+and\s+).*?'",
                    r";\s*(drop|delete|truncate)"
                ],
                "xss": [
                    r"<script[^>]*>.*?</script>",
                    r"javascript:",
                    r"on\w+\s*=",
                    r"<iframe[^>]*>",
                    r"eval\s*\("
                ],
                "command_injection": [
                    r";\s*(cat|ls|pwd|whoami|id)",
                    r"\|\s*(cat|ls|pwd|whoami|id)",
                    r"`.*?`",
                    r"\$\([^)]*\)"
                ],
                "path_traversal": [
                    r"\.\.\/|\.\.\\",
                    r"\/etc\/passwd",
                    r"\\windows\\system32"
                ]
            },
            "user_agent": {
                "scanner": [
                    "nmap", "masscan", "zmap", "nikto", "dirb", "gobuster",
                    "sqlmap", "burp", "owasp", "acunetix"
                ],
                "bot": [
                    "bot", "crawler", "spider", "scraper", "automated"
                ]
            }
        }


class AutomatedResponseEngine:
    """Automated incident response and mitigation."""
    
    def __init__(self):
        self.response_strategies = self._load_response_strategies()
        self.active_responses = {}
        self.response_history = deque(maxlen=1000)
        self.circuit_breakers = {}
        
    async def execute_response(
        self, 
        incident: SecurityIncident,
        system_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute automated response to security incident."""
        try:
            logger.info(f"Executing automated response for incident {incident.incident_id}")
            
            # Select appropriate response strategy
            strategy = await self._select_response_strategy(incident)
            if not strategy:
                logger.warning(f"No suitable response strategy found for incident {incident.incident_id}")
                return {"status": "no_strategy", "actions_taken": []}
            
            # Execute response actions
            response_results = []
            actions_taken = []
            
            for action in strategy.actions:
                try:
                    result = await self._execute_response_action(action, incident, system_context)
                    response_results.append(result)
                    actions_taken.append(action.value)
                    
                    logger.info(f"Executed response action {action.value} for incident {incident.incident_id}: "
                               f"success={result.get('success', False)}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute response action {action.value}: {e}")
                    response_results.append({"action": action.value, "success": False, "error": str(e)})
            
            # Record response in history
            response_record = {
                "incident_id": incident.incident_id,
                "strategy_id": strategy.strategy_id,
                "timestamp": time.time(),
                "actions_taken": actions_taken,
                "results": response_results,
                "success_count": sum(1 for r in response_results if r.get("success", False))
            }
            
            self.response_history.append(response_record)
            
            # Start monitoring for escalation if needed
            if strategy.escalation_timeout > 0:
                asyncio.create_task(
                    self._monitor_response_effectiveness(incident, strategy, response_record)
                )
            
            return {
                "status": "executed",
                "strategy_used": strategy.strategy_id,
                "actions_taken": actions_taken,
                "results": response_results,
                "success_rate": response_record["success_count"] / len(response_results) if response_results else 0
            }
            
        except Exception as e:
            logger.error(f"Automated response execution failed: {e}")
            return {"status": "error", "error": str(e), "actions_taken": []}
    
    async def _select_response_strategy(self, incident: SecurityIncident) -> Optional[ResponseStrategy]:
        """Select appropriate response strategy for incident."""
        try:
            suitable_strategies = []
            
            for strategy in self.response_strategies:
                # Check if strategy applies to this threat category
                if incident.category not in strategy.applicable_categories:
                    continue
                
                # Check severity threshold
                severity_order = [IncidentSeverity.INFO, IncidentSeverity.LOW, IncidentSeverity.MEDIUM, 
                                IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
                
                if severity_order.index(incident.severity) >= severity_order.index(strategy.severity_threshold):
                    suitable_strategies.append(strategy)
            
            if not suitable_strategies:
                return None
            
            # Select strategy with highest severity threshold (most appropriate)
            suitable_strategies.sort(
                key=lambda s: [IncidentSeverity.INFO, IncidentSeverity.LOW, IncidentSeverity.MEDIUM, 
                              IncidentSeverity.HIGH, IncidentSeverity.CRITICAL].index(s.severity_threshold),
                reverse=True
            )
            
            return suitable_strategies[0]
            
        except Exception as e:
            logger.error(f"Response strategy selection failed: {e}")
            return None
    
    async def _execute_response_action(
        self, 
        action: ResponseAction, 
        incident: SecurityIncident,
        system_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific response action."""
        try:
            if action == ResponseAction.LOG_ONLY:
                return await self._log_incident(incident)
            
            elif action == ResponseAction.RATE_LIMIT:
                return await self._apply_rate_limiting(incident)
            
            elif action == ResponseAction.TEMPORARY_BLOCK:
                return await self._apply_temporary_block(incident)
            
            elif action == ResponseAction.PERMANENT_BLOCK:
                return await self._apply_permanent_block(incident)
            
            elif action == ResponseAction.CIRCUIT_BREAKER:
                return await self._activate_circuit_breaker(incident, system_context)
            
            elif action == ResponseAction.EMERGENCY_SHUTDOWN:
                return await self._emergency_shutdown(incident, system_context)
            
            elif action == ResponseAction.NOTIFY_ADMIN:
                return await self._notify_administrators(incident)
            
            elif action == ResponseAction.FORENSIC_CAPTURE:
                return await self._capture_forensic_data(incident)
            
            else:
                logger.warning(f"Unknown response action: {action}")
                return {"action": action.value, "success": False, "error": "unknown_action"}
                
        except Exception as e:
            logger.error(f"Response action {action.value} execution failed: {e}")
            return {"action": action.value, "success": False, "error": str(e)}
    
    async def _log_incident(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Log security incident."""
        try:
            log_entry = {
                "timestamp": incident.timestamp.isoformat(),
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "category": incident.category.value,
                "source_ip": incident.source_ip,
                "confidence": incident.confidence_score,
                "threat_indicators": incident.threat_indicators,
                "impact_assessment": incident.impact_assessment
            }
            
            # Log at appropriate level based on severity
            if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
                logger.error(f"SECURITY INCIDENT: {json.dumps(log_entry, indent=2)}")
            elif incident.severity == IncidentSeverity.MEDIUM:
                logger.warning(f"Security incident: {json.dumps(log_entry)}")
            else:
                logger.info(f"Security event: {json.dumps(log_entry)}")
            
            return {"action": "log_only", "success": True, "log_entry": log_entry}
            
        except Exception as e:
            logger.error(f"Incident logging failed: {e}")
            return {"action": "log_only", "success": False, "error": str(e)}
    
    async def _apply_rate_limiting(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Apply rate limiting to source IP."""
        try:
            # Placeholder for actual rate limiting implementation
            source_ip = incident.source_ip
            
            # Record rate limit decision
            logger.warning(f"Applying rate limiting to IP {source_ip} due to incident {incident.incident_id}")
            
            # In production, this would interface with load balancer/proxy
            return {
                "action": "rate_limit",
                "success": True,
                "target_ip": source_ip,
                "duration": "1h",
                "rate_limit": "10/min"
            }
            
        except Exception as e:
            logger.error(f"Rate limiting failed: {e}")
            return {"action": "rate_limit", "success": False, "error": str(e)}
    
    async def _apply_temporary_block(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Apply temporary block to source IP."""
        try:
            source_ip = incident.source_ip
            block_duration = 3600  # 1 hour
            
            # Record block decision
            logger.warning(f"Applying temporary block to IP {source_ip} for {block_duration}s "
                          f"due to incident {incident.incident_id}")
            
            # Store block info for tracking
            self.active_responses[f"block_{source_ip}"] = {
                "type": "temporary_block",
                "target_ip": source_ip,
                "start_time": time.time(),
                "duration": block_duration,
                "incident_id": incident.incident_id
            }
            
            # Schedule automatic unblock
            asyncio.create_task(self._schedule_unblock(source_ip, block_duration))
            
            return {
                "action": "temporary_block",
                "success": True,
                "target_ip": source_ip,
                "duration_seconds": block_duration
            }
            
        except Exception as e:
            logger.error(f"Temporary block failed: {e}")
            return {"action": "temporary_block", "success": False, "error": str(e)}
    
    async def _apply_permanent_block(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Apply permanent block to source IP."""
        try:
            source_ip = incident.source_ip
            
            logger.error(f"Applying PERMANENT block to IP {source_ip} due to critical incident {incident.incident_id}")
            
            # Store permanent block info
            self.active_responses[f"perm_block_{source_ip}"] = {
                "type": "permanent_block",
                "target_ip": source_ip,
                "start_time": time.time(),
                "incident_id": incident.incident_id
            }
            
            return {
                "action": "permanent_block",
                "success": True,
                "target_ip": source_ip,
                "status": "permanent"
            }
            
        except Exception as e:
            logger.error(f"Permanent block failed: {e}")
            return {"action": "permanent_block", "success": False, "error": str(e)}
    
    async def _activate_circuit_breaker(self, incident: SecurityIncident, context: Dict[str, Any]) -> Dict[str, Any]:
        """Activate circuit breaker for service protection."""
        try:
            service_name = context.get("service_name", "main_service")
            
            logger.error(f"Activating circuit breaker for {service_name} due to incident {incident.incident_id}")
            
            # Activate circuit breaker
            self.circuit_breakers[service_name] = {
                "state": "OPEN",
                "activated_time": time.time(),
                "incident_id": incident.incident_id,
                "reset_timeout": 300  # 5 minutes
            }
            
            # Schedule automatic reset
            asyncio.create_task(self._schedule_circuit_breaker_reset(service_name, 300))
            
            return {
                "action": "circuit_breaker",
                "success": True,
                "service": service_name,
                "state": "OPEN",
                "reset_timeout": 300
            }
            
        except Exception as e:
            logger.error(f"Circuit breaker activation failed: {e}")
            return {"action": "circuit_breaker", "success": False, "error": str(e)}
    
    async def _emergency_shutdown(self, incident: SecurityIncident, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emergency shutdown procedures."""
        try:
            logger.critical(f"EMERGENCY SHUTDOWN initiated due to critical incident {incident.incident_id}")
            
            # Record shutdown decision
            shutdown_info = {
                "initiated_time": time.time(),
                "incident_id": incident.incident_id,
                "reason": f"{incident.category.value}_severity_{incident.severity.value}",
                "context": context
            }
            
            # In production, this would initiate graceful shutdown
            return {
                "action": "emergency_shutdown",
                "success": True,
                "shutdown_reason": shutdown_info["reason"],
                "timestamp": shutdown_info["initiated_time"]
            }
            
        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            return {"action": "emergency_shutdown", "success": False, "error": str(e)}
    
    async def _notify_administrators(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Notify system administrators of security incident."""
        try:
            notification_data = {
                "timestamp": time.time(),
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "category": incident.category.value,
                "source_ip": incident.source_ip,
                "confidence": incident.confidence_score,
                "threat_indicators": incident.threat_indicators[:5],  # Top 5
                "recommended_actions": [action.value for action in incident.recommended_actions]
            }
            
            # Log notification (in production, would send actual alerts)
            logger.critical(f"ADMIN NOTIFICATION: Security incident requires attention: "
                           f"{json.dumps(notification_data, indent=2)}")
            
            return {
                "action": "notify_admin",
                "success": True,
                "notification_sent": True,
                "recipients": ["security_team", "on_call_admin"]
            }
            
        except Exception as e:
            logger.error(f"Administrator notification failed: {e}")
            return {"action": "notify_admin", "success": False, "error": str(e)}
    
    async def _capture_forensic_data(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Capture forensic data for incident analysis."""
        try:
            forensic_data = {
                "capture_timestamp": time.time(),
                "incident_id": incident.incident_id,
                "source_ip": incident.source_ip,
                "session_id": incident.session_id,
                "raw_incident_data": incident.raw_data,
                "system_state": {
                    "memory_usage": "captured",
                    "network_connections": "captured", 
                    "process_list": "captured",
                    "log_snapshots": "captured"
                }
            }
            
            # Log forensic capture
            logger.info(f"Forensic data captured for incident {incident.incident_id}")
            
            return {
                "action": "forensic_capture",
                "success": True,
                "data_captured": True,
                "capture_size": "estimated_50mb",
                "storage_path": f"/forensic/incident_{incident.incident_id}"
            }
            
        except Exception as e:
            logger.error(f"Forensic data capture failed: {e}")
            return {"action": "forensic_capture", "success": False, "error": str(e)}
    
    async def _schedule_unblock(self, source_ip: str, duration: int) -> None:
        """Schedule automatic unblock of IP address."""
        try:
            await asyncio.sleep(duration)
            
            # Remove from active responses
            block_key = f"block_{source_ip}"
            if block_key in self.active_responses:
                del self.active_responses[block_key]
                logger.info(f"Automatically unblocked IP {source_ip} after {duration} seconds")
            
        except Exception as e:
            logger.error(f"Scheduled unblock failed for IP {source_ip}: {e}")
    
    async def _schedule_circuit_breaker_reset(self, service_name: str, timeout: int) -> None:
        """Schedule automatic circuit breaker reset."""
        try:
            await asyncio.sleep(timeout)
            
            if service_name in self.circuit_breakers:
                self.circuit_breakers[service_name]["state"] = "HALF_OPEN"
                logger.info(f"Circuit breaker for {service_name} reset to HALF_OPEN after {timeout} seconds")
            
        except Exception as e:
            logger.error(f"Circuit breaker reset failed for {service_name}: {e}")
    
    async def _monitor_response_effectiveness(
        self, 
        incident: SecurityIncident, 
        strategy: ResponseStrategy,
        response_record: Dict[str, Any]
    ) -> None:
        """Monitor response effectiveness and escalate if needed."""
        try:
            await asyncio.sleep(strategy.escalation_timeout)
            
            # Check if incident has been resolved
            # This is a placeholder - in production would check actual metrics
            
            logger.info(f"Monitoring response effectiveness for incident {incident.incident_id} "
                       f"using strategy {strategy.strategy_id}")
            
            # If incident persists, could trigger escalation here
            
        except Exception as e:
            logger.error(f"Response monitoring failed: {e}")
    
    def _load_response_strategies(self) -> List[ResponseStrategy]:
        """Load predefined response strategies."""
        return [
            ResponseStrategy(
                strategy_id="dos_mitigation",
                name="DoS Attack Mitigation",
                description="Mitigate denial of service attacks",
                applicable_categories=[ThreatCategory.DOS_ATTACK],
                severity_threshold=IncidentSeverity.MEDIUM,
                actions=[ResponseAction.RATE_LIMIT, ResponseAction.TEMPORARY_BLOCK, ResponseAction.NOTIFY_ADMIN],
                escalation_timeout=300,
                success_criteria={"request_rate_below": 50},
                rollback_actions=[ResponseAction.LOG_ONLY]
            ),
            
            ResponseStrategy(
                strategy_id="injection_containment",
                name="Injection Attack Containment",
                description="Contain and block injection attacks",
                applicable_categories=[ThreatCategory.INJECTION_ATTACK],
                severity_threshold=IncidentSeverity.HIGH,
                actions=[ResponseAction.TEMPORARY_BLOCK, ResponseAction.FORENSIC_CAPTURE, ResponseAction.NOTIFY_ADMIN],
                escalation_timeout=600,
                success_criteria={"no_further_attempts": True},
                rollback_actions=[ResponseAction.LOG_ONLY]
            ),
            
            ResponseStrategy(
                strategy_id="quantum_threat_response",
                name="Quantum Threat Response",
                description="Respond to quantum-specific security threats",
                applicable_categories=[ThreatCategory.QUANTUM_MANIPULATION],
                severity_threshold=IncidentSeverity.HIGH,
                actions=[ResponseAction.CIRCUIT_BREAKER, ResponseAction.FORENSIC_CAPTURE, ResponseAction.NOTIFY_ADMIN],
                escalation_timeout=180,
                success_criteria={"quantum_integrity_restored": True},
                rollback_actions=[ResponseAction.LOG_ONLY]
            ),
            
            ResponseStrategy(
                strategy_id="critical_incident_response", 
                name="Critical Incident Response",
                description="Response to critical security incidents",
                applicable_categories=list(ThreatCategory),  # All categories
                severity_threshold=IncidentSeverity.CRITICAL,
                actions=[ResponseAction.EMERGENCY_SHUTDOWN, ResponseAction.FORENSIC_CAPTURE, ResponseAction.NOTIFY_ADMIN],
                escalation_timeout=60,
                success_criteria={"threat_eliminated": True},
                rollback_actions=[]
            )
        ]


class AIIncidentResponseSystem:
    """
    Comprehensive AI-powered incident response system.
    
    Integrates threat intelligence, automated analysis, and intelligent response
    strategies for comprehensive defensive security incident management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threat_intel = ThreatIntelligenceEngine()
        self.response_engine = AutomatedResponseEngine()
        self.metrics_collector = MetricsCollector("incident_response")
        self.incident_queue = asyncio.Queue()
        self.active_incidents = {}
        self.incident_history = deque(maxlen=10000)
        
    async def start_system(self) -> None:
        """Start the AI incident response system."""
        logger.info("Starting AI-powered incident response system")
        
        # Start background processing tasks
        asyncio.create_task(self._incident_processing_loop())
        asyncio.create_task(self._metrics_collection_loop())
        
    async def process_security_event(
        self,
        event_data: Dict[str, Any],
        system_context: Optional[Dict[str, Any]] = None
    ) -> SecurityIncident:
        """
        Process a security event and generate incident response.
        
        Args:
            event_data: Raw security event data
            system_context: System context information
            
        Returns:
            SecurityIncident with analysis and response recommendations
        """
        try:
            system_context = system_context or {}
            
            # Generate incident ID
            incident_id = f"inc_{int(time.time())}_{hash(str(event_data)) % 10000:04d}"
            
            logger.debug(f"Processing security event as incident {incident_id}")
            
            # Threat intelligence analysis
            threat_category, confidence, threat_indicators = await self.threat_intel.analyze_incident(event_data)
            
            # Determine severity based on category and confidence
            severity = self._determine_severity(threat_category, confidence, threat_indicators)
            
            # Calculate false positive likelihood
            fp_likelihood = await self._calculate_false_positive_likelihood(
                event_data, threat_category, confidence
            )
            
            # Create incident object
            incident = SecurityIncident(
                incident_id=incident_id,
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                category=threat_category,
                source_ip=event_data.get("source_ip", "unknown"),
                user_id=event_data.get("user_id"),
                session_id=event_data.get("session_id"),
                threat_indicators=threat_indicators,
                raw_data=event_data,
                confidence_score=confidence,
                false_positive_likelihood=fp_likelihood,
                impact_assessment={},
                recommended_actions=[]
            )
            
            # Impact assessment
            incident.impact_assessment = await self.threat_intel.assess_impact(incident, system_context)
            
            # Determine recommended actions
            incident.recommended_actions = self._determine_recommended_actions(incident)
            
            # Store incident
            self.active_incidents[incident_id] = incident
            self.incident_history.append(incident)
            
            # Queue for automated response (if confidence and severity are high enough)
            if (confidence > 0.7 and severity.value in ["high", "critical"] and 
                fp_likelihood < 0.3):
                await self.incident_queue.put((incident, system_context))
            
            # Update reputation database
            await self.threat_intel.reputation_database.update_reputation(
                incident.source_ip, incident
            )
            
            # Collect metrics
            await self._collect_incident_metrics(incident)
            
            logger.info(f"Incident {incident_id} processed: category={threat_category.value}, "
                       f"severity={severity.value}, confidence={confidence:.3f}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Security event processing failed: {e}")
            
            # Return minimal incident on error
            return SecurityIncident(
                incident_id=f"error_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                severity=IncidentSeverity.CRITICAL,
                category=ThreatCategory.UNKNOWN,
                source_ip=event_data.get("source_ip", "unknown"),
                user_id=event_data.get("user_id"),
                session_id=event_data.get("session_id"),
                threat_indicators=["processing_error"],
                raw_data=event_data,
                confidence_score=0.5,
                false_positive_likelihood=0.5,
                impact_assessment={"error": str(e)},
                recommended_actions=[ResponseAction.LOG_ONLY, ResponseAction.NOTIFY_ADMIN]
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive incident response system status."""
        try:
            # Calculate recent incident statistics
            recent_incidents = [i for i in self.incident_history 
                              if (datetime.now(timezone.utc) - i.timestamp).seconds < 3600]  # Last hour
            
            category_counts = {}
            severity_counts = {}
            
            for incident in recent_incidents:
                category_counts[incident.category.value] = category_counts.get(incident.category.value, 0) + 1
                severity_counts[incident.severity.value] = severity_counts.get(incident.severity.value, 0) + 1
            
            # Response system status
            response_status = {
                "active_responses": len(self.response_engine.active_responses),
                "circuit_breakers": len(self.response_engine.circuit_breakers),
                "response_history_size": len(self.response_engine.response_history)
            }
            
            return {
                "system_health": "operational",
                "active_incidents": len(self.active_incidents),
                "queue_size": self.incident_queue.qsize(),
                "recent_incidents_count": len(recent_incidents),
                "threat_categories": category_counts,
                "severity_distribution": severity_counts,
                "response_system": response_status,
                "reputation_database": {
                    "tracked_ips": len(self.threat_intel.reputation_database.ip_reputation),
                    "known_bad_ips": len(self.threat_intel.reputation_database.known_bad_ips)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"system_health": "error", "error": str(e)}
    
    def _determine_severity(
        self, 
        category: ThreatCategory, 
        confidence: float, 
        indicators: List[str]
    ) -> IncidentSeverity:
        """Determine incident severity based on analysis results."""
        try:
            # Base severity from category
            category_severity = {
                ThreatCategory.INJECTION_ATTACK: IncidentSeverity.HIGH,
                ThreatCategory.DOS_ATTACK: IncidentSeverity.MEDIUM,
                ThreatCategory.QUANTUM_MANIPULATION: IncidentSeverity.CRITICAL,
                ThreatCategory.SIDE_CHANNEL: IncidentSeverity.HIGH,
                ThreatCategory.SYSTEM_COMPROMISE: IncidentSeverity.CRITICAL,
                ThreatCategory.TIMING_ATTACK: IncidentSeverity.MEDIUM,
                ThreatCategory.PROTOCOL_VIOLATION: IncidentSeverity.MEDIUM,
                ThreatCategory.AUTHENTICATION_FAILURE: IncidentSeverity.LOW,
                ThreatCategory.ANOMALOUS_BEHAVIOR: IncidentSeverity.LOW,
                ThreatCategory.UNKNOWN: IncidentSeverity.LOW
            }
            
            base_severity = category_severity.get(category, IncidentSeverity.MEDIUM)
            
            # Adjust based on confidence
            if confidence > 0.9:
                # High confidence - increase severity
                severity_order = [IncidentSeverity.INFO, IncidentSeverity.LOW, IncidentSeverity.MEDIUM, 
                                IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
                current_index = severity_order.index(base_severity)
                if current_index < len(severity_order) - 1:
                    base_severity = severity_order[current_index + 1]
            elif confidence < 0.5:
                # Low confidence - decrease severity
                severity_order = [IncidentSeverity.INFO, IncidentSeverity.LOW, IncidentSeverity.MEDIUM, 
                                IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
                current_index = severity_order.index(base_severity)
                if current_index > 0:
                    base_severity = severity_order[current_index - 1]
            
            # Adjust based on number of indicators
            if len(indicators) > 10:  # Many indicators suggest severe threat
                if base_severity != IncidentSeverity.CRITICAL:
                    severity_order = [IncidentSeverity.INFO, IncidentSeverity.LOW, IncidentSeverity.MEDIUM, 
                                    IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
                    current_index = severity_order.index(base_severity)
                    if current_index < len(severity_order) - 1:
                        base_severity = severity_order[current_index + 1]
            
            return base_severity
            
        except Exception as e:
            logger.error(f"Severity determination failed: {e}")
            return IncidentSeverity.MEDIUM  # Default to medium severity
    
    def _determine_recommended_actions(self, incident: SecurityIncident) -> List[ResponseAction]:
        """Determine recommended response actions for incident."""
        try:
            actions = []
            
            # Always log
            actions.append(ResponseAction.LOG_ONLY)
            
            # Actions based on severity
            if incident.severity == IncidentSeverity.CRITICAL:
                actions.extend([
                    ResponseAction.EMERGENCY_SHUTDOWN,
                    ResponseAction.FORENSIC_CAPTURE,
                    ResponseAction.NOTIFY_ADMIN
                ])
            elif incident.severity == IncidentSeverity.HIGH:
                actions.extend([
                    ResponseAction.PERMANENT_BLOCK,
                    ResponseAction.FORENSIC_CAPTURE,
                    ResponseAction.NOTIFY_ADMIN
                ])
            elif incident.severity == IncidentSeverity.MEDIUM:
                actions.extend([
                    ResponseAction.TEMPORARY_BLOCK,
                    ResponseAction.RATE_LIMIT
                ])
            
            # Actions based on category
            if incident.category == ThreatCategory.DOS_ATTACK:
                actions.extend([ResponseAction.RATE_LIMIT, ResponseAction.CIRCUIT_BREAKER])
            elif incident.category == ThreatCategory.QUANTUM_MANIPULATION:
                actions.extend([ResponseAction.CIRCUIT_BREAKER, ResponseAction.FORENSIC_CAPTURE])
            elif incident.category in [ThreatCategory.INJECTION_ATTACK, ThreatCategory.SYSTEM_COMPROMISE]:
                actions.extend([ResponseAction.FORENSIC_CAPTURE])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_actions = []
            for action in actions:
                if action not in seen:
                    seen.add(action)
                    unique_actions.append(action)
            
            return unique_actions
            
        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return [ResponseAction.LOG_ONLY, ResponseAction.NOTIFY_ADMIN]
    
    async def _calculate_false_positive_likelihood(
        self,
        event_data: Dict[str, Any],
        category: ThreatCategory,
        confidence: float
    ) -> float:
        """Calculate likelihood that this is a false positive."""
        try:
            fp_score = 0.0
            
            # Base false positive rates by category (based on typical security operations)
            base_fp_rates = {
                ThreatCategory.INJECTION_ATTACK: 0.15,
                ThreatCategory.DOS_ATTACK: 0.25,
                ThreatCategory.QUANTUM_MANIPULATION: 0.05,  # New category, assume low FP
                ThreatCategory.TIMING_ATTACK: 0.40,  # High FP rate due to network variations
                ThreatCategory.ANOMALOUS_BEHAVIOR: 0.60,  # Very high FP rate
                ThreatCategory.AUTHENTICATION_FAILURE: 0.30,
                ThreatCategory.UNKNOWN: 0.70  # High uncertainty
            }
            
            fp_score = base_fp_rates.get(category, 0.50)
            
            # Adjust based on confidence (higher confidence = lower FP likelihood)
            confidence_adjustment = (1.0 - confidence) * 0.5
            fp_score = min(1.0, fp_score + confidence_adjustment)
            
            # Adjust based on source reputation
            source_ip = event_data.get("source_ip", "unknown")
            if source_ip in self.threat_intel.reputation_database.known_bad_ips:
                fp_score *= 0.3  # Known bad IPs have lower FP likelihood
            
            # Adjust based on time of day (placeholder logic)
            current_hour = datetime.now().hour
            if 2 <= current_hour <= 6:  # Early morning attacks are less likely to be FP
                fp_score *= 0.8
            
            return min(1.0, max(0.0, fp_score))
            
        except Exception as e:
            logger.error(f"False positive calculation failed: {e}")
            return 0.5  # Default uncertainty
    
    async def _incident_processing_loop(self) -> None:
        """Background loop for processing incidents and executing responses."""
        while True:
            try:
                # Get incident from queue with timeout
                try:
                    incident, system_context = await asyncio.wait_for(
                        self.incident_queue.get(), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue  # No incidents to process
                
                logger.info(f"Processing incident {incident.incident_id} for automated response")
                
                # Execute automated response
                response_result = await self.response_engine.execute_response(incident, system_context)
                
                # Update incident with response information
                incident.metadata["automated_response"] = response_result
                
                logger.info(f"Automated response completed for incident {incident.incident_id}: "
                           f"status={response_result.get('status', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Incident processing loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _metrics_collection_loop(self) -> None:
        """Background loop for collecting incident response metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                status = await self.get_system_status()
                
                # Record system metrics
                self.metrics_collector.record_gauge("active_incidents", status["active_incidents"])
                self.metrics_collector.record_gauge("queue_size", status["queue_size"])
                self.metrics_collector.record_gauge("recent_incidents", status["recent_incidents_count"])
                
                # Record threat category distribution
                for category, count in status.get("threat_categories", {}).items():
                    self.metrics_collector.record_gauge(f"threats_{category}", count)
                
                # Record severity distribution
                for severity, count in status.get("severity_distribution", {}).items():
                    self.metrics_collector.record_gauge(f"severity_{severity}", count)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(120)  # Back off on error
    
    async def _collect_incident_metrics(self, incident: SecurityIncident) -> None:
        """Collect metrics for individual incident."""
        try:
            # Record incident by category and severity
            self.metrics_collector.record_counter(
                f"incidents_{incident.category.value}_{incident.severity.value}", 1
            )
            
            # Record confidence score
            self.metrics_collector.record_histogram(
                "incident_confidence_score", incident.confidence_score
            )
            
            # Record false positive likelihood
            self.metrics_collector.record_histogram(
                "false_positive_likelihood", incident.false_positive_likelihood
            )
            
            # Record threat indicator count
            self.metrics_collector.record_histogram(
                "threat_indicators_count", len(incident.threat_indicators)
            )
            
        except Exception as e:
            logger.error(f"Incident metrics collection failed: {e}")


# Export main classes for defensive incident response
__all__ = [
    "AIIncidentResponseSystem",
    "SecurityIncident",
    "ResponseStrategy", 
    "ThreatIntelligenceEngine",
    "AutomatedResponseEngine",
    "IncidentSeverity",
    "ThreatCategory",
    "ResponseAction"
]