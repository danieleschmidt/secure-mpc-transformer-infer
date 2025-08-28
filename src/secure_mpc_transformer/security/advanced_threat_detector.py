"""
Advanced AI-Powered Threat Detection System

Enterprise-grade threat detection using machine learning and behavioral analysis
for the secure MPC transformer system.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, deque
import hashlib
import statistics

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackVector(Enum):
    """Known attack vectors."""
    TIMING_ATTACK = "timing_attack"
    SIDE_CHANNEL = "side_channel"
    DATA_INJECTION = "data_injection"
    REPLAY_ATTACK = "replay_attack"
    MPC_MANIPULATION = "mpc_manipulation"
    QUANTUM_INTERFERENCE = "quantum_interference"
    PROTOCOL_DEVIATION = "protocol_deviation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class ThreatEvent:
    """Represents a detected threat event."""
    timestamp: float
    threat_level: ThreatLevel
    attack_vector: AttackVector
    confidence: float
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp,
            "threat_level": self.threat_level.value,
            "attack_vector": self.attack_vector.value,
            "confidence": self.confidence,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "request_payload": self.request_payload,
            "context": self.context
        }


class BehavioralAnalyzer:
    """Analyzes user behavior patterns to detect anomalies."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.1):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.user_patterns: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.isolation_forest = IsolationForest(
            contamination=sensitivity,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        
    def record_request(self, user_id: str, features: Dict[str, float]):
        """Record a request for behavioral analysis."""
        self.user_patterns[user_id].append({
            'timestamp': time.time(),
            **features
        })
        
        # Retrain model periodically with new data
        if len(self.user_patterns[user_id]) >= 50:
            self._retrain_model(user_id)
    
    def analyze_request(self, user_id: str, features: Dict[str, float]) -> float:
        """Analyze a request and return anomaly score (0-1)."""
        if not self._is_fitted or user_id not in self.user_patterns:
            return 0.0
            
        # Extract feature vector
        feature_vector = self._extract_features(features)
        
        # Get anomaly score
        anomaly_score = self.isolation_forest.decision_function([feature_vector])[0]
        
        # Convert to 0-1 probability
        normalized_score = max(0, -anomaly_score)
        return min(1, normalized_score)
    
    def _extract_features(self, features: Dict[str, float]) -> List[float]:
        """Extract numerical features for ML analysis."""
        return [
            features.get('request_size', 0),
            features.get('response_time', 0),
            features.get('cpu_usage', 0),
            features.get('memory_usage', 0),
            features.get('network_latency', 0),
            features.get('quantum_coherence', 0.5)
        ]
    
    def _retrain_model(self, user_id: str):
        """Retrain the anomaly detection model."""
        patterns = list(self.user_patterns[user_id])
        if len(patterns) < 20:
            return
            
        # Prepare training data
        features = [self._extract_features(p) for p in patterns]
        
        # Fit scaler and model
        try:
            scaled_features = self.scaler.fit_transform(features)
            self.isolation_forest.fit(scaled_features)
            self._is_fitted = True
        except Exception as e:
            logger.warning(f"Failed to retrain model for {user_id}: {e}")


class QuantumSecurityMonitor:
    """Monitors quantum-specific security threats."""
    
    def __init__(self):
        self.coherence_history: deque = deque(maxlen=1000)
        self.entanglement_violations: List[float] = []
        self.state_manipulation_attempts = 0
        
    def monitor_quantum_state(self, quantum_metrics: Dict[str, float]) -> List[ThreatEvent]:
        """Monitor quantum state for security threats."""
        threats = []
        
        coherence = quantum_metrics.get('coherence', 0.5)
        entanglement = quantum_metrics.get('entanglement', 0.0)
        fidelity = quantum_metrics.get('fidelity', 1.0)
        
        self.coherence_history.append(coherence)
        
        # Detect coherence attacks
        if len(self.coherence_history) >= 10:
            recent_variance = statistics.variance(list(self.coherence_history)[-10:])
            if recent_variance > 0.1:  # Threshold for coherence manipulation
                threats.append(ThreatEvent(
                    timestamp=time.time(),
                    threat_level=ThreatLevel.HIGH,
                    attack_vector=AttackVector.QUANTUM_INTERFERENCE,
                    confidence=min(0.9, recent_variance * 5),
                    context={
                        'coherence_variance': recent_variance,
                        'current_coherence': coherence
                    }
                ))
        
        # Detect entanglement violations
        if entanglement < 0.0 or entanglement > 1.0:
            self.entanglement_violations.append(time.time())
            threats.append(ThreatEvent(
                timestamp=time.time(),
                threat_level=ThreatLevel.CRITICAL,
                attack_vector=AttackVector.QUANTUM_INTERFERENCE,
                confidence=0.95,
                context={
                    'invalid_entanglement': entanglement,
                    'violation_count': len(self.entanglement_violations)
                }
            ))
        
        # Detect state fidelity attacks
        if fidelity < 0.8:
            threats.append(ThreatEvent(
                timestamp=time.time(),
                threat_level=ThreatLevel.MEDIUM,
                attack_vector=AttackVector.MPC_MANIPULATION,
                confidence=1.0 - fidelity,
                context={'fidelity': fidelity}
            ))
            
        return threats


class TimingAttackDetector:
    """Detects timing-based attacks on MPC operations."""
    
    def __init__(self, baseline_samples: int = 100):
        self.timing_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=baseline_samples)
        )
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        self.timing_history[operation].append(duration)
        
        # Update baseline statistics
        if len(self.timing_history[operation]) >= 20:
            timings = list(self.timing_history[operation])
            self.baseline_stats[operation] = {
                'mean': statistics.mean(timings),
                'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
                'median': statistics.median(timings)
            }
    
    def analyze_timing(self, operation: str, duration: float) -> Optional[ThreatEvent]:
        """Analyze timing and detect potential attacks."""
        if operation not in self.baseline_stats:
            return None
            
        stats = self.baseline_stats[operation]
        
        # Calculate z-score
        if stats['stdev'] > 0:
            z_score = abs(duration - stats['mean']) / stats['stdev']
        else:
            z_score = 0
            
        # Detect timing anomalies
        if z_score > 3.0:  # 3 standard deviations
            confidence = min(0.9, z_score / 10.0)
            return ThreatEvent(
                timestamp=time.time(),
                threat_level=ThreatLevel.HIGH if z_score > 5 else ThreatLevel.MEDIUM,
                attack_vector=AttackVector.TIMING_ATTACK,
                confidence=confidence,
                context={
                    'operation': operation,
                    'duration': duration,
                    'expected_mean': stats['mean'],
                    'z_score': z_score
                }
            )
        
        return None


class AdvancedThreatDetector:
    """Main threat detection orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize sub-detectors
        self.behavioral_analyzer = BehavioralAnalyzer(
            sensitivity=self.config.get('behavioral_sensitivity', 0.1)
        )
        self.quantum_monitor = QuantumSecurityMonitor()
        self.timing_detector = TimingAttackDetector(
            baseline_samples=self.config.get('timing_baseline_samples', 100)
        )
        
        # Threat event storage
        self.threat_events: deque = deque(maxlen=10000)
        self.active_threats: Set[str] = set()
        
        # Rate limiting and IP tracking
        self.ip_request_counts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.blocked_ips: Set[str] = set()
        
        logger.info("Advanced Threat Detector initialized")
    
    async def analyze_request(
        self,
        request_data: Dict[str, Any],
        quantum_metrics: Optional[Dict[str, float]] = None,
        timing_data: Optional[Dict[str, float]] = None
    ) -> List[ThreatEvent]:
        """Comprehensive threat analysis of a request."""
        threats = []
        current_time = time.time()
        
        # Extract request metadata
        source_ip = request_data.get('source_ip', 'unknown')
        user_agent = request_data.get('user_agent', 'unknown')
        user_id = request_data.get('user_id', source_ip)
        
        # Rate limiting check
        self.ip_request_counts[source_ip].append(current_time)
        recent_requests = [
            t for t in self.ip_request_counts[source_ip] 
            if current_time - t < 60  # Last minute
        ]
        
        if len(recent_requests) > 100:  # More than 100 requests per minute
            threats.append(ThreatEvent(
                timestamp=current_time,
                threat_level=ThreatLevel.HIGH,
                attack_vector=AttackVector.RESOURCE_EXHAUSTION,
                confidence=0.9,
                source_ip=source_ip,
                context={'request_rate': len(recent_requests)}
            ))
            self.blocked_ips.add(source_ip)
        
        # Behavioral analysis
        if user_id and 'features' in request_data:
            self.behavioral_analyzer.record_request(user_id, request_data['features'])
            anomaly_score = self.behavioral_analyzer.analyze_request(
                user_id, request_data['features']
            )
            
            if anomaly_score > 0.7:  # High anomaly threshold
                threats.append(ThreatEvent(
                    timestamp=current_time,
                    threat_level=ThreatLevel.MEDIUM,
                    attack_vector=AttackVector.PROTOCOL_DEVIATION,
                    confidence=anomaly_score,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    context={'anomaly_score': anomaly_score}
                ))
        
        # Quantum security monitoring
        if quantum_metrics:
            quantum_threats = self.quantum_monitor.monitor_quantum_state(quantum_metrics)
            threats.extend(quantum_threats)
        
        # Timing attack detection
        if timing_data:
            for operation, duration in timing_data.items():
                self.timing_detector.record_timing(operation, duration)
                timing_threat = self.timing_detector.analyze_timing(operation, duration)
                if timing_threat:
                    threats.append(timing_threat)
        
        # Store detected threats
        for threat in threats:
            self.threat_events.append(threat)
            threat_key = f"{threat.attack_vector.value}_{threat.source_ip or 'unknown'}"
            self.active_threats.add(threat_key)
        
        # Log critical threats
        critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
        if critical_threats:
            logger.critical(f"Critical threats detected: {len(critical_threats)}")
            for threat in critical_threats:
                logger.critical(f"Threat: {threat.to_dict()}")
        
        return threats
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get comprehensive threat intelligence summary."""
        recent_threats = [
            t for t in self.threat_events 
            if time.time() - t.timestamp < 3600  # Last hour
        ]
        
        # Threat level distribution
        threat_levels = {}
        for level in ThreatLevel:
            threat_levels[level.value] = len([
                t for t in recent_threats if t.threat_level == level
            ])
        
        # Attack vector distribution
        attack_vectors = {}
        for vector in AttackVector:
            attack_vectors[vector.value] = len([
                t for t in recent_threats if t.attack_vector == vector
            ])
        
        # Top threat sources
        source_counts = defaultdict(int)
        for threat in recent_threats:
            if threat.source_ip:
                source_counts[threat.source_ip] += 1
        
        top_sources = sorted(
            source_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_threats_last_hour': len(recent_threats),
            'threat_levels': threat_levels,
            'attack_vectors': attack_vectors,
            'top_threat_sources': dict(top_sources),
            'active_threats': len(self.active_threats),
            'blocked_ips': len(self.blocked_ips),
            'quantum_coherence_history': list(self.quantum_monitor.coherence_history)[-50:],
            'last_updated': time.time()
        }
    
    def is_blocked(self, source_ip: str) -> bool:
        """Check if an IP is blocked."""
        return source_ip in self.blocked_ips
    
    def unblock_ip(self, source_ip: str) -> bool:
        """Unblock an IP address."""
        if source_ip in self.blocked_ips:
            self.blocked_ips.remove(source_ip)
            logger.info(f"Unblocked IP: {source_ip}")
            return True
        return False


# Global instance
_threat_detector: Optional[AdvancedThreatDetector] = None


def get_threat_detector() -> AdvancedThreatDetector:
    """Get the global threat detector instance."""
    global _threat_detector
    if _threat_detector is None:
        _threat_detector = AdvancedThreatDetector()
    return _threat_detector


async def analyze_request_security(
    request_data: Dict[str, Any],
    quantum_metrics: Optional[Dict[str, float]] = None,
    timing_data: Optional[Dict[str, float]] = None
) -> List[ThreatEvent]:
    """Convenience function for threat analysis."""
    detector = get_threat_detector()
    return await detector.analyze_request(request_data, quantum_metrics, timing_data)