"""Security service for MPC operations and monitoring."""

import time
import logging
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading

from ..utils.validators import SecurityValidator
from ..utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event data structure."""
    
    event_id: str
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: float
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class ThreatDetection:
    """Threat detection result."""
    
    threat_type: str
    confidence: float
    risk_score: int  # 1-100
    description: str
    recommended_action: str
    evidence: Dict[str, Any]


class SecurityAuditor:
    """Security auditing and monitoring."""
    
    def __init__(self):
        self.audit_log: List[SecurityEvent] = []
        self.threat_patterns = self._load_threat_patterns()
        self.rate_limits = defaultdict(deque)
        self.blocked_ips = set()
        self.max_audit_entries = 10000
        self._lock = threading.Lock()
        
    def log_security_event(self, event_type: str, severity: str, source_ip: str,
                          details: Dict[str, Any], user_id: Optional[str] = None):
        """Log a security event."""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            source_ip=source_ip,
            user_id=user_id,
            details=details
        )
        
        with self._lock:
            self.audit_log.append(event)
            
            # Maintain size limit
            if len(self.audit_log) > self.max_audit_entries:
                self.audit_log = self.audit_log[-self.max_audit_entries:]
        
        logger.info(f"Security event logged: {event_type} - {severity}")
        
        # Trigger automatic response for critical events
        if severity == "CRITICAL":
            self._handle_critical_event(event)
    
    def detect_threats(self, request_data: Dict[str, Any], source_ip: str) -> List[ThreatDetection]:
        """Detect potential security threats."""
        threats = []
        
        # Rate limiting check
        rate_threat = self._check_rate_limiting(source_ip)
        if rate_threat:
            threats.append(rate_threat)
        
        # Input validation threats
        input_threats = self._detect_input_threats(request_data)
        threats.extend(input_threats)
        
        # Protocol manipulation threats
        protocol_threats = self._detect_protocol_threats(request_data)
        threats.extend(protocol_threats)
        
        # Timing attack detection
        timing_threats = self._detect_timing_attacks(request_data, source_ip)
        threats.extend(timing_threats)
        
        return threats
    
    def _check_rate_limiting(self, source_ip: str) -> Optional[ThreatDetection]:
        """Check for rate limiting violations."""
        current_time = time.time()
        window_size = 60  # 60 seconds
        max_requests = 100  # requests per minute
        
        # Clean old entries
        ip_requests = self.rate_limits[source_ip]
        while ip_requests and ip_requests[0] < current_time - window_size:
            ip_requests.popleft()
        
        # Add current request
        ip_requests.append(current_time)
        
        if len(ip_requests) > max_requests:
            return ThreatDetection(
                threat_type="RATE_LIMITING_VIOLATION",
                confidence=1.0,
                risk_score=70,
                description=f"Source {source_ip} exceeded rate limit: {len(ip_requests)} requests/min",
                recommended_action="BLOCK_IP",
                evidence={"request_count": len(ip_requests), "window_size": window_size}
            )
        
        return None
    
    def _detect_input_threats(self, request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Detect threats in input data."""
        threats = []
        
        # Check text input for malicious patterns
        text_input = request_data.get('text', '')
        if isinstance(text_input, str):
            # SQL injection patterns
            sql_patterns = ['union select', 'drop table', 'exec(', 'script>']
            for pattern in sql_patterns:
                if pattern.lower() in text_input.lower():
                    threats.append(ThreatDetection(
                        threat_type="SQL_INJECTION_ATTEMPT",
                        confidence=0.8,
                        risk_score=90,
                        description=f"Potential SQL injection pattern detected: {pattern}",
                        recommended_action="BLOCK_REQUEST",
                        evidence={"pattern": pattern, "input": text_input[:100]}
                    ))
            
            # Unusually long input
            if len(text_input) > 50000:
                threats.append(ThreatDetection(
                    threat_type="EXCESSIVE_INPUT_SIZE",
                    confidence=0.9,
                    risk_score=60,
                    description=f"Unusually large input: {len(text_input)} characters",
                    recommended_action="VALIDATE_INPUT",
                    evidence={"input_length": len(text_input)}
                ))
        
        return threats
    
    def _detect_protocol_threats(self, request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Detect protocol manipulation threats."""
        threats = []
        
        protocol_config = request_data.get('protocol_config', {})
        
        # Check for unusual protocol parameters
        if 'security_level' in protocol_config:
            security_level = protocol_config['security_level']
            if security_level < 80:
                threats.append(ThreatDetection(
                    threat_type="WEAK_SECURITY_PARAMETERS",
                    confidence=0.9,
                    risk_score=80,
                    description=f"Weak security level requested: {security_level}",
                    recommended_action="REJECT_REQUEST",
                    evidence={"security_level": security_level}
                ))
        
        # Check for suspicious party configuration
        if 'num_parties' in protocol_config:
            num_parties = protocol_config['num_parties']
            if num_parties > 10 or num_parties < 2:
                threats.append(ThreatDetection(
                    threat_type="SUSPICIOUS_PARTY_CONFIG",
                    confidence=0.8,
                    risk_score=60,
                    description=f"Unusual number of parties: {num_parties}",
                    recommended_action="VALIDATE_CONFIG",
                    evidence={"num_parties": num_parties}
                ))
        
        return threats
    
    def _detect_timing_attacks(self, request_data: Dict[str, Any], source_ip: str) -> List[ThreatDetection]:
        """Detect potential timing attacks."""
        threats = []
        
        # Simple timing attack detection based on request patterns
        current_time = time.time()
        recent_requests = [t for t in self.rate_limits[source_ip] if t > current_time - 10]
        
        if len(recent_requests) > 20:  # Many requests in short time
            intervals = [recent_requests[i] - recent_requests[i-1] for i in range(1, len(recent_requests))]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            if avg_interval < 0.1:  # Very fast requests
                threats.append(ThreatDetection(
                    threat_type="POTENTIAL_TIMING_ATTACK",
                    confidence=0.6,
                    risk_score=70,
                    description=f"Rapid sequential requests detected: avg interval {avg_interval:.3f}s",
                    recommended_action="MONITOR_CLOSELY",
                    evidence={"avg_interval": avg_interval, "request_count": len(recent_requests)}
                ))
        
        return threats
    
    def _handle_critical_event(self, event: SecurityEvent):
        """Handle critical security events."""
        if event.event_type == "RATE_LIMITING_VIOLATION":
            self.blocked_ips.add(event.source_ip)
            logger.critical(f"IP {event.source_ip} blocked due to critical security event")
        
        # Additional automatic responses can be added here
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = secrets.token_hex(8)
        return f"sec_{timestamp}_{random_part}"
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            "sql_injection": [
                "union select", "drop table", "insert into", "delete from",
                "exec(", "execute(", "sp_", "xp_", "' or 1=1", "' or '1'='1"
            ],
            "xss": [
                "<script", "javascript:", "onload=", "onerror=", "onclick=",
                "eval(", "alert(", "document.cookie"
            ],
            "command_injection": [
                "; rm -rf", "| cat", "$(", "`", "&& rm", "|| rm"
            ]
        }
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_events = [e for e in self.audit_log if e.timestamp > cutoff_time]
        
        events_by_type = defaultdict(int)
        events_by_severity = defaultdict(int)
        
        for event in recent_events:
            events_by_type[event.event_type] += 1
            events_by_severity[event.severity] += 1
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "events_by_type": dict(events_by_type),
            "events_by_severity": dict(events_by_severity),
            "blocked_ips": list(self.blocked_ips),
            "active_threats": len([e for e in recent_events if not e.resolved])
        }


class PrivacyAccountant:
    """Track privacy budget and leakage."""
    
    def __init__(self, epsilon_budget: float = 1.0, delta: float = 1e-5):
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.epsilon_spent = 0.0
        self.privacy_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def spend_privacy_budget(self, epsilon: float, operation: str, details: Dict[str, Any]):
        """Record privacy budget expenditure."""
        with self._lock:
            if self.epsilon_spent + epsilon > self.epsilon_budget:
                raise ValueError(f"Privacy budget exceeded: {self.epsilon_spent + epsilon} > {self.epsilon_budget}")
            
            self.epsilon_spent += epsilon
            
            privacy_entry = {
                "timestamp": time.time(),
                "epsilon": epsilon,
                "operation": operation,
                "details": details,
                "total_epsilon": self.epsilon_spent
            }
            
            self.privacy_log.append(privacy_entry)
        
        logger.info(f"Privacy budget spent: {epsilon:.6f} for {operation}")
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.epsilon_budget - self.epsilon_spent
    
    def reset_budget(self):
        """Reset privacy budget (use with caution)."""
        with self._lock:
            self.epsilon_spent = 0.0
            self.privacy_log.clear()
        
        logger.warning("Privacy budget reset")
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get privacy accounting summary."""
        with self._lock:
            return {
                "epsilon_budget": self.epsilon_budget,
                "epsilon_spent": self.epsilon_spent,
                "epsilon_remaining": self.epsilon_budget - self.epsilon_spent,
                "delta": self.delta,
                "operations_count": len(self.privacy_log),
                "budget_utilization": (self.epsilon_spent / self.epsilon_budget) * 100
            }


class SecurityService:
    """Main security service coordinating all security functions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = SecurityValidator()
        self.auditor = SecurityAuditor()
        self.privacy_accountant = PrivacyAccountant(
            epsilon_budget=self.config.get('privacy_epsilon_budget', 1.0),
            delta=self.config.get('privacy_delta', 1e-5)
        )
        self.metrics = MetricsCollector()
        
        # Security configuration
        self.enable_threat_detection = self.config.get('enable_threat_detection', True)
        self.enable_audit_logging = self.config.get('enable_audit_logging', True)
        self.enable_privacy_accounting = self.config.get('enable_privacy_accounting', True)
        
        logger.info("SecurityService initialized")
    
    def validate_request(self, request_data: Dict[str, Any], source_ip: str,
                        user_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate incoming request for security compliance."""
        errors = []
        
        try:
            # Input validation
            if 'text' in request_data:
                self._validate_text_input(request_data['text'])
            
            # Protocol validation
            if 'protocol_config' in request_data:
                self._validate_protocol_config(request_data['protocol_config'])
            
            # Threat detection
            if self.enable_threat_detection:
                threats = self.auditor.detect_threats(request_data, source_ip)
                
                for threat in threats:
                    if threat.risk_score > 80:  # High risk threshold
                        errors.append(f"Security threat detected: {threat.description}")
                        
                        if self.enable_audit_logging:
                            self.auditor.log_security_event(
                                event_type=threat.threat_type,
                                severity="HIGH" if threat.risk_score > 90 else "MEDIUM",
                                source_ip=source_ip,
                                user_id=user_id,
                                details=threat.evidence
                            )
            
            # Check if IP is blocked
            if source_ip in self.auditor.blocked_ips:
                errors.append("Request blocked: source IP is on blocklist")
            
            # Update metrics
            self.metrics.increment_counter("security_validations_total")
            if errors:
                self.metrics.increment_counter("security_validations_failed")
            else:
                self.metrics.increment_counter("security_validations_passed")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Security validation error: {str(e)}")
            self.metrics.increment_counter("security_validation_errors")
            return False, [f"Security validation failed: {str(e)}"]
    
    def _validate_text_input(self, text: str):
        """Validate text input for security."""
        if len(text) > 100000:  # 100K character limit
            raise ValueError("Text input exceeds maximum length")
        
        # Check for potentially malicious patterns
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*='
        ]
        
        import re
        for pattern in malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(f"Potentially malicious pattern detected: {pattern}")
    
    def _validate_protocol_config(self, config: Dict[str, Any]):
        """Validate protocol configuration."""
        if 'security_level' in config:
            security_level = config['security_level']
            if not isinstance(security_level, int) or security_level < 80:
                raise ValueError(f"Invalid security level: {security_level}")
        
        if 'num_parties' in config:
            num_parties = config['num_parties']
            if not isinstance(num_parties, int) or not (2 <= num_parties <= 10):
                raise ValueError(f"Invalid number of parties: {num_parties}")
    
    def record_privacy_expenditure(self, epsilon: float, operation: str, 
                                  details: Dict[str, Any]):
        """Record privacy budget expenditure."""
        if self.enable_privacy_accounting:
            try:
                self.privacy_accountant.spend_privacy_budget(epsilon, operation, details)
                self.metrics.observe_histogram("privacy_epsilon_spent", epsilon)
            except ValueError as e:
                logger.error(f"Privacy budget error: {str(e)}")
                raise
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        status = {
            "timestamp": time.time(),
            "service_status": "active",
            "threat_detection_enabled": self.enable_threat_detection,
            "audit_logging_enabled": self.enable_audit_logging,
            "privacy_accounting_enabled": self.enable_privacy_accounting
        }
        
        # Security summary
        if self.enable_audit_logging:
            status["security_summary"] = self.auditor.get_security_summary()
        
        # Privacy status
        if self.enable_privacy_accounting:
            status["privacy_summary"] = self.privacy_accountant.get_privacy_summary()
        
        # Metrics
        status["security_metrics"] = {
            "validations_total": self.metrics.get_counter_value("security_validations_total"),
            "validations_passed": self.metrics.get_counter_value("security_validations_passed"),
            "validations_failed": self.metrics.get_counter_value("security_validations_failed"),
            "validation_errors": self.metrics.get_counter_value("security_validation_errors")
        }
        
        return status
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            "report_timestamp": time.time(),
            "time_period_hours": hours,
            "service_configuration": {
                "threat_detection": self.enable_threat_detection,
                "audit_logging": self.enable_audit_logging,
                "privacy_accounting": self.enable_privacy_accounting
            }
        }
        
        # Security events summary
        if self.enable_audit_logging:
            report["security_events"] = self.auditor.get_security_summary(hours)
        
        # Privacy analysis
        if self.enable_privacy_accounting:
            report["privacy_analysis"] = self.privacy_accountant.get_privacy_summary()
        
        # Recommendations
        report["recommendations"] = self._generate_security_recommendations()
        
        return report
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        # Check privacy budget utilization
        if self.enable_privacy_accounting:
            budget_util = (self.privacy_accountant.epsilon_spent / 
                          self.privacy_accountant.epsilon_budget) * 100
            
            if budget_util > 80:
                recommendations.append("Privacy budget utilization is high (>80%). Consider resetting or increasing budget.")
            elif budget_util > 90:
                recommendations.append("CRITICAL: Privacy budget almost exhausted. Immediate action required.")
        
        # Check recent security events
        if self.enable_audit_logging:
            recent_summary = self.auditor.get_security_summary(hours=1)
            
            if recent_summary["total_events"] > 100:
                recommendations.append("High number of security events in the last hour. Investigate potential attack.")
            
            if "CRITICAL" in recent_summary["events_by_severity"]:
                recommendations.append("Critical security events detected. Review and respond immediately.")
        
        # General recommendations
        if not self.enable_threat_detection:
            recommendations.append("Threat detection is disabled. Enable for better security.")
        
        if not self.enable_audit_logging:
            recommendations.append("Audit logging is disabled. Enable for security monitoring.")
        
        return recommendations
    
    def emergency_lockdown(self, reason: str):
        """Emergency security lockdown."""
        logger.critical(f"EMERGENCY LOCKDOWN INITIATED: {reason}")
        
        # Clear privacy budget to prevent further operations
        if self.enable_privacy_accounting:
            self.privacy_accountant.epsilon_spent = self.privacy_accountant.epsilon_budget
        
        # Log emergency event
        if self.enable_audit_logging:
            self.auditor.log_security_event(
                event_type="EMERGENCY_LOCKDOWN",
                severity="CRITICAL",
                source_ip="SYSTEM",
                user_id="SYSTEM",
                details={"reason": reason, "initiated_by": "security_service"}
            )
        
        self.metrics.increment_counter("emergency_lockdowns")
        
        # Additional lockdown procedures can be implemented here