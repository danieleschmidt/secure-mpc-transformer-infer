"""Advanced threat detection and security intelligence system."""

import ipaddress
import json
import logging
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackVector(Enum):
    """Known attack vectors."""
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    TIMING_ATTACK = "timing_attack"
    PROTOCOL_MANIPULATION = "protocol_manipulation"
    DATA_EXFILTRATION = "data_exfiltration"
    SESSION_HIJACKING = "session_hijacking"
    REPLAY_ATTACK = "replay_attack"
    SIDE_CHANNEL = "side_channel"


@dataclass
class ThreatIntelligence:
    """Threat intelligence data."""

    threat_id: str
    attack_vector: AttackVector
    threat_level: ThreatLevel
    confidence_score: float  # 0.0 - 1.0
    source_ips: list[str]
    attack_patterns: list[str]
    timeline: list[float]
    mitigation_actions: list[str]
    false_positive_probability: float
    related_threats: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['attack_vector'] = self.attack_vector.value
        data['threat_level'] = self.threat_level.value
        return data


@dataclass
class SecurityEvent:
    """Enhanced security event with intelligence."""

    event_id: str
    timestamp: float
    source_ip: str
    user_agent: str | None
    endpoint: str
    method: str
    status_code: int
    response_time: float
    request_size: int
    response_size: int
    headers: dict[str, str]
    payload_hash: str | None
    geolocation: dict[str, str] | None
    threat_indicators: list[str]
    risk_score: int  # 0-100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class GeolocationService:
    """IP geolocation and reputation service."""

    def __init__(self):
        self._ip_cache = {}
        self._reputation_cache = {}
        self._known_malicious_ips = set()
        self._known_safe_ips = set()

        # Initialize with common threat intelligence
        self._load_threat_intelligence()

    def get_ip_info(self, ip: str) -> dict[str, str]:
        """Get geolocation info for IP address."""
        if ip in self._ip_cache:
            return self._ip_cache[ip]

        try:
            # For production, integrate with real geolocation service
            # This is a mock implementation
            ip_obj = ipaddress.ip_address(ip)

            if ip_obj.is_private:
                info = {"country": "Private", "region": "Internal", "risk": "low"}
            elif ip_obj.is_loopback:
                info = {"country": "Localhost", "region": "Local", "risk": "low"}
            else:
                # Mock geolocation data
                info = {
                    "country": "Unknown",
                    "region": "Unknown",
                    "city": "Unknown",
                    "risk": "medium"
                }

            self._ip_cache[ip] = info
            return info

        except ValueError:
            return {"country": "Invalid", "region": "Invalid", "risk": "high"}

    def get_reputation_score(self, ip: str) -> float:
        """Get IP reputation score (0.0 = malicious, 1.0 = safe)."""
        if ip in self._reputation_cache:
            return self._reputation_cache[ip]

        score = 0.5  # Neutral by default

        if ip in self._known_malicious_ips:
            score = 0.0
        elif ip in self._known_safe_ips:
            score = 1.0
        else:
            # Analyze IP characteristics
            try:
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback:
                    score = 0.8  # Internal IPs are generally safer
                elif self._is_known_cloud_provider(ip):
                    score = 0.6  # Cloud providers have mixed reputation

            except ValueError:
                score = 0.1  # Invalid IPs are suspicious

        self._reputation_cache[ip] = score
        return score

    def _load_threat_intelligence(self):
        """Load known threat intelligence."""
        # Known malicious IP ranges (examples)
        malicious_patterns = [
            "192.0.2.0/24",  # RFC 3330 test network
            "203.0.113.0/24"  # RFC 3330 test network
        ]

        for pattern in malicious_patterns:
            try:
                network = ipaddress.ip_network(pattern, strict=False)
                for ip in network.hosts():
                    self._known_malicious_ips.add(str(ip))
            except ValueError:
                pass

    def _is_known_cloud_provider(self, ip: str) -> bool:
        """Check if IP belongs to known cloud provider."""
        # This would typically use cloud provider IP ranges
        cloud_ranges = [
            "54.0.0.0/8",    # AWS
            "104.16.0.0/12", # Cloudflare
            "8.8.8.0/24"     # Google
        ]

        try:
            ip_obj = ipaddress.ip_address(ip)
            for range_str in cloud_ranges:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip_obj in network:
                    return True
        except ValueError:
            pass

        return False


class BehaviorAnalyzer:
    """Analyze user behavior patterns for anomalies."""

    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.user_patterns = defaultdict(lambda: {
            "requests": deque(),
            "endpoints": defaultdict(int),
            "user_agents": set(),
            "avg_response_time": 0.0,
            "error_rate": 0.0
        })
        self._lock = threading.Lock()

    def analyze_behavior(self, event: SecurityEvent) -> list[str]:
        """Analyze behavior and return anomaly indicators."""
        anomalies = []
        current_time = time.time()

        with self._lock:
            pattern = self.user_patterns[event.source_ip]

            # Clean old requests
            while pattern["requests"] and pattern["requests"][0]["timestamp"] < current_time - self.window_size:
                old_req = pattern["requests"].popleft()
                pattern["endpoints"][old_req["endpoint"]] -= 1
                if pattern["endpoints"][old_req["endpoint"]] <= 0:
                    del pattern["endpoints"][old_req["endpoint"]]

            # Add current request
            request_data = {
                "timestamp": event.timestamp,
                "endpoint": event.endpoint,
                "response_time": event.response_time,
                "status_code": event.status_code
            }
            pattern["requests"].append(request_data)
            pattern["endpoints"][event.endpoint] += 1

            if event.user_agent:
                pattern["user_agents"].add(event.user_agent)

            # Analyze patterns
            recent_requests = list(pattern["requests"])

            # Check request frequency
            if len(recent_requests) > 100:  # More than 100 requests in 5 minutes
                anomalies.append("high_request_frequency")

            # Check endpoint diversity
            if len(pattern["endpoints"]) == 1 and len(recent_requests) > 20:
                anomalies.append("single_endpoint_focus")

            # Check error rate
            error_count = sum(1 for req in recent_requests if req["status_code"] >= 400)
            error_rate = error_count / len(recent_requests) if recent_requests else 0
            if error_rate > 0.5:
                anomalies.append("high_error_rate")

            # Check response time patterns
            response_times = [req["response_time"] for req in recent_requests]
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                if event.response_time < avg_time * 0.1:  # Unusually fast
                    anomalies.append("unusually_fast_response")
                elif event.response_time > avg_time * 10:  # Unusually slow
                    anomalies.append("unusually_slow_response")

            # Check user agent diversity
            if len(pattern["user_agents"]) > 10:  # Too many different user agents
                anomalies.append("user_agent_rotation")

        return anomalies


class SignatureDatabase:
    """Database of attack signatures and patterns."""

    def __init__(self):
        self.signatures = {
            AttackVector.SQL_INJECTION: [
                r"(?i)(union\s+select|drop\s+table|exec\s*\(|xp_cmdshell)",
                r"(?i)(\'\s*or\s+1\s*=\s*1|\'\s*or\s+\'1\'\s*=\s*\'1)",
                r"(?i)(insert\s+into|update\s+.*\s+set|delete\s+from)",
                r"(?i)(waitfor\s+delay|sleep\s*\(|pg_sleep)"
            ],
            AttackVector.XSS: [
                r"(?i)(<script[^>]*>|javascript:|vbscript:|onload\s*=)",
                r"(?i)(onerror\s*=|onclick\s*=|onmouseover\s*=)",
                r"(?i)(document\.cookie|document\.write|eval\s*\()",
                r"(?i)(<iframe[^>]*>|<object[^>]*>|<embed[^>]*>)"
            ],
            AttackVector.PROTOCOL_MANIPULATION: [
                r"(?i)(security_level.*[0-7][0-9]|num_parties.*[1-9][0-9])",
                r"(?i)(protocol.*test|protocol.*debug|protocol.*bypass)",
                r"(?i)(malicious.*true|adversary.*true|corrupt.*true)"
            ],
            AttackVector.DATA_EXFILTRATION: [
                r"(?i)(extract|exfiltrate|dump|leak).*data",
                r"(?i)(download|export|backup).*database",
                r"(?i)(select.*from.*information_schema|show\s+tables)"
            ]
        }

    def match_signatures(self, text: str, headers: dict[str, str]) -> list[tuple[AttackVector, str]]:
        """Match text against known attack signatures."""
        matches = []

        # Check request body/parameters
        for attack_vector, patterns in self.signatures.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    matches.append((attack_vector, pattern))

        # Check headers
        for header_name, header_value in headers.items():
            for attack_vector, patterns in self.signatures.items():
                for pattern in patterns:
                    if re.search(pattern, header_value):
                        matches.append((attack_vector, f"header:{header_name}:{pattern}"))

        return matches


class AdvancedThreatDetector:
    """Advanced threat detection system with machine learning capabilities."""

    def __init__(self):
        self.geolocation = GeolocationService()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.signature_db = SignatureDatabase()

        # Threat tracking
        self.active_threats = {}
        self.threat_history = deque(maxlen=10000)
        self.blocked_ips = set()
        self.suspicious_ips = defaultdict(float)  # IP -> suspicion score

        # Rate limiting tracking
        self.request_windows = defaultdict(deque)
        self.failed_auth_attempts = defaultdict(deque)

        # DDoS detection
        self.connection_counts = defaultdict(int)
        self.ddos_thresholds = {
            "requests_per_minute": 120,
            "unique_ips_threshold": 100,
            "error_rate_threshold": 0.8,
            "bandwidth_threshold": 10 * 1024 * 1024  # 10MB/min
        }

        self._lock = threading.Lock()
        logger.info("Advanced threat detection system initialized")

    def analyze_request(self, event: SecurityEvent) -> ThreatIntelligence:
        """Comprehensive threat analysis of incoming request."""
        threat_indicators = []
        confidence_scores = []
        attack_vectors = []
        evidence = {}

        # IP reputation analysis
        ip_reputation = self.geolocation.get_reputation_score(event.source_ip)
        ip_info = self.geolocation.get_ip_info(event.source_ip)
        evidence["ip_reputation"] = ip_reputation
        evidence["ip_geolocation"] = ip_info

        if ip_reputation < 0.3:
            threat_indicators.append("low_ip_reputation")
            confidence_scores.append(0.8)

        # Geolocation risk assessment
        if ip_info.get("risk") == "high":
            threat_indicators.append("high_risk_geolocation")
            confidence_scores.append(0.7)

        # Behavior analysis
        behavior_anomalies = self.behavior_analyzer.analyze_behavior(event)
        if behavior_anomalies:
            threat_indicators.extend(behavior_anomalies)
            confidence_scores.extend([0.6] * len(behavior_anomalies))
            evidence["behavior_anomalies"] = behavior_anomalies

        # Signature matching
        request_text = json.dumps(event.to_dict())
        signature_matches = self.signature_db.match_signatures(request_text, event.headers)
        if signature_matches:
            for attack_vector, pattern in signature_matches:
                threat_indicators.append(f"signature_match_{attack_vector.value}")
                attack_vectors.append(attack_vector)
                confidence_scores.append(0.9)
            evidence["signature_matches"] = [(av.value, pattern) for av, pattern in signature_matches]

        # Rate limiting analysis
        rate_limit_threats = self._analyze_rate_limiting(event)
        if rate_limit_threats:
            threat_indicators.extend(rate_limit_threats)
            confidence_scores.extend([0.8] * len(rate_limit_threats))
            evidence["rate_limit_violations"] = rate_limit_threats

        # DDoS detection
        ddos_indicators = self._detect_ddos_patterns(event)
        if ddos_indicators:
            threat_indicators.extend(ddos_indicators)
            confidence_scores.extend([0.9] * len(ddos_indicators))
            attack_vectors.append(AttackVector.DDoS)
            evidence["ddos_indicators"] = ddos_indicators

        # Protocol-specific threats
        protocol_threats = self._analyze_protocol_threats(event)
        if protocol_threats:
            threat_indicators.extend(protocol_threats)
            confidence_scores.extend([0.7] * len(protocol_threats))
            attack_vectors.append(AttackVector.PROTOCOL_MANIPULATION)
            evidence["protocol_threats"] = protocol_threats

        # Timing attack detection
        timing_threats = self._detect_timing_attacks(event)
        if timing_threats:
            threat_indicators.extend(timing_threats)
            confidence_scores.append(0.6)
            attack_vectors.append(AttackVector.TIMING_ATTACK)
            evidence["timing_indicators"] = timing_threats

        # Calculate overall threat assessment
        if not threat_indicators:
            threat_level = ThreatLevel.LOW
            overall_confidence = 0.1
            primary_attack_vector = None
        else:
            overall_confidence = max(confidence_scores) if confidence_scores else 0.1

            # Determine primary attack vector
            if attack_vectors:
                primary_attack_vector = max(set(attack_vectors), key=attack_vectors.count)
            else:
                primary_attack_vector = AttackVector.BRUTE_FORCE  # Default

            # Determine threat level
            max_confidence = max(confidence_scores) if confidence_scores else 0.0
            if max_confidence >= 0.9:
                threat_level = ThreatLevel.CRITICAL
            elif max_confidence >= 0.7:
                threat_level = ThreatLevel.HIGH
            elif max_confidence >= 0.5:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW

        # Generate threat intelligence
        threat_id = self._generate_threat_id()
        threat_intel = ThreatIntelligence(
            threat_id=threat_id,
            attack_vector=primary_attack_vector or AttackVector.BRUTE_FORCE,
            threat_level=threat_level,
            confidence_score=overall_confidence,
            source_ips=[event.source_ip],
            attack_patterns=threat_indicators,
            timeline=[event.timestamp],
            mitigation_actions=self._suggest_mitigation_actions(threat_level, primary_attack_vector),
            false_positive_probability=self._calculate_false_positive_probability(threat_indicators, overall_confidence),
            related_threats=[],
            evidence=evidence
        )

        # Store threat intelligence
        with self._lock:
            self.active_threats[threat_id] = threat_intel
            self.threat_history.append(threat_intel)

            # Update suspicion scores
            self.suspicious_ips[event.source_ip] += overall_confidence

            # Auto-block if threat is critical
            if threat_level == ThreatLevel.CRITICAL and overall_confidence > 0.85:
                self.blocked_ips.add(event.source_ip)
                logger.critical(f"Auto-blocked IP {event.source_ip} due to critical threat: {threat_id}")

        return threat_intel

    def _analyze_rate_limiting(self, event: SecurityEvent) -> list[str]:
        """Analyze rate limiting violations."""
        threats = []
        current_time = time.time()
        window_size = 60  # 1 minute window

        # Clean old requests
        ip_requests = self.request_windows[event.source_ip]
        while ip_requests and ip_requests[0] < current_time - window_size:
            ip_requests.popleft()

        # Add current request
        ip_requests.append(current_time)

        # Check thresholds
        if len(ip_requests) > self.ddos_thresholds["requests_per_minute"]:
            threats.append("rate_limit_exceeded")

        # Check failed authentication attempts
        if event.status_code == 401:
            failed_attempts = self.failed_auth_attempts[event.source_ip]
            while failed_attempts and failed_attempts[0] < current_time - 300:  # 5 minute window
                failed_attempts.popleft()
            failed_attempts.append(current_time)

            if len(failed_attempts) > 10:  # More than 10 failed attempts in 5 minutes
                threats.append("brute_force_authentication")

        return threats

    def _detect_ddos_patterns(self, event: SecurityEvent) -> list[str]:
        """Detect DDoS attack patterns."""
        indicators = []
        current_time = time.time()

        # Track connection counts
        self.connection_counts[event.source_ip] += 1

        # Check for volumetric attacks
        recent_requests = []
        for ip, requests in self.request_windows.items():
            recent_requests.extend([r for r in requests if r > current_time - 60])

        if len(recent_requests) > self.ddos_thresholds["requests_per_minute"] * 5:
            indicators.append("volumetric_attack")

        # Check unique IP threshold (distributed DDoS)
        unique_ips = len([ip for ip, requests in self.request_windows.items()
                         if requests and requests[-1] > current_time - 60])

        if unique_ips > self.ddos_thresholds["unique_ips_threshold"]:
            indicators.append("distributed_ddos")

        # Check for application layer attacks
        if event.endpoint.startswith("/api/") and event.response_time > 5000:  # 5 second response
            indicators.append("application_layer_attack")

        return indicators

    def _analyze_protocol_threats(self, event: SecurityEvent) -> list[str]:
        """Analyze MPC protocol-specific threats."""
        threats = []

        # Check for protocol manipulation attempts
        if "protocol" in event.endpoint.lower():
            if event.status_code >= 400:
                threats.append("protocol_manipulation_attempt")

        # Check for unusual protocol parameters in headers
        protocol_headers = [h for h in event.headers.keys() if "protocol" in h.lower()]
        if len(protocol_headers) > 3:
            threats.append("excessive_protocol_headers")

        # Check for potential side-channel attacks
        if event.endpoint.startswith("/api/inference") and event.response_time > 10000:
            threats.append("potential_side_channel")

        return threats

    def _detect_timing_attacks(self, event: SecurityEvent) -> list[str]:
        """Detect timing-based attacks."""
        indicators = []

        # Get historical response times for this endpoint
        similar_requests = []
        for threat in list(self.threat_history)[-100:]:  # Last 100 threats
            if hasattr(threat, 'evidence') and 'endpoint' in threat.evidence:
                if threat.evidence.get('endpoint') == event.endpoint:
                    similar_requests.append(threat.evidence.get('response_time', 0))

        if similar_requests and len(similar_requests) > 5:
            avg_response_time = sum(similar_requests) / len(similar_requests)

            # Check for timing probes (very fast consecutive requests)
            if event.response_time < avg_response_time * 0.1:
                indicators.append("timing_probe_fast")

            # Check for timing analysis patterns
            recent_times = similar_requests[-10:]  # Last 10 similar requests
            if len(set(recent_times)) == 1:  # All exactly the same time
                indicators.append("timing_analysis_pattern")

        return indicators

    def _generate_threat_id(self) -> str:
        """Generate unique threat ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = secrets.token_hex(4)
        return f"threat_{timestamp}_{random_part}"

    def _suggest_mitigation_actions(self, threat_level: ThreatLevel,
                                   attack_vector: AttackVector | None) -> list[str]:
        """Suggest mitigation actions based on threat assessment."""
        actions = []

        if threat_level == ThreatLevel.CRITICAL:
            actions.extend([
                "block_source_ip",
                "trigger_incident_response",
                "notify_security_team",
                "increase_monitoring"
            ])
        elif threat_level == ThreatLevel.HIGH:
            actions.extend([
                "rate_limit_source_ip",
                "increase_logging",
                "monitor_closely"
            ])
        elif threat_level == ThreatLevel.MEDIUM:
            actions.extend([
                "log_event",
                "monitor_pattern"
            ])

        # Attack vector specific actions
        if attack_vector == AttackVector.DDoS:
            actions.extend(["activate_ddos_protection", "contact_upstream_provider"])
        elif attack_vector == AttackVector.SQL_INJECTION:
            actions.extend(["sanitize_input", "review_sql_queries"])
        elif attack_vector == AttackVector.TIMING_ATTACK:
            actions.extend(["add_random_delay", "review_timing_leaks"])

        return list(set(actions))  # Remove duplicates

    def _calculate_false_positive_probability(self, indicators: list[str],
                                            confidence: float) -> float:
        """Calculate probability of false positive."""
        # Base false positive rate
        base_rate = 0.1

        # Adjust based on indicators
        high_confidence_indicators = [
            "signature_match_sql_injection",
            "signature_match_xss",
            "volumetric_attack",
            "distributed_ddos"
        ]

        low_fp_count = sum(1 for indicator in indicators if indicator in high_confidence_indicators)
        adjustment = max(0, base_rate - (low_fp_count * 0.02))

        # Adjust based on confidence
        confidence_adjustment = (1.0 - confidence) * 0.3

        return min(0.5, adjustment + confidence_adjustment)

    def get_threat_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get threat intelligence summary."""
        cutoff_time = time.time() - (hours * 3600)

        recent_threats = [t for t in self.threat_history
                         if t.timeline and max(t.timeline) > cutoff_time]

        # Analyze threat patterns
        threat_levels = defaultdict(int)
        attack_vectors = defaultdict(int)
        source_ips = set()

        for threat in recent_threats:
            threat_levels[threat.threat_level.value] += 1
            attack_vectors[threat.attack_vector.value] += 1
            source_ips.update(threat.source_ips)

        return {
            "time_period_hours": hours,
            "total_threats": len(recent_threats),
            "threat_levels": dict(threat_levels),
            "attack_vectors": dict(attack_vectors),
            "unique_source_ips": len(source_ips),
            "blocked_ips": len(self.blocked_ips),
            "active_threats": len(self.active_threats),
            "highest_threat_level": max(threat_levels.keys()) if threat_levels else "none",
            "most_common_attack": max(attack_vectors.items(), key=lambda x: x[1])[0] if attack_vectors else "none"
        }

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips

    def block_ip(self, ip: str, reason: str):
        """Block an IP address."""
        with self._lock:
            self.blocked_ips.add(ip)
            logger.warning(f"IP {ip} blocked: {reason}")

    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        with self._lock:
            self.blocked_ips.discard(ip)
            logger.info(f"IP {ip} unblocked")

    async def cleanup_old_data(self):
        """Clean up old data to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 hours

        with self._lock:
            # Clean old request windows
            for ip in list(self.request_windows.keys()):
                requests = self.request_windows[ip]
                while requests and requests[0] < cutoff_time:
                    requests.popleft()

                if not requests:
                    del self.request_windows[ip]

            # Clean old failed auth attempts
            for ip in list(self.failed_auth_attempts.keys()):
                attempts = self.failed_auth_attempts[ip]
                while attempts and attempts[0] < cutoff_time:
                    attempts.popleft()

                if not attempts:
                    del self.failed_auth_attempts[ip]

            # Clean old threats
            old_threat_ids = [
                threat_id for threat_id, threat in self.active_threats.items()
                if threat.timeline and max(threat.timeline) < cutoff_time
            ]

            for threat_id in old_threat_ids:
                del self.active_threats[threat_id]

            # Decay suspicion scores
            for ip in self.suspicious_ips:
                self.suspicious_ips[ip] *= 0.95  # 5% decay

            # Remove low suspicion IPs
            low_suspicion_ips = [
                ip for ip, score in self.suspicious_ips.items() if score < 0.1
            ]
            for ip in low_suspicion_ips:
                del self.suspicious_ips[ip]

        logger.debug("Threat detector data cleanup completed")


# Global threat detector instance
threat_detector = AdvancedThreatDetector()
