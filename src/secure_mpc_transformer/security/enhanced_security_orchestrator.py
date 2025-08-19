"""Enhanced Security Orchestrator - Generation 2 Robustness Implementation."""

import hashlib
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Input validation result."""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    QUARANTINED = "quarantined"


@dataclass
class SecurityEvent:
    """Security event record."""
    id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: str | None
    user_id: str | None
    details: dict[str, Any]
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ValidationPattern:
    """Input validation pattern."""
    name: str
    pattern: str
    threat_level: ThreatLevel
    description: str
    action: str = "block"  # block, quarantine, log


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()

        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time

        # Get or create request history for identifier
        if identifier not in self.requests:
            self.requests[identifier] = []

        request_times = self.requests[identifier]

        # Remove requests outside the window
        cutoff_time = current_time - self.window_seconds
        request_times[:] = [t for t in request_times if t > cutoff_time]

        # Check if under limit
        if len(request_times) < self.max_requests:
            request_times.append(current_time)
            return True

        return False

    def _cleanup_old_entries(self, current_time: float):
        """Remove old entries to prevent memory leak."""
        cutoff_time = current_time - self.window_seconds * 2

        for identifier in list(self.requests.keys()):
            request_times = self.requests[identifier]
            request_times[:] = [t for t in request_times if t > cutoff_time]

            # Remove empty entries
            if not request_times:
                del self.requests[identifier]


class InputValidator:
    """Advanced input validation system."""

    def __init__(self):
        self.patterns: list[ValidationPattern] = []
        self.blocked_patterns: set[str] = set()
        self.quarantine_cache: dict[str, float] = {}

        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default security patterns."""
        default_patterns = [
            ValidationPattern(
                name="xss_script",
                pattern=r"<script[^>]*>.*?</script>",
                threat_level=ThreatLevel.HIGH,
                description="Cross-site scripting attempt",
                action="block"
            ),
            ValidationPattern(
                name="sql_injection",
                pattern=r"(?i)(union|select|insert|update|delete|drop|exec|script)\s+",
                threat_level=ThreatLevel.HIGH,
                description="SQL injection attempt",
                action="block"
            ),
            ValidationPattern(
                name="command_injection",
                pattern=r"(?i)(exec|eval|system|shell_exec|passthru|`)",
                threat_level=ThreatLevel.CRITICAL,
                description="Command injection attempt",
                action="block"
            ),
            ValidationPattern(
                name="path_traversal",
                pattern=r"\.\.\/|\.\.\\\\",
                threat_level=ThreatLevel.MEDIUM,
                description="Path traversal attempt",
                action="block"
            ),
            ValidationPattern(
                name="suspicious_encoding",
                pattern=r"(%0a|%0d|%00|%2e%2e|%252e|%c0%ae)",
                threat_level=ThreatLevel.MEDIUM,
                description="Suspicious URL encoding",
                action="quarantine"
            ),
            ValidationPattern(
                name="large_payload",
                pattern=r".{10000,}",  # More than 10k characters
                threat_level=ThreatLevel.LOW,
                description="Unusually large payload",
                action="log"
            )
        ]

        self.patterns.extend(default_patterns)
        logger.info(f"Loaded {len(default_patterns)} default security patterns")

    def add_pattern(self, pattern: ValidationPattern):
        """Add custom validation pattern."""
        self.patterns.append(pattern)
        logger.info(f"Added security pattern: {pattern.name}")

    def validate_input(self, text: str, context: dict[str, Any] = None) -> dict[str, Any]:
        """Validate input text against security patterns."""
        context = context or {}
        violations = []
        max_threat_level = ThreatLevel.LOW

        # Check input hash for quarantine
        input_hash = hashlib.sha256(text.encode()).hexdigest()
        if input_hash in self.quarantine_cache:
            quarantine_time = self.quarantine_cache[input_hash]
            if time.time() - quarantine_time < 3600:  # 1 hour quarantine
                return {
                    "allowed": False,
                    "result": ValidationResult.QUARANTINED,
                    "reason": "Input previously quarantined",
                    "violations": [],
                    "threat_level": ThreatLevel.MEDIUM.value
                }

        # Check against patterns
        for pattern in self.patterns:
            try:
                if re.search(pattern.pattern, text, re.IGNORECASE | re.MULTILINE):
                    violations.append({
                        "pattern": pattern.name,
                        "description": pattern.description,
                        "threat_level": pattern.threat_level.value,
                        "action": pattern.action
                    })

                    # Update max threat level
                    if self._is_higher_threat(pattern.threat_level, max_threat_level):
                        max_threat_level = pattern.threat_level

            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern.name}: {e}")

        # Determine action based on violations
        if not violations:
            return {
                "allowed": True,
                "result": ValidationResult.ALLOWED,
                "violations": [],
                "threat_level": ThreatLevel.LOW.value
            }

        # Find the most severe action
        actions = [v["action"] for v in violations]

        if "block" in actions:
            result = ValidationResult.BLOCKED
            allowed = False
        elif "quarantine" in actions:
            result = ValidationResult.QUARANTINED
            allowed = False
            # Add to quarantine cache
            self.quarantine_cache[input_hash] = time.time()
        else:
            # Only "log" actions
            result = ValidationResult.ALLOWED
            allowed = True

        return {
            "allowed": allowed,
            "result": result,
            "violations": violations,
            "threat_level": max_threat_level.value,
            "reason": f"Input matched {len(violations)} security patterns"
        }

    def _is_higher_threat(self, level1: ThreatLevel, level2: ThreatLevel) -> bool:
        """Check if level1 is higher threat than level2."""
        threat_order = {
            ThreatLevel.LOW: 0,
            ThreatLevel.MEDIUM: 1,
            ThreatLevel.HIGH: 2,
            ThreatLevel.CRITICAL: 3
        }
        return threat_order[level1] > threat_order[level2]


class SessionManager:
    """Secure session management."""

    def __init__(self, session_timeout: int = 3600):  # 1 hour
        self.sessions: dict[str, dict[str, Any]] = {}
        self.session_timeout = session_timeout
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def create_session(self, user_id: str, metadata: dict[str, Any] = None) -> str:
        """Create new secure session."""
        session_id = self._generate_session_id()

        session_data = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "metadata": metadata or {},
            "request_count": 0
        }

        self.sessions[session_id] = session_data
        logger.info(f"Created session for user {user_id}: {session_id}")

        return session_id

    def validate_session(self, session_id: str) -> dict[str, Any] | None:
        """Validate and update session."""
        current_time = time.time()

        # Cleanup expired sessions periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_sessions(current_time)
            self.last_cleanup = current_time

        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check if session expired
        if current_time - session["last_accessed"] > self.session_timeout:
            del self.sessions[session_id]
            logger.info(f"Session expired: {session_id}")
            return None

        # Update last accessed
        session["last_accessed"] = current_time
        session["request_count"] += 1

        return session.copy()

    def revoke_session(self, session_id: str) -> bool:
        """Revoke session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session revoked: {session_id}")
            return True
        return False

    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID."""
        return secrets.token_urlsafe(32)

    def _cleanup_expired_sessions(self, current_time: float):
        """Remove expired sessions."""
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if current_time - session["last_accessed"] > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class SecurityEventLogger:
    """Security event logging and analysis."""

    def __init__(self, max_events: int = 10000):
        self.events: list[SecurityEvent] = []
        self.max_events = max_events
        self.event_handlers: list[Callable[[SecurityEvent], None]] = []

    def add_event_handler(self, handler: Callable[[SecurityEvent], None]):
        """Add event handler."""
        self.event_handlers.append(handler)

    def log_event(self, event_type: str, threat_level: ThreatLevel, details: dict[str, Any],
                  source_ip: str = None, user_id: str = None, blocked: bool = False):
        """Log security event."""
        event = SecurityEvent(
            id=f"sec_{int(time.time() * 1000)}_{secrets.token_hex(4)}",
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            details=details,
            blocked=blocked
        )

        # Add to events list
        self.events.append(event)

        # Trim events if too many
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Notify handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Security event handler failed: {e}")

        logger.info(f"Security event logged: {event_type} [{threat_level.value}] - {blocked}")

    def get_events(self, limit: int = 100, event_type: str = None,
                   threat_level: ThreatLevel = None) -> list[dict[str, Any]]:
        """Get recent security events."""
        filtered_events = self.events

        # Filter by event type
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        # Filter by threat level
        if threat_level:
            filtered_events = [e for e in filtered_events if e.threat_level == threat_level]

        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)

        return [event.to_dict() for event in filtered_events[:limit]]

    def get_threat_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get threat summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]

        # Count by threat level
        threat_counts = {level.value: 0 for level in ThreatLevel}
        blocked_count = 0

        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
            if event.blocked:
                blocked_count += 1

        return {
            "total_events": len(recent_events),
            "blocked_events": blocked_count,
            "threat_counts": threat_counts,
            "time_period_hours": hours,
            "summary_timestamp": time.time()
        }


class EnhancedSecurityOrchestrator:
    """Enhanced security orchestration system."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get("rate_limit_requests", 100),
            window_seconds=self.config.get("rate_limit_window", 60)
        )

        self.input_validator = InputValidator()
        self.session_manager = SessionManager(
            session_timeout=self.config.get("session_timeout", 3600)
        )
        self.event_logger = SecurityEventLogger(
            max_events=self.config.get("max_security_events", 10000)
        )

        # Security state
        self.blocked_ips: set[str] = set()
        self.threat_threshold = self.config.get("threat_threshold", 5)
        self.auto_block_enabled = self.config.get("auto_block_enabled", True)

        # Setup default event handlers
        self.event_logger.add_event_handler(self._default_event_handler)

        logger.info("Enhanced Security Orchestrator initialized")

    async def validate_request(self, text: str, source_ip: str = None,
                             user_id: str = None, session_id: str = None) -> dict[str, Any]:
        """Comprehensive request validation."""
        start_time = time.time()

        # Check IP blocklist
        if source_ip and source_ip in self.blocked_ips:
            self.event_logger.log_event(
                event_type="blocked_ip_access",
                threat_level=ThreatLevel.HIGH,
                details={"blocked_ip": source_ip, "text_length": len(text)},
                source_ip=source_ip,
                user_id=user_id,
                blocked=True
            )

            return {
                "allowed": False,
                "result": ValidationResult.BLOCKED,
                "reason": "IP address blocked",
                "processing_time_ms": (time.time() - start_time) * 1000
            }

        # Rate limiting
        identifier = source_ip or user_id or "anonymous"
        if not self.rate_limiter.is_allowed(identifier):
            self.event_logger.log_event(
                event_type="rate_limit_exceeded",
                threat_level=ThreatLevel.MEDIUM,
                details={"identifier": identifier, "text_length": len(text)},
                source_ip=source_ip,
                user_id=user_id,
                blocked=True
            )

            return {
                "allowed": False,
                "result": ValidationResult.RATE_LIMITED,
                "reason": "Rate limit exceeded",
                "processing_time_ms": (time.time() - start_time) * 1000
            }

        # Session validation
        session_info = None
        if session_id:
            session_info = self.session_manager.validate_session(session_id)
            if not session_info:
                self.event_logger.log_event(
                    event_type="invalid_session",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"session_id": session_id, "text_length": len(text)},
                    source_ip=source_ip,
                    user_id=user_id,
                    blocked=True
                )

                return {
                    "allowed": False,
                    "result": ValidationResult.BLOCKED,
                    "reason": "Invalid session",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }

        # Input validation
        validation_result = self.input_validator.validate_input(text, {
            "source_ip": source_ip,
            "user_id": user_id,
            "session_info": session_info
        })

        # Log security event
        if not validation_result["allowed"]:
            threat_level = ThreatLevel(validation_result["threat_level"])

            self.event_logger.log_event(
                event_type="input_validation_failed",
                threat_level=threat_level,
                details={
                    "violations": validation_result["violations"],
                    "text_length": len(text),
                    "validation_result": validation_result["result"].value
                },
                source_ip=source_ip,
                user_id=user_id,
                blocked=True
            )

            # Auto-block IP if too many high-threat events
            if (self.auto_block_enabled and source_ip and
                threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]):
                await self._evaluate_auto_block(source_ip)

        # Add processing time
        validation_result["processing_time_ms"] = (time.time() - start_time) * 1000

        return validation_result

    async def _evaluate_auto_block(self, source_ip: str):
        """Evaluate if IP should be auto-blocked."""
        # Get recent high-threat events from this IP
        recent_events = self.event_logger.get_events(limit=50)
        ip_threats = [
            e for e in recent_events
            if (e.get("source_ip") == source_ip and
                e.get("threat_level") in ["high", "critical"] and
                e.get("timestamp", 0) > time.time() - 3600)  # Last hour
        ]

        if len(ip_threats) >= self.threat_threshold:
            self.blocked_ips.add(source_ip)

            self.event_logger.log_event(
                event_type="auto_ip_block",
                threat_level=ThreatLevel.CRITICAL,
                details={
                    "blocked_ip": source_ip,
                    "threat_events": len(ip_threats),
                    "threshold": self.threat_threshold
                },
                source_ip=source_ip,
                blocked=True
            )

            logger.warning(f"Auto-blocked IP {source_ip} after {len(ip_threats)} threat events")

    def create_session(self, user_id: str, metadata: dict[str, Any] = None) -> str:
        """Create secure session."""
        return self.session_manager.create_session(user_id, metadata)

    def validate_session(self, session_id: str) -> dict[str, Any] | None:
        """Validate session."""
        return self.session_manager.validate_session(session_id)

    def revoke_session(self, session_id: str) -> bool:
        """Revoke session."""
        return self.session_manager.revoke_session(session_id)

    def block_ip(self, ip_address: str, reason: str = "Manual block"):
        """Manually block IP address."""
        self.blocked_ips.add(ip_address)

        self.event_logger.log_event(
            event_type="manual_ip_block",
            threat_level=ThreatLevel.HIGH,
            details={"blocked_ip": ip_address, "reason": reason},
            source_ip=ip_address,
            blocked=True
        )

        logger.info(f"Manually blocked IP {ip_address}: {reason}")

    def unblock_ip(self, ip_address: str, reason: str = "Manual unblock"):
        """Unblock IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)

            self.event_logger.log_event(
                event_type="ip_unblock",
                threat_level=ThreatLevel.LOW,
                details={"unblocked_ip": ip_address, "reason": reason},
                source_ip=ip_address,
                blocked=False
            )

            logger.info(f"Unblocked IP {ip_address}: {reason}")
            return True
        return False

    def get_security_status(self) -> dict[str, Any]:
        """Get current security status."""
        threat_summary = self.event_logger.get_threat_summary(hours=24)

        return {
            "blocked_ips": len(self.blocked_ips),
            "active_sessions": len(self.session_manager.sessions),
            "threat_summary": threat_summary,
            "rate_limiter_status": {
                "tracked_identifiers": len(self.rate_limiter.requests),
                "max_requests": self.rate_limiter.max_requests,
                "window_seconds": self.rate_limiter.window_seconds
            },
            "security_patterns": len(self.input_validator.patterns),
            "auto_block_enabled": self.auto_block_enabled,
            "threat_threshold": self.threat_threshold,
            "timestamp": time.time()
        }

    def _default_event_handler(self, event: SecurityEvent):
        """Default security event handler."""
        # Log high and critical events
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.warning(
                f"Security Event: {event.event_type} [{event.threat_level.value}] "
                f"from {event.source_ip or 'unknown'} - Blocked: {event.blocked}"
            )

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent security events."""
        return self.event_logger.get_events(limit=limit)

    def add_validation_pattern(self, pattern: ValidationPattern):
        """Add custom validation pattern."""
        self.input_validator.add_pattern(pattern)

    def shutdown(self):
        """Shutdown security orchestrator."""
        logger.info("Enhanced Security Orchestrator shutting down")

        # Clear sensitive data
        self.session_manager.sessions.clear()
        self.blocked_ips.clear()

        logger.info("Security orchestrator shutdown complete")
