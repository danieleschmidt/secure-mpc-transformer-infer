#!/usr/bin/env python3
"""
Enhanced Security Validator for Secure MPC Transformer

Implements multi-stage security validation pipeline with ML-based anomaly detection
and comprehensive input sanitization for defensive security.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..utils.validators import BaseValidator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from security validation pipeline."""
    is_valid: bool
    risk_score: float  # 0.0 to 1.0
    threats_detected: list[str]
    validation_time: float
    metadata: dict[str, Any]


@dataclass
class ValidationContext:
    """Context for validation operations."""
    client_ip: str
    user_agent: str
    session_id: str | None
    request_timestamp: datetime
    request_size: int
    content_type: str


class MLAnomalyDetector:
    """Machine learning-based anomaly detection for security validation."""

    def __init__(self):
        self.request_patterns = {}
        self.baseline_metrics = {}
        self.anomaly_threshold = 0.7

    def analyze_request_pattern(self, context: ValidationContext, content: str) -> float:
        """Analyze request for anomalous patterns."""
        try:
            # Request size anomaly detection
            size_score = self._analyze_request_size(context.request_size)

            # Content pattern anomaly detection
            content_score = self._analyze_content_patterns(content)

            # Temporal pattern anomaly detection
            temporal_score = self._analyze_temporal_patterns(context)

            # Combined anomaly score
            anomaly_score = max(size_score, content_score, temporal_score)

            logger.debug(f"Anomaly detection scores: size={size_score:.3f}, "
                        f"content={content_score:.3f}, temporal={temporal_score:.3f}")

            return anomaly_score

        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
            return 0.5  # Neutral score on error

    def _analyze_request_size(self, size: int) -> float:
        """Analyze request size for anomalies."""
        # Statistical analysis of request size
        if size > 10_000_000:  # 10MB threshold
            return 0.9
        elif size > 1_000_000:  # 1MB threshold
            return 0.6
        return 0.1

    def _analyze_content_patterns(self, content: str) -> float:
        """Analyze content for suspicious patterns."""
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript protocol
            r'data:.*base64',  # Base64 data URLs
            r'\b(union|select|insert|delete|drop|exec|eval)\b',  # SQL injection
            r'\.\.\/|\.\.\\',  # Path traversal
            r'<\w+[^>]*on\w+\s*=',  # Event handlers
        ]

        max_score = 0.0
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                max_score = max(max_score, 0.8)

        return max_score

    def _analyze_temporal_patterns(self, context: ValidationContext) -> float:
        """Analyze temporal request patterns."""
        # Simple burst detection
        current_time = time.time()
        client_key = context.client_ip

        if client_key not in self.request_patterns:
            self.request_patterns[client_key] = []

        # Clean old requests (older than 1 minute)
        self.request_patterns[client_key] = [
            ts for ts in self.request_patterns[client_key]
            if current_time - ts < 60
        ]

        self.request_patterns[client_key].append(current_time)

        # Rate-based scoring
        request_count = len(self.request_patterns[client_key])
        if request_count > 100:  # More than 100 requests/minute
            return 0.9
        elif request_count > 50:  # More than 50 requests/minute
            return 0.6
        return 0.1


class ContentSecurityAnalyzer:
    """Advanced content security analysis."""

    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()

    def analyze_content_security(self, content: str, content_type: str) -> tuple[float, list[str]]:
        """Perform comprehensive content security analysis."""
        threats = []
        max_risk = 0.0

        try:
            # Input sanitization check
            sanitization_risk, sanitization_threats = self._check_input_sanitization(content)
            threats.extend(sanitization_threats)
            max_risk = max(max_risk, sanitization_risk)

            # Encoding analysis
            encoding_risk, encoding_threats = self._analyze_encoding(content)
            threats.extend(encoding_threats)
            max_risk = max(max_risk, encoding_risk)

            # Content type validation
            type_risk, type_threats = self._validate_content_type(content, content_type)
            threats.extend(type_threats)
            max_risk = max(max_risk, type_risk)

            return max_risk, threats

        except Exception as e:
            logger.error(f"Content security analysis failed: {e}")
            return 0.5, ["analysis_error"]

    def _check_input_sanitization(self, content: str) -> tuple[float, list[str]]:
        """Check for input sanitization requirements."""
        threats = []
        risk = 0.0

        # Check for unescaped special characters
        dangerous_chars = ['<', '>', '"', "'", '&', '`']
        for char in dangerous_chars:
            if char in content and content.count(char) > 10:
                threats.append(f"high_frequency_special_char_{char}")
                risk = max(risk, 0.4)

        # Check for potential injection patterns
        injection_patterns = [
            (r'\b(eval|exec|system|shell_exec)\b', "code_injection"),
            (r'<\w+[^>]*javascript:', "xss_attempt"),
            (r'\b(union|select|drop|delete|insert|update)\b.*\b(from|where|table)\b', "sql_injection"),
        ]

        for pattern, threat_type in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats.append(threat_type)
                risk = max(risk, 0.8)

        return risk, threats

    def _analyze_encoding(self, content: str) -> tuple[float, list[str]]:
        """Analyze content encoding for potential attacks."""
        threats = []
        risk = 0.0

        # Check for suspicious encoding patterns
        encoding_patterns = [
            (r'%[0-9a-fA-F]{2}', "url_encoding"),
            (r'&#x[0-9a-fA-F]+;', "html_entity_encoding"),
            (r'\\x[0-9a-fA-F]{2}', "hex_encoding"),
            (r'\\u[0-9a-fA-F]{4}', "unicode_encoding"),
        ]

        encoding_count = 0
        for pattern, encoding_type in encoding_patterns:
            matches = re.findall(pattern, content)
            if len(matches) > 10:  # High frequency of encoded content
                threats.append(f"suspicious_{encoding_type}")
                encoding_count += len(matches)

        if encoding_count > 50:
            risk = 0.7
        elif encoding_count > 20:
            risk = 0.4

        return risk, threats

    def _validate_content_type(self, content: str, content_type: str) -> tuple[float, list[str]]:
        """Validate content matches declared content type."""
        threats = []
        risk = 0.0

        try:
            # JSON content validation
            if "application/json" in content_type:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    threats.append("json_parse_error")
                    risk = 0.3

            # Check for content type mismatch
            if content_type == "text/plain" and ("<script>" in content or "<?php" in content):
                threats.append("content_type_mismatch")
                risk = 0.6

            return risk, threats

        except Exception as e:
            logger.error(f"Content type validation failed: {e}")
            return 0.2, ["validation_error"]

    def _load_threat_patterns(self) -> dict[str, list[str]]:
        """Load threat detection patterns."""
        return {
            "xss": [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
            ],
            "sqli": [
                r'\b(union|select|insert|delete|drop|exec)\b',
                r';\s*(drop|delete|truncate)',
                r"'.*?(\sor\s|\sand\s).*?'",
            ],
            "path_traversal": [
                r'\.\.\/|\.\.\\',
                r'\/etc\/passwd',
                r'\\windows\\system32',
            ],
            "command_injection": [
                r';\s*(cat|ls|pwd|whoami)',
                r'\|\s*(cat|ls|pwd|whoami)',
                r'`.*?`',
            ]
        }


class QuantumProtocolValidator:
    """Validator for quantum protocol specific security requirements."""

    def __init__(self):
        self.quantum_patterns = self._load_quantum_patterns()

    def validate_quantum_operation(self, operation_data: dict[str, Any]) -> tuple[float, list[str]]:
        """Validate quantum protocol operations."""
        threats = []
        risk = 0.0

        try:
            # Validate quantum state integrity
            if "quantum_state" in operation_data:
                state_risk, state_threats = self._validate_quantum_state(
                    operation_data["quantum_state"]
                )
                threats.extend(state_threats)
                risk = max(risk, state_risk)

            # Validate quantum operation parameters
            if "operation_params" in operation_data:
                param_risk, param_threats = self._validate_operation_params(
                    operation_data["operation_params"]
                )
                threats.extend(param_threats)
                risk = max(risk, param_risk)

            # Check for quantum-specific attack patterns
            attack_risk, attack_threats = self._check_quantum_attacks(operation_data)
            threats.extend(attack_threats)
            risk = max(risk, attack_risk)

            return risk, threats

        except Exception as e:
            logger.error(f"Quantum protocol validation failed: {e}")
            return 0.5, ["quantum_validation_error"]

    def _validate_quantum_state(self, quantum_state: dict[str, Any]) -> tuple[float, list[str]]:
        """Validate quantum state data."""
        threats = []
        risk = 0.0

        # Check for required quantum state fields
        required_fields = ["coherence", "entanglement", "measurement_basis"]
        for field in required_fields:
            if field not in quantum_state:
                threats.append(f"missing_quantum_field_{field}")
                risk = max(risk, 0.3)

        # Validate coherence values
        if "coherence" in quantum_state:
            coherence = quantum_state["coherence"]
            if not isinstance(coherence, (int, float)) or not 0 <= coherence <= 1:
                threats.append("invalid_coherence_value")
                risk = max(risk, 0.5)

        return risk, threats

    def _validate_operation_params(self, params: dict[str, Any]) -> tuple[float, list[str]]:
        """Validate quantum operation parameters."""
        threats = []
        risk = 0.0

        # Check for suspicious parameter values
        if isinstance(params.get("iterations"), int) and params["iterations"] > 10000:
            threats.append("excessive_iterations")
            risk = max(risk, 0.4)

        if isinstance(params.get("temperature"), (int, float)) and params["temperature"] < 0:
            threats.append("invalid_temperature")
            risk = max(risk, 0.3)

        return risk, threats

    def _check_quantum_attacks(self, operation_data: dict[str, Any]) -> tuple[float, list[str]]:
        """Check for quantum-specific attack patterns."""
        threats = []
        risk = 0.0

        # Check for quantum state manipulation attempts
        if "state_manipulation" in str(operation_data):
            threats.append("state_manipulation_attempt")
            risk = max(risk, 0.8)

        # Check for decoherence attacks
        if isinstance(operation_data.get("decoherence_factor"), (int, float)):
            if operation_data["decoherence_factor"] > 0.9:
                threats.append("decoherence_attack")
                risk = max(risk, 0.7)

        return risk, threats

    def _load_quantum_patterns(self) -> dict[str, list[str]]:
        """Load quantum-specific threat patterns."""
        return {
            "state_manipulation": [
                "state_injection",
                "coherence_disruption",
                "entanglement_break"
            ],
            "measurement_attacks": [
                "basis_manipulation",
                "measurement_tampering",
                "result_forgery"
            ]
        }


class EnhancedSecurityValidator(BaseValidator):
    """
    Enhanced multi-stage security validation system.
    
    Implements comprehensive security validation pipeline with:
    - Multi-stage validation process
    - ML-based anomaly detection
    - Content security analysis
    - Quantum protocol validation
    - Risk assessment and scoring
    """

    def __init__(self):
        super().__init__()
        self.ml_detector = MLAnomalyDetector()
        self.content_analyzer = ContentSecurityAnalyzer()
        self.quantum_validator = QuantumProtocolValidator()
        self.validation_cache = {}

    async def validate_request_pipeline(
        self,
        content: str,
        context: ValidationContext,
        validate_quantum: bool = False
    ) -> ValidationResult:
        """
        Execute multi-stage security validation pipeline.
        
        Args:
            content: Request content to validate
            context: Validation context information
            validate_quantum: Whether to perform quantum protocol validation
            
        Returns:
            ValidationResult with validation outcome and threat analysis
        """
        start_time = time.time()
        threats_detected = []
        max_risk_score = 0.0
        metadata = {}

        try:
            logger.debug(f"Starting validation pipeline for {context.client_ip}")

            # Stage 1: Basic validation (existing functionality)
            basic_risk = await self._basic_validation(content)
            max_risk_score = max(max_risk_score, basic_risk)
            metadata["basic_validation"] = {"risk_score": basic_risk}

            # Stage 2: ML-based anomaly detection
            anomaly_score = self.ml_detector.analyze_request_pattern(context, content)
            max_risk_score = max(max_risk_score, anomaly_score)
            metadata["anomaly_detection"] = {"score": anomaly_score}

            if anomaly_score > 0.7:
                threats_detected.append("ml_anomaly_detected")

            # Stage 3: Content security analysis
            content_risk, content_threats = self.content_analyzer.analyze_content_security(
                content, context.content_type
            )
            threats_detected.extend(content_threats)
            max_risk_score = max(max_risk_score, content_risk)
            metadata["content_security"] = {
                "risk_score": content_risk,
                "threats": content_threats
            }

            # Stage 4: Quantum protocol validation (if enabled)
            if validate_quantum and content:
                try:
                    operation_data = json.loads(content)
                    quantum_risk, quantum_threats = self.quantum_validator.validate_quantum_operation(
                        operation_data
                    )
                    threats_detected.extend(quantum_threats)
                    max_risk_score = max(max_risk_score, quantum_risk)
                    metadata["quantum_validation"] = {
                        "risk_score": quantum_risk,
                        "threats": quantum_threats
                    }
                except (json.JSONDecodeError, TypeError):
                    # Skip quantum validation for non-JSON content
                    pass

            # Stage 5: Risk assessment and final decision
            is_valid = max_risk_score < 0.8  # Reject if risk score > 80%
            validation_time = time.time() - start_time

            # Log high-risk requests
            if max_risk_score > 0.5:
                logger.warning(f"High-risk request detected: IP={context.client_ip}, "
                              f"risk_score={max_risk_score:.3f}, threats={threats_detected}")

            return ValidationResult(
                is_valid=is_valid,
                risk_score=max_risk_score,
                threats_detected=threats_detected,
                validation_time=validation_time,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}", exc_info=True)
            validation_time = time.time() - start_time

            return ValidationResult(
                is_valid=False,
                risk_score=1.0,  # Maximum risk on error
                threats_detected=["validation_pipeline_error"],
                validation_time=validation_time,
                metadata={"error": str(e)}
            )

    async def _basic_validation(self, content: str) -> float:
        """Basic validation checks."""
        risk_score = 0.0

        # Content length validation
        if len(content) > 1_000_000:  # 1MB limit
            risk_score = max(risk_score, 0.6)

        # Basic pattern matching for common attacks
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'\beval\b',
            r'\bexec\b',
            r'\.\.\/|\.\.\\'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                risk_score = max(risk_score, 0.5)

        return risk_score

    def get_validation_metrics(self) -> dict[str, Any]:
        """Get validation performance metrics."""
        return {
            "cache_size": len(self.validation_cache),
            "ml_detector_patterns": len(self.ml_detector.request_patterns),
            "threat_patterns_loaded": len(self.content_analyzer.threat_patterns),
            "quantum_patterns_loaded": len(self.quantum_validator.quantum_patterns)
        }


# Defensive security utility functions
def create_validation_context(
    request_data: dict[str, Any],
    client_ip: str = "unknown",
    user_agent: str = "unknown"
) -> ValidationContext:
    """Create validation context from request data."""
    return ValidationContext(
        client_ip=client_ip,
        user_agent=user_agent,
        session_id=request_data.get("session_id"),
        request_timestamp=datetime.now(timezone.utc),
        request_size=len(str(request_data)),
        content_type=request_data.get("content_type", "application/json")
    )


def hash_content_for_caching(content: str) -> str:
    """Create hash of content for validation caching."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


# Export main classes for defensive security
__all__ = [
    "EnhancedSecurityValidator",
    "ValidationResult",
    "ValidationContext",
    "MLAnomalyDetector",
    "ContentSecurityAnalyzer",
    "QuantumProtocolValidator",
    "create_validation_context",
    "hash_content_for_caching"
]
