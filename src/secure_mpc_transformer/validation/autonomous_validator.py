"""
Autonomous Validation Framework - Generation 2 Implementation

Comprehensive input validation, security checks, and data integrity
verification for autonomous SDLC execution with defensive security focus.
"""

import hashlib
import ipaddress
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation failures"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ThreatCategory(Enum):
    """Categories of security threats"""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    MALFORMED_DATA = "malformed_data"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    severity: ValidationSeverity
    threat_category: ThreatCategory | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    confidence_score: float = 1.0


@dataclass
class ValidationPolicy:
    """Validation policy configuration"""
    max_string_length: int = 10000
    max_list_size: int = 1000
    max_dict_depth: int = 10
    allowed_file_extensions: list[str] = field(default_factory=lambda: ['.txt', '.json', '.yaml'])
    blocked_patterns: list[str] = field(default_factory=list)
    require_https: bool = True
    allow_private_ips: bool = False
    max_execution_time: float = 30.0


class AutonomousValidator:
    """
    Comprehensive validation framework for autonomous systems.
    
    Provides defensive security validation with threat detection,
    input sanitization, and integrity verification.
    """

    def __init__(self, policy: ValidationPolicy | None = None):
        self.policy = policy or ValidationPolicy()
        self.validation_cache: dict[str, ValidationResult] = {}
        self.threat_patterns = self._initialize_threat_patterns()
        self.validation_stats = {
            "total_validations": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "validation_time_ms": 0
        }

        logger.info("AutonomousValidator initialized with defensive security policies")

    def _initialize_threat_patterns(self) -> dict[ThreatCategory, list[Pattern]]:
        """Initialize threat detection patterns"""
        return {
            ThreatCategory.INJECTION: [
                re.compile(r"(?i)(union\s+select|drop\s+table|insert\s+into)", re.IGNORECASE),
                re.compile(r"(?i)(<script|javascript:|vbscript:)", re.IGNORECASE),
                re.compile(r"(?i)(exec|eval|system|passthru)", re.IGNORECASE),
            ],
            ThreatCategory.XSS: [
                re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
                re.compile(r"javascript:[^\"']*", re.IGNORECASE),
                re.compile(r"on\w+\s*=", re.IGNORECASE),
            ],
            ThreatCategory.PATH_TRAVERSAL: [
                re.compile(r"\.\./"),
                re.compile(r"\.\.\\\\"),
                re.compile(r"(?i)(\/etc\/passwd|\/etc\/shadow)"),
            ],
            ThreatCategory.COMMAND_INJECTION: [
                re.compile(r"[;&|`$]"),
                re.compile(r"(?i)(rm\s+-rf|format\s+c:)"),
                re.compile(r"(?i)(wget|curl)\s+http"),
            ]
        }

    async def validate_input(self, data: Any, context: dict[str, Any] | None = None) -> ValidationResult:
        """
        Comprehensive input validation with threat detection.
        
        Args:
            data: Input data to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult with security assessment
        """
        start_time = time.time()
        context = context or {}

        # Generate cache key
        cache_key = self._generate_cache_key(data, context)

        # Check cache first
        if cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            logger.debug("Using cached validation result")
            return cached_result

        try:
            # Perform comprehensive validation
            result = await self._perform_validation(data, context)

            # Cache result
            self.validation_cache[cache_key] = result

            # Update statistics
            self.validation_stats["total_validations"] += 1
            self.validation_stats["validation_time_ms"] += (time.time() - start_time) * 1000

            if result.threat_category:
                self.validation_stats["threats_detected"] += 1

            return result

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                threat_category=None,
                message=f"Validation failed: {str(e)}",
                confidence_score=0.0
            )

    async def _perform_validation(self, data: Any, context: dict[str, Any]) -> ValidationResult:
        """Perform the actual validation logic"""

        # Type-specific validation
        if isinstance(data, str):
            return await self._validate_string(data, context)
        elif isinstance(data, dict):
            return await self._validate_dict(data, context)
        elif isinstance(data, list):
            return await self._validate_list(data, context)
        elif isinstance(data, (int, float)):
            return await self._validate_number(data, context)
        elif data is None:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                threat_category=None,
                message="Null value validated"
            )
        else:
            return await self._validate_object(data, context)

    async def _validate_string(self, text: str, context: dict[str, Any]) -> ValidationResult:
        """Validate string input with threat detection"""

        # Length check
        if len(text) > self.policy.max_string_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                threat_category=ThreatCategory.RESOURCE_EXHAUSTION,
                message=f"String too long: {len(text)} > {self.policy.max_string_length}",
                remediation="Truncate input or increase length limit"
            )

        # Threat pattern detection
        for threat_category, patterns in self.threat_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    confidence = self._calculate_threat_confidence(match, text, threat_category)

                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        threat_category=threat_category,
                        message=f"Potential {threat_category.value} detected: {match.group()}",
                        details={
                            "matched_pattern": pattern.pattern,
                            "match_position": match.span(),
                            "context_snippet": text[max(0, match.start()-20):match.end()+20]
                        },
                        remediation="Sanitize input or reject request",
                        confidence_score=confidence
                    )

        # URL validation if context suggests it's a URL
        if context.get("type") == "url":
            return await self._validate_url(text)

        # Email validation if context suggests it's an email
        if context.get("type") == "email":
            return await self._validate_email(text)

        # File path validation if context suggests it's a path
        if context.get("type") == "filepath":
            return await self._validate_filepath(text)

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message="String validation passed"
        )

    async def _validate_dict(self, data: dict[str, Any], context: dict[str, Any]) -> ValidationResult:
        """Validate dictionary input"""

        # Check depth to prevent stack overflow
        depth = self._calculate_dict_depth(data)
        if depth > self.policy.max_dict_depth:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                threat_category=ThreatCategory.RESOURCE_EXHAUSTION,
                message=f"Dictionary too deep: {depth} > {self.policy.max_dict_depth}",
                remediation="Reduce nesting depth"
            )

        # Validate each key and value
        for key, value in data.items():
            # Validate key
            key_result = await self.validate_input(key, {"type": "dict_key"})
            if not key_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    severity=key_result.severity,
                    threat_category=key_result.threat_category,
                    message=f"Invalid dictionary key: {key_result.message}",
                    details={"invalid_key": key}
                )

            # Validate value
            value_result = await self.validate_input(value, {"parent_key": key})
            if not value_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    severity=value_result.severity,
                    threat_category=value_result.threat_category,
                    message=f"Invalid value for key '{key}': {value_result.message}",
                    details={"invalid_key": key, "invalid_value": str(value)[:100]}
                )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message=f"Dictionary validation passed ({len(data)} items)"
        )

    async def _validate_list(self, data: list[Any], context: dict[str, Any]) -> ValidationResult:
        """Validate list input"""

        # Size check
        if len(data) > self.policy.max_list_size:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                threat_category=ThreatCategory.RESOURCE_EXHAUSTION,
                message=f"List too large: {len(data)} > {self.policy.max_list_size}",
                remediation="Reduce list size or process in batches"
            )

        # Validate each item
        for i, item in enumerate(data):
            item_result = await self.validate_input(item, {"list_index": i})
            if not item_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    severity=item_result.severity,
                    threat_category=item_result.threat_category,
                    message=f"Invalid list item at index {i}: {item_result.message}",
                    details={"invalid_index": i}
                )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message=f"List validation passed ({len(data)} items)"
        )

    async def _validate_number(self, data: int | float, context: dict[str, Any]) -> ValidationResult:
        """Validate numeric input"""

        # Range checks based on context
        min_val = context.get("min_value")
        max_val = context.get("max_value")

        if min_val is not None and data < min_val:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                threat_category=None,
                message=f"Number below minimum: {data} < {min_val}"
            )

        if max_val is not None and data > max_val:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                threat_category=None,
                message=f"Number above maximum: {data} > {max_val}"
            )

        # Check for infinity and NaN
        if isinstance(data, float):
            if not isinstance(data, (int, float)) or data != data:  # NaN check
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    threat_category=ThreatCategory.MALFORMED_DATA,
                    message="Invalid float value (NaN or infinity)"
                )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message="Number validation passed"
        )

    async def _validate_object(self, data: Any, context: dict[str, Any]) -> ValidationResult:
        """Validate generic object"""

        # Check for potentially dangerous object types
        dangerous_types = ['module', 'function', 'method', 'builtin_function_or_method']

        if type(data).__name__ in dangerous_types:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                threat_category=ThreatCategory.UNAUTHORIZED_ACCESS,
                message=f"Dangerous object type detected: {type(data).__name__}",
                remediation="Remove dangerous object from input"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message=f"Object validation passed (type: {type(data).__name__})"
        )

    async def _validate_url(self, url: str) -> ValidationResult:
        """Validate URL with security checks"""

        try:
            parsed = urlparse(url)

            # Scheme validation
            if self.policy.require_https and parsed.scheme != 'https':
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    threat_category=ThreatCategory.UNAUTHORIZED_ACCESS,
                    message="HTTP not allowed, HTTPS required",
                    remediation="Use HTTPS protocol"
                )

            # Hostname validation
            if parsed.hostname:
                try:
                    ip = ipaddress.ip_address(parsed.hostname)
                    if not self.policy.allow_private_ips and ip.is_private:
                        return ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            threat_category=ThreatCategory.UNAUTHORIZED_ACCESS,
                            message="Private IP addresses not allowed",
                            remediation="Use public IP or domain name"
                        )
                except ValueError:
                    # Not an IP address, check domain
                    if not self._is_valid_domain(parsed.hostname):
                        return ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            threat_category=ThreatCategory.MALFORMED_DATA,
                            message="Invalid domain name",
                            remediation="Use valid domain name"
                        )

            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                threat_category=None,
                message="URL validation passed"
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                threat_category=ThreatCategory.MALFORMED_DATA,
                message=f"Invalid URL format: {str(e)}"
            )

    async def _validate_email(self, email: str) -> ValidationResult:
        """Validate email address"""

        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )

        if not email_pattern.match(email):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                threat_category=ThreatCategory.MALFORMED_DATA,
                message="Invalid email format"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message="Email validation passed"
        )

    async def _validate_filepath(self, filepath: str) -> ValidationResult:
        """Validate file path with security checks"""

        # Path traversal check
        if '..' in filepath or filepath.startswith('/'):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                threat_category=ThreatCategory.PATH_TRAVERSAL,
                message="Path traversal attempt detected",
                remediation="Use relative paths without '..' components"
            )

        # File extension check
        if self.policy.allowed_file_extensions:
            extension = filepath.lower().split('.')[-1] if '.' in filepath else ''
            if f'.{extension}' not in self.policy.allowed_file_extensions:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    threat_category=ThreatCategory.UNAUTHORIZED_ACCESS,
                    message=f"File extension not allowed: .{extension}",
                    details={"allowed_extensions": self.policy.allowed_file_extensions}
                )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            threat_category=None,
            message="File path validation passed"
        )

    def _calculate_threat_confidence(self, match: re.Match, text: str,
                                   threat_category: ThreatCategory) -> float:
        """Calculate confidence score for threat detection"""

        # Base confidence based on pattern match
        confidence = 0.7

        # Increase confidence for exact matches
        if match.group() == text:
            confidence += 0.2

        # Adjust based on threat category
        critical_threats = [ThreatCategory.INJECTION, ThreatCategory.COMMAND_INJECTION]
        if threat_category in critical_threats:
            confidence += 0.1

        # Consider context around the match
        start, end = match.span()
        context_before = text[max(0, start-10):start]
        context_after = text[end:end+10]

        # Look for suspicious context patterns
        suspicious_context = ['eval', 'exec', 'system', 'shell', 'cmd']
        for pattern in suspicious_context:
            if pattern in context_before.lower() or pattern in context_after.lower():
                confidence += 0.1
                break

        return min(confidence, 1.0)

    def _calculate_dict_depth(self, data: dict[str, Any], current_depth: int = 1) -> int:
        """Calculate maximum depth of nested dictionary"""

        max_depth = current_depth

        for value in data.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)

        return max_depth

    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name format"""

        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'
            r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        )

        return bool(domain_pattern.match(domain))

    def _generate_cache_key(self, data: Any, context: dict[str, Any]) -> str:
        """Generate cache key for validation result"""

        # Create a hash of the input data and context
        data_str = json.dumps(data, sort_keys=True, default=str)
        context_str = json.dumps(context, sort_keys=True, default=str)

        combined = f"{data_str}:{context_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data by removing/escaping dangerous content"""

        if isinstance(data, str):
            return self._sanitize_string(data)
        elif isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        else:
            return data

    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input"""

        # Remove potential script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Escape HTML characters
        html_escape = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;'
        }

        for char, escape in html_escape.items():
            text = text.replace(char, escape)

        return text

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics"""

        stats = self.validation_stats.copy()

        if stats["total_validations"] > 0:
            stats["avg_validation_time_ms"] = (
                stats["validation_time_ms"] / stats["total_validations"]
            )
            stats["threat_detection_rate"] = (
                stats["threats_detected"] / stats["total_validations"]
            )
        else:
            stats["avg_validation_time_ms"] = 0
            stats["threat_detection_rate"] = 0

        stats["cache_size"] = len(self.validation_cache)

        return stats

    def reset_cache(self) -> None:
        """Reset validation cache"""
        self.validation_cache.clear()
        logger.info("Validation cache reset")

    def add_threat_pattern(self, category: ThreatCategory, pattern: str) -> None:
        """Add custom threat detection pattern"""

        compiled_pattern = re.compile(pattern, re.IGNORECASE)

        if category not in self.threat_patterns:
            self.threat_patterns[category] = []

        self.threat_patterns[category].append(compiled_pattern)
        logger.info(f"Added threat pattern for {category.value}: {pattern}")

    def update_policy(self, **kwargs) -> None:
        """Update validation policy"""

        for key, value in kwargs.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                logger.info(f"Updated policy {key} = {value}")
            else:
                logger.warning(f"Unknown policy parameter: {key}")
