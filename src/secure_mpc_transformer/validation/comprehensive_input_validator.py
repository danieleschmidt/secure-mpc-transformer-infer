"""
Comprehensive Input Validation System with ML-Enhanced Security
"""

import base64
import hashlib
import html
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

class ThreatType(Enum):
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    SCRIPT_INJECTION = "script_injection"
    BUFFER_OVERFLOW = "buffer_overflow"
    FORMAT_STRING = "format_string"
    DESERIALIZATION = "deserialization"
    SSRF = "ssrf"
    PROTOTYPE_POLLUTION = "prototype_pollution"

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

@dataclass
class ValidationResult:
    is_valid: bool
    risk_score: float  # 0.0 to 1.0
    threats_detected: list[ThreatType] = field(default_factory=list)
    validation_details: dict[str, Any] = field(default_factory=dict)
    sanitized_input: str | None = None
    confidence: float = 1.0
    processing_time: float = 0.0
    warnings: list[str] = field(default_factory=list)

@dataclass
class ValidationPattern:
    threat_type: ThreatType
    pattern: str
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    flags: int = 0

class ComprehensiveInputValidator:
    """
    Comprehensive input validation system with ML-enhanced threat detection,
    multiple validation levels, and advanced sanitization capabilities.
    """

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.validation_level = ValidationLevel(
            self.config.get("validation_level", "standard")
        )

        # Pattern databases
        self.threat_patterns = self._initialize_threat_patterns()
        self.whitelist_patterns = self._initialize_whitelist_patterns()
        self.encoding_patterns = self._initialize_encoding_patterns()

        # ML-based detection (if available)
        self.ml_detector = None
        self._initialize_ml_detector()

        # Statistics tracking
        self.validation_stats = {
            'total_validations': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'processing_time_total': 0.0
        }

        # Caching for performance
        self.validation_cache = {}
        self.cache_max_size = self.config.get("cache_size", 10000)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour

        # Content type specific validators
        self.content_validators = {
            'json': self._validate_json_content,
            'xml': self._validate_xml_content,
            'html': self._validate_html_content,
            'url': self._validate_url_content,
            'email': self._validate_email_content,
            'sql': self._validate_sql_content
        }

    def _initialize_threat_patterns(self) -> dict[ThreatType, list[ValidationPattern]]:
        """Initialize comprehensive threat detection patterns"""
        patterns = {
            ThreatType.SQL_INJECTION: [
                ValidationPattern(
                    ThreatType.SQL_INJECTION,
                    r"('|(\\')|(;)|(\||\\|\^)|(\/\*.*\*\/)|(--)|(\b(alter|create|delete|drop|exec(ute)?|insert|select|union|update)\b))",
                    0.8, 0.9,
                    "Basic SQL injection patterns",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.SQL_INJECTION,
                    r"\b(and|or)\s+\d+\s*=\s*\d+",
                    0.9, 0.95,
                    "Boolean-based SQL injection",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.SQL_INJECTION,
                    r"(sleep|waitfor|benchmark|pg_sleep)\s*\(",
                    0.9, 0.9,
                    "Time-based SQL injection",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.SQL_INJECTION,
                    r"(load_file|into\s+outfile|into\s+dumpfile)",
                    1.0, 0.95,
                    "File-based SQL injection",
                    re.IGNORECASE
                )
            ],

            ThreatType.XSS: [
                ValidationPattern(
                    ThreatType.XSS,
                    r"<\s*script[^>]*>.*?</\s*script\s*>",
                    0.9, 0.9,
                    "Script tag injection",
                    re.IGNORECASE | re.DOTALL
                ),
                ValidationPattern(
                    ThreatType.XSS,
                    r"javascript\s*:",
                    0.8, 0.85,
                    "JavaScript protocol injection",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.XSS,
                    r"on\w+\s*=\s*[\"'].*[\"']",
                    0.7, 0.8,
                    "Event handler injection",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.XSS,
                    r"<\s*(iframe|object|embed|applet)[^>]*>",
                    0.6, 0.75,
                    "Dangerous HTML tags",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.XSS,
                    r"expression\s*\(",
                    0.8, 0.85,
                    "CSS expression injection",
                    re.IGNORECASE
                )
            ],

            ThreatType.COMMAND_INJECTION: [
                ValidationPattern(
                    ThreatType.COMMAND_INJECTION,
                    r"[;&|`$(){}[\]\\]",
                    0.7, 0.8,
                    "Command injection metacharacters"
                ),
                ValidationPattern(
                    ThreatType.COMMAND_INJECTION,
                    r"\b(cat|ls|ps|id|pwd|whoami|uname|wget|curl|nc|netcat|telnet)\b",
                    0.8, 0.85,
                    "Common system commands",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.COMMAND_INJECTION,
                    r"(\\x[0-9a-f]{2}|%[0-9a-f]{2})",
                    0.6, 0.7,
                    "Hex encoded commands",
                    re.IGNORECASE
                )
            ],

            ThreatType.PATH_TRAVERSAL: [
                ValidationPattern(
                    ThreatType.PATH_TRAVERSAL,
                    r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                    0.9, 0.9,
                    "Directory traversal patterns",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.PATH_TRAVERSAL,
                    r"(\/etc\/passwd|\/etc\/shadow|\\windows\\system32)",
                    1.0, 0.95,
                    "System file access attempts",
                    re.IGNORECASE
                )
            ],

            ThreatType.LDAP_INJECTION: [
                ValidationPattern(
                    ThreatType.LDAP_INJECTION,
                    r"[()&|!*]",
                    0.6, 0.7,
                    "LDAP injection metacharacters"
                ),
                ValidationPattern(
                    ThreatType.LDAP_INJECTION,
                    r"(\*\)|\(\|)",
                    0.8, 0.85,
                    "LDAP filter injection patterns"
                )
            ],

            ThreatType.XML_INJECTION: [
                ValidationPattern(
                    ThreatType.XML_INJECTION,
                    r"<!ENTITY\s+\w+\s+SYSTEM",
                    0.9, 0.9,
                    "XML external entity injection",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.XML_INJECTION,
                    r"<!\[CDATA\[.*?\]\]>",
                    0.6, 0.7,
                    "CDATA section injection",
                    re.IGNORECASE | re.DOTALL
                )
            ],

            ThreatType.DESERIALIZATION: [
                ValidationPattern(
                    ThreatType.DESERIALIZATION,
                    r"(rO0AB|aced00|java\.lang\.Runtime|eval\s*\()",
                    0.9, 0.85,
                    "Java deserialization patterns",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.DESERIALIZATION,
                    r"(__reduce__|pickle\.loads|cPickle\.loads)",
                    0.9, 0.9,
                    "Python pickle injection",
                    re.IGNORECASE
                )
            ],

            ThreatType.SSRF: [
                ValidationPattern(
                    ThreatType.SSRF,
                    r"(file://|ftp://|gopher://|dict://|ldap://|jar://)",
                    0.8, 0.8,
                    "Dangerous URL schemes",
                    re.IGNORECASE
                ),
                ValidationPattern(
                    ThreatType.SSRF,
                    r"(localhost|127\.0\.0\.1|0\.0\.0\.0|::1|\[::1\])",
                    0.7, 0.75,
                    "Local network access attempts",
                    re.IGNORECASE
                )
            ]
        }

        return patterns

    def _initialize_whitelist_patterns(self) -> dict[str, list[str]]:
        """Initialize whitelist patterns for safe content"""
        return {
            'safe_html_tags': [
                'p', 'br', 'strong', 'em', 'u', 'i', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
            ],
            'safe_attributes': [
                'class', 'id', 'title', 'alt', 'href', 'src'
            ],
            'safe_protocols': [
                'http', 'https', 'mailto', 'ftp'
            ]
        }

    def _initialize_encoding_patterns(self) -> dict[str, str]:
        """Initialize encoding detection patterns"""
        return {
            'url_encoded': r'%[0-9a-f]{2}',
            'html_encoded': r'&([a-z]+|#[0-9]+|#x[0-9a-f]+);',
            'unicode_escaped': r'\\u[0-9a-f]{4}',
            'hex_encoded': r'\\x[0-9a-f]{2}',
            'base64': r'[A-Za-z0-9+/]{20,}={0,2}'
        }

    def _initialize_ml_detector(self) -> None:
        """Initialize ML-based threat detector if available"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Placeholder - in production would load pre-trained model
            self.ml_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
            self.ml_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

            # Would train with known malicious/benign samples
            logger.info("ML threat detector initialized")

        except ImportError:
            logger.info("scikit-learn not available, ML detection disabled")

    async def validate_input(
        self,
        input_data: str,
        content_type: str = "text",
        context: dict[str, Any] = None
    ) -> ValidationResult:
        """
        Comprehensive input validation with threat detection and sanitization
        
        Args:
            input_data: Input string to validate
            content_type: Type of content (text, json, xml, html, url, etc.)
            context: Additional context for validation
            
        Returns:
            ValidationResult with validation status and details
        """
        start_time = time.time()
        context = context or {}

        try:
            # Check cache first
            cache_key = self._generate_cache_key(input_data, content_type)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

            # Initialize result
            result = ValidationResult(
                is_valid=True,
                risk_score=0.0,
                threats_detected=[],
                validation_details={
                    'input_length': len(input_data),
                    'content_type': content_type,
                    'validation_level': self.validation_level.value
                }
            )

            # Skip validation for empty input
            if not input_data or not input_data.strip():
                result.validation_details['empty_input'] = True
                result.sanitized_input = ""
                return result

            # Normalize and decode input
            normalized_input = await self._normalize_input(input_data)
            result.validation_details['normalization'] = {
                'original_length': len(input_data),
                'normalized_length': len(normalized_input),
                'encoding_detected': self._detect_encoding(input_data)
            }

            # Pattern-based threat detection
            pattern_results = await self._pattern_based_validation(normalized_input)
            result.threats_detected.extend(pattern_results['threats'])
            result.risk_score = max(result.risk_score, pattern_results['max_risk'])
            result.validation_details['pattern_analysis'] = pattern_results

            # Content-specific validation
            if content_type in self.content_validators:
                content_result = await self.content_validators[content_type](normalized_input)
                result.threats_detected.extend(content_result.get('threats', []))
                result.risk_score = max(result.risk_score, content_result.get('risk_score', 0.0))
                result.validation_details[f'{content_type}_analysis'] = content_result

            # ML-based detection (if available)
            if self.ml_detector:
                ml_result = await self._ml_based_validation(normalized_input)
                result.risk_score = max(result.risk_score, ml_result.get('risk_score', 0.0))
                result.validation_details['ml_analysis'] = ml_result

            # Context-based validation
            if context:
                context_result = await self._context_based_validation(normalized_input, context)
                result.risk_score = max(result.risk_score, context_result.get('risk_score', 0.0))
                result.validation_details['context_analysis'] = context_result

            # Determine final validation result
            risk_threshold = self._get_risk_threshold()
            result.is_valid = result.risk_score < risk_threshold

            # Generate sanitized input
            if not result.is_valid or self.config.get('always_sanitize', False):
                result.sanitized_input = await self._sanitize_input(
                    normalized_input,
                    result.threats_detected,
                    content_type
                )

            # Calculate confidence
            result.confidence = await self._calculate_confidence(result)

            # Processing time
            result.processing_time = time.time() - start_time

            # Cache result
            self._cache_result(cache_key, result)

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"Input validation failed: {e}")

            return ValidationResult(
                is_valid=False,
                risk_score=1.0,
                threats_detected=[],
                validation_details={'error': str(e)},
                processing_time=time.time() - start_time,
                warnings=[f"Validation error: {str(e)}"]
            )

    async def _normalize_input(self, input_data: str) -> str:
        """Normalize input by decoding various encodings"""
        try:
            normalized = input_data

            # Unicode normalization
            normalized = unicodedata.normalize('NFKC', normalized)

            # URL decoding
            try:
                normalized = unquote(normalized)
            except:
                pass

            # HTML entity decoding
            try:
                normalized = html.unescape(normalized)
            except:
                pass

            # Multiple rounds of decoding for nested encoding
            for _ in range(3):  # Maximum 3 rounds to prevent infinite loops
                prev_normalized = normalized

                # URL decode again
                try:
                    normalized = unquote(normalized)
                except:
                    pass

                # HTML decode again
                try:
                    normalized = html.unescape(normalized)
                except:
                    pass

                # Base64 detection and decoding
                if self._looks_like_base64(normalized):
                    try:
                        decoded = base64.b64decode(normalized).decode('utf-8', errors='ignore')
                        if decoded and len(decoded) > 10:  # Sanity check
                            normalized = decoded
                    except:
                        pass

                # Stop if no change occurred
                if normalized == prev_normalized:
                    break

            return normalized

        except Exception as e:
            logger.error(f"Input normalization failed: {e}")
            return input_data

    def _detect_encoding(self, input_data: str) -> list[str]:
        """Detect encoding types used in input"""
        encodings = []

        for encoding_name, pattern in self.encoding_patterns.items():
            if re.search(pattern, input_data, re.IGNORECASE):
                encodings.append(encoding_name)

        return encodings

    def _looks_like_base64(self, data: str) -> bool:
        """Check if string looks like base64 encoded data"""
        if len(data) < 4 or len(data) % 4 != 0:
            return False

        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        return bool(base64_pattern.match(data))

    async def _pattern_based_validation(self, input_data: str) -> dict[str, Any]:
        """Pattern-based threat detection"""
        threats = []
        max_risk = 0.0
        pattern_matches = []

        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern.pattern, input_data, pattern.flags))

                    if matches:
                        threats.append(threat_type)
                        max_risk = max(max_risk, pattern.severity)

                        pattern_matches.append({
                            'threat_type': threat_type.value,
                            'pattern_description': pattern.description,
                            'severity': pattern.severity,
                            'confidence': pattern.confidence,
                            'matches': len(matches),
                            'match_positions': [(m.start(), m.end()) for m in matches[:5]]  # First 5
                        })

                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {threat_type}: {e}")

        return {
            'threats': list(set(threats)),  # Remove duplicates
            'max_risk': max_risk,
            'pattern_matches': pattern_matches,
            'total_matches': len(pattern_matches)
        }

    async def _validate_json_content(self, input_data: str) -> dict[str, Any]:
        """Validate JSON content"""
        result = {'threats': [], 'risk_score': 0.0}

        try:
            # Try to parse as JSON
            json_data = json.loads(input_data)

            # Check for dangerous patterns in JSON
            json_str = json.dumps(json_data)

            # Look for function calls or code execution patterns
            dangerous_patterns = [
                r'eval\s*\(',
                r'Function\s*\(',
                r'constructor\s*\(',
                r'__proto__',
                r'prototype\.constructor'
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, json_str, re.IGNORECASE):
                    result['threats'].append(ThreatType.SCRIPT_INJECTION)
                    result['risk_score'] = max(result['risk_score'], 0.8)

            # Check for prototype pollution attempts
            if self._check_prototype_pollution(json_data):
                result['threats'].append(ThreatType.PROTOTYPE_POLLUTION)
                result['risk_score'] = max(result['risk_score'], 0.7)

        except json.JSONDecodeError:
            result['json_parse_error'] = True
            result['risk_score'] = 0.1  # Slightly risky due to malformed JSON

        return result

    def _check_prototype_pollution(self, data: Any, max_depth: int = 5) -> bool:
        """Check for prototype pollution patterns in JSON data"""
        if max_depth <= 0:
            return False

        if isinstance(data, dict):
            dangerous_keys = ['__proto__', 'constructor', 'prototype']

            for key in data.keys():
                if str(key).lower() in dangerous_keys:
                    return True

                if self._check_prototype_pollution(data[key], max_depth - 1):
                    return True

        elif isinstance(data, list):
            for item in data:
                if self._check_prototype_pollution(item, max_depth - 1):
                    return True

        return False

    async def _validate_xml_content(self, input_data: str) -> dict[str, Any]:
        """Validate XML content"""
        result = {'threats': [], 'risk_score': 0.0}

        # Check for XML External Entity (XXE) patterns
        xxe_patterns = [
            r'<!ENTITY[^>]+SYSTEM[^>]+>',
            r'<!ENTITY[^>]+PUBLIC[^>]+>',
            r'ENTITY\s+\%[^;]+;'
        ]

        for pattern in xxe_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                result['threats'].append(ThreatType.XML_INJECTION)
                result['risk_score'] = max(result['risk_score'], 0.9)

        # Check for CDATA abuse
        if re.search(r'<!\[CDATA\[.*?\]\]>', input_data, re.IGNORECASE | re.DOTALL):
            cdata_content = re.findall(r'<!\[CDATA\[(.*?)\]\]>', input_data, re.DOTALL)
            for content in cdata_content:
                # Recursively validate CDATA content
                cdata_result = await self._pattern_based_validation(content)
                if cdata_result['threats']:
                    result['threats'].extend(cdata_result['threats'])
                    result['risk_score'] = max(result['risk_score'], cdata_result['max_risk'])

        return result

    async def _validate_html_content(self, input_data: str) -> dict[str, Any]:
        """Validate HTML content"""
        result = {'threats': [], 'risk_score': 0.0}

        # Check for dangerous HTML tags
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'applet', 'form', 'meta']

        for tag in dangerous_tags:
            pattern = rf'<\s*{tag}[^>]*>'
            if re.search(pattern, input_data, re.IGNORECASE):
                result['threats'].append(ThreatType.XSS)
                result['risk_score'] = max(result['risk_score'], 0.8)

        # Check for event handlers
        event_handlers = [
            'onload', 'onclick', 'onmouseover', 'onmouseout', 'onfocus', 'onblur',
            'onchange', 'onsubmit', 'onerror', 'onresize'
        ]

        for handler in event_handlers:
            pattern = rf'{handler}\s*='
            if re.search(pattern, input_data, re.IGNORECASE):
                result['threats'].append(ThreatType.XSS)
                result['risk_score'] = max(result['risk_score'], 0.7)

        return result

    async def _validate_url_content(self, input_data: str) -> dict[str, Any]:
        """Validate URL content"""
        result = {'threats': [], 'risk_score': 0.0}

        try:
            parsed = urlparse(input_data)

            # Check for dangerous schemes
            dangerous_schemes = ['file', 'ftp', 'gopher', 'dict', 'ldap', 'jar']

            if parsed.scheme.lower() in dangerous_schemes:
                result['threats'].append(ThreatType.SSRF)
                result['risk_score'] = max(result['risk_score'], 0.8)

            # Check for localhost/internal network access
            if parsed.hostname:
                dangerous_hosts = [
                    'localhost', '127.0.0.1', '0.0.0.0', '::1',
                    '10.', '172.16.', '172.17.', '172.18.', '172.19.',
                    '172.20.', '172.21.', '172.22.', '172.23.',
                    '172.24.', '172.25.', '172.26.', '172.27.',
                    '172.28.', '172.29.', '172.30.', '172.31.',
                    '192.168.'
                ]

                hostname_lower = parsed.hostname.lower()
                for dangerous_host in dangerous_hosts:
                    if hostname_lower.startswith(dangerous_host):
                        result['threats'].append(ThreatType.SSRF)
                        result['risk_score'] = max(result['risk_score'], 0.7)
                        break

        except Exception as e:
            result['url_parse_error'] = str(e)
            result['risk_score'] = 0.2

        return result

    async def _validate_email_content(self, input_data: str) -> dict[str, Any]:
        """Validate email content"""
        result = {'threats': [], 'risk_score': 0.0}

        # Basic email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(email_pattern, input_data):
            result['invalid_format'] = True
            result['risk_score'] = 0.3

        # Check for header injection patterns
        header_injection_patterns = [
            r'[\r\n]',
            r'%0[ad]',
            r'content-type:',
            r'bcc:',
            r'cc:'
        ]

        for pattern in header_injection_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                result['threats'].append(ThreatType.SCRIPT_INJECTION)
                result['risk_score'] = max(result['risk_score'], 0.8)

        return result

    async def _validate_sql_content(self, input_data: str) -> dict[str, Any]:
        """Validate SQL-like content"""
        result = {'threats': [], 'risk_score': 0.0}

        # This is for when content is expected to be SQL (like stored procedures)
        # More permissive than general SQL injection detection

        dangerous_sql = [
            r'\bexec\s+xp_',
            r'\bsp_executesql\b',
            r'\bxp_cmdshell\b',
            r'\bbulk\s+insert\b',
            r'\bopenrowset\b'
        ]

        for pattern in dangerous_sql:
            if re.search(pattern, input_data, re.IGNORECASE):
                result['threats'].append(ThreatType.SQL_INJECTION)
                result['risk_score'] = max(result['risk_score'], 0.9)

        return result

    async def _ml_based_validation(self, input_data: str) -> dict[str, Any]:
        """ML-based threat detection (placeholder)"""
        result = {'risk_score': 0.0}

        # In production, would use trained ML models
        # For now, simple heuristics

        try:
            # Feature extraction
            features = {
                'length': len(input_data),
                'entropy': self._calculate_entropy(input_data),
                'special_char_ratio': len(re.findall(r'[^a-zA-Z0-9\s]', input_data)) / max(len(input_data), 1),
                'digit_ratio': len(re.findall(r'\d', input_data)) / max(len(input_data), 1),
                'uppercase_ratio': len(re.findall(r'[A-Z]', input_data)) / max(len(input_data), 1)
            }

            # Simple scoring based on heuristics
            risk_score = 0.0

            # High entropy might indicate encoded payload
            if features['entropy'] > 4.5:
                risk_score += 0.2

            # High special character ratio
            if features['special_char_ratio'] > 0.3:
                risk_score += 0.3

            # Unusual length patterns
            if features['length'] > 1000 or features['length'] < 1:
                risk_score += 0.1

            result['risk_score'] = min(1.0, risk_score)
            result['features'] = features

        except Exception as e:
            logger.error(f"ML validation failed: {e}")
            result['error'] = str(e)

        return result

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string"""
        if not data:
            return 0.0

        import math
        from collections import Counter

        counts = Counter(data)
        length = len(data)

        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        return entropy

    async def _context_based_validation(self, input_data: str, context: dict[str, Any]) -> dict[str, Any]:
        """Context-aware validation based on usage context"""
        result = {'risk_score': 0.0}

        # Context-specific validation rules
        user_role = context.get('user_role', 'anonymous')
        request_source = context.get('request_source', 'unknown')
        session_trust = context.get('session_trust_score', 0.5)

        # Adjust risk based on user role
        if user_role == 'anonymous':
            result['risk_score'] += 0.1
        elif user_role == 'admin':
            result['risk_score'] -= 0.1  # Admin users get slight benefit

        # Adjust risk based on request source
        if request_source == 'external':
            result['risk_score'] += 0.1
        elif request_source == 'internal':
            result['risk_score'] -= 0.05

        # Adjust risk based on session trust
        result['risk_score'] += (1.0 - session_trust) * 0.2

        # Ensure risk score stays in bounds
        result['risk_score'] = max(0.0, min(1.0, result['risk_score']))

        result['context_factors'] = {
            'user_role_adjustment': user_role,
            'source_adjustment': request_source,
            'trust_score_used': session_trust
        }

        return result

    async def _sanitize_input(
        self,
        input_data: str,
        threats: list[ThreatType],
        content_type: str
    ) -> str:
        """Sanitize input based on detected threats and content type"""
        try:
            sanitized = input_data

            # HTML/XSS sanitization
            if ThreatType.XSS in threats or content_type == 'html':
                sanitized = self._sanitize_html(sanitized)

            # SQL injection sanitization
            if ThreatType.SQL_INJECTION in threats:
                sanitized = self._sanitize_sql(sanitized)

            # Command injection sanitization
            if ThreatType.COMMAND_INJECTION in threats:
                sanitized = self._sanitize_command_injection(sanitized)

            # Path traversal sanitization
            if ThreatType.PATH_TRAVERSAL in threats:
                sanitized = self._sanitize_path_traversal(sanitized)

            # General dangerous character removal
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                sanitized = self._aggressive_sanitization(sanitized)

            return sanitized

        except Exception as e:
            logger.error(f"Input sanitization failed: {e}")
            return ""  # Return empty string on sanitization failure

    def _sanitize_html(self, input_data: str) -> str:
        """Sanitize HTML content"""
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', input_data, flags=re.IGNORECASE | re.DOTALL)

        # Remove dangerous tags
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'applet', 'form', 'meta']
        for tag in dangerous_tags:
            sanitized = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(f'<{tag}[^>]*/?>', '', sanitized, flags=re.IGNORECASE)

        # Remove event handlers
        sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)

        # Remove javascript: protocols
        sanitized = re.sub(r'javascript\s*:', '', sanitized, flags=re.IGNORECASE)

        return sanitized

    def _sanitize_sql(self, input_data: str) -> str:
        """Sanitize SQL injection patterns"""
        # Remove SQL comments
        sanitized = re.sub(r'(--|/\*.*?\*/)', '', input_data, flags=re.DOTALL)

        # Remove dangerous SQL keywords
        dangerous_sql = [
            'exec', 'execute', 'sp_', 'xp_', 'union', 'select', 'insert', 'update', 'delete',
            'drop', 'create', 'alter', 'truncate'
        ]

        for keyword in dangerous_sql:
            sanitized = re.sub(f'\\b{keyword}\\b', '', sanitized, flags=re.IGNORECASE)

        # Escape single quotes
        sanitized = sanitized.replace("'", "''")

        return sanitized

    def _sanitize_command_injection(self, input_data: str) -> str:
        """Sanitize command injection patterns"""
        # Remove dangerous shell metacharacters
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '{', '}', '[', ']', '\\']

        sanitized = input_data
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')

        return sanitized

    def _sanitize_path_traversal(self, input_data: str) -> str:
        """Sanitize path traversal patterns"""
        # Remove directory traversal patterns
        sanitized = re.sub(r'\.\.[\\/]', '', input_data)
        sanitized = re.sub(r'%2e%2e%2f', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'%2e%2e%5c', '', sanitized, flags=re.IGNORECASE)

        return sanitized

    def _aggressive_sanitization(self, input_data: str) -> str:
        """Aggressive sanitization for strict/paranoid levels"""
        # Allow only alphanumeric characters, spaces, and basic punctuation
        allowed_chars = re.compile(r'[^a-zA-Z0-9\s\-_.,!?@#$%^&*()=+\[\]{}:;"\']')
        sanitized = allowed_chars.sub('', input_data)

        # Limit length
        max_length = self.config.get('max_sanitized_length', 1000)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def _get_risk_threshold(self) -> float:
        """Get risk threshold based on validation level"""
        thresholds = {
            ValidationLevel.BASIC: 0.9,
            ValidationLevel.STANDARD: 0.7,
            ValidationLevel.STRICT: 0.5,
            ValidationLevel.PARANOID: 0.3
        }

        return thresholds.get(self.validation_level, 0.7)

    async def _calculate_confidence(self, result: ValidationResult) -> float:
        """Calculate confidence score for validation result"""
        base_confidence = 0.8

        # Adjust based on number of detection methods used
        detection_methods = len([
            key for key in result.validation_details.keys()
            if key.endswith('_analysis')
        ])

        confidence_boost = min(0.2, detection_methods * 0.05)

        # Adjust based on pattern match confidence
        pattern_analysis = result.validation_details.get('pattern_analysis', {})
        pattern_matches = pattern_analysis.get('pattern_matches', [])

        if pattern_matches:
            avg_pattern_confidence = sum(
                match.get('confidence', 0.5) for match in pattern_matches
            ) / len(pattern_matches)
            confidence_boost += avg_pattern_confidence * 0.1

        return min(1.0, base_confidence + confidence_boost)

    def _generate_cache_key(self, input_data: str, content_type: str) -> str:
        """Generate cache key for validation result"""
        content = f"{input_data}{content_type}{self.validation_level.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> ValidationResult | None:
        """Get cached validation result if available and not expired"""
        if cache_key in self.validation_cache:
            cached_data = self.validation_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['result']
            else:
                del self.validation_cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: ValidationResult) -> None:
        """Cache validation result"""
        # Implement LRU-like cache cleanup
        if len(self.validation_cache) >= self.cache_max_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self.validation_cache.keys(),
                key=lambda k: self.validation_cache[k]['timestamp']
            )[:self.cache_max_size // 4]  # Remove 25%

            for key in oldest_keys:
                del self.validation_cache[key]

        self.validation_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

    def _update_stats(self, result: ValidationResult) -> None:
        """Update validation statistics"""
        self.validation_stats['total_validations'] += 1

        if result.threats_detected:
            self.validation_stats['threats_detected'] += 1

        self.validation_stats['processing_time_total'] += result.processing_time

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation statistics"""
        total_validations = self.validation_stats['total_validations']

        return {
            'total_validations': total_validations,
            'threats_detected': self.validation_stats['threats_detected'],
            'threat_detection_rate': (
                self.validation_stats['threats_detected'] / max(total_validations, 1)
            ),
            'average_processing_time': (
                self.validation_stats['processing_time_total'] / max(total_validations, 1)
            ),
            'cache_size': len(self.validation_cache),
            'validation_level': self.validation_level.value,
            'ml_detection_enabled': self.ml_detector is not None
        }
