"""Input validation utilities for secure MPC operations."""

import re
import torch
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    field: str
    value: Any
    message: str
    
    def __str__(self):
        return f"ValidationError in field '{self.field}': {self.message} (value: {self.value})"


class InputValidator:
    """Validator for inference request inputs."""
    
    def __init__(self):
        self.max_text_length = 10000  # Maximum input text length
        self.max_sequence_length = 512  # Maximum tokenized sequence length
        self.supported_models = {
            "bert-base-uncased", "bert-large-uncased",
            "roberta-base", "roberta-large",
            "distilbert-base-uncased",
            "gpt2", "gpt2-medium", "gpt2-large"
        }
        
        # Security constraints
        self.min_security_level = 80
        self.max_security_level = 256
        self.supported_protocols = {"aby3", "semi_honest_3pc", "malicious_3pc"}
        
    def validate_inference_request(self, request) -> bool:
        """Validate inference request object."""
        self._validate_text_input(request.text)
        self._validate_model_name(request.model_name)
        self._validate_max_length(request.max_length)
        self._validate_batch_size(request.batch_size)
        
        if request.protocol_config:
            self._validate_protocol_config(request.protocol_config)
        
        return True
    
    def _validate_text_input(self, text: str):
        """Validate text input."""
        if not isinstance(text, str):
            raise ValidationError("text", text, "Input must be a string")
        
        if not text.strip():
            raise ValidationError("text", text, "Input text cannot be empty")
        
        if len(text) > self.max_text_length:
            raise ValidationError("text", len(text), 
                                f"Text length exceeds maximum of {self.max_text_length} characters")
        
        # Check for potentially malicious content
        if self._contains_suspicious_patterns(text):
            raise ValidationError("text", text, "Input contains suspicious patterns")
    
    def _validate_model_name(self, model_name: str):
        """Validate model name."""
        if not isinstance(model_name, str):
            raise ValidationError("model_name", model_name, "Model name must be a string")
        
        if model_name not in self.supported_models:
            logger.warning(f"Model {model_name} not in supported list: {self.supported_models}")
    
    def _validate_max_length(self, max_length: Optional[int]):
        """Validate maximum sequence length."""
        if max_length is not None:
            if not isinstance(max_length, int):
                raise ValidationError("max_length", max_length, "Max length must be an integer")
            
            if max_length <= 0:
                raise ValidationError("max_length", max_length, "Max length must be positive")
            
            if max_length > self.max_sequence_length:
                raise ValidationError("max_length", max_length, 
                                    f"Max length exceeds limit of {self.max_sequence_length}")
    
    def _validate_batch_size(self, batch_size: int):
        """Validate batch size."""
        if not isinstance(batch_size, int):
            raise ValidationError("batch_size", batch_size, "Batch size must be an integer")
        
        if batch_size <= 0:
            raise ValidationError("batch_size", batch_size, "Batch size must be positive")
        
        if batch_size > 32:  # Reasonable limit for secure computation
            raise ValidationError("batch_size", batch_size, "Batch size too large for secure computation")
    
    def _validate_protocol_config(self, config: Dict[str, Any]):
        """Validate protocol configuration."""
        if "protocol_name" in config:
            protocol = config["protocol_name"]
            if protocol not in self.supported_protocols:
                raise ValidationError("protocol_name", protocol, 
                                    f"Unsupported protocol. Supported: {self.supported_protocols}")
        
        if "security_level" in config:
            security_level = config["security_level"]
            if not isinstance(security_level, int):
                raise ValidationError("security_level", security_level, "Security level must be an integer")
            
            if not (self.min_security_level <= security_level <= self.max_security_level):
                raise ValidationError("security_level", security_level, 
                                    f"Security level must be between {self.min_security_level} and {self.max_security_level}")
        
        if "num_parties" in config:
            num_parties = config["num_parties"]
            if not isinstance(num_parties, int):
                raise ValidationError("num_parties", num_parties, "Number of parties must be an integer")
            
            if num_parties < 2 or num_parties > 10:
                raise ValidationError("num_parties", num_parties, "Number of parties must be between 2 and 10")
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns in input text."""
        # SQL injection patterns
        sql_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|exec|execute)",
            r"[';].*--",
            r"\bor\s+1\s*=\s*1\b"
        ]
        
        # Script injection patterns
        script_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*="
        ]
        
        # Command injection patterns
        command_patterns = [
            r"[;&|`]",
            r"\$\(",
            r"\`[^`]*\`"
        ]
        
        all_patterns = sql_patterns + script_patterns + command_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, text):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return True
        
        return False
    
    def validate_tensor_input(self, tensor: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
        """Validate tensor inputs."""
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError("tensor", type(tensor), "Input must be a torch.Tensor")
        
        if expected_shape and tensor.shape != expected_shape:
            raise ValidationError("tensor_shape", tensor.shape, 
                                f"Expected shape {expected_shape}, got {tensor.shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(tensor).any():
            raise ValidationError("tensor", "contains_nan", "Tensor contains NaN values")
        
        if torch.isinf(tensor).any():
            raise ValidationError("tensor", "contains_inf", "Tensor contains infinite values")
        
        return True
    
    def sanitize_text_input(self, text: str) -> str:
        """Sanitize text input by removing potentially harmful content."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove control characters except whitespace
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class ProtocolValidator:
    """Validator for MPC protocol parameters and states."""
    
    def __init__(self):
        self.supported_field_sizes = [2**31 - 1, 2**61 - 1, 2**127 - 1]  # Common prime field sizes
        self.min_parties = 2
        self.max_parties = 10
        
    def validate_protocol_setup(self, protocol_name: str, party_id: int, num_parties: int, 
                               field_size: Optional[int] = None) -> bool:
        """Validate protocol setup parameters."""
        self._validate_protocol_name(protocol_name)
        self._validate_party_config(party_id, num_parties)
        
        if field_size is not None:
            self._validate_field_size(field_size)
        
        return True
    
    def _validate_protocol_name(self, protocol_name: str):
        """Validate protocol name."""
        supported_protocols = {"aby3", "semi_honest_3pc", "malicious_3pc", "bgw", "gmw"}
        
        if protocol_name not in supported_protocols:
            raise ValidationError("protocol_name", protocol_name, 
                                f"Unsupported protocol. Supported: {supported_protocols}")
    
    def _validate_party_config(self, party_id: int, num_parties: int):
        """Validate party configuration."""
        if not isinstance(party_id, int):
            raise ValidationError("party_id", party_id, "Party ID must be an integer")
        
        if not isinstance(num_parties, int):
            raise ValidationError("num_parties", num_parties, "Number of parties must be an integer")
        
        if not (self.min_parties <= num_parties <= self.max_parties):
            raise ValidationError("num_parties", num_parties, 
                                f"Number of parties must be between {self.min_parties} and {self.max_parties}")
        
        if not (0 <= party_id < num_parties):
            raise ValidationError("party_id", party_id, 
                                f"Party ID must be between 0 and {num_parties - 1}")
    
    def _validate_field_size(self, field_size: int):
        """Validate finite field size."""
        if field_size not in self.supported_field_sizes:
            logger.warning(f"Field size {field_size} not in recommended list: {self.supported_field_sizes}")
    
    def validate_secret_shares(self, shares: List[torch.Tensor], expected_threshold: int) -> bool:
        """Validate secret sharing parameters."""
        if not isinstance(shares, list):
            raise ValidationError("shares", type(shares), "Shares must be a list")
        
        if len(shares) < expected_threshold:
            raise ValidationError("shares", len(shares), 
                                f"Insufficient shares: need at least {expected_threshold}, got {len(shares)}")
        
        # Validate share consistency
        if not shares:
            raise ValidationError("shares", shares, "Empty shares list")
        
        reference_shape = shares[0].shape
        reference_dtype = shares[0].dtype
        
        for i, share in enumerate(shares[1:], 1):
            if share.shape != reference_shape:
                raise ValidationError(f"share_{i}_shape", share.shape, 
                                    f"Inconsistent share shape: expected {reference_shape}")
            
            if share.dtype != reference_dtype:
                raise ValidationError(f"share_{i}_dtype", share.dtype, 
                                    f"Inconsistent share dtype: expected {reference_dtype}")
        
        return True
    
    def validate_communication_round(self, round_data: Dict[str, Any], party_id: int, 
                                   expected_parties: List[int]) -> bool:
        """Validate data for a communication round."""
        if not isinstance(round_data, dict):
            raise ValidationError("round_data", type(round_data), "Round data must be a dictionary")
        
        # Check required fields
        required_fields = ["sender", "receiver", "message_type", "payload"]
        for field in required_fields:
            if field not in round_data:
                raise ValidationError(f"missing_{field}", round_data.keys(), 
                                    f"Missing required field: {field}")
        
        # Validate sender/receiver
        sender = round_data["sender"]
        receiver = round_data["receiver"]
        
        if sender not in expected_parties:
            raise ValidationError("sender", sender, f"Invalid sender: not in {expected_parties}")
        
        if receiver not in expected_parties:
            raise ValidationError("receiver", receiver, f"Invalid receiver: not in {expected_parties}")
        
        if sender == receiver:
            raise ValidationError("sender_receiver", (sender, receiver), 
                                "Sender and receiver cannot be the same")
        
        return True


class SecurityValidator:
    """Validator for security-related parameters and constraints."""
    
    def __init__(self):
        self.min_key_size = 80  # Minimum security level in bits
        self.max_key_size = 256  # Maximum practical security level
        
    def validate_security_parameters(self, security_level: int, protocol_type: str) -> bool:
        """Validate security parameters."""
        if not (self.min_key_size <= security_level <= self.max_key_size):
            raise ValidationError("security_level", security_level, 
                                f"Security level must be between {self.min_key_size} and {self.max_key_size}")
        
        # Protocol-specific security validations
        if protocol_type == "semi_honest" and security_level < 128:
            logger.warning(f"Low security level ({security_level}) for semi-honest protocol")
        
        if protocol_type == "malicious" and security_level < 128:
            raise ValidationError("security_level", security_level, 
                                "Malicious security requires at least 128-bit security")
        
        return True
    
    def validate_privacy_budget(self, epsilon: float, delta: float) -> bool:
        """Validate differential privacy parameters."""
        if not isinstance(epsilon, (int, float)):
            raise ValidationError("epsilon", epsilon, "Epsilon must be a number")
        
        if not isinstance(delta, (int, float)):
            raise ValidationError("delta", delta, "Delta must be a number")
        
        if epsilon <= 0:
            raise ValidationError("epsilon", epsilon, "Epsilon must be positive")
        
        if not (0 <= delta <= 1):
            raise ValidationError("delta", delta, "Delta must be between 0 and 1")
        
        # Warn about weak privacy parameters
        if epsilon > 10:
            logger.warning(f"Large epsilon value ({epsilon}) provides weak privacy")
        
        if delta > 1e-5:
            logger.warning(f"Large delta value ({delta}) provides weak privacy")
        
        return True
    
    def validate_encryption_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate homomorphic encryption parameters."""
        required_params = ["poly_modulus_degree", "coeff_modulus", "plain_modulus"]
        
        for param in required_params:
            if param not in params:
                raise ValidationError(f"missing_{param}", params.keys(), 
                                    f"Missing encryption parameter: {param}")
        
        # Validate polynomial modulus degree (must be power of 2)
        poly_degree = params["poly_modulus_degree"]
        if not isinstance(poly_degree, int) or poly_degree <= 0:
            raise ValidationError("poly_modulus_degree", poly_degree, 
                                "Polynomial modulus degree must be a positive integer")
        
        if not (poly_degree & (poly_degree - 1)) == 0:
            raise ValidationError("poly_modulus_degree", poly_degree, 
                                "Polynomial modulus degree must be a power of 2")
        
        return True


class NetworkValidator:
    """Validator for network and communication parameters."""
    
    @staticmethod
    def validate_party_config(party_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate multi-party configuration."""
        errors = []
        
        required_fields = ["party_id", "num_parties", "endpoints"]
        for field in required_fields:
            if field not in party_config:
                errors.append(ValidationError(
                    field=field,
                    value=None,
                    error_type="missing_field",
                    message=f"Required field {field} missing from party config",
                    severity="critical"
                ))
        
        if "endpoints" in party_config:
            endpoints = party_config["endpoints"]
            if not isinstance(endpoints, list):
                errors.append(ValidationError(
                    field="endpoints",
                    value=endpoints,
                    error_type="type_error",
                    message="Endpoints must be list",
                    severity="critical"
                ))
            else:
                for i, endpoint in enumerate(endpoints):
                    if not isinstance(endpoint, str):
                        errors.append(ValidationError(
                            field=f"endpoints[{i}]",
                            value=endpoint,
                            error_type="type_error",
                            message="Endpoint must be string",
                            severity="error"
                        ))
                    elif not NetworkValidator.is_valid_endpoint(endpoint):
                        errors.append(ValidationError(
                            field=f"endpoints[{i}]",
                            value=endpoint,
                            error_type="format_error",
                            message=f"Invalid endpoint format: {endpoint}",
                            severity="error"
                        ))
        
        return errors
    
    @staticmethod
    def is_valid_endpoint(endpoint: str) -> bool:
        """Check if endpoint has valid format."""
        # Simple validation for host:port format
        pattern = r'^[a-zA-Z0-9.-]+:\d{1,5}$'
        return re.match(pattern, endpoint) is not None


# Global validator instance
default_validator = InputValidator()