"""Advanced schema validation and data sanitization system."""

import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import functools
import inspect
from datetime import datetime
import html
import urllib.parse

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FieldType(Enum):
    """Supported field types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    REGEX = "regex"


@dataclass
class ValidationError:
    """Represents a validation error."""
    
    field_path: str
    error_type: str
    message: str
    severity: ValidationSeverity
    received_value: Any
    expected_type: Optional[str] = None
    constraint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'field_path': self.field_path,
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'received_value': str(self.received_value) if self.received_value is not None else None,
            'expected_type': self.expected_type,
            'constraint': self.constraint
        }


@dataclass
class FieldConstraints:
    """Constraints for field validation."""
    
    # Basic constraints
    required: bool = False
    nullable: bool = True
    default_value: Any = None
    
    # Type-specific constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    
    # Array/object constraints
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: bool = False
    item_schema: Optional['FieldSchema'] = None
    
    # Custom validation
    custom_validator: Optional[Callable[[Any], bool]] = None
    custom_sanitizer: Optional[Callable[[Any], Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'required': self.required,
            'nullable': self.nullable,
            'default_value': self.default_value,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'pattern': self.pattern,
            'allowed_values': self.allowed_values,
            'min_items': self.min_items,
            'max_items': self.max_items,
            'unique_items': self.unique_items,
            'has_custom_validator': self.custom_validator is not None,
            'has_custom_sanitizer': self.custom_sanitizer is not None
        }


@dataclass
class FieldSchema:
    """Schema definition for a field."""
    
    field_type: FieldType
    constraints: FieldConstraints = field(default_factory=FieldConstraints)
    description: Optional[str] = None
    examples: List[Any] = field(default_factory=list)
    
    # For object types
    properties: Optional[Dict[str, 'FieldSchema']] = None
    additional_properties: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = {
            'field_type': self.field_type.value,
            'constraints': self.constraints.to_dict(),
            'description': self.description,
            'examples': self.examples,
            'additional_properties': self.additional_properties
        }
        
        if self.properties:
            data['properties'] = {k: v.to_dict() for k, v in self.properties.items()}
        
        return data


class DataSanitizer:
    """Data sanitization utilities."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        # HTML escape
        value = html.escape(value)
        
        # Truncate if needed
        if max_length and len(value) > max_length:
            value = value[:max_length]
        
        return value
    
    @staticmethod
    def sanitize_email(email: str) -> str:
        """Sanitize email input."""
        if not isinstance(email, str):
            email = str(email)
        
        # Basic sanitization
        email = email.strip().lower()
        
        # Remove dangerous characters
        email = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', email)
        
        return email
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL input."""
        if not isinstance(url, str):
            url = str(url)
        
        # Basic sanitization
        url = url.strip()
        
        # Ensure safe protocols
        if url and not re.match(r'^https?://', url, re.IGNORECASE):
            if not url.startswith('//'):
                url = 'http://' + url
        
        # URL encode dangerous characters
        try:
            parsed = urllib.parse.urlparse(url)
            sanitized = urllib.parse.urlunparse(parsed)
            return sanitized
        except Exception:
            # If parsing fails, return empty string
            return ''
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename input."""
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure it's not empty
        if not filename:
            filename = 'unnamed_file'
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def sanitize_json(data: Any) -> Any:
        """Recursively sanitize JSON data."""
        if isinstance(data, dict):
            return {
                DataSanitizer.sanitize_string(str(k)): DataSanitizer.sanitize_json(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [DataSanitizer.sanitize_json(item) for item in data]
        elif isinstance(data, str):
            return DataSanitizer.sanitize_string(data)
        else:
            return data


class TypeValidator:
    """Type validation utilities."""
    
    @staticmethod
    def validate_string(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate string value."""
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                return False, "Cannot convert to string"
        
        # Length constraints
        if constraints.min_length is not None and len(value) < constraints.min_length:
            return False, f"String too short (minimum {constraints.min_length} characters)"
        
        if constraints.max_length is not None and len(value) > constraints.max_length:
            return False, f"String too long (maximum {constraints.max_length} characters)"
        
        # Pattern matching
        if constraints.pattern:
            try:
                if not re.match(constraints.pattern, value):
                    return False, f"String does not match pattern: {constraints.pattern}"
            except re.error as e:
                return False, f"Invalid regex pattern: {str(e)}"
        
        # Allowed values
        if constraints.allowed_values is not None and value not in constraints.allowed_values:
            return False, f"Value not in allowed list: {constraints.allowed_values}"
        
        return True, None
    
    @staticmethod
    def validate_integer(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate integer value."""
        if isinstance(value, bool):
            return False, "Boolean is not a valid integer"
        
        if not isinstance(value, int):
            try:
                if isinstance(value, str):
                    value = int(value)
                elif isinstance(value, float):
                    if value.is_integer():
                        value = int(value)
                    else:
                        return False, "Float is not an integer"
                else:
                    value = int(value)
            except (ValueError, TypeError):
                return False, "Cannot convert to integer"
        
        # Range constraints
        if constraints.min_value is not None and value < constraints.min_value:
            return False, f"Value too small (minimum {constraints.min_value})"
        
        if constraints.max_value is not None and value > constraints.max_value:
            return False, f"Value too large (maximum {constraints.max_value})"
        
        # Allowed values
        if constraints.allowed_values is not None and value not in constraints.allowed_values:
            return False, f"Value not in allowed list: {constraints.allowed_values}"
        
        return True, None
    
    @staticmethod
    def validate_float(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate float value."""
        if isinstance(value, bool):
            return False, "Boolean is not a valid float"
        
        if not isinstance(value, (int, float)):
            try:
                if isinstance(value, str):
                    value = float(value)
                else:
                    value = float(value)
            except (ValueError, TypeError):
                return False, "Cannot convert to float"
        
        # Check for special values
        if not isinstance(value, (int, float)) or not (value == value):  # NaN check
            return False, "Invalid float value (NaN or infinity)"
        
        # Range constraints
        if constraints.min_value is not None and value < constraints.min_value:
            return False, f"Value too small (minimum {constraints.min_value})"
        
        if constraints.max_value is not None and value > constraints.max_value:
            return False, f"Value too large (maximum {constraints.max_value})"
        
        return True, None
    
    @staticmethod
    def validate_boolean(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate boolean value."""
        if isinstance(value, bool):
            return True, None
        
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True, None
            elif value.lower() in ('false', '0', 'no', 'off'):
                return True, None
        
        if isinstance(value, int):
            if value in (0, 1):
                return True, None
        
        return False, "Cannot convert to boolean"
    
    @staticmethod
    def validate_email(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate email address."""
        if not isinstance(value, str):
            return False, "Email must be a string"
        
        # Basic email regex (simplified)
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, value):
            return False, "Invalid email format"
        
        # Additional length check
        if len(value) > 254:  # RFC 5321 limit
            return False, "Email address too long"
        
        return True, None
    
    @staticmethod
    def validate_url(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate URL."""
        if not isinstance(value, str):
            return False, "URL must be a string"
        
        try:
            parsed = urllib.parse.urlparse(value)
            if not all([parsed.scheme, parsed.netloc]):
                return False, "Invalid URL format"
            
            # Check for safe protocols
            if parsed.scheme not in ('http', 'https'):
                return False, f"Unsafe protocol: {parsed.scheme}"
            
            return True, None
        
        except Exception as e:
            return False, f"Invalid URL: {str(e)}"
    
    @staticmethod
    def validate_uuid(value: Any, constraints: FieldConstraints) -> Tuple[bool, Optional[str]]:
        """Validate UUID."""
        if not isinstance(value, str):
            return False, "UUID must be a string"
        
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        
        if not re.match(uuid_pattern, value):
            return False, "Invalid UUID format"
        
        return True, None
    
    @staticmethod
    def validate_array(value: Any, constraints: FieldConstraints, item_schema: Optional[FieldSchema] = None) -> Tuple[bool, Optional[str]]:
        """Validate array value."""
        if not isinstance(value, list):
            return False, "Value must be an array"
        
        # Size constraints
        if constraints.min_items is not None and len(value) < constraints.min_items:
            return False, f"Array too small (minimum {constraints.min_items} items)"
        
        if constraints.max_items is not None and len(value) > constraints.max_items:
            return False, f"Array too large (maximum {constraints.max_items} items)"
        
        # Unique items
        if constraints.unique_items and len(value) != len(set(str(item) for item in value)):
            return False, "Array items must be unique"
        
        return True, None


class SchemaValidator:
    """Main schema validation engine."""
    
    def __init__(self):
        self.type_validators = {
            FieldType.STRING: TypeValidator.validate_string,
            FieldType.INTEGER: TypeValidator.validate_integer,
            FieldType.FLOAT: TypeValidator.validate_float,
            FieldType.BOOLEAN: TypeValidator.validate_boolean,
            FieldType.EMAIL: TypeValidator.validate_email,
            FieldType.URL: TypeValidator.validate_url,
            FieldType.UUID: TypeValidator.validate_uuid,
            FieldType.ARRAY: TypeValidator.validate_array,
        }
        
        self.sanitizers = {
            FieldType.STRING: DataSanitizer.sanitize_string,
            FieldType.EMAIL: DataSanitizer.sanitize_email,
            FieldType.URL: DataSanitizer.sanitize_url,
        }
        
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'sanitizations_applied': 0
        }
    
    def validate(self, data: Dict[str, Any], schema: Dict[str, FieldSchema],
                 sanitize: bool = True) -> Tuple[bool, List[ValidationError], Dict[str, Any]]:
        """Validate data against schema."""
        errors = []
        sanitized_data = {}
        
        self.validation_stats['total_validations'] += 1
        
        # Check required fields
        for field_name, field_schema in schema.items():
            if field_schema.constraints.required and field_name not in data:
                errors.append(ValidationError(
                    field_path=field_name,
                    error_type="missing_required_field",
                    message=f"Required field '{field_name}' is missing",
                    severity=ValidationSeverity.ERROR,
                    received_value=None,
                    expected_type=field_schema.field_type.value
                ))
        
        # Validate existing fields
        for field_name, value in data.items():
            if field_name in schema:
                field_errors, sanitized_value = self._validate_field(
                    field_name, value, schema[field_name], sanitize
                )
                errors.extend(field_errors)
                sanitized_data[field_name] = sanitized_value
            else:
                # Handle additional properties
                sanitized_data[field_name] = value
        
        # Add default values for missing optional fields
        for field_name, field_schema in schema.items():
            if (field_name not in data and
                not field_schema.constraints.required and
                field_schema.constraints.default_value is not None):
                sanitized_data[field_name] = field_schema.constraints.default_value
        
        # Update stats
        if errors:
            self.validation_stats['failed_validations'] += 1
        else:
            self.validation_stats['successful_validations'] += 1
        
        return len(errors) == 0, errors, sanitized_data
    
    def _validate_field(self, field_path: str, value: Any, schema: FieldSchema,
                       sanitize: bool) -> Tuple[List[ValidationError], Any]:
        """Validate a single field."""
        errors = []
        sanitized_value = value
        
        # Check nullable
        if value is None:
            if not schema.constraints.nullable:
                errors.append(ValidationError(
                    field_path=field_path,
                    error_type="null_not_allowed",
                    message=f"Field '{field_path}' cannot be null",
                    severity=ValidationSeverity.ERROR,
                    received_value=value,
                    expected_type=schema.field_type.value
                ))
            return errors, sanitized_value
        
        # Sanitize if requested
        if sanitize and schema.field_type in self.sanitizers:
            try:
                if schema.field_type == FieldType.STRING and schema.constraints.max_length:
                    sanitized_value = self.sanitizers[schema.field_type](
                        value, schema.constraints.max_length
                    )
                else:
                    sanitized_value = self.sanitizers[schema.field_type](value)
                
                if sanitized_value != value:
                    self.validation_stats['sanitizations_applied'] += 1
                
            except Exception as e:
                errors.append(ValidationError(
                    field_path=field_path,
                    error_type="sanitization_failed",
                    message=f"Failed to sanitize field '{field_path}': {str(e)}",
                    severity=ValidationSeverity.WARNING,
                    received_value=value
                ))
        
        # Type validation
        if schema.field_type in self.type_validators:
            is_valid, error_message = self.type_validators[schema.field_type](
                sanitized_value, schema.constraints
            )
            
            if not is_valid:
                errors.append(ValidationError(
                    field_path=field_path,
                    error_type="type_validation_failed",
                    message=error_message,
                    severity=ValidationSeverity.ERROR,
                    received_value=sanitized_value,
                    expected_type=schema.field_type.value
                ))
        
        # Custom validation
        if schema.constraints.custom_validator:
            try:
                if not schema.constraints.custom_validator(sanitized_value):
                    errors.append(ValidationError(
                        field_path=field_path,
                        error_type="custom_validation_failed",
                        message=f"Custom validation failed for field '{field_path}'",
                        severity=ValidationSeverity.ERROR,
                        received_value=sanitized_value
                    ))
            except Exception as e:
                errors.append(ValidationError(
                    field_path=field_path,
                    error_type="custom_validation_error",
                    message=f"Custom validator error for field '{field_path}': {str(e)}",
                    severity=ValidationSeverity.WARNING,
                    received_value=sanitized_value
                ))
        
        # Custom sanitization
        if sanitize and schema.constraints.custom_sanitizer:
            try:
                sanitized_value = schema.constraints.custom_sanitizer(sanitized_value)
                self.validation_stats['sanitizations_applied'] += 1
            except Exception as e:
                errors.append(ValidationError(
                    field_path=field_path,
                    error_type="custom_sanitization_error",
                    message=f"Custom sanitizer error for field '{field_path}': {str(e)}",
                    severity=ValidationSeverity.WARNING,
                    received_value=sanitized_value
                ))
        
        # Validate nested objects
        if schema.field_type == FieldType.OBJECT and schema.properties:
            if isinstance(sanitized_value, dict):
                nested_valid, nested_errors, nested_sanitized = self.validate(
                    sanitized_value, schema.properties, sanitize
                )
                
                # Adjust field paths for nested errors
                for error in nested_errors:
                    error.field_path = f"{field_path}.{error.field_path}"
                
                errors.extend(nested_errors)
                sanitized_value = nested_sanitized
            else:
                errors.append(ValidationError(
                    field_path=field_path,
                    error_type="type_mismatch",
                    message=f"Expected object for field '{field_path}'",
                    severity=ValidationSeverity.ERROR,
                    received_value=sanitized_value,
                    expected_type="object"
                ))
        
        # Validate arrays
        elif schema.field_type == FieldType.ARRAY:
            if isinstance(sanitized_value, list) and schema.constraints.item_schema:
                sanitized_items = []
                for i, item in enumerate(sanitized_value):
                    item_errors, sanitized_item = self._validate_field(
                        f"{field_path}[{i}]", item, schema.constraints.item_schema, sanitize
                    )
                    errors.extend(item_errors)
                    sanitized_items.append(sanitized_item)
                
                sanitized_value = sanitized_items
        
        return errors, sanitized_value
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()


# Global validator instance
schema_validator = SchemaValidator()


def validate_schema(schema: Dict[str, FieldSchema], sanitize: bool = True):
    """Decorator for schema validation."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract data from first argument (assuming it's a dict)
            if args and isinstance(args[0], dict):
                data = args[0]
                is_valid, errors, sanitized_data = schema_validator.validate(
                    data, schema, sanitize
                )
                
                if not is_valid:
                    error_details = [error.to_dict() for error in errors]
                    raise ValueError(f"Validation failed: {error_details}")
                
                # Replace first argument with sanitized data
                args = (sanitized_data,) + args[1:]
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Predefined schemas for common use cases
class CommonSchemas:
    """Common validation schemas."""
    
    @staticmethod
    def user_input_schema() -> Dict[str, FieldSchema]:
        """Schema for user input validation."""
        return {
            'text': FieldSchema(
                field_type=FieldType.STRING,
                constraints=FieldConstraints(
                    required=True,
                    min_length=1,
                    max_length=10000,
                    pattern=r'^[^<>]*$'  # No HTML tags
                ),
                description="User input text"
            ),
            'max_length': FieldSchema(
                field_type=FieldType.INTEGER,
                constraints=FieldConstraints(
                    min_value=1,
                    max_value=1000,
                    default_value=512
                ),
                description="Maximum sequence length"
            )
        }
    
    @staticmethod
    def mpc_config_schema() -> Dict[str, FieldSchema]:
        """Schema for MPC configuration."""
        return {
            'protocol_name': FieldSchema(
                field_type=FieldType.STRING,
                constraints=FieldConstraints(
                    required=True,
                    allowed_values=['aby3', 'semi_honest_3pc', 'malicious_3pc']
                )
            ),
            'security_level': FieldSchema(
                field_type=FieldType.INTEGER,
                constraints=FieldConstraints(
                    required=True,
                    min_value=80,
                    max_value=256
                )
            ),
            'num_parties': FieldSchema(
                field_type=FieldType.INTEGER,
                constraints=FieldConstraints(
                    required=True,
                    min_value=2,
                    max_value=10
                )
            )
        }
    
    @staticmethod
    def api_request_schema() -> Dict[str, FieldSchema]:
        """Schema for API request validation."""
        return {
            'request_id': FieldSchema(
                field_type=FieldType.UUID,
                constraints=FieldConstraints(required=True)
            ),
            'timestamp': FieldSchema(
                field_type=FieldType.INTEGER,
                constraints=FieldConstraints(required=True)
            ),
            'user_id': FieldSchema(
                field_type=FieldType.STRING,
                constraints=FieldConstraints(
                    max_length=256,
                    pattern=r'^[a-zA-Z0-9_-]+$'
                )
            )
        }