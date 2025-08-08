"""Production secrets management and secure configuration handling."""

import os
import json
import base64
import secrets
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import hmac

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    TOKEN = "token"
    CONNECTION_STRING = "connection_string"


class SecretSource(Enum):
    """Sources for secrets."""
    ENVIRONMENT = "environment"
    FILE = "file"
    VAULT = "vault"
    KUBERNETES = "kubernetes"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_SECRET_MANAGER = "gcp_secret_manager"


@dataclass
class SecretMetadata:
    """Metadata for a secret."""
    
    name: str
    secret_type: SecretType
    source: SecretSource
    created_at: float
    expires_at: Optional[float] = None
    last_accessed: Optional[float] = None
    access_count: int = 0
    encrypted: bool = True
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (excluding sensitive data)."""
        return {
            'name': self.name,
            'secret_type': self.secret_type.value,
            'source': self.source.value,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'encrypted': self.encrypted,
            'tags': self.tags
        }


class SecretStore:
    """Secure storage for secrets."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key is None:
            encryption_key = self._derive_key_from_env()
        
        self.fernet = Fernet(encryption_key)
        self.secrets: Dict[str, bytes] = {}  # Encrypted secret values
        self.metadata: Dict[str, SecretMetadata] = {}
        self._lock = threading.RLock()
        
    def _derive_key_from_env(self) -> bytes:
        """Derive encryption key from environment."""
        # Try to get master key from environment
        env_key = os.environ.get('SECRETS_MASTER_KEY')
        if env_key:
            try:
                return base64.b64decode(env_key.encode())
            except Exception:
                logger.warning("Invalid SECRETS_MASTER_KEY format, generating new key")
        
        # Generate key from system information (not recommended for production)
        system_info = f"{os.uname().nodename}{os.getpid()}"
        
        password = system_info.encode()
        salt = b'secure_mpc_salt_2024'  # In production, use a random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        logger.warning("Generated encryption key from system info. Set SECRETS_MASTER_KEY for production.")
        return key
    
    def store_secret(self, name: str, value: str, metadata: SecretMetadata) -> bool:
        """Store a secret securely."""
        try:
            encrypted_value = self.fernet.encrypt(value.encode())
            
            with self._lock:
                self.secrets[name] = encrypted_value
                self.metadata[name] = metadata
            
            logger.debug(f"Stored secret: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {e}")
            return False
    
    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret."""
        with self._lock:
            if name not in self.secrets:
                return None
            
            try:
                encrypted_value = self.secrets[name]
                decrypted_value = self.fernet.decrypt(encrypted_value).decode()
                
                # Update metadata
                if name in self.metadata:
                    metadata = self.metadata[name]
                    metadata.last_accessed = time.time()
                    metadata.access_count += 1
                    
                    # Check expiration
                    if metadata.expires_at and time.time() > metadata.expires_at:
                        logger.warning(f"Secret {name} has expired")
                        del self.secrets[name]
                        del self.metadata[name]
                        return None
                
                return decrypted_value
                
            except Exception as e:
                logger.error(f"Failed to decrypt secret {name}: {e}")
                return None
    
    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        with self._lock:
            if name in self.secrets:
                del self.secrets[name]
                if name in self.metadata:
                    del self.metadata[name]
                logger.info(f"Deleted secret: {name}")
                return True
            return False
    
    def list_secrets(self) -> List[SecretMetadata]:
        """List all secret metadata (no values)."""
        with self._lock:
            return list(self.metadata.values())
    
    def cleanup_expired_secrets(self) -> int:
        """Remove expired secrets."""
        current_time = time.time()
        expired = []
        
        with self._lock:
            for name, metadata in self.metadata.items():
                if metadata.expires_at and current_time > metadata.expires_at:
                    expired.append(name)
            
            for name in expired:
                del self.secrets[name]
                del self.metadata[name]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired secrets")
        
        return len(expired)


class EnvironmentSecretsProvider:
    """Load secrets from environment variables."""
    
    def __init__(self, prefix: str = "SECRET_"):
        self.prefix = prefix
    
    def load_secrets(self) -> Dict[str, str]:
        """Load secrets from environment."""
        secrets_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                secret_name = key[len(self.prefix):].lower()
                secrets_dict[secret_name] = value
        
        return secrets_dict


class FileSecretsProvider:
    """Load secrets from files."""
    
    def __init__(self, secrets_dir: str = "/etc/secrets"):
        self.secrets_dir = Path(secrets_dir)
    
    def load_secrets(self) -> Dict[str, str]:
        """Load secrets from files."""
        secrets_dict = {}
        
        if not self.secrets_dir.exists():
            return secrets_dict
        
        try:
            for secret_file in self.secrets_dir.iterdir():
                if secret_file.is_file() and not secret_file.name.startswith('.'):
                    try:
                        content = secret_file.read_text().strip()
                        secrets_dict[secret_file.name] = content
                    except Exception as e:
                        logger.error(f"Failed to read secret file {secret_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to scan secrets directory: {e}")
        
        return secrets_dict


class KubernetesSecretsProvider:
    """Load secrets from Kubernetes secrets."""
    
    def __init__(self, secrets_path: str = "/var/run/secrets"):
        self.secrets_path = Path(secrets_path)
    
    def load_secrets(self) -> Dict[str, str]:
        """Load secrets from Kubernetes mounted secrets."""
        secrets_dict = {}
        
        if not self.secrets_path.exists():
            return secrets_dict
        
        try:
            # Look for mounted secret directories
            for secret_dir in self.secrets_path.iterdir():
                if secret_dir.is_dir():
                    for secret_file in secret_dir.iterdir():
                        if secret_file.is_file():
                            try:
                                content = secret_file.read_text().strip()
                                secret_name = f"{secret_dir.name}_{secret_file.name}"
                                secrets_dict[secret_name] = content
                            except Exception as e:
                                logger.error(f"Failed to read Kubernetes secret {secret_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to scan Kubernetes secrets: {e}")
        
        return secrets_dict


class SecretsManager:
    """Main secrets management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize secret store
        self.secret_store = SecretStore()
        
        # Initialize providers
        self.providers = []
        self._initialize_providers()
        
        # Secret validation rules
        self.validation_rules = {
            SecretType.API_KEY: self._validate_api_key,
            SecretType.DATABASE_PASSWORD: self._validate_password,
            SecretType.JWT_SECRET: self._validate_jwt_secret,
            SecretType.ENCRYPTION_KEY: self._validate_encryption_key
        }
        
        # Load secrets from providers
        self._load_all_secrets()
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info("Secrets manager initialized")
    
    def _initialize_providers(self):
        """Initialize secret providers based on configuration."""
        
        # Environment provider (always enabled)
        env_prefix = self.config.get('env_prefix', 'SECRET_')
        self.providers.append(EnvironmentSecretsProvider(env_prefix))
        
        # File provider
        if self.config.get('enable_file_secrets', True):
            secrets_dir = self.config.get('secrets_dir', '/etc/secrets')
            self.providers.append(FileSecretsProvider(secrets_dir))
        
        # Kubernetes provider
        if self.config.get('enable_kubernetes_secrets', True):
            k8s_path = self.config.get('kubernetes_secrets_path', '/var/run/secrets')
            self.providers.append(KubernetesSecretsProvider(k8s_path))
    
    def _load_all_secrets(self):
        """Load secrets from all providers."""
        all_secrets = {}
        
        for provider in self.providers:
            try:
                provider_secrets = provider.load_secrets()
                all_secrets.update(provider_secrets)
                logger.debug(f"Loaded {len(provider_secrets)} secrets from {type(provider).__name__}")
            except Exception as e:
                logger.error(f"Failed to load secrets from {type(provider).__name__}: {e}")
        
        # Store secrets with metadata
        for name, value in all_secrets.items():
            secret_type = self._infer_secret_type(name, value)
            source = SecretSource.ENVIRONMENT  # Simplified for now
            
            metadata = SecretMetadata(
                name=name,
                secret_type=secret_type,
                source=source,
                created_at=time.time(),
                tags=[]
            )
            
            self.secret_store.store_secret(name, value, metadata)
        
        logger.info(f"Loaded {len(all_secrets)} secrets from {len(self.providers)} providers")
    
    def _infer_secret_type(self, name: str, value: str) -> SecretType:
        """Infer secret type from name and value."""
        name_lower = name.lower()
        
        if 'api' in name_lower and 'key' in name_lower:
            return SecretType.API_KEY
        elif 'password' in name_lower or 'pwd' in name_lower:
            return SecretType.DATABASE_PASSWORD
        elif 'jwt' in name_lower and 'secret' in name_lower:
            return SecretType.JWT_SECRET
        elif 'encrypt' in name_lower and 'key' in name_lower:
            return SecretType.ENCRYPTION_KEY
        elif 'cert' in name_lower or 'certificate' in name_lower:
            return SecretType.CERTIFICATE
        elif 'private' in name_lower and 'key' in name_lower:
            return SecretType.PRIVATE_KEY
        elif 'token' in name_lower:
            return SecretType.TOKEN
        elif 'connection' in name_lower or 'conn' in name_lower:
            return SecretType.CONNECTION_STRING
        else:
            return SecretType.API_KEY  # Default
    
    def get_secret(self, name: str, required: bool = True) -> Optional[str]:
        """Get a secret by name."""
        value = self.secret_store.get_secret(name)
        
        if value is None and required:
            logger.error(f"Required secret '{name}' not found")
            raise ValueError(f"Required secret '{name}' not found")
        
        return value
    
    def set_secret(self, name: str, value: str, secret_type: SecretType,
                   expires_in: Optional[float] = None, tags: List[str] = None) -> bool:
        """Set a secret programmatically."""
        
        # Validate secret value
        if secret_type in self.validation_rules:
            is_valid, error_message = self.validation_rules[secret_type](value)
            if not is_valid:
                logger.error(f"Secret validation failed for {name}: {error_message}")
                return False
        
        # Create metadata
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
        
        metadata = SecretMetadata(
            name=name,
            secret_type=secret_type,
            source=SecretSource.ENVIRONMENT,  # Programmatically set
            created_at=time.time(),
            expires_at=expires_at,
            tags=tags or []
        )
        
        return self.secret_store.store_secret(name, value, metadata)
    
    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        return self.secret_store.delete_secret(name)
    
    def rotate_secret(self, name: str, new_value: str) -> bool:
        """Rotate a secret to a new value."""
        metadata_list = self.secret_store.list_secrets()
        metadata = next((m for m in metadata_list if m.name == name), None)
        
        if not metadata:
            logger.error(f"Cannot rotate secret {name}: not found")
            return False
        
        # Validate new value
        if metadata.secret_type in self.validation_rules:
            is_valid, error_message = self.validation_rules[metadata.secret_type](new_value)
            if not is_valid:
                logger.error(f"Secret rotation validation failed for {name}: {error_message}")
                return False
        
        # Update the secret
        metadata.created_at = time.time()
        metadata.access_count = 0
        metadata.last_accessed = None
        
        success = self.secret_store.store_secret(name, new_value, metadata)
        if success:
            logger.info(f"Rotated secret: {name}")
        
        return success
    
    def generate_secret(self, name: str, secret_type: SecretType, length: int = 32,
                       expires_in: Optional[float] = None) -> str:
        """Generate a new secure secret."""
        
        if secret_type == SecretType.API_KEY:
            # Generate URL-safe random string
            secret_value = secrets.token_urlsafe(length)
        
        elif secret_type == SecretType.JWT_SECRET:
            # Generate hex string for JWT
            secret_value = secrets.token_hex(length)
        
        elif secret_type == SecretType.ENCRYPTION_KEY:
            # Generate base64-encoded key for Fernet
            key = Fernet.generate_key()
            secret_value = key.decode()
        
        else:
            # Default: URL-safe random string
            secret_value = secrets.token_urlsafe(length)
        
        # Store the generated secret
        success = self.set_secret(name, secret_value, secret_type, expires_in)
        if not success:
            raise RuntimeError(f"Failed to store generated secret: {name}")
        
        logger.info(f"Generated new secret: {name}")
        return secret_value
    
    def _validate_api_key(self, value: str) -> tuple[bool, Optional[str]]:
        """Validate API key format and strength."""
        if len(value) < 16:
            return False, "API key too short (minimum 16 characters)"
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            return False, "API key contains invalid characters"
        
        return True, None
    
    def _validate_password(self, value: str) -> tuple[bool, Optional[str]]:
        """Validate password strength."""
        if len(value) < 8:
            return False, "Password too short (minimum 8 characters)"
        
        # Check for basic complexity
        has_upper = any(c.isupper() for c in value)
        has_lower = any(c.islower() for c in value)
        has_digit = any(c.isdigit() for c in value)
        
        if not (has_upper and has_lower and has_digit):
            return False, "Password must contain uppercase, lowercase, and digits"
        
        return True, None
    
    def _validate_jwt_secret(self, value: str) -> tuple[bool, Optional[str]]:
        """Validate JWT secret."""
        if len(value) < 32:
            return False, "JWT secret too short (minimum 32 characters)"
        
        return True, None
    
    def _validate_encryption_key(self, value: str) -> tuple[bool, Optional[str]]:
        """Validate encryption key."""
        try:
            # Try to use as Fernet key
            Fernet(value.encode())
            return True, None
        except Exception:
            return False, "Invalid Fernet encryption key format"
    
    def _start_cleanup_task(self):
        """Start background cleanup of expired secrets."""
        def cleanup_worker():
            while True:
                try:
                    self.secret_store.cleanup_expired_secrets()
                    time.sleep(3600)  # Clean up every hour
                except Exception as e:
                    logger.error(f"Secret cleanup error: {e}")
                    time.sleep(3600)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get_secrets_summary(self) -> Dict[str, Any]:
        """Get summary of managed secrets."""
        metadata_list = self.secret_store.list_secrets()
        
        by_type = {}
        by_source = {}
        total_accesses = 0
        expired_count = 0
        current_time = time.time()
        
        for metadata in metadata_list:
            # Count by type
            type_name = metadata.secret_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            
            # Count by source
            source_name = metadata.source.value
            by_source[source_name] = by_source.get(source_name, 0) + 1
            
            # Sum accesses
            total_accesses += metadata.access_count
            
            # Count expired
            if metadata.expires_at and current_time > metadata.expires_at:
                expired_count += 1
        
        return {
            'total_secrets': len(metadata_list),
            'by_type': by_type,
            'by_source': by_source,
            'total_accesses': total_accesses,
            'expired_secrets': expired_count,
            'providers_count': len(self.providers)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of secrets system."""
        try:
            # Test secret store functionality
            test_secret = "health_check_secret"
            test_value = secrets.token_hex(16)
            
            # Store test secret
            metadata = SecretMetadata(
                name=test_secret,
                secret_type=SecretType.API_KEY,
                source=SecretSource.ENVIRONMENT,
                created_at=time.time()
            )
            
            store_success = self.secret_store.store_secret(test_secret, test_value, metadata)
            
            # Retrieve test secret
            retrieved_value = self.secret_store.get_secret(test_secret)
            retrieve_success = retrieved_value == test_value
            
            # Clean up test secret
            self.secret_store.delete_secret(test_secret)
            
            return {
                'healthy': store_success and retrieve_success,
                'store_test': store_success,
                'retrieve_test': retrieve_success,
                'secrets_count': len(self.secret_store.list_secrets()),
                'providers_count': len(self.providers)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }


# Global secrets manager
secrets_manager = SecretsManager()


def require_secret(secret_name: str):
    """Decorator to ensure a secret is available."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            secret_value = secrets_manager.get_secret(secret_name, required=True)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Import at the end to avoid circular imports
import re
import functools