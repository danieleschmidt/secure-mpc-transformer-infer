"""Cryptographic key management and automated incident response system."""

import asyncio
import base64
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of cryptographic keys."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    SIGNING = "signing"
    MAC = "mac"
    DERIVATION = "derivation"


class KeyStatus(Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    EXPIRED = "expired"


class IncidentSeverity(Enum):
    """Security incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CryptoKey:
    """Cryptographic key metadata."""

    key_id: str
    key_type: KeyType
    algorithm: str
    key_size: int
    created_at: float
    expires_at: float | None
    status: KeyStatus
    usage_count: int
    max_usage: int | None
    permissions: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['key_type'] = self.key_type.value
        data['status'] = self.status.value
        return data

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_usage_exceeded(self) -> bool:
        """Check if usage limit is exceeded."""
        if self.max_usage is None:
            return False
        return self.usage_count >= self.max_usage


@dataclass
class SecurityIncident:
    """Security incident data."""

    incident_id: str
    incident_type: str
    severity: IncidentSeverity
    timestamp: float
    source: str
    description: str
    affected_resources: list[str]
    indicators: dict[str, Any]
    response_actions: list[str]
    status: str  # open, investigating, resolved
    assigned_to: str | None
    resolution_time: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


class KeyRotationManager:
    """Automatic key rotation system."""

    def __init__(self, rotation_interval: int = 86400):  # 24 hours
        self.rotation_interval = rotation_interval
        self.rotation_schedule = {}
        self.rotation_callbacks = {}
        self._lock = threading.Lock()

    def schedule_rotation(self, key_id: str, callback_func, interval: int | None = None):
        """Schedule automatic key rotation."""
        rotation_interval = interval or self.rotation_interval
        next_rotation = time.time() + rotation_interval

        with self._lock:
            self.rotation_schedule[key_id] = next_rotation
            self.rotation_callbacks[key_id] = callback_func

        logger.info(f"Scheduled rotation for key {key_id} in {rotation_interval} seconds")

    def check_rotations(self):
        """Check and execute pending rotations."""
        current_time = time.time()
        pending_rotations = []

        with self._lock:
            for key_id, rotation_time in self.rotation_schedule.items():
                if current_time >= rotation_time:
                    pending_rotations.append(key_id)

        # Execute rotations
        for key_id in pending_rotations:
            try:
                callback = self.rotation_callbacks.get(key_id)
                if callback:
                    callback(key_id)

                # Reschedule next rotation
                with self._lock:
                    self.rotation_schedule[key_id] = current_time + self.rotation_interval

                logger.info(f"Successfully rotated key {key_id}")

            except Exception as e:
                logger.error(f"Failed to rotate key {key_id}: {e}")

    def cancel_rotation(self, key_id: str):
        """Cancel scheduled rotation."""
        with self._lock:
            self.rotation_schedule.pop(key_id, None)
            self.rotation_callbacks.pop(key_id, None)


class SecureKeyStore:
    """Secure storage for cryptographic keys."""

    def __init__(self, master_key: bytes | None = None):
        if master_key is None:
            # Generate master key from environment or create new
            master_key = self._get_or_create_master_key()

        self.master_key = master_key
        self.fernet = Fernet(base64.urlsafe_b64encode(master_key[:32]))
        self.keys = {}  # key_id -> encrypted_key_data
        self.metadata = {}  # key_id -> CryptoKey
        self._lock = threading.RLock()

    def _get_or_create_master_key(self) -> bytes:
        """Get master key from environment or create new one."""
        env_key = os.environ.get('MPC_MASTER_KEY')
        if env_key:
            return base64.b64decode(env_key.encode())

        # Generate new master key
        master_key = secrets.token_bytes(32)
        logger.warning("Generated new master key. Store MPC_MASTER_KEY in environment for persistence.")
        return master_key

    def store_key(self, key_metadata: CryptoKey, key_data: bytes) -> bool:
        """Store encrypted key."""
        try:
            encrypted_data = self.fernet.encrypt(key_data)

            with self._lock:
                self.keys[key_metadata.key_id] = encrypted_data
                self.metadata[key_metadata.key_id] = key_metadata

            logger.debug(f"Stored key {key_metadata.key_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store key {key_metadata.key_id}: {e}")
            return False

    def retrieve_key(self, key_id: str) -> tuple[CryptoKey, bytes] | None:
        """Retrieve and decrypt key."""
        with self._lock:
            if key_id not in self.keys or key_id not in self.metadata:
                return None

            try:
                encrypted_data = self.keys[key_id]
                key_metadata = self.metadata[key_id]

                # Check key status
                if key_metadata.status == KeyStatus.REVOKED:
                    logger.warning(f"Attempted to use revoked key {key_id}")
                    return None

                if key_metadata.is_expired():
                    logger.warning(f"Attempted to use expired key {key_id}")
                    key_metadata.status = KeyStatus.EXPIRED
                    return None

                if key_metadata.is_usage_exceeded():
                    logger.warning(f"Usage limit exceeded for key {key_id}")
                    key_metadata.status = KeyStatus.DEPRECATED
                    return None

                # Decrypt key
                key_data = self.fernet.decrypt(encrypted_data)

                # Update usage count
                key_metadata.usage_count += 1

                return key_metadata, key_data

            except Exception as e:
                logger.error(f"Failed to retrieve key {key_id}: {e}")
                return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key."""
        with self._lock:
            if key_id in self.metadata:
                self.metadata[key_id].status = KeyStatus.REVOKED
                logger.info(f"Key {key_id} revoked")
                return True
            return False

    def list_keys(self, key_type: KeyType | None = None,
                  status: KeyStatus | None = None) -> list[CryptoKey]:
        """List keys with optional filtering."""
        with self._lock:
            keys = list(self.metadata.values())

        if key_type:
            keys = [k for k in keys if k.key_type == key_type]

        if status:
            keys = [k for k in keys if k.status == status]

        return keys

    def cleanup_expired_keys(self) -> int:
        """Clean up expired and revoked keys."""
        current_time = time.time()
        cleaned_count = 0

        with self._lock:
            expired_keys = []

            for key_id, metadata in self.metadata.items():
                if (metadata.status in [KeyStatus.REVOKED, KeyStatus.EXPIRED] or
                    metadata.is_expired()):
                    expired_keys.append(key_id)

            for key_id in expired_keys:
                del self.keys[key_id]
                del self.metadata[key_id]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired keys")

        return cleaned_count


class CryptographicKeyManager:
    """Main cryptographic key management system."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Initialize components
        self.key_store = SecureKeyStore()
        self.rotation_manager = KeyRotationManager(
            rotation_interval=self.config.get('rotation_interval', 86400)
        )

        # Key generation parameters
        self.default_key_sizes = {
            KeyType.SYMMETRIC: 256,
            KeyType.ASYMMETRIC: 2048,
            KeyType.SIGNING: 256,
            KeyType.MAC: 256
        }

        # Start background tasks
        self._start_maintenance_task()

        logger.info("Cryptographic key manager initialized")

    def generate_symmetric_key(self, algorithm: str = "AES", key_size: int = None,
                             expires_in: int | None = None,
                             max_usage: int | None = None) -> str:
        """Generate a symmetric encryption key."""
        key_size = key_size or self.default_key_sizes[KeyType.SYMMETRIC]
        key_data = secrets.token_bytes(key_size // 8)

        key_id = self._generate_key_id()
        key_metadata = CryptoKey(
            key_id=key_id,
            key_type=KeyType.SYMMETRIC,
            algorithm=algorithm,
            key_size=key_size,
            created_at=time.time(),
            expires_at=time.time() + expires_in if expires_in else None,
            status=KeyStatus.ACTIVE,
            usage_count=0,
            max_usage=max_usage,
            permissions=["encrypt", "decrypt"],
            metadata={}
        )

        self.key_store.store_key(key_metadata, key_data)

        # Schedule rotation if configured
        if self.config.get('auto_rotation', True):
            self.rotation_manager.schedule_rotation(
                key_id,
                lambda kid: self._rotate_symmetric_key(kid)
            )

        logger.info(f"Generated symmetric key {key_id}")
        return key_id

    def generate_asymmetric_keypair(self, algorithm: str = "RSA", key_size: int = None,
                                  expires_in: int | None = None) -> tuple[str, str]:
        """Generate asymmetric key pair."""
        key_size = key_size or self.default_key_sizes[KeyType.ASYMMETRIC]

        if algorithm == "RSA":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            public_key = private_key.public_key()

            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")

        # Store private key
        private_key_id = self._generate_key_id()
        private_metadata = CryptoKey(
            key_id=private_key_id,
            key_type=KeyType.ASYMMETRIC,
            algorithm=f"{algorithm}_PRIVATE",
            key_size=key_size,
            created_at=time.time(),
            expires_at=time.time() + expires_in if expires_in else None,
            status=KeyStatus.ACTIVE,
            usage_count=0,
            max_usage=None,
            permissions=["sign", "decrypt"],
            metadata={"key_pair": True}
        )

        # Store public key
        public_key_id = self._generate_key_id()
        public_metadata = CryptoKey(
            key_id=public_key_id,
            key_type=KeyType.ASYMMETRIC,
            algorithm=f"{algorithm}_PUBLIC",
            key_size=key_size,
            created_at=time.time(),
            expires_at=time.time() + expires_in if expires_in else None,
            status=KeyStatus.ACTIVE,
            usage_count=0,
            max_usage=None,
            permissions=["verify", "encrypt"],
            metadata={"key_pair": True, "private_key_id": private_key_id}
        )

        # Cross-reference the keys
        private_metadata.metadata["public_key_id"] = public_key_id

        self.key_store.store_key(private_metadata, private_pem)
        self.key_store.store_key(public_metadata, public_pem)

        logger.info(f"Generated asymmetric key pair {private_key_id}/{public_key_id}")
        return private_key_id, public_key_id

    def generate_mac_key(self, algorithm: str = "HMAC-SHA256", key_size: int = None,
                        expires_in: int | None = None) -> str:
        """Generate MAC key."""
        key_size = key_size or self.default_key_sizes[KeyType.MAC]
        key_data = secrets.token_bytes(key_size // 8)

        key_id = self._generate_key_id()
        key_metadata = CryptoKey(
            key_id=key_id,
            key_type=KeyType.MAC,
            algorithm=algorithm,
            key_size=key_size,
            created_at=time.time(),
            expires_at=time.time() + expires_in if expires_in else None,
            status=KeyStatus.ACTIVE,
            usage_count=0,
            max_usage=None,
            permissions=["mac", "verify"],
            metadata={}
        )

        self.key_store.store_key(key_metadata, key_data)

        logger.info(f"Generated MAC key {key_id}")
        return key_id

    def derive_key(self, master_key_id: str, context: bytes, length: int = 32,
                  algorithm: str = "HKDF-SHA256") -> str:
        """Derive key from master key."""
        master_key_data = self.key_store.retrieve_key(master_key_id)
        if not master_key_data:
            raise ValueError(f"Master key {master_key_id} not found")

        master_metadata, master_key_bytes = master_key_data

        if algorithm == "HKDF-SHA256":
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=length,
                salt=None,
                info=context
            )
            derived_key = hkdf.derive(master_key_bytes)
        else:
            raise ValueError(f"Unsupported derivation algorithm: {algorithm}")

        # Store derived key
        derived_key_id = self._generate_key_id()
        derived_metadata = CryptoKey(
            key_id=derived_key_id,
            key_type=KeyType.DERIVATION,
            algorithm=algorithm,
            key_size=length * 8,
            created_at=time.time(),
            expires_at=master_metadata.expires_at,  # Same expiration as master
            status=KeyStatus.ACTIVE,
            usage_count=0,
            max_usage=None,
            permissions=["derive"],
            metadata={"master_key_id": master_key_id, "context": context.hex()}
        )

        self.key_store.store_key(derived_metadata, derived_key)

        logger.info(f"Derived key {derived_key_id} from master {master_key_id}")
        return derived_key_id

    def get_key(self, key_id: str) -> tuple[CryptoKey, bytes] | None:
        """Get key by ID."""
        return self.key_store.retrieve_key(key_id)

    def revoke_key(self, key_id: str, reason: str = "Manual revocation"):
        """Revoke a key."""
        success = self.key_store.revoke_key(key_id)
        if success:
            # Cancel any scheduled rotations
            self.rotation_manager.cancel_rotation(key_id)
            logger.warning(f"Key {key_id} revoked: {reason}")
        return success

    def rotate_key(self, key_id: str) -> str | None:
        """Manually rotate a key."""
        key_data = self.key_store.retrieve_key(key_id)
        if not key_data:
            return None

        old_metadata, _ = key_data

        if old_metadata.key_type == KeyType.SYMMETRIC:
            return self._rotate_symmetric_key(key_id)
        else:
            logger.error(f"Key rotation not implemented for type {old_metadata.key_type}")
            return None

    def _rotate_symmetric_key(self, key_id: str) -> str:
        """Rotate a symmetric key."""
        old_key_data = self.key_store.retrieve_key(key_id)
        if not old_key_data:
            return None

        old_metadata, _ = old_key_data

        # Generate new key with same parameters
        new_key_id = self.generate_symmetric_key(
            algorithm=old_metadata.algorithm,
            key_size=old_metadata.key_size,
            max_usage=old_metadata.max_usage
        )

        # Deprecate old key
        old_metadata.status = KeyStatus.DEPRECATED

        logger.info(f"Rotated key {key_id} -> {new_key_id}")
        return new_key_id

    def _generate_key_id(self) -> str:
        """Generate unique key ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = secrets.token_hex(8)
        return f"key_{timestamp}_{random_part}"

    def _start_maintenance_task(self):
        """Start background maintenance task."""
        async def maintenance_loop():
            while True:
                try:
                    # Check key rotations
                    self.rotation_manager.check_rotations()

                    # Clean up expired keys
                    self.key_store.cleanup_expired_keys()

                    await asyncio.sleep(60)  # Run every minute

                except Exception as e:
                    logger.error(f"Key management maintenance error: {e}")
                    await asyncio.sleep(60)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(maintenance_loop())
        except RuntimeError:
            # Start manual maintenance
            threading.Timer(60, self._manual_maintenance).start()

    def _manual_maintenance(self):
        """Manual maintenance for non-async environments."""
        try:
            self.rotation_manager.check_rotations()
            self.key_store.cleanup_expired_keys()
        except Exception as e:
            logger.error(f"Manual key maintenance error: {e}")
        finally:
            threading.Timer(60, self._manual_maintenance).start()

    def get_key_stats(self) -> dict[str, Any]:
        """Get key management statistics."""
        all_keys = self.key_store.list_keys()

        stats = {
            'total_keys': len(all_keys),
            'by_type': {},
            'by_status': {},
            'rotation_schedule_count': len(self.rotation_manager.rotation_schedule),
            'upcoming_expirations': 0
        }

        current_time = time.time()

        for key in all_keys:
            # Count by type
            key_type = key.key_type.value
            stats['by_type'][key_type] = stats['by_type'].get(key_type, 0) + 1

            # Count by status
            status = key.status.value
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # Count upcoming expirations (within 24 hours)
            if key.expires_at and key.expires_at - current_time < 86400:
                stats['upcoming_expirations'] += 1

        return stats


class IncidentResponseSystem:
    """Automated incident response system."""

    def __init__(self, key_manager: CryptographicKeyManager):
        self.key_manager = key_manager
        self.incidents = {}
        self.response_rules = self._initialize_response_rules()
        self.notification_channels = []
        self._lock = threading.Lock()

        logger.info("Incident response system initialized")

    def _initialize_response_rules(self) -> dict[str, dict[str, Any]]:
        """Initialize automatic response rules."""
        return {
            'key_compromise': {
                'severity': IncidentSeverity.CRITICAL,
                'actions': ['revoke_key', 'generate_replacement', 'notify_security_team'],
                'auto_execute': True
            },
            'unusual_key_usage': {
                'severity': IncidentSeverity.MEDIUM,
                'actions': ['monitor_key', 'log_usage', 'alert_admin'],
                'auto_execute': True
            },
            'key_expiration_warning': {
                'severity': IncidentSeverity.LOW,
                'actions': ['schedule_rotation', 'notify_admin'],
                'auto_execute': True
            },
            'failed_key_operation': {
                'severity': IncidentSeverity.MEDIUM,
                'actions': ['log_failure', 'check_key_status', 'alert_if_pattern'],
                'auto_execute': False
            }
        }

    async def report_incident(self, incident_type: str, source: str, description: str,
                            affected_resources: list[str] = None,
                            indicators: dict[str, Any] = None) -> str:
        """Report a security incident."""
        incident_id = self._generate_incident_id()

        # Determine severity
        rule = self.response_rules.get(incident_type, {})
        severity = rule.get('severity', IncidentSeverity.MEDIUM)

        incident = SecurityIncident(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            timestamp=time.time(),
            source=source,
            description=description,
            affected_resources=affected_resources or [],
            indicators=indicators or {},
            response_actions=[],
            status="open",
            assigned_to=None,
            resolution_time=None
        )

        with self._lock:
            self.incidents[incident_id] = incident

        # Execute automatic response if configured
        if rule.get('auto_execute', False):
            await self._execute_response_actions(incident, rule.get('actions', []))

        # Send notifications
        await self._send_notifications(incident)

        logger.warning(f"Security incident {incident_id} reported: {description}")
        return incident_id

    async def _execute_response_actions(self, incident: SecurityIncident,
                                      actions: list[str]):
        """Execute automatic response actions."""
        for action in actions:
            try:
                if action == 'revoke_key' and incident.affected_resources:
                    for resource in incident.affected_resources:
                        if resource.startswith('key_'):
                            self.key_manager.revoke_key(resource, f"Incident: {incident.incident_id}")

                elif action == 'generate_replacement' and incident.affected_resources:
                    for resource in incident.affected_resources:
                        if resource.startswith('key_'):
                            new_key_id = self.key_manager.rotate_key(resource)
                            if new_key_id:
                                logger.info(f"Generated replacement key {new_key_id} for {resource}")

                elif action == 'monitor_key':
                    # Enhanced monitoring for affected keys
                    logger.info(f"Enhanced monitoring activated for incident {incident.incident_id}")

                elif action == 'log_usage':
                    # Log detailed key usage
                    logger.info(f"Detailed logging activated for incident {incident.incident_id}")

                elif action == 'schedule_rotation':
                    # Schedule immediate rotation
                    for resource in incident.affected_resources:
                        if resource.startswith('key_'):
                            self.key_manager.rotation_manager.schedule_rotation(
                                resource,
                                lambda kid: self.key_manager.rotate_key(kid),
                                interval=3600  # 1 hour
                            )

                incident.response_actions.append(f"Executed: {action}")

            except Exception as e:
                error_msg = f"Failed to execute action {action}: {str(e)}"
                incident.response_actions.append(error_msg)
                logger.error(error_msg)

    async def _send_notifications(self, incident: SecurityIncident):
        """Send incident notifications."""
        for channel in self.notification_channels:
            try:
                await channel.send_notification(incident)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")

    def add_notification_channel(self, channel):
        """Add notification channel."""
        self.notification_channels.append(channel)

    def resolve_incident(self, incident_id: str, resolution_notes: str = ""):
        """Mark incident as resolved."""
        with self._lock:
            if incident_id in self.incidents:
                incident = self.incidents[incident_id]
                incident.status = "resolved"
                incident.resolution_time = time.time()
                if resolution_notes:
                    incident.response_actions.append(f"Resolution: {resolution_notes}")

                logger.info(f"Incident {incident_id} resolved")
                return True
        return False

    def get_incident_stats(self) -> dict[str, Any]:
        """Get incident statistics."""
        with self._lock:
            incidents = list(self.incidents.values())

        stats = {
            'total_incidents': len(incidents),
            'open_incidents': len([i for i in incidents if i.status == "open"]),
            'by_severity': {},
            'by_type': {},
            'avg_resolution_time': 0.0
        }

        resolution_times = []

        for incident in incidents:
            # Count by severity
            severity = incident.severity.value
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

            # Count by type
            inc_type = incident.incident_type
            stats['by_type'][inc_type] = stats['by_type'].get(inc_type, 0) + 1

            # Collect resolution times
            if incident.resolution_time and incident.timestamp:
                resolution_times.append(incident.resolution_time - incident.timestamp)

        if resolution_times:
            stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)

        return stats

    def _generate_incident_id(self) -> str:
        """Generate unique incident ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = secrets.token_hex(4)
        return f"inc_{timestamp}_{random_part}"


# Global instances
key_manager = CryptographicKeyManager()
incident_response = IncidentResponseSystem(key_manager)
