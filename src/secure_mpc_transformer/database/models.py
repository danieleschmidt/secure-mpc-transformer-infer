"""Database models for secure MPC transformer."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SessionStatus(Enum):
    """Computation session status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProtocolType(Enum):
    """MPC protocol types."""
    SEMI_HONEST_3PC = "semi_honest_3pc"
    MALICIOUS_3PC = "malicious_3pc"
    ABY3 = "aby3"
    REPLICATED_3PC = "replicated_3pc"
    FANTASTIC_FOUR = "fantastic_four"


class AuditEventType(Enum):
    """Types of audit events."""
    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    PROTOCOL_INITIALIZED = "protocol_initialized"
    VALUE_SHARED = "value_shared"
    VALUE_RECONSTRUCTED = "value_reconstructed"
    SECURE_OPERATION = "secure_operation"
    SECURITY_VIOLATION = "security_violation"
    AUTHENTICATION_FAILED = "authentication_failed"
    MAC_VERIFICATION_FAILED = "mac_verification_failed"
    PROOF_GENERATION = "proof_generation"
    PROOF_VERIFICATION = "proof_verification"
    NETWORK_ERROR = "network_error"
    PERFORMANCE_WARNING = "performance_warning"


@dataclass
class ComputationSession:
    """Represents a secure computation session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    protocol_type: ProtocolType = ProtocolType.SEMI_HONEST_3PC
    num_parties: int = 3
    party_ids: list[int] = field(default_factory=list)
    status: SessionStatus = SessionStatus.PENDING

    # Input/Output
    input_text: str = ""
    input_tokens: list[int] | None = None
    sequence_length: int = 0

    # Configuration
    security_config: dict[str, Any] = field(default_factory=dict)
    performance_config: dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Performance metrics
    latency_ms: float | None = None
    gpu_utilization: float | None = None
    memory_usage_mb: float | None = None
    communication_rounds: int | None = None
    bytes_transmitted: int | None = None

    # Error handling
    error_message: str | None = None
    error_traceback: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)

        # Convert enums to strings
        data['protocol_type'] = self.protocol_type.value
        data['status'] = self.status.value

        # Convert datetime objects to ISO strings
        for field_name in ['created_at', 'started_at', 'completed_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComputationSession":
        """Create instance from dictionary."""
        # Convert string enums back to enum objects
        if 'protocol_type' in data:
            data['protocol_type'] = ProtocolType(data['protocol_type'])
        if 'status' in data:
            data['status'] = SessionStatus(data['status'])

        # Convert ISO strings back to datetime objects
        for field_name in ['created_at', 'started_at', 'completed_at']:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)

    def start(self) -> None:
        """Mark session as started."""
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self, latency_ms: float, **metrics) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.latency_ms = latency_ms

        # Update performance metrics
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def fail(self, error_message: str, error_traceback: str | None = None) -> None:
        """Mark session as failed."""
        self.status = SessionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        self.error_traceback = error_traceback

    def get_duration_ms(self) -> float | None:
        """Get session duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


@dataclass
class InferenceResult:
    """Stores the result of a secure inference computation."""

    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""

    # Output data
    logits: list[float] | None = None
    predicted_tokens: list[int] | None = None
    output_text: str = ""
    confidence_scores: list[float] | None = None

    # Privacy metrics
    privacy_epsilon: float | None = None
    privacy_delta: float | None = None
    privacy_spent: float | None = None

    # Computational metrics
    total_operations: int = 0
    arithmetic_operations: int = 0
    boolean_operations: int = 0
    conversions: int = 0

    # Performance metrics
    computation_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0

    # Security metrics
    mac_verifications: int = 0
    proof_generations: int = 0
    proof_verifications: int = 0
    security_violations: int = 0

    # Storage
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceResult":
        """Create instance from dictionary."""
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

    def add_operation_stats(self, operation_type: str, count: int = 1) -> None:
        """Add operation statistics."""
        self.total_operations += count

        if operation_type in ['add', 'multiply', 'matmul']:
            self.arithmetic_operations += count
        elif operation_type in ['and', 'or', 'xor', 'comparison']:
            self.boolean_operations += count
        elif operation_type in ['a2b', 'b2a', 'share_conversion']:
            self.conversions += count

    def add_security_event(self, event_type: str, count: int = 1) -> None:
        """Add security event statistics."""
        if event_type == 'mac_verification':
            self.mac_verifications += count
        elif event_type == 'proof_generation':
            self.proof_generations += count
        elif event_type == 'proof_verification':
            self.proof_verifications += count
        elif event_type == 'security_violation':
            self.security_violations += count


@dataclass
class AuditLog:
    """Audit log entry for security monitoring."""

    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    party_id: int = 0

    # Event details
    event_type: AuditEventType = AuditEventType.SECURE_OPERATION
    event_description: str = ""
    event_data: dict[str, Any] = field(default_factory=dict)

    # Context
    protocol_type: ProtocolType | None = None
    operation_type: str | None = None
    security_level: int | None = None

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Risk assessment
    risk_level: str = "low"  # low, medium, high, critical
    requires_investigation: bool = False

    # Source information
    source_ip: str | None = None
    user_agent: str | None = None
    request_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)

        # Convert enums to strings
        data['event_type'] = self.event_type.value
        if self.protocol_type:
            data['protocol_type'] = self.protocol_type.value

        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditLog":
        """Create instance from dictionary."""
        # Convert string enums back to enum objects
        if 'event_type' in data:
            data['event_type'] = AuditEventType(data['event_type'])
        if 'protocol_type' in data and data['protocol_type']:
            data['protocol_type'] = ProtocolType(data['protocol_type'])

        # Convert ISO string back to datetime
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        return cls(**data)

    @classmethod
    def create_security_event(cls, event_type: AuditEventType, description: str,
                            session_id: str | None = None, party_id: int = 0,
                            risk_level: str = "medium", **kwargs) -> "AuditLog":
        """Create security audit log entry."""
        return cls(
            session_id=session_id,
            party_id=party_id,
            event_type=event_type,
            event_description=description,
            risk_level=risk_level,
            requires_investigation=risk_level in ['high', 'critical'],
            **kwargs
        )

    @classmethod
    def create_performance_event(cls, description: str, session_id: str | None = None,
                               performance_data: dict[str, Any] | None = None) -> "AuditLog":
        """Create performance audit log entry."""
        return cls(
            session_id=session_id,
            event_type=AuditEventType.PERFORMANCE_WARNING,
            event_description=description,
            event_data=performance_data or {},
            risk_level="low"
        )


# SQL Schema definitions for PostgreSQL/SQLite
CREATE_TABLES_SQL = {
    'computation_sessions': """
        CREATE TABLE IF NOT EXISTS computation_sessions (
            session_id VARCHAR(36) PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            protocol_type VARCHAR(50) NOT NULL,
            num_parties INTEGER NOT NULL,
            party_ids TEXT,  -- JSON array
            status VARCHAR(20) NOT NULL,
            input_text TEXT,
            input_tokens TEXT,  -- JSON array
            sequence_length INTEGER,
            security_config TEXT,  -- JSON
            performance_config TEXT,  -- JSON
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            latency_ms REAL,
            gpu_utilization REAL,
            memory_usage_mb REAL,
            communication_rounds INTEGER,
            bytes_transmitted BIGINT,
            error_message TEXT,
            error_traceback TEXT,
            metadata TEXT  -- JSON
        )
    """,

    'inference_results': """
        CREATE TABLE IF NOT EXISTS inference_results (
            result_id VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL,
            logits TEXT,  -- JSON array
            predicted_tokens TEXT,  -- JSON array
            output_text TEXT,
            confidence_scores TEXT,  -- JSON array
            privacy_epsilon REAL,
            privacy_delta REAL,
            privacy_spent REAL,
            total_operations INTEGER DEFAULT 0,
            arithmetic_operations INTEGER DEFAULT 0,
            boolean_operations INTEGER DEFAULT 0,
            conversions INTEGER DEFAULT 0,
            computation_time_ms REAL DEFAULT 0,
            communication_time_ms REAL DEFAULT 0,
            preprocessing_time_ms REAL DEFAULT 0,
            mac_verifications INTEGER DEFAULT 0,
            proof_generations INTEGER DEFAULT 0,
            proof_verifications INTEGER DEFAULT 0,
            security_violations INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            metadata TEXT,  -- JSON
            FOREIGN KEY (session_id) REFERENCES computation_sessions(session_id)
        )
    """,

    'audit_logs': """
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(36),
            party_id INTEGER NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            event_description TEXT NOT NULL,
            event_data TEXT,  -- JSON
            protocol_type VARCHAR(50),
            operation_type VARCHAR(50),
            security_level INTEGER,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            requires_investigation BOOLEAN DEFAULT FALSE,
            source_ip VARCHAR(45),
            user_agent TEXT,
            request_id VARCHAR(36),
            metadata TEXT,  -- JSON
            FOREIGN KEY (session_id) REFERENCES computation_sessions(session_id)
        )
    """
}

# Index definitions for better query performance
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON computation_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON computation_sessions(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_protocol ON computation_sessions(protocol_type)",
    "CREATE INDEX IF NOT EXISTS idx_results_session_id ON inference_results(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_results_created_at ON inference_results(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_session_id ON audit_logs(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_logs(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_audit_risk_level ON audit_logs(risk_level)",
    "CREATE INDEX IF NOT EXISTS idx_audit_party_id ON audit_logs(party_id)"
]
