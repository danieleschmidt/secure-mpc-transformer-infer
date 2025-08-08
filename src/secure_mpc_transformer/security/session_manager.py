"""Secure session management with advanced security features."""

import time
import secrets
import hashlib
import json
import asyncio
from typing import Dict, Optional, Any, Set, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Session data structure."""
    
    session_id: str
    user_id: Optional[str]
    created_at: float
    last_accessed: float
    expires_at: float
    source_ip: str
    user_agent: str
    permissions: Set[str]
    mfa_verified: bool
    risk_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['permissions'] = list(self.permissions)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create from dictionary."""
        data['permissions'] = set(data.get('permissions', []))
        return cls(**data)


@dataclass 
class SessionToken:
    """JWT token structure for sessions."""
    
    session_id: str
    user_id: Optional[str]
    issued_at: float
    expires_at: float
    permissions: List[str]
    risk_level: str
    fingerprint: str


class SessionSecurity:
    """Session security utilities."""
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate cryptographically secure session ID."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: str = None) -> str:
        """Hash sensitive data with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}".encode()
        hash_obj = hashlib.sha256(combined)
        return f"{salt}:{hash_obj.hexdigest()}"
    
    @staticmethod
    def verify_hashed_data(data: str, hashed: str) -> bool:
        """Verify hashed data."""
        try:
            salt, hash_value = hashed.split(':', 1)
            expected = SessionSecurity.hash_sensitive_data(data, salt)
            return expected == hashed
        except ValueError:
            return False
    
    @staticmethod
    def create_fingerprint(user_agent: str, ip: str, additional_data: str = "") -> str:
        """Create device fingerprint."""
        combined = f"{user_agent}:{ip}:{additional_data}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def encrypt_session_data(data: str, key: bytes) -> str:
        """Encrypt session data."""
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    @staticmethod
    def decrypt_session_data(encrypted_data: str, key: bytes) -> str:
        """Decrypt session data."""
        try:
            fernet = Fernet(key)
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt session data: {e}")
            return None


class SessionStore:
    """In-memory session store with persistence options."""
    
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes
        self._sessions: Dict[str, SessionData] = {}
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self._ip_sessions: Dict[str, Set[str]] = {}    # ip -> session_ids
        self._lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def store_session(self, session: SessionData) -> bool:
        """Store session data."""
        with self._lock:
            self._sessions[session.session_id] = session
            
            # Index by user ID
            if session.user_id:
                if session.user_id not in self._user_sessions:
                    self._user_sessions[session.user_id] = set()
                self._user_sessions[session.user_id].add(session.session_id)
            
            # Index by IP
            if session.source_ip not in self._ip_sessions:
                self._ip_sessions[session.source_ip] = set()
            self._ip_sessions[session.source_ip].add(session.session_id)
            
            logger.debug(f"Session {session.session_id} stored")
            return True
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, **updates) -> bool:
        """Update session data."""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.last_accessed = time.time()
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                
                # Remove from user index
                if session.user_id and session.user_id in self._user_sessions:
                    self._user_sessions[session.user_id].discard(session_id)
                    if not self._user_sessions[session.user_id]:
                        del self._user_sessions[session.user_id]
                
                # Remove from IP index
                if session.source_ip in self._ip_sessions:
                    self._ip_sessions[session.source_ip].discard(session_id)
                    if not self._ip_sessions[session.source_ip]:
                        del self._ip_sessions[session.source_ip]
                
                # Remove session
                del self._sessions[session_id]
                logger.debug(f"Session {session_id} deleted")
                return True
            return False
    
    def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user."""
        with self._lock:
            session_ids = self._user_sessions.get(user_id, set())
            return [self._sessions[sid] for sid in session_ids if sid in self._sessions]
    
    def get_ip_sessions(self, ip: str) -> List[SessionData]:
        """Get all sessions for an IP."""
        with self._lock:
            session_ids = self._ip_sessions.get(ip, set())
            return [self._sessions[sid] for sid in session_ids if sid in self._sessions]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.expires_at < current_time:
                    expired_sessions.append(session_id)
        
        # Delete expired sessions
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    self.cleanup_expired_sessions()
                    await asyncio.sleep(self._cleanup_interval)
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
        
        # Start cleanup task
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            pass
        
        if loop and loop.is_running():
            self._cleanup_task = asyncio.create_task(cleanup_loop())
        else:
            # Schedule for later when event loop is available
            threading.Timer(self._cleanup_interval, self._manual_cleanup).start()
    
    def _manual_cleanup(self):
        """Manual cleanup for non-async environments."""
        try:
            self.cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Manual session cleanup error: {e}")
        finally:
            # Schedule next cleanup
            threading.Timer(self._cleanup_interval, self._manual_cleanup).start()


class SecureSessionManager:
    """Advanced secure session management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Session configuration
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour
        self.max_sessions_per_user = self.config.get('max_sessions_per_user', 5)
        self.max_sessions_per_ip = self.config.get('max_sessions_per_ip', 10)
        self.require_mfa = self.config.get('require_mfa', True)
        
        # JWT configuration
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(64))
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        
        # Encryption key for session data
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = self.config.get('encryption_salt', b'salt_1234567890ab').ljust(16, b'\0')[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.encryption_key = key
        
        # Session store
        self.store = SessionStore(cleanup_interval=self.config.get('cleanup_interval', 300))
        
        # Security tracking
        self.failed_logins = {}
        self.suspicious_ips = set()
        self.blocked_ips = set()
        
        logger.info("Secure session manager initialized")
    
    async def create_session(self, user_id: Optional[str], source_ip: str,
                           user_agent: str, permissions: Set[str] = None,
                           metadata: Dict[str, Any] = None) -> SessionToken:
        """Create a new secure session."""
        current_time = time.time()
        
        # Check IP blocking
        if source_ip in self.blocked_ips:
            raise ValueError(f"IP {source_ip} is blocked")
        
        # Check session limits
        if user_id:
            user_sessions = self.store.get_user_sessions(user_id)
            active_sessions = [s for s in user_sessions if s.expires_at > current_time]
            
            if len(active_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest_session = min(active_sessions, key=lambda s: s.created_at)
                self.store.delete_session(oldest_session.session_id)
                logger.info(f"Removed oldest session for user {user_id}")
        
        # Check IP session limits
        ip_sessions = self.store.get_ip_sessions(source_ip)
        active_ip_sessions = [s for s in ip_sessions if s.expires_at > current_time]
        
        if len(active_ip_sessions) >= self.max_sessions_per_ip:
            logger.warning(f"IP {source_ip} has too many active sessions")
            if source_ip not in self.suspicious_ips:
                self.suspicious_ips.add(source_ip)
        
        # Create session data
        session_id = SessionSecurity.generate_session_id()
        fingerprint = SessionSecurity.create_fingerprint(user_agent, source_ip)
        
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_accessed=current_time,
            expires_at=current_time + self.session_timeout,
            source_ip=source_ip,
            user_agent=user_agent,
            permissions=permissions or set(),
            mfa_verified=not self.require_mfa,  # Will need MFA verification
            risk_score=self._calculate_risk_score(source_ip, user_agent),
            metadata=metadata or {}
        )
        
        # Store session
        self.store.store_session(session_data)
        
        # Create JWT token
        token = SessionToken(
            session_id=session_id,
            user_id=user_id,
            issued_at=current_time,
            expires_at=current_time + self.session_timeout,
            permissions=list(permissions or []),
            risk_level=self._get_risk_level(session_data.risk_score),
            fingerprint=fingerprint
        )
        
        logger.info(f"Created session {session_id} for user {user_id or 'anonymous'}")
        return token
    
    def encode_token(self, token: SessionToken) -> str:
        """Encode session token as JWT."""
        payload = {
            'session_id': token.session_id,
            'user_id': token.user_id,
            'iat': int(token.issued_at),
            'exp': int(token.expires_at),
            'permissions': token.permissions,
            'risk_level': token.risk_level,
            'fingerprint': token.fingerprint
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def decode_token(self, token: str) -> Optional[SessionToken]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            return SessionToken(
                session_id=payload['session_id'],
                user_id=payload.get('user_id'),
                issued_at=payload['iat'],
                expires_at=payload['exp'],
                permissions=payload.get('permissions', []),
                risk_level=payload.get('risk_level', 'medium'),
                fingerprint=payload.get('fingerprint', '')
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def validate_session(self, token: str, source_ip: str,
                             user_agent: str) -> Optional[SessionData]:
        """Validate session token and return session data."""
        # Decode token
        session_token = self.decode_token(token)
        if not session_token:
            return None
        
        # Get session data
        session_data = self.store.get_session(session_token.session_id)
        if not session_data:
            logger.warning(f"Session {session_token.session_id} not found")
            return None
        
        # Check expiration
        current_time = time.time()
        if session_data.expires_at < current_time:
            self.store.delete_session(session_token.session_id)
            logger.info(f"Session {session_token.session_id} expired")
            return None
        
        # Validate fingerprint
        expected_fingerprint = SessionSecurity.create_fingerprint(
            session_data.user_agent, session_data.source_ip
        )
        if session_token.fingerprint != expected_fingerprint:
            logger.warning(f"Session {session_token.session_id} fingerprint mismatch")
            # Don't immediately reject - could be proxy or NAT
            session_data.risk_score += 0.2
        
        # Check IP consistency
        if session_data.source_ip != source_ip:
            logger.warning(f"Session {session_token.session_id} IP changed: {session_data.source_ip} -> {source_ip}")
            session_data.risk_score += 0.3
        
        # Update last accessed time
        self.store.update_session(
            session_token.session_id,
            last_accessed=current_time,
            risk_score=session_data.risk_score
        )
        
        # Check if risk score is too high
        if session_data.risk_score > 0.8:
            logger.warning(f"Session {session_token.session_id} has high risk score: {session_data.risk_score}")
            # Could trigger additional authentication
        
        return session_data
    
    async def refresh_session(self, token: str) -> Optional[str]:
        """Refresh session and return new token."""
        session_token = self.decode_token(token)
        if not session_token:
            return None
        
        session_data = self.store.get_session(session_token.session_id)
        if not session_data:
            return None
        
        current_time = time.time()
        
        # Check if session is still valid and not too close to expiration
        if session_data.expires_at < current_time + 300:  # 5 minutes before expiry
            # Extend session
            new_expires_at = current_time + self.session_timeout
            self.store.update_session(
                session_token.session_id,
                expires_at=new_expires_at,
                last_accessed=current_time
            )
            
            # Create new token
            new_token = SessionToken(
                session_id=session_token.session_id,
                user_id=session_token.user_id,
                issued_at=current_time,
                expires_at=new_expires_at,
                permissions=session_token.permissions,
                risk_level=session_token.risk_level,
                fingerprint=session_token.fingerprint
            )
            
            return self.encode_token(new_token)
        
        return None
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate a specific session."""
        result = self.store.delete_session(session_id)
        if result:
            logger.info(f"Session {session_id} terminated")
        return result
    
    async def terminate_user_sessions(self, user_id: str) -> int:
        """Terminate all sessions for a user."""
        user_sessions = self.store.get_user_sessions(user_id)
        count = 0
        
        for session in user_sessions:
            if self.store.delete_session(session.session_id):
                count += 1
        
        logger.info(f"Terminated {count} sessions for user {user_id}")
        return count
    
    async def terminate_ip_sessions(self, ip: str) -> int:
        """Terminate all sessions for an IP address."""
        ip_sessions = self.store.get_ip_sessions(ip)
        count = 0
        
        for session in ip_sessions:
            if self.store.delete_session(session.session_id):
                count += 1
        
        logger.info(f"Terminated {count} sessions for IP {ip}")
        return count
    
    def block_ip(self, ip: str, reason: str = "Security violation"):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        # Terminate existing sessions
        asyncio.create_task(self.terminate_ip_sessions(ip))
        logger.warning(f"Blocked IP {ip}: {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
        self.suspicious_ips.discard(ip)
        logger.info(f"Unblocked IP {ip}")
    
    def _calculate_risk_score(self, source_ip: str, user_agent: str) -> float:
        """Calculate initial risk score for session."""
        score = 0.0
        
        # IP-based risk factors
        if source_ip in self.suspicious_ips:
            score += 0.3
        
        if source_ip in self.failed_logins:
            failed_count = len(self.failed_logins[source_ip])
            score += min(0.4, failed_count * 0.1)
        
        # User agent risk factors
        if not user_agent or len(user_agent) < 10:
            score += 0.2
        
        # Check for common bot patterns
        bot_patterns = ['bot', 'crawler', 'spider', 'scraper']
        if any(pattern in user_agent.lower() for pattern in bot_patterns):
            score += 0.3
        
        return min(1.0, score)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level."""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def record_failed_login(self, ip: str):
        """Record failed login attempt."""
        current_time = time.time()
        
        if ip not in self.failed_logins:
            self.failed_logins[ip] = []
        
        self.failed_logins[ip].append(current_time)
        
        # Clean old failed attempts (older than 1 hour)
        cutoff = current_time - 3600
        self.failed_logins[ip] = [t for t in self.failed_logins[ip] if t > cutoff]
        
        # Check if IP should be blocked
        if len(self.failed_logins[ip]) >= 10:  # 10 failed attempts in 1 hour
            self.block_ip(ip, "Too many failed login attempts")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        current_time = time.time()
        all_sessions = list(self.store._sessions.values())
        
        active_sessions = [s for s in all_sessions if s.expires_at > current_time]
        expired_sessions = [s for s in all_sessions if s.expires_at <= current_time]
        
        # Risk distribution
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for session in active_sessions:
            risk_level = self._get_risk_level(session.risk_score)
            risk_distribution[risk_level] += 1
        
        return {
            'total_sessions': len(all_sessions),
            'active_sessions': len(active_sessions),
            'expired_sessions': len(expired_sessions),
            'unique_users': len(set(s.user_id for s in active_sessions if s.user_id)),
            'unique_ips': len(set(s.source_ip for s in active_sessions)),
            'risk_distribution': risk_distribution,
            'blocked_ips': len(self.blocked_ips),
            'suspicious_ips': len(self.suspicious_ips),
            'failed_login_ips': len(self.failed_logins)
        }