"""
Base classes and common structures for compliance frameworks.

This module provides the foundational classes and data structures
used across different compliance frameworks (GDPR, CCPA, PDPA).
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

class DataSubjectRightType(Enum):
    """Common data subject rights across compliance frameworks."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    RESTRICTION = "restriction"
    PORTABILITY = "portability"
    OBJECTION = "objection"
    OPT_OUT = "opt_out"  # CCPA-specific
    KNOW = "know"  # CCPA-specific

class DataCategory(Enum):
    """General data categories for classification."""
    PERSONAL_IDENTIFIERS = "personal_identifiers"
    CONTACT_INFORMATION = "contact_information"
    DEMOGRAPHIC_DATA = "demographic_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    BEHAVIORAL_DATA = "behavioral_data"
    TECHNICAL_DATA = "technical_data"
    COMMUNICATION_DATA = "communication_data"

class ProcessingPurpose(Enum):
    """Common data processing purposes."""
    SERVICE_PROVISION = "service_provision"
    AUTHENTICATION = "authentication"
    SECURITY_MONITORING = "security_monitoring"
    FRAUD_PREVENTION = "fraud_prevention"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    RESEARCH = "research"
    LEGAL_COMPLIANCE = "legal_compliance"
    BUSINESS_OPERATIONS = "business_operations"

@dataclass
class DataSubjectRequest:
    """Data subject request for exercising privacy rights."""
    request_id: str
    data_subject_id: str
    request_type: str
    status: str = "pending"
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.details is None:
            self.details = {}
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

@dataclass
class DataClassification:
    """Data classification with sensitivity and regulatory scope."""
    classification_id: str
    data_type: str
    sensitivity_level: str  # "public", "internal", "confidential", "restricted"
    regulatory_scope: List[str]  # ["gdpr", "ccpa", "pdpa"]
    retention_period: Optional[timedelta] = None
    encryption_required: bool = False
    access_controls: List[str] = None
    
    def __post_init__(self):
        if self.access_controls is None:
            self.access_controls = []
        if not self.classification_id:
            self.classification_id = str(uuid.uuid4())

@dataclass
class ConsentRecord:
    """Base consent record structure."""
    consent_id: str
    data_subject_id: str
    granted: bool
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    consent_method: str = "explicit"
    consent_version: str = "1.0"
    
    def __post_init__(self):
        if not self.consent_id:
            self.consent_id = str(uuid.uuid4())

@dataclass
class AuditLog:
    """Audit log entry for compliance tracking."""
    log_id: str
    timestamp: datetime
    event_type: str
    data_subject_id: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = None
    compliance_framework: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if not self.log_id:
            self.log_id = str(uuid.uuid4())

@dataclass
class PrivacyNotice:
    """Privacy notice/policy information."""
    notice_id: str
    version: str
    effective_date: datetime
    jurisdiction: str
    language: str
    content: Dict[str, Any]
    last_updated: datetime
    
    def __post_init__(self):
        if not self.notice_id:
            self.notice_id = str(uuid.uuid4())

@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""
    policy_id: str
    data_category: str
    retention_period: timedelta
    retention_basis: str  # Legal basis for retention
    disposal_method: str  # "deletion", "anonymization", "pseudonymization"
    exceptions: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if not self.policy_id:
            self.policy_id = str(uuid.uuid4())

class ComplianceFramework(ABC):
    """Abstract base class for compliance frameworks."""
    
    def __init__(self, framework_name: str, config: Dict[str, Any]):
        """
        Initialize compliance framework.
        
        Args:
            framework_name: Name of the compliance framework
            config: Framework configuration
        """
        self.framework_name = framework_name
        self.config = config
        self.audit_logs: List[AuditLog] = []
        self.data_classifications: Dict[str, DataClassification] = {}
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self.privacy_notices: Dict[str, PrivacyNotice] = {}
    
    @abstractmethod
    def process_data_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """
        Process a data subject request.
        
        Args:
            request: Data subject request to process
            
        Returns:
            Processing result
        """
        pass
    
    @abstractmethod
    def validate_processing(self, 
                           data_category: str, 
                           purpose: str, 
                           legal_basis: str,
                           **kwargs) -> bool:
        """
        Validate if data processing is compliant.
        
        Args:
            data_category: Category of data being processed
            purpose: Purpose of processing
            legal_basis: Legal basis for processing
            **kwargs: Additional validation parameters
            
        Returns:
            True if processing is compliant
        """
        pass
    
    @abstractmethod
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Returns:
            Compliance report dictionary
        """
        pass
    
    def add_data_classification(self, classification: DataClassification) -> None:
        """Add a data classification."""
        self.data_classifications[classification.classification_id] = classification
        self.log_audit_event("data_classification_added", 
                           details={"classification_id": classification.classification_id})
    
    def add_retention_policy(self, policy: DataRetentionPolicy) -> None:
        """Add a data retention policy."""
        self.retention_policies[policy.policy_id] = policy
        self.log_audit_event("retention_policy_added",
                           details={"policy_id": policy.policy_id})
    
    def add_privacy_notice(self, notice: PrivacyNotice) -> None:
        """Add a privacy notice."""
        self.privacy_notices[notice.notice_id] = notice
        self.log_audit_event("privacy_notice_added",
                           details={"notice_id": notice.notice_id})
    
    def log_audit_event(self, 
                       event_type: str,
                       data_subject_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            data_subject_id: Data subject identifier (if applicable)
            user_id: User identifier (if applicable)
            details: Additional event details
            
        Returns:
            Audit log ID
        """
        audit_log = AuditLog(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            data_subject_id=data_subject_id,
            user_id=user_id,
            details=details or {},
            compliance_framework=self.framework_name
        )
        
        self.audit_logs.append(audit_log)
        return audit_log.log_id
    
    def get_audit_logs(self, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      event_type: Optional[str] = None,
                      data_subject_id: Optional[str] = None) -> List[AuditLog]:
        """
        Retrieve audit logs with filtering.
        
        Args:
            start_date: Filter logs from this date
            end_date: Filter logs until this date
            event_type: Filter by event type
            data_subject_id: Filter by data subject
            
        Returns:
            List of matching audit logs
        """
        filtered_logs = self.audit_logs
        
        if start_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
        
        if end_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
        
        if event_type:
            filtered_logs = [log for log in filtered_logs if log.event_type == event_type]
        
        if data_subject_id:
            filtered_logs = [log for log in filtered_logs if log.data_subject_id == data_subject_id]
        
        return filtered_logs
    
    def get_retention_policy(self, data_category: str) -> Optional[DataRetentionPolicy]:
        """
        Get retention policy for a data category.
        
        Args:
            data_category: Data category to get policy for
            
        Returns:
            Retention policy if found
        """
        for policy in self.retention_policies.values():
            if policy.data_category == data_category:
                return policy
        return None
    
    def is_data_expired(self, data_created_at: datetime, data_category: str) -> bool:
        """
        Check if data has expired based on retention policy.
        
        Args:
            data_created_at: When the data was created
            data_category: Category of the data
            
        Returns:
            True if data has expired
        """
        policy = self.get_retention_policy(data_category)
        if not policy:
            return False
        
        expiry_date = data_created_at + policy.retention_period
        return datetime.utcnow() > expiry_date
    
    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get information about the compliance framework.
        
        Returns:
            Framework information dictionary
        """
        return {
            "name": self.framework_name,
            "config": self.config,
            "total_audit_logs": len(self.audit_logs),
            "total_classifications": len(self.data_classifications),
            "total_retention_policies": len(self.retention_policies),
            "total_privacy_notices": len(self.privacy_notices)
        }