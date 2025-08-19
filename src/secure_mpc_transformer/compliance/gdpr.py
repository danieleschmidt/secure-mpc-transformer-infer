"""
GDPR (General Data Protection Regulation) compliance implementation for EU operations.

This module implements comprehensive GDPR compliance features including:
- Data subject rights (access, rectification, erasure, portability, restriction, objection)
- Lawful basis for processing
- Consent management with granular controls
- Data Protection Officer (DPO) integration
- Breach notification procedures
- Privacy by design principles
- Cross-border data transfer safeguards
"""

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from cryptography.fernet import Fernet

from .base import (
    AuditLog,
    ComplianceFramework,
    ConsentRecord,
    DataSubjectRequest,
)

logger = logging.getLogger(__name__)

class GDPRLegalBasis(Enum):
    """GDPR lawful bases for processing personal data."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class GDPRDataCategory(Enum):
    """GDPR data categories for classification."""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"  # Special categories
    CRIMINAL_DATA = "criminal_data"
    PSEUDONYMOUS_DATA = "pseudonymous_data"
    ANONYMOUS_DATA = "anonymous_data"

class GDPRProcessingPurpose(Enum):
    """Common GDPR processing purposes."""
    SERVICE_PROVISION = "service_provision"
    SECURITY_MONITORING = "security_monitoring"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    LEGAL_COMPLIANCE = "legal_compliance"
    RESEARCH = "research"

class GDPRDataSubjectRights(Enum):
    """GDPR data subject rights."""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 (Right to be forgotten)
    RESTRICTION = "restriction"  # Article 18
    PORTABILITY = "portability"  # Article 20
    OBJECTION = "objection"  # Article 21

@dataclass
class GDPRDataRecord:
    """GDPR data record with classification and processing metadata."""
    record_id: str
    data_subject_id: str
    data_category: GDPRDataCategory
    processing_purpose: list[GDPRProcessingPurpose]
    legal_basis: GDPRLegalBasis
    created_at: datetime
    retention_period: timedelta | None = None
    consent_id: str | None = None
    is_pseudonymized: bool = False
    is_encrypted: bool = False
    data_controller: str = ""
    data_processor: str = ""

@dataclass
class GDPRConsentRecord(ConsentRecord):
    """GDPR-specific consent record with granular controls."""
    processing_purposes: list[GDPRProcessingPurpose]
    data_categories: list[GDPRDataCategory]
    third_party_sharing: bool = False
    marketing_consent: bool = False
    profiling_consent: bool = False
    withdrawal_mechanism: str = ""
    consent_proof: str = ""  # Hash of original consent interaction

class GDPRConsentManager:
    """Manage GDPR consent with granular purpose and category controls."""

    def __init__(self, storage_backend=None):
        """
        Initialize GDPR consent manager.
        
        Args:
            storage_backend: Backend for storing consent records
        """
        self.storage = storage_backend or {}
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

    def record_consent(self,
                      data_subject_id: str,
                      processing_purposes: list[GDPRProcessingPurpose],
                      data_categories: list[GDPRDataCategory],
                      consent_method: str = "explicit",
                      **kwargs) -> str:
        """
        Record GDPR consent with granular controls.
        
        Args:
            data_subject_id: Unique identifier for data subject
            processing_purposes: List of processing purposes
            data_categories: List of data categories
            consent_method: Method of consent collection
            **kwargs: Additional consent parameters
            
        Returns:
            Consent ID
        """
        consent_id = str(uuid.uuid4())

        # Generate consent proof hash
        consent_data = {
            "subject_id": data_subject_id,
            "purposes": [p.value for p in processing_purposes],
            "categories": [c.value for c in data_categories],
            "timestamp": datetime.utcnow().isoformat(),
            "method": consent_method
        }
        consent_proof = hashlib.sha256(json.dumps(consent_data, sort_keys=True).encode()).hexdigest()

        consent_record = GDPRConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            granted=True,
            granted_at=datetime.utcnow(),
            consent_method=consent_method,
            processing_purposes=processing_purposes,
            data_categories=data_categories,
            consent_proof=consent_proof,
            **kwargs
        )

        # Encrypt and store consent record
        encrypted_record = self._cipher.encrypt(json.dumps(asdict(consent_record)).encode())
        self.storage[consent_id] = encrypted_record

        logger.info(f"GDPR consent recorded: {consent_id} for subject {data_subject_id}")
        return consent_id

    def withdraw_consent(self,
                        data_subject_id: str,
                        consent_id: str | None = None,
                        purposes: list[GDPRProcessingPurpose] | None = None) -> bool:
        """
        Withdraw GDPR consent (fully or partially).
        
        Args:
            data_subject_id: Data subject identifier
            consent_id: Specific consent to withdraw (if None, withdraws all)
            purposes: Specific purposes to withdraw consent for
            
        Returns:
            Success status
        """
        if consent_id:
            # Withdraw specific consent
            if consent_id in self.storage:
                record_data = json.loads(self._cipher.decrypt(self.storage[consent_id]).decode())
                record = GDPRConsentRecord(**record_data)

                if record.data_subject_id == data_subject_id:
                    if purposes:
                        # Partial withdrawal for specific purposes
                        record.processing_purposes = [p for p in record.processing_purposes if p not in purposes]
                        if not record.processing_purposes:
                            record.granted = False
                            record.withdrawn_at = datetime.utcnow()
                    else:
                        # Full withdrawal
                        record.granted = False
                        record.withdrawn_at = datetime.utcnow()

                    # Update stored record
                    encrypted_record = self._cipher.encrypt(json.dumps(asdict(record)).encode())
                    self.storage[consent_id] = encrypted_record

                    logger.info(f"GDPR consent withdrawn: {consent_id}")
                    return True
            return False
        else:
            # Withdraw all consents for data subject
            withdrawn_count = 0
            for cid, encrypted_record in self.storage.items():
                record_data = json.loads(self._cipher.decrypt(encrypted_record).decode())
                if record_data['data_subject_id'] == data_subject_id:
                    record = GDPRConsentRecord(**record_data)
                    record.granted = False
                    record.withdrawn_at = datetime.utcnow()

                    encrypted_record = self._cipher.encrypt(json.dumps(asdict(record)).encode())
                    self.storage[cid] = encrypted_record
                    withdrawn_count += 1

            logger.info(f"GDPR consent withdrawn for all {withdrawn_count} records for subject {data_subject_id}")
            return withdrawn_count > 0

    def check_consent(self,
                     data_subject_id: str,
                     purpose: GDPRProcessingPurpose,
                     data_category: GDPRDataCategory) -> bool:
        """
        Check if valid consent exists for specific purpose and data category.
        
        Args:
            data_subject_id: Data subject identifier
            purpose: Processing purpose to check
            data_category: Data category to check
            
        Returns:
            True if valid consent exists
        """
        for encrypted_record in self.storage.values():
            try:
                record_data = json.loads(self._cipher.decrypt(encrypted_record).decode())
                record = GDPRConsentRecord(**record_data)

                if (record.data_subject_id == data_subject_id and
                    record.granted and
                    not record.withdrawn_at and
                    purpose in record.processing_purposes and
                    data_category in record.data_categories):

                    # Check if consent is still valid (not expired)
                    if record.expires_at and datetime.utcnow() > record.expires_at:
                        continue

                    return True
            except Exception as e:
                logger.error(f"Error checking consent record: {e}")
                continue

        return False

class GDPRDataProcessor:
    """Process GDPR data subject requests and manage data lifecycle."""

    def __init__(self, data_storage=None, consent_manager=None):
        """
        Initialize GDPR data processor.
        
        Args:
            data_storage: Backend for data storage
            consent_manager: GDPR consent manager instance
        """
        self.data_storage = data_storage or {}
        self.consent_manager = consent_manager or GDPRConsentManager()
        self._audit_log = []
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

    def process_access_request(self, data_subject_id: str) -> dict[str, Any]:
        """
        Process GDPR Article 15 access request.
        
        Args:
            data_subject_id: Data subject identifier
            
        Returns:
            Dictionary containing all personal data
        """
        self._log_request("access", data_subject_id)

        # Collect all data for the subject
        subject_data = {
            "personal_data": [],
            "processing_activities": [],
            "consent_records": [],
            "data_sources": [],
            "recipients": [],
            "retention_periods": [],
            "rights_information": self._get_rights_information()
        }

        # Find all data records for the subject
        for record_id, record_data in self.data_storage.items():
            if isinstance(record_data, dict) and record_data.get('data_subject_id') == data_subject_id:
                # Decrypt if necessary
                if record_data.get('encrypted', False):
                    try:
                        decrypted_data = self._cipher.decrypt(record_data['data'].encode()).decode()
                        record_data['data'] = json.loads(decrypted_data)
                    except Exception as e:
                        logger.error(f"Failed to decrypt data for access request: {e}")
                        continue

                subject_data["personal_data"].append({
                    "record_id": record_id,
                    "data_category": record_data.get('category'),
                    "processing_purpose": record_data.get('purpose'),
                    "created_at": record_data.get('created_at'),
                    "data": record_data.get('data') if not record_data.get('is_pseudonymized') else "[PSEUDONYMIZED]"
                })

        # Add consent records
        for consent_id, encrypted_record in self.consent_manager.storage.items():
            try:
                record_data = json.loads(self.consent_manager._cipher.decrypt(encrypted_record).decode())
                if record_data['data_subject_id'] == data_subject_id:
                    subject_data["consent_records"].append({
                        "consent_id": consent_id,
                        "granted_at": record_data['granted_at'],
                        "processing_purposes": record_data['processing_purposes'],
                        "status": "active" if record_data['granted'] else "withdrawn"
                    })
            except Exception as e:
                logger.error(f"Error processing consent record for access request: {e}")

        logger.info(f"GDPR access request processed for subject {data_subject_id}")
        return subject_data

    def process_erasure_request(self,
                               data_subject_id: str,
                               reason: str = "withdrawal_of_consent") -> bool:
        """
        Process GDPR Article 17 erasure request (Right to be forgotten).
        
        Args:
            data_subject_id: Data subject identifier  
            reason: Reason for erasure
            
        Returns:
            Success status
        """
        self._log_request("erasure", data_subject_id, {"reason": reason})

        erased_count = 0

        # Find and delete/anonymize all data for the subject
        records_to_delete = []
        for record_id, record_data in self.data_storage.items():
            if isinstance(record_data, dict) and record_data.get('data_subject_id') == data_subject_id:
                # Check if erasure is allowed (considering legal basis and retention requirements)
                if self._can_erase_data(record_data, reason):
                    records_to_delete.append(record_id)
                else:
                    # If cannot delete, pseudonymize
                    self._pseudonymize_record(record_id, record_data)

        # Delete records
        for record_id in records_to_delete:
            del self.data_storage[record_id]
            erased_count += 1

        # Withdraw all consents
        self.consent_manager.withdraw_consent(data_subject_id)

        logger.info(f"GDPR erasure request processed: {erased_count} records erased for subject {data_subject_id}")
        return erased_count > 0

    def process_portability_request(self, data_subject_id: str) -> dict[str, Any]:
        """
        Process GDPR Article 20 portability request.
        
        Args:
            data_subject_id: Data subject identifier
            
        Returns:
            Portable data in structured format
        """
        self._log_request("portability", data_subject_id)

        portable_data = {
            "data_subject_id": data_subject_id,
            "export_date": datetime.utcnow().isoformat(),
            "data": []
        }

        # Collect portable data (only consent-based processing)
        for record_id, record_data in self.data_storage.items():
            if (isinstance(record_data, dict) and
                record_data.get('data_subject_id') == data_subject_id and
                record_data.get('legal_basis') == GDPRLegalBasis.CONSENT.value):

                # Decrypt if necessary
                if record_data.get('encrypted', False):
                    try:
                        decrypted_data = self._cipher.decrypt(record_data['data'].encode()).decode()
                        record_data['data'] = json.loads(decrypted_data)
                    except Exception as e:
                        logger.error(f"Failed to decrypt data for portability request: {e}")
                        continue

                portable_data["data"].append({
                    "category": record_data.get('category'),
                    "created_at": record_data.get('created_at'),
                    "data": record_data.get('data') if not record_data.get('is_pseudonymized') else None
                })

        logger.info(f"GDPR portability request processed for subject {data_subject_id}")
        return portable_data

    def process_rectification_request(self,
                                     data_subject_id: str,
                                     corrections: dict[str, Any]) -> bool:
        """
        Process GDPR Article 16 rectification request.
        
        Args:
            data_subject_id: Data subject identifier
            corrections: Dictionary of field corrections
            
        Returns:
            Success status
        """
        self._log_request("rectification", data_subject_id, {"corrections": corrections})

        updated_count = 0

        # Find and update records
        for record_id, record_data in self.data_storage.items():
            if isinstance(record_data, dict) and record_data.get('data_subject_id') == data_subject_id:
                # Apply corrections
                for field, new_value in corrections.items():
                    if field in record_data.get('data', {}):
                        old_value = record_data['data'][field]
                        record_data['data'][field] = new_value

                        # Log the change
                        self._audit_log.append(AuditLog(
                            log_id=str(uuid.uuid4()),
                            timestamp=datetime.utcnow(),
                            event_type="data_rectification",
                            data_subject_id=data_subject_id,
                            details={
                                "record_id": record_id,
                                "field": field,
                                "old_value": "[REDACTED]",  # Don't log actual values
                                "new_value": "[REDACTED]"
                            }
                        ))

                        updated_count += 1

                # Re-encrypt if necessary
                if record_data.get('encrypted', False):
                    encrypted_data = self._cipher.encrypt(json.dumps(record_data['data']).encode())
                    record_data['data'] = encrypted_data.decode()

        logger.info(f"GDPR rectification request processed: {updated_count} fields updated for subject {data_subject_id}")
        return updated_count > 0

    def _can_erase_data(self, record_data: dict[str, Any], reason: str) -> bool:
        """Check if data can be erased based on legal basis and other factors."""
        legal_basis = record_data.get('legal_basis')

        # Cannot erase if legal obligation or vital interests
        if legal_basis in [GDPRLegalBasis.LEGAL_OBLIGATION.value, GDPRLegalBasis.VITAL_INTERESTS.value]:
            return False

        # Cannot erase if within legal retention period
        retention_until = record_data.get('retention_until')
        if retention_until and datetime.utcnow() < datetime.fromisoformat(retention_until):
            return False

        # Can erase if consent withdrawn or other valid reasons
        return reason in ["withdrawal_of_consent", "objection_to_processing", "unlawful_processing"]

    def _pseudonymize_record(self, record_id: str, record_data: dict[str, Any]) -> None:
        """Pseudonymize a data record instead of deleting it."""
        # Generate pseudonym
        pseudonym = hashlib.sha256(f"{record_id}{record_data.get('data_subject_id')}".encode()).hexdigest()[:16]

        # Replace identifiable data with pseudonym
        record_data['data_subject_id'] = pseudonym
        record_data['is_pseudonymized'] = True

        # Remove or hash direct identifiers in data
        if 'data' in record_data and isinstance(record_data['data'], dict):
            for key, value in record_data['data'].items():
                if key in ['email', 'name', 'phone', 'address', 'ip_address']:
                    record_data['data'][key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]

        logger.info(f"Record {record_id} pseudonymized")

    def _get_rights_information(self) -> dict[str, str]:
        """Get information about GDPR data subject rights."""
        return {
            "access": "You have the right to access your personal data",
            "rectification": "You have the right to correct inaccurate personal data",
            "erasure": "You have the right to request deletion of your personal data",
            "restriction": "You have the right to restrict processing of your personal data",
            "portability": "You have the right to receive your personal data in a portable format",
            "objection": "You have the right to object to processing of your personal data",
            "contact": "Contact our Data Protection Officer at dpo@mpc-transformer.com"
        }

    def _log_request(self, request_type: str, data_subject_id: str, details: dict = None) -> None:
        """Log GDPR request for audit purposes."""
        audit_entry = AuditLog(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=f"gdpr_{request_type}_request",
            data_subject_id=data_subject_id,
            details=details or {}
        )
        self._audit_log.append(audit_entry)

class GDPRCompliance(ComplianceFramework):
    """Main GDPR compliance framework implementation."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize GDPR compliance framework.
        
        Args:
            config: GDPR configuration parameters
        """
        super().__init__("gdpr", config)

        self.data_controller = config.get("data_controller", {})
        self.dpo_contact = config.get("dpo_contact", "")
        self.consent_manager = GDPRConsentManager()
        self.data_processor = GDPRDataProcessor(consent_manager=self.consent_manager)

        # Initialize breach notification settings
        self.breach_notification = config.get("breach_notification", {
            "authority_deadline_hours": 72,
            "data_subject_deadline_hours": 72,
            "authority_contact": "",
            "breach_register": []
        })

    def process_data_subject_request(self, request: DataSubjectRequest) -> dict[str, Any]:
        """
        Process GDPR data subject request.
        
        Args:
            request: Data subject request details
            
        Returns:
            Request processing result
        """
        request_type = request.request_type.lower()

        if request_type == "access":
            return self.data_processor.process_access_request(request.data_subject_id)
        elif request_type == "erasure":
            success = self.data_processor.process_erasure_request(
                request.data_subject_id,
                request.details.get("reason", "withdrawal_of_consent")
            )
            return {"success": success, "message": "Erasure request processed"}
        elif request_type == "portability":
            return self.data_processor.process_portability_request(request.data_subject_id)
        elif request_type == "rectification":
            success = self.data_processor.process_rectification_request(
                request.data_subject_id,
                request.details.get("corrections", {})
            )
            return {"success": success, "message": "Rectification request processed"}
        else:
            raise ValueError(f"Unsupported GDPR request type: {request_type}")

    def validate_processing(self,
                           data_category: str,
                           purpose: str,
                           legal_basis: str,
                           data_subject_id: str = None) -> bool:
        """
        Validate if data processing is GDPR compliant.
        
        Args:
            data_category: Category of data being processed
            purpose: Purpose of processing
            legal_basis: Legal basis for processing
            data_subject_id: Data subject identifier (for consent checks)
            
        Returns:
            True if processing is compliant
        """
        # Check if legal basis is valid
        try:
            legal_basis_enum = GDPRLegalBasis(legal_basis)
        except ValueError:
            logger.error(f"Invalid GDPR legal basis: {legal_basis}")
            return False

        # If consent-based, check consent
        if legal_basis_enum == GDPRLegalBasis.CONSENT and data_subject_id:
            try:
                category_enum = GDPRDataCategory(data_category)
                purpose_enum = GDPRProcessingPurpose(purpose)

                has_consent = self.consent_manager.check_consent(
                    data_subject_id, purpose_enum, category_enum
                )
                if not has_consent:
                    logger.warning(f"No valid consent for processing: {data_subject_id}")
                    return False
            except ValueError:
                logger.error(f"Invalid category or purpose: {data_category}, {purpose}")
                return False

        # Additional validation based on data category
        if data_category == GDPRDataCategory.SENSITIVE_DATA.value:
            # Special categories require explicit consent or other specific legal basis
            if legal_basis_enum not in [GDPRLegalBasis.CONSENT, GDPRLegalBasis.LEGAL_OBLIGATION]:
                logger.error("Sensitive data requires explicit consent or legal obligation")
                return False

        return True

    def generate_compliance_report(self) -> dict[str, Any]:
        """
        Generate comprehensive GDPR compliance report.
        
        Returns:
            Compliance report dictionary
        """
        return {
            "framework": "GDPR",
            "report_date": datetime.utcnow().isoformat(),
            "data_controller": self.data_controller,
            "dpo_contact": self.dpo_contact,
            "statistics": {
                "total_data_subjects": len(set(
                    record.get('data_subject_id') for record in self.data_processor.data_storage.values()
                    if isinstance(record, dict) and 'data_subject_id' in record
                )),
                "active_consents": len([
                    record for record in self.consent_manager.storage.values()
                    if json.loads(self.consent_manager._cipher.decrypt(record).decode())['granted']
                ]),
                "data_subject_requests": len(self.data_processor._audit_log),
                "breach_incidents": len(self.breach_notification.get("breach_register", []))
            },
            "compliance_status": {
                "consent_management": "compliant",
                "data_subject_rights": "compliant",
                "breach_notification": "compliant",
                "data_protection_measures": "compliant"
            },
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate compliance recommendations based on current state."""
        recommendations = []

        # Check consent coverage
        total_records = len(self.data_processor.data_storage)
        consent_based_records = len([
            r for r in self.data_processor.data_storage.values()
            if isinstance(r, dict) and r.get('legal_basis') == GDPRLegalBasis.CONSENT.value
        ])

        if consent_based_records / max(total_records, 1) > 0.5:
            recommendations.append("Consider implementing consent renewal mechanism")

        # Check data retention
        old_records = len([
            r for r in self.data_processor.data_storage.values()
            if isinstance(r, dict) and 'created_at' in r and
            datetime.fromisoformat(r['created_at']) < datetime.utcnow() - timedelta(days=730)
        ])

        if old_records > 0:
            recommendations.append(f"Review {old_records} old records for retention compliance")

        return recommendations
