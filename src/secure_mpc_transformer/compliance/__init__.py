"""
Compliance module for GDPR, CCPA, and PDPA regulatory requirements.

This module provides comprehensive compliance features including:
- GDPR (General Data Protection Regulation) for EU
- CCPA (California Consumer Privacy Act) for California
- PDPA (Personal Data Protection Act) for APAC regions

Features:
- Data classification and tagging
- Consent management
- Data subject rights (access, rectification, erasure, portability)
- Audit logging and reporting
- Data retention and anonymization
- Privacy by design implementations
"""

from .gdpr import GDPRCompliance, GDPRDataProcessor, GDPRConsentManager
from .ccpa import CCPACompliance, CCPADataProcessor, CCPAPrivacyRights
from .pdpa import PDPACompliance, PDPADataProcessor, PDPAPrivacyManager
from .base import (
    ComplianceFramework,
    DataSubjectRequest,
    DataClassification,
    ConsentRecord,
    AuditLog
)
from .data_processor import (
    GlobalDataProcessor,
    DataRetentionManager,
    AnonymizationEngine
)
from .audit import ComplianceAuditor, AuditReporter

__all__ = [
    # GDPR
    "GDPRCompliance",
    "GDPRDataProcessor", 
    "GDPRConsentManager",
    
    # CCPA
    "CCPACompliance",
    "CCPADataProcessor",
    "CCPAPrivacyRights",
    
    # PDPA
    "PDPACompliance", 
    "PDPADataProcessor",
    "PDPAPrivacyManager",
    
    # Base classes
    "ComplianceFramework",
    "DataSubjectRequest",
    "DataClassification",
    "ConsentRecord",
    "AuditLog",
    
    # Data processing
    "GlobalDataProcessor",
    "DataRetentionManager", 
    "AnonymizationEngine",
    
    # Auditing
    "ComplianceAuditor",
    "AuditReporter"
]

# Supported compliance frameworks
SUPPORTED_FRAMEWORKS = {
    "gdpr": {
        "name": "General Data Protection Regulation",
        "jurisdiction": "European Union",
        "effective_date": "2018-05-25",
        "class": "GDPRCompliance"
    },
    "ccpa": {
        "name": "California Consumer Privacy Act", 
        "jurisdiction": "California, USA",
        "effective_date": "2020-01-01",
        "class": "CCPACompliance"
    },
    "pdpa": {
        "name": "Personal Data Protection Act",
        "jurisdiction": "Singapore, Thailand, Malaysia",
        "effective_date": "2021-01-01", 
        "class": "PDPACompliance"
    }
}

def get_compliance_framework(framework_name: str, config: dict = None):
    """
    Get a compliance framework instance by name.
    
    Args:
        framework_name: Name of the compliance framework ("gdpr", "ccpa", "pdpa")
        config: Configuration dictionary for the framework
        
    Returns:
        Compliance framework instance
        
    Raises:
        ValueError: If framework_name is not supported
    """
    if framework_name.lower() not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported compliance framework: {framework_name}")
    
    framework_info = SUPPORTED_FRAMEWORKS[framework_name.lower()]
    config = config or {}
    
    if framework_name.lower() == "gdpr":
        return GDPRCompliance(config)
    elif framework_name.lower() == "ccpa":
        return CCPACompliance(config)
    elif framework_name.lower() == "pdpa":
        return PDPACompliance(config)
    else:
        raise ValueError(f"Framework class not implemented: {framework_info['class']}")

def get_supported_frameworks():
    """Get list of supported compliance frameworks."""
    return SUPPORTED_FRAMEWORKS.copy()

def is_framework_supported(framework_name: str) -> bool:
    """Check if a compliance framework is supported."""
    return framework_name.lower() in SUPPORTED_FRAMEWORKS