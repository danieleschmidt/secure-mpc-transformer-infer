"""
Autonomous Global Manager - Global-First Implementation

Comprehensive internationalization, compliance, and cross-platform support
for autonomous SDLC execution with defensive security focus.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import locale
import platform

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Personal Data Protection Act
    ISO_27001 = "iso_27001"  # Information Security Management
    SOC_2 = "soc_2"  # Service Organization Control 2
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard


class SupportedLocale(Enum):
    """Supported locales for internationalization"""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    FR_FR = "fr_FR"  # French (France)
    DE_DE = "de_DE"  # German (Germany)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    KO_KR = "ko_KR"  # Korean (South Korea)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    IT_IT = "it_IT"  # Italian (Italy)
    RU_RU = "ru_RU"  # Russian (Russia)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)
    HI_IN = "hi_IN"  # Hindi (India)


@dataclass
class ComplianceRule:
    """Individual compliance rule definition"""
    id: str
    framework: ComplianceFramework
    category: str
    title: str
    description: str
    requirements: List[str]
    severity: str  # critical, high, medium, low
    applicable_regions: List[str]
    validation_function: Optional[str] = None


@dataclass
class LocalizationEntry:
    """Localization entry for translations"""
    key: str
    locale: SupportedLocale
    value: str
    context: Optional[str] = None
    pluralization_rules: Optional[Dict[str, str]] = None


@dataclass
class GlobalConfig:
    """Configuration for global deployment"""
    default_locale: SupportedLocale = SupportedLocale.EN_US
    supported_locales: List[SupportedLocale] = field(default_factory=lambda: [
        SupportedLocale.EN_US, SupportedLocale.ES_ES, SupportedLocale.FR_FR,
        SupportedLocale.DE_DE, SupportedLocale.JA_JP, SupportedLocale.ZH_CN
    ])
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=lambda: [
        ComplianceFramework.GDPR, ComplianceFramework.CCPA, ComplianceFramework.ISO_27001
    ])
    enable_rtl_support: bool = True  # Right-to-left language support
    enable_timezone_conversion: bool = True
    enable_currency_conversion: bool = True
    data_residency_requirements: Dict[str, List[str]] = field(default_factory=dict)


class AutonomousGlobalManager:
    """
    Comprehensive global deployment manager.
    
    Implements internationalization, compliance frameworks,
    and cross-platform support for autonomous systems.
    """
    
    def __init__(self, config: Optional[GlobalConfig] = None,
                 project_root: Optional[Path] = None):
        self.config = config or GlobalConfig()
        self.project_root = project_root or Path.cwd()
        
        # Localization data
        self.translations: Dict[str, Dict[str, str]] = {}
        self.localization_cache: Dict[str, LocalizationEntry] = {}
        
        # Compliance rules
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        
        # Platform detection
        self.platform_info = self._detect_platform_info()
        
        # Global deployment tracking
        self.deployment_regions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"AutonomousGlobalManager initialized for {len(self.config.supported_locales)} locales")
        
        # Initialize default data
        asyncio.create_task(self._initialize_global_data())
    
    def _detect_platform_info(self) -> Dict[str, Any]:
        """Detect current platform information"""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "locale": locale.getdefaultlocale(),
            "encoding": locale.getpreferredencoding()
        }
    
    async def _initialize_global_data(self) -> None:
        """Initialize global localization and compliance data"""
        try:
            await self._load_translations()
            await self._load_compliance_rules()
            logger.info("Global data initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize global data: {e}")
    
    async def _load_translations(self) -> None:
        """Load translation data for supported locales"""
        
        # Default translations for common terms
        default_translations = {
            "en_US": {
                "welcome": "Welcome",
                "error": "Error",
                "success": "Success", 
                "loading": "Loading",
                "cancel": "Cancel",
                "confirm": "Confirm",
                "save": "Save",
                "delete": "Delete",
                "edit": "Edit",
                "create": "Create",
                "update": "Update",
                "security_warning": "Security Warning",
                "validation_failed": "Validation Failed",
                "processing": "Processing",
                "completed": "Completed",
                "autonomous_execution": "Autonomous Execution",
                "quality_gates": "Quality Gates",
                "performance_metrics": "Performance Metrics"
            },
            "es_ES": {
                "welcome": "Bienvenido",
                "error": "Error",
                "success": "Éxito",
                "loading": "Cargando",
                "cancel": "Cancelar",
                "confirm": "Confirmar",
                "save": "Guardar",
                "delete": "Eliminar",
                "edit": "Editar",
                "create": "Crear",
                "update": "Actualizar",
                "security_warning": "Advertencia de Seguridad",
                "validation_failed": "Validación Fallida",
                "processing": "Procesando",
                "completed": "Completado",
                "autonomous_execution": "Ejecución Autónoma",
                "quality_gates": "Puertas de Calidad",
                "performance_metrics": "Métricas de Rendimiento"
            },
            "fr_FR": {
                "welcome": "Bienvenue",
                "error": "Erreur",
                "success": "Succès",
                "loading": "Chargement",
                "cancel": "Annuler",
                "confirm": "Confirmer",
                "save": "Sauvegarder",
                "delete": "Supprimer",
                "edit": "Modifier",
                "create": "Créer",
                "update": "Mettre à jour",
                "security_warning": "Avertissement de Sécurité",
                "validation_failed": "Validation Échouée",
                "processing": "Traitement",
                "completed": "Terminé",
                "autonomous_execution": "Exécution Autonome",
                "quality_gates": "Portes de Qualité",
                "performance_metrics": "Métriques de Performance"
            },
            "de_DE": {
                "welcome": "Willkommen",
                "error": "Fehler",
                "success": "Erfolg",
                "loading": "Laden",
                "cancel": "Abbrechen",
                "confirm": "Bestätigen",
                "save": "Speichern",
                "delete": "Löschen",
                "edit": "Bearbeiten",
                "create": "Erstellen",
                "update": "Aktualisieren",
                "security_warning": "Sicherheitswarnung",
                "validation_failed": "Validierung Fehlgeschlagen",
                "processing": "Verarbeitung",
                "completed": "Abgeschlossen",
                "autonomous_execution": "Autonome Ausführung",
                "quality_gates": "Qualitätstore",
                "performance_metrics": "Leistungsmetriken"
            },
            "ja_JP": {
                "welcome": "ようこそ",
                "error": "エラー",
                "success": "成功",
                "loading": "読み込み中",
                "cancel": "キャンセル",
                "confirm": "確認",
                "save": "保存",
                "delete": "削除",
                "edit": "編集",
                "create": "作成",
                "update": "更新",
                "security_warning": "セキュリティ警告",
                "validation_failed": "検証失敗",
                "processing": "処理中",
                "completed": "完了",
                "autonomous_execution": "自律実行",
                "quality_gates": "品質ゲート",
                "performance_metrics": "パフォーマンス指標"
            },
            "zh_CN": {
                "welcome": "欢迎",
                "error": "错误",
                "success": "成功",
                "loading": "加载中",
                "cancel": "取消",
                "confirm": "确认",
                "save": "保存",
                "delete": "删除",
                "edit": "编辑",
                "create": "创建",
                "update": "更新",
                "security_warning": "安全警告",
                "validation_failed": "验证失败",
                "processing": "处理中",
                "completed": "已完成",
                "autonomous_execution": "自主执行",
                "quality_gates": "质量门",
                "performance_metrics": "性能指标"
            }
        }
        
        self.translations = default_translations
        
        # Try to load from translation files if they exist
        i18n_dir = self.project_root / "src" / "secure_mpc_transformer" / "i18n" / "translations"
        
        if i18n_dir.exists():
            for locale_file in i18n_dir.glob("*.json"):
                locale_code = locale_file.stem
                try:
                    with open(locale_file, 'r', encoding='utf-8') as f:
                        file_translations = json.load(f)
                        if locale_code in self.translations:
                            self.translations[locale_code].update(file_translations)
                        else:
                            self.translations[locale_code] = file_translations
                    logger.debug(f"Loaded translations for {locale_code}")
                except Exception as e:
                    logger.warning(f"Failed to load translations from {locale_file}: {e}")
    
    async def _load_compliance_rules(self) -> None:
        """Load compliance rules for different frameworks"""
        
        # GDPR compliance rules
        gdpr_rules = [
            ComplianceRule(
                id="gdpr_data_minimization",
                framework=ComplianceFramework.GDPR,
                category="data_protection",
                title="Data Minimization",
                description="Personal data must be adequate, relevant and limited to what is necessary",
                requirements=[
                    "Collect only necessary personal data",
                    "Implement data retention policies",
                    "Regular data audits and cleanup"
                ],
                severity="high",
                applicable_regions=["EU", "EEA"],
                validation_function="validate_data_minimization"
            ),
            ComplianceRule(
                id="gdpr_consent_management",
                framework=ComplianceFramework.GDPR,
                category="consent",
                title="Consent Management",
                description="Valid consent must be freely given, specific, informed and unambiguous",
                requirements=[
                    "Implement consent collection mechanisms",
                    "Provide consent withdrawal options",
                    "Maintain consent records"
                ],
                severity="critical",
                applicable_regions=["EU", "EEA"],
                validation_function="validate_consent_management"
            ),
            ComplianceRule(
                id="gdpr_data_encryption",
                framework=ComplianceFramework.GDPR,
                category="technical_measures",
                title="Data Encryption",
                description="Implement appropriate technical measures to protect personal data",
                requirements=[
                    "Encrypt personal data at rest",
                    "Encrypt personal data in transit",
                    "Use strong encryption algorithms"
                ],
                severity="high",
                applicable_regions=["EU", "EEA"],
                validation_function="validate_data_encryption"
            )
        ]
        
        # ISO 27001 compliance rules
        iso27001_rules = [
            ComplianceRule(
                id="iso27001_access_control",
                framework=ComplianceFramework.ISO_27001,
                category="access_control",
                title="Access Control Policy",
                description="Implement comprehensive access control mechanisms",
                requirements=[
                    "Define access control policy",
                    "Implement role-based access control",
                    "Regular access reviews"
                ],
                severity="high",
                applicable_regions=["Global"],
                validation_function="validate_access_control"
            ),
            ComplianceRule(
                id="iso27001_incident_management",
                framework=ComplianceFramework.ISO_27001,
                category="incident_management",
                title="Information Security Incident Management",
                description="Establish incident management procedures",
                requirements=[
                    "Define incident response procedures",
                    "Implement incident detection mechanisms",
                    "Maintain incident logs"
                ],
                severity="critical",
                applicable_regions=["Global"],
                validation_function="validate_incident_management"
            )
        ]
        
        # CCPA compliance rules
        ccpa_rules = [
            ComplianceRule(
                id="ccpa_privacy_rights",
                framework=ComplianceFramework.CCPA,
                category="consumer_rights",
                title="Consumer Privacy Rights",
                description="Provide consumers with privacy rights regarding their personal information",
                requirements=[
                    "Right to know about personal information collected",
                    "Right to delete personal information",
                    "Right to opt-out of sale of personal information"
                ],
                severity="high",
                applicable_regions=["California", "US"],
                validation_function="validate_privacy_rights"
            )
        ]
        
        # Store all rules
        all_rules = gdpr_rules + iso27001_rules + ccpa_rules
        for rule in all_rules:
            self.compliance_rules[rule.id] = rule
        
        logger.info(f"Loaded {len(all_rules)} compliance rules")
    
    def translate(self, key: str, locale: Optional[Union[str, SupportedLocale]] = None,
                  context: Optional[str] = None, **kwargs) -> str:
        """
        Translate a key to the specified locale.
        
        Args:
            key: Translation key
            locale: Target locale (defaults to default locale)
            context: Optional context for disambiguation
            **kwargs: Variables for string interpolation
            
        Returns:
            Translated string
        """
        
        # Determine target locale
        if locale is None:
            target_locale = self.config.default_locale.value
        elif isinstance(locale, SupportedLocale):
            target_locale = locale.value
        else:
            target_locale = str(locale)
        
        # Get translation
        translation = self._get_translation(key, target_locale)
        
        # Apply string interpolation if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation interpolation failed for key '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, locale: str) -> str:
        """Get translation for key and locale"""
        
        # Check cache first
        cache_key = f"{locale}:{key}"
        if cache_key in self.localization_cache:
            return self.localization_cache[cache_key].value
        
        # Look up translation
        if locale in self.translations and key in self.translations[locale]:
            translation = self.translations[locale][key]
        elif self.config.default_locale.value in self.translations and key in self.translations[self.config.default_locale.value]:
            # Fallback to default locale
            translation = self.translations[self.config.default_locale.value][key]
            logger.debug(f"Using fallback translation for key '{key}' in locale '{locale}'")
        else:
            # No translation found, return key
            translation = key
            logger.warning(f"No translation found for key '{key}' in locale '{locale}'")
        
        # Cache the result
        self.localization_cache[cache_key] = LocalizationEntry(
            key=key,
            locale=SupportedLocale(locale) if locale in [l.value for l in SupportedLocale] else SupportedLocale.EN_US,
            value=translation
        )
        
        return translation
    
    async def validate_compliance(self, frameworks: Optional[List[ComplianceFramework]] = None,
                                region: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate compliance with specified frameworks.
        
        Args:
            frameworks: List of frameworks to validate (defaults to configured frameworks)
            region: Specific region to validate for
            
        Returns:
            Compliance validation results
        """
        
        if frameworks is None:
            frameworks = self.config.compliance_frameworks
        
        logger.info(f"Validating compliance for frameworks: {[f.value for f in frameworks]}")
        
        validation_results = {
            "overall_status": "compliant",
            "framework_results": {},
            "violations": [],
            "recommendations": [],
            "validation_time": time.time()
        }
        
        total_rules = 0
        passed_rules = 0
        
        for framework in frameworks:
            framework_rules = [
                rule for rule in self.compliance_rules.values()
                if rule.framework == framework
            ]
            
            if region:
                framework_rules = [
                    rule for rule in framework_rules
                    if region in rule.applicable_regions or "Global" in rule.applicable_regions
                ]
            
            framework_result = await self._validate_framework_compliance(framework, framework_rules)
            validation_results["framework_results"][framework.value] = framework_result
            
            total_rules += framework_result["total_rules"]
            passed_rules += framework_result["passed_rules"]
            
            # Collect violations
            validation_results["violations"].extend(framework_result["violations"])
            validation_results["recommendations"].extend(framework_result["recommendations"])
        
        # Determine overall status
        if validation_results["violations"]:
            critical_violations = [v for v in validation_results["violations"] if v["severity"] == "critical"]
            if critical_violations:
                validation_results["overall_status"] = "non_compliant"
            else:
                validation_results["overall_status"] = "partial_compliance"
        
        validation_results["compliance_score"] = passed_rules / total_rules if total_rules > 0 else 1.0
        
        return validation_results
    
    async def _validate_framework_compliance(self, framework: ComplianceFramework,
                                           rules: List[ComplianceRule]) -> Dict[str, Any]:
        """Validate compliance for a specific framework"""
        
        result = {
            "framework": framework.value,
            "total_rules": len(rules),
            "passed_rules": 0,
            "failed_rules": 0,
            "violations": [],
            "recommendations": []
        }
        
        for rule in rules:
            try:
                is_compliant = await self._validate_compliance_rule(rule)
                
                if is_compliant:
                    result["passed_rules"] += 1
                else:
                    result["failed_rules"] += 1
                    result["violations"].append({
                        "rule_id": rule.id,
                        "title": rule.title,
                        "severity": rule.severity,
                        "description": rule.description,
                        "requirements": rule.requirements
                    })
                    
                    # Add recommendations for failed rules
                    result["recommendations"].extend([
                        f"Address {rule.title}: {req}" for req in rule.requirements
                    ])
                    
            except Exception as e:
                logger.error(f"Failed to validate rule {rule.id}: {e}")
                result["failed_rules"] += 1
        
        return result
    
    async def _validate_compliance_rule(self, rule: ComplianceRule) -> bool:
        """Validate a specific compliance rule"""
        
        # This is a simplified validation - in production, each rule would have
        # specific validation logic
        
        if rule.validation_function:
            try:
                # In a real implementation, this would call the specific validation function
                validation_method = getattr(self, rule.validation_function, None)
                if validation_method:
                    return await validation_method(rule)
                else:
                    logger.warning(f"Validation function {rule.validation_function} not found")
                    return False
            except Exception as e:
                logger.error(f"Validation function {rule.validation_function} failed: {e}")
                return False
        
        # Default validation based on rule category
        if rule.category == "data_protection":
            return await self._validate_data_protection(rule)
        elif rule.category == "access_control":
            return await self._validate_access_control(rule)
        elif rule.category == "technical_measures":
            return await self._validate_technical_measures(rule)
        else:
            # Generic validation
            return True  # Assume compliant for unknown categories
    
    async def _validate_data_protection(self, rule: ComplianceRule) -> bool:
        """Validate data protection compliance"""
        
        # Check for data encryption
        has_encryption = False
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(keyword in content.lower() for keyword in ['encrypt', 'crypto', 'hash', 'secure']):
                    has_encryption = True
                    break
            except Exception:
                continue
        
        return has_encryption
    
    async def _validate_access_control(self, rule: ComplianceRule) -> bool:
        """Validate access control compliance"""
        
        # Check for authentication/authorization code
        has_auth = False
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(keyword in content.lower() for keyword in ['auth', 'login', 'permission', 'role']):
                    has_auth = True
                    break
            except Exception:
                continue
        
        return has_auth
    
    async def _validate_technical_measures(self, rule: ComplianceRule) -> bool:
        """Validate technical security measures"""
        
        # Check for security implementations
        has_security = False
        security_files = ["security", "validation", "resilience"]
        
        for security_term in security_files:
            if any(self.project_root.rglob(f"*{security_term}*.py")):
                has_security = True
                break
        
        return has_security
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales with metadata"""
        
        locale_metadata = {
            SupportedLocale.EN_US: {"name": "English (US)", "rtl": False, "region": "Americas"},
            SupportedLocale.EN_GB: {"name": "English (UK)", "rtl": False, "region": "Europe"},
            SupportedLocale.ES_ES: {"name": "Español (España)", "rtl": False, "region": "Europe"},
            SupportedLocale.ES_MX: {"name": "Español (México)", "rtl": False, "region": "Americas"},
            SupportedLocale.FR_FR: {"name": "Français (France)", "rtl": False, "region": "Europe"},
            SupportedLocale.DE_DE: {"name": "Deutsch (Deutschland)", "rtl": False, "region": "Europe"},
            SupportedLocale.JA_JP: {"name": "日本語 (日本)", "rtl": False, "region": "Asia"},
            SupportedLocale.ZH_CN: {"name": "中文 (简体)", "rtl": False, "region": "Asia"},
            SupportedLocale.ZH_TW: {"name": "中文 (繁體)", "rtl": False, "region": "Asia"},
            SupportedLocale.KO_KR: {"name": "한국어 (대한민국)", "rtl": False, "region": "Asia"},
            SupportedLocale.PT_BR: {"name": "Português (Brasil)", "rtl": False, "region": "Americas"},
            SupportedLocale.IT_IT: {"name": "Italiano (Italia)", "rtl": False, "region": "Europe"},
            SupportedLocale.RU_RU: {"name": "Русский (Россия)", "rtl": False, "region": "Europe"},
            SupportedLocale.AR_SA: {"name": "العربية (السعودية)", "rtl": True, "region": "Middle East"},
            SupportedLocale.HI_IN: {"name": "हिन्दी (भारत)", "rtl": False, "region": "Asia"}
        }
        
        supported = []
        for locale in self.config.supported_locales:
            metadata = locale_metadata.get(locale, {"name": locale.value, "rtl": False, "region": "Unknown"})
            supported.append({
                "code": locale.value,
                "name": metadata["name"],
                "rtl": metadata["rtl"],
                "region": metadata["region"],
                "has_translations": locale.value in self.translations
            })
        
        return supported
    
    def get_compliance_frameworks(self) -> List[Dict[str, Any]]:
        """Get list of supported compliance frameworks"""
        
        framework_metadata = {
            ComplianceFramework.GDPR: {
                "name": "General Data Protection Regulation",
                "region": "EU/EEA",
                "description": "EU regulation on data protection and privacy"
            },
            ComplianceFramework.CCPA: {
                "name": "California Consumer Privacy Act",
                "region": "California, US",
                "description": "California state statute on consumer privacy rights"
            },
            ComplianceFramework.PDPA: {
                "name": "Personal Data Protection Act",
                "region": "Singapore, Thailand",
                "description": "Data protection regulations in ASEAN countries"
            },
            ComplianceFramework.ISO_27001: {
                "name": "ISO/IEC 27001",
                "region": "Global",
                "description": "International standard for information security management"
            },
            ComplianceFramework.SOC_2: {
                "name": "SOC 2",
                "region": "Global",
                "description": "Auditing procedure for service organizations"
            },
            ComplianceFramework.HIPAA: {
                "name": "Health Insurance Portability and Accountability Act",
                "region": "US",
                "description": "US legislation for healthcare data protection"
            },
            ComplianceFramework.PCI_DSS: {
                "name": "Payment Card Industry Data Security Standard",
                "region": "Global",
                "description": "Security standard for payment card data"
            }
        }
        
        frameworks = []
        for framework in ComplianceFramework:
            metadata = framework_metadata.get(framework, {
                "name": framework.value,
                "region": "Unknown",
                "description": "No description available"
            })
            
            rule_count = len([r for r in self.compliance_rules.values() if r.framework == framework])
            
            frameworks.append({
                "code": framework.value,
                "name": metadata["name"],
                "region": metadata["region"],
                "description": metadata["description"],
                "rule_count": rule_count,
                "enabled": framework in self.config.compliance_frameworks
            })
        
        return frameworks
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get current platform information"""
        return self.platform_info.copy()
    
    async def generate_deployment_config(self, regions: List[str]) -> Dict[str, Any]:
        """Generate deployment configuration for multiple regions"""
        
        deployment_config = {
            "global_config": {
                "default_locale": self.config.default_locale.value,
                "supported_locales": [l.value for l in self.config.supported_locales],
                "compliance_frameworks": [f.value for f in self.config.compliance_frameworks],
                "enable_rtl_support": self.config.enable_rtl_support,
                "enable_timezone_conversion": self.config.enable_timezone_conversion,
                "enable_currency_conversion": self.config.enable_currency_conversion
            },
            "regional_configs": {},
            "compliance_requirements": {},
            "localization_data": self.translations
        }
        
        # Regional configuration mapping
        region_configs = {
            "eu-west": {
                "primary_locale": "en_GB",
                "secondary_locales": ["fr_FR", "de_DE", "es_ES", "it_IT"],
                "compliance_frameworks": ["gdpr", "iso_27001"],
                "data_residency": "EU",
                "timezone": "Europe/London",
                "currency": "EUR"
            },
            "us-east": {
                "primary_locale": "en_US",
                "secondary_locales": ["es_MX"],
                "compliance_frameworks": ["ccpa", "soc_2"],
                "data_residency": "US",
                "timezone": "America/New_York",
                "currency": "USD"
            },
            "asia-pacific": {
                "primary_locale": "en_US",
                "secondary_locales": ["ja_JP", "zh_CN", "ko_KR", "hi_IN"],
                "compliance_frameworks": ["pdpa", "iso_27001"],
                "data_residency": "APAC",
                "timezone": "Asia/Tokyo",
                "currency": "JPY"
            },
            "americas": {
                "primary_locale": "en_US",
                "secondary_locales": ["es_ES", "pt_BR"],
                "compliance_frameworks": ["ccpa", "iso_27001"],
                "data_residency": "Americas",
                "timezone": "America/Sao_Paulo",
                "currency": "USD"
            }
        }
        
        # Configure each region
        for region in regions:
            if region in region_configs:
                deployment_config["regional_configs"][region] = region_configs[region]
            else:
                # Default configuration for unknown regions
                deployment_config["regional_configs"][region] = {
                    "primary_locale": "en_US",
                    "secondary_locales": [],
                    "compliance_frameworks": ["iso_27001"],
                    "data_residency": "Global",
                    "timezone": "UTC",
                    "currency": "USD"
                }
        
        # Add compliance requirements for each framework
        for framework in self.config.compliance_frameworks:
            framework_rules = [r for r in self.compliance_rules.values() if r.framework == framework]
            deployment_config["compliance_requirements"][framework.value] = [
                {
                    "id": rule.id,
                    "title": rule.title,
                    "category": rule.category,
                    "severity": rule.severity,
                    "requirements": rule.requirements
                }
                for rule in framework_rules
            ]
        
        return deployment_config
    
    def get_globalization_metrics(self) -> Dict[str, Any]:
        """Get globalization implementation metrics"""
        
        translation_coverage = {}
        for locale in [l.value for l in self.config.supported_locales]:
            if locale in self.translations:
                translation_coverage[locale] = len(self.translations[locale])
            else:
                translation_coverage[locale] = 0
        
        return {
            "supported_locales": len(self.config.supported_locales),
            "available_translations": len(self.translations),
            "translation_coverage": translation_coverage,
            "compliance_frameworks": len(self.config.compliance_frameworks),
            "compliance_rules": len(self.compliance_rules),
            "platform_compatibility": {
                "system": self.platform_info["system"],
                "architecture": self.platform_info["architecture"][0] if self.platform_info["architecture"] else "unknown",
                "python_version": self.platform_info["python_version"]
            },
            "features": {
                "rtl_support": self.config.enable_rtl_support,
                "timezone_conversion": self.config.enable_timezone_conversion,
                "currency_conversion": self.config.enable_currency_conversion
            },
            "cache_stats": {
                "localization_cache_size": len(self.localization_cache)
            }
        }
    
    def clear_cache(self) -> None:
        """Clear localization cache"""
        self.localization_cache.clear()
        logger.info("Localization cache cleared")