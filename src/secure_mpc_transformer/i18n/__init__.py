"""
Internationalization (i18n) module for Secure MPC Transformer

Provides multi-language support for:
- English (en)
- Spanish (es) 
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)

Features:
- Dynamic language switching
- Regional formatting (currency, dates, numbers)
- Timezone handling
- Translation management
- Locale-aware validation
"""

from .translator import Translator, get_translator
from .formatters import (
    CurrencyFormatter,
    DateTimeFormatter, 
    NumberFormatter,
    get_formatter
)
from .locale_manager import LocaleManager, get_locale_manager
from .validation import LocaleValidator, validate_locale_input

__all__ = [
    "Translator",
    "get_translator", 
    "CurrencyFormatter",
    "DateTimeFormatter",
    "NumberFormatter", 
    "get_formatter",
    "LocaleManager",
    "get_locale_manager",
    "LocaleValidator",
    "validate_locale_input"
]

# Supported languages with regional variants
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "native_name": "English",
        "regions": ["US", "GB", "CA", "AU"],
        "default_region": "US",
        "currency": "USD",
        "rtl": False
    },
    "es": {
        "name": "Spanish", 
        "native_name": "Español",
        "regions": ["ES", "MX", "AR", "CO", "CL"],
        "default_region": "ES",
        "currency": "EUR",
        "rtl": False
    },
    "fr": {
        "name": "French",
        "native_name": "Français", 
        "regions": ["FR", "CA", "BE", "CH"],
        "default_region": "FR",
        "currency": "EUR",
        "rtl": False
    },
    "de": {
        "name": "German",
        "native_name": "Deutsch",
        "regions": ["DE", "AT", "CH"],
        "default_region": "DE", 
        "currency": "EUR",
        "rtl": False
    },
    "ja": {
        "name": "Japanese",
        "native_name": "日本語",
        "regions": ["JP"],
        "default_region": "JP",
        "currency": "JPY", 
        "rtl": False
    },
    "zh": {
        "name": "Chinese",
        "native_name": "中文",
        "regions": ["CN", "TW", "HK", "SG"],
        "default_region": "CN",
        "currency": "CNY",
        "rtl": False
    }
}

# Default configuration
DEFAULT_LANGUAGE = "en"
DEFAULT_REGION = "US" 
DEFAULT_TIMEZONE = "UTC"

def get_supported_languages():
    """Get list of supported languages with metadata."""
    return SUPPORTED_LANGUAGES.copy()

def is_language_supported(language_code: str) -> bool:
    """Check if a language code is supported."""
    return language_code.lower() in SUPPORTED_LANGUAGES

def get_language_info(language_code: str) -> dict:
    """Get information about a specific language."""
    if not is_language_supported(language_code):
        raise ValueError(f"Language '{language_code}' is not supported")
    return SUPPORTED_LANGUAGES[language_code.lower()].copy()