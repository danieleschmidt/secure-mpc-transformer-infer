"""
Translation system for multi-language support
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class Translator:
    """Multi-language translation manager with caching and fallback support."""
    
    def __init__(self, 
                 translations_dir: Optional[Path] = None,
                 default_language: str = "en",
                 fallback_language: str = "en"):
        """
        Initialize translator with translation files.
        
        Args:
            translations_dir: Directory containing translation JSON files
            default_language: Default language code
            fallback_language: Fallback language for missing translations
        """
        self.default_language = default_language
        self.fallback_language = fallback_language
        self._current_language = default_language
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
        if translations_dir is None:
            translations_dir = Path(__file__).parent / "translations"
        
        self.translations_dir = Path(translations_dir)
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load all translation files from the translations directory."""
        if not self.translations_dir.exists():
            logger.warning(f"Translations directory not found: {self.translations_dir}")
            return
            
        for lang_file in self.translations_dir.glob("*.json"):
            language = lang_file.stem
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self._translations[language] = json.load(f)
                logger.info(f"Loaded translations for language: {language}")
            except Exception as e:
                logger.error(f"Failed to load translations for {language}: {e}")
    
    def set_language(self, language_code: str) -> None:
        """Set the current language for translations."""
        with self._lock:
            if language_code in self._translations:
                self._current_language = language_code
                logger.info(f"Language changed to: {language_code}")
            else:
                logger.warning(f"Language {language_code} not available, using {self._current_language}")
    
    def get_language(self) -> str:
        """Get the current language code."""
        return self._current_language
    
    def translate(self, 
                  key: str, 
                  language: Optional[str] = None,
                  **kwargs) -> str:
        """
        Translate a key to the specified or current language.
        
        Args:
            key: Translation key (supports dot notation for nested keys)
            language: Target language code (uses current if not specified)
            **kwargs: Variables for string formatting
            
        Returns:
            Translated string with variables substituted
        """
        target_language = language or self._current_language
        
        # Get translation from target language
        translation = self._get_translation(key, target_language)
        
        # Fallback to default language if not found
        if translation is None and target_language != self.fallback_language:
            translation = self._get_translation(key, self.fallback_language)
            
        # Return key if no translation found
        if translation is None:
            logger.warning(f"Translation not found for key: {key} in language: {target_language}")
            return key
        
        # Format string with provided variables
        try:
            return translation.format(**kwargs) if kwargs else translation
        except (KeyError, ValueError) as e:
            logger.error(f"Translation formatting error for key {key}: {e}")
            return translation
    
    def _get_translation(self, key: str, language: str) -> Optional[str]:
        """Get a translation for a specific key and language."""
        if language not in self._translations:
            return None
            
        translations = self._translations[language]
        
        # Support dot notation for nested keys (e.g., "errors.validation.required")
        keys = key.split('.')
        current = translations
        
        try:
            for k in keys:
                current = current[k]
            return str(current) if current is not None else None
        except (KeyError, TypeError):
            return None
    
    def get_available_languages(self) -> list:
        """Get list of available language codes."""
        return list(self._translations.keys())
    
    def has_translation(self, key: str, language: Optional[str] = None) -> bool:
        """Check if a translation exists for a key in the specified language."""
        target_language = language or self._current_language
        return self._get_translation(key, target_language) is not None
    
    def get_translations_for_language(self, language: str) -> Dict[str, Any]:
        """Get all translations for a specific language."""
        return self._translations.get(language, {}).copy()
    
    def add_translation(self, language: str, key: str, value: str) -> None:
        """Dynamically add a translation."""
        with self._lock:
            if language not in self._translations:
                self._translations[language] = {}
            
            # Support dot notation for nested keys
            keys = key.split('.')
            current = self._translations[language]
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            logger.debug(f"Added translation {language}.{key}: {value}")

    def bulk_add_translations(self, language: str, translations: Dict[str, str]) -> None:
        """Add multiple translations at once."""
        with self._lock:
            if language not in self._translations:
                self._translations[language] = {}
            
            for key, value in translations.items():
                self.add_translation(language, key, value)


# Global translator instance
_global_translator: Optional[Translator] = None
_translator_lock = Lock()

def get_translator(translations_dir: Optional[Path] = None) -> Translator:
    """Get the global translator instance (singleton)."""
    global _global_translator
    
    with _translator_lock:
        if _global_translator is None:
            _global_translator = Translator(translations_dir)
        return _global_translator

def t(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation."""
    translator = get_translator()
    return translator.translate(key, language, **kwargs)