"""
Regional formatting utilities for currency, dates, and numbers
"""

import locale
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass
import pytz
import logging

logger = logging.getLogger(__name__)

@dataclass
class LocaleSettings:
    """Locale-specific formatting settings."""
    language: str
    region: str
    currency_symbol: str
    currency_position: str  # "before" or "after"
    decimal_separator: str
    thousands_separator: str
    date_format: str
    time_format: str
    timezone: str

# Regional locale settings
LOCALE_SETTINGS: Dict[str, LocaleSettings] = {
    "en-US": LocaleSettings(
        language="en",
        region="US",
        currency_symbol="$",
        currency_position="before",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%m/%d/%Y",
        time_format="%I:%M %p",
        timezone="America/New_York"
    ),
    "en-GB": LocaleSettings(
        language="en",
        region="GB",
        currency_symbol="£",
        currency_position="before",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%d/%m/%Y",
        time_format="%H:%M",
        timezone="Europe/London"
    ),
    "es-ES": LocaleSettings(
        language="es",
        region="ES",
        currency_symbol="€",
        currency_position="after",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        time_format="%H:%M",
        timezone="Europe/Madrid"
    ),
    "es-MX": LocaleSettings(
        language="es",
        region="MX",
        currency_symbol="$",
        currency_position="before",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%d/%m/%Y",
        time_format="%H:%M",
        timezone="America/Mexico_City"
    ),
    "fr-FR": LocaleSettings(
        language="fr",
        region="FR",
        currency_symbol="€",
        currency_position="after",
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d/%m/%Y",
        time_format="%H:%M",
        timezone="Europe/Paris"
    ),
    "fr-CA": LocaleSettings(
        language="fr",
        region="CA",
        currency_symbol="$",
        currency_position="before",
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d/%m/%Y",
        time_format="%H:%M",
        timezone="America/Toronto"
    ),
    "de-DE": LocaleSettings(
        language="de",
        region="DE",
        currency_symbol="€",
        currency_position="after",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d.%m.%Y",
        time_format="%H:%M",
        timezone="Europe/Berlin"
    ),
    "ja-JP": LocaleSettings(
        language="ja",
        region="JP",
        currency_symbol="¥",
        currency_position="before",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y/%m/%d",
        time_format="%H:%M",
        timezone="Asia/Tokyo"
    ),
    "zh-CN": LocaleSettings(
        language="zh",
        region="CN",
        currency_symbol="¥",
        currency_position="before",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y/%m/%d",
        time_format="%H:%M",
        timezone="Asia/Shanghai"
    ),
    "zh-TW": LocaleSettings(
        language="zh",
        region="TW",
        currency_symbol="NT$",
        currency_position="before",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y/%m/%d",
        time_format="%H:%M",
        timezone="Asia/Taipei"
    )
}

class CurrencyFormatter:
    """Format currency values according to regional conventions."""
    
    def __init__(self, locale_code: str = "en-US"):
        """
        Initialize currency formatter for a specific locale.
        
        Args:
            locale_code: Locale code (e.g., "en-US", "fr-FR")
        """
        self.locale_code = locale_code
        self.settings = LOCALE_SETTINGS.get(locale_code, LOCALE_SETTINGS["en-US"])
    
    def format(self, 
               amount: Union[float, Decimal, int], 
               precision: int = 2,
               show_symbol: bool = True) -> str:
        """
        Format a currency amount.
        
        Args:
            amount: Amount to format
            precision: Number of decimal places
            show_symbol: Whether to show currency symbol
            
        Returns:
            Formatted currency string
        """
        # Convert to Decimal for precise calculations
        if not isinstance(amount, Decimal):
            amount = Decimal(str(amount))
        
        # Round to specified precision
        rounded_amount = round(amount, precision)
        
        # Format the number with separators
        formatted_number = self._format_number(rounded_amount, precision)
        
        if not show_symbol:
            return formatted_number
        
        # Add currency symbol based on position
        if self.settings.currency_position == "before":
            return f"{self.settings.currency_symbol}{formatted_number}"
        else:
            return f"{formatted_number} {self.settings.currency_symbol}"
    
    def _format_number(self, amount: Decimal, precision: int) -> str:
        """Format number with locale-specific separators."""
        # Split into integer and decimal parts
        amount_str = f"{amount:.{precision}f}"
        integer_part, decimal_part = amount_str.split('.')
        
        # Add thousands separators
        if len(integer_part) > 3 and self.settings.thousands_separator:
            # Insert thousands separator every 3 digits from right
            chars = list(reversed(integer_part))
            for i in range(3, len(chars), 4):
                chars.insert(i, self.settings.thousands_separator)
            integer_part = ''.join(reversed(chars))
        
        # Combine with decimal separator
        if precision > 0:
            return f"{integer_part}{self.settings.decimal_separator}{decimal_part}"
        else:
            return integer_part

class DateTimeFormatter:
    """Format dates and times according to regional conventions."""
    
    def __init__(self, locale_code: str = "en-US"):
        """
        Initialize datetime formatter for a specific locale.
        
        Args:
            locale_code: Locale code (e.g., "en-US", "fr-FR")
        """
        self.locale_code = locale_code
        self.settings = LOCALE_SETTINGS.get(locale_code, LOCALE_SETTINGS["en-US"])
        self.timezone = pytz.timezone(self.settings.timezone)
    
    def format_date(self, dt: datetime, format_style: str = "medium") -> str:
        """
        Format a date according to locale conventions.
        
        Args:
            dt: Datetime object to format
            format_style: Format style ("short", "medium", "long", "full")
            
        Returns:
            Formatted date string
        """
        # Convert to local timezone if datetime is timezone-aware
        if dt.tzinfo is not None:
            dt = dt.astimezone(self.timezone)
        
        # Get format based on style
        date_formats = {
            "short": self.settings.date_format,
            "medium": self._get_medium_date_format(),
            "long": self._get_long_date_format(),
            "full": self._get_full_date_format()
        }
        
        format_str = date_formats.get(format_style, self.settings.date_format)
        return dt.strftime(format_str)
    
    def format_time(self, dt: datetime, include_seconds: bool = False) -> str:
        """
        Format a time according to locale conventions.
        
        Args:
            dt: Datetime object to format
            include_seconds: Whether to include seconds
            
        Returns:
            Formatted time string
        """
        # Convert to local timezone if datetime is timezone-aware
        if dt.tzinfo is not None:
            dt = dt.astimezone(self.timezone)
        
        time_format = self.settings.time_format
        if include_seconds:
            if "%I" in time_format:  # 12-hour format
                time_format = time_format.replace("%M", "%M:%S")
            else:  # 24-hour format
                time_format = time_format.replace("%M", "%M:%S")
        
        return dt.strftime(time_format)
    
    def format_datetime(self, 
                       dt: datetime, 
                       date_style: str = "medium",
                       include_seconds: bool = False) -> str:
        """
        Format a datetime according to locale conventions.
        
        Args:
            dt: Datetime object to format
            date_style: Date format style
            include_seconds: Whether to include seconds in time
            
        Returns:
            Formatted datetime string
        """
        date_part = self.format_date(dt, date_style)
        time_part = self.format_time(dt, include_seconds)
        
        # Combine date and time based on locale
        if self.locale_code.startswith("en"):
            return f"{date_part} {time_part}"
        elif self.locale_code.startswith(("fr", "es", "de")):
            return f"{date_part} à {time_part}" if self.locale_code.startswith("fr") else f"{date_part} {time_part}"
        elif self.locale_code.startswith(("ja", "zh")):
            return f"{date_part} {time_part}"
        else:
            return f"{date_part} {time_part}"
    
    def format_relative_time(self, dt: datetime) -> str:
        """
        Format a datetime as relative time (e.g., "2 hours ago").
        
        Args:
            dt: Datetime to format
            
        Returns:
            Relative time string
        """
        now = datetime.now(self.timezone)
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)
        
        diff = now - dt
        seconds = int(diff.total_seconds())
        
        if seconds < 60:
            return "now" if seconds < 10 else f"{seconds} seconds ago"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds < 2629746:  # ~1 month
            weeks = seconds // 604800
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif seconds < 31556952:  # ~1 year
            months = seconds // 2629746
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = seconds // 31556952
            return f"{years} year{'s' if years != 1 else ''} ago"
    
    def _get_medium_date_format(self) -> str:
        """Get medium date format for locale."""
        formats = {
            "en": "%b %d, %Y",
            "es": "%d %b %Y", 
            "fr": "%d %b %Y",
            "de": "%d. %b %Y",
            "ja": "%Y年%m月%d日",
            "zh": "%Y年%m月%d日"
        }
        return formats.get(self.settings.language, formats["en"])
    
    def _get_long_date_format(self) -> str:
        """Get long date format for locale."""
        formats = {
            "en": "%B %d, %Y",
            "es": "%d de %B de %Y",
            "fr": "%d %B %Y", 
            "de": "%d. %B %Y",
            "ja": "%Y年%m月%d日",
            "zh": "%Y年%m月%d日"
        }
        return formats.get(self.settings.language, formats["en"])
    
    def _get_full_date_format(self) -> str:
        """Get full date format for locale."""
        formats = {
            "en": "%A, %B %d, %Y",
            "es": "%A, %d de %B de %Y",
            "fr": "%A %d %B %Y",
            "de": "%A, %d. %B %Y", 
            "ja": "%Y年%m月%d日 %A",
            "zh": "%Y年%m月%d日 %A"
        }
        return formats.get(self.settings.language, formats["en"])

class NumberFormatter:
    """Format numbers according to regional conventions."""
    
    def __init__(self, locale_code: str = "en-US"):
        """
        Initialize number formatter for a specific locale.
        
        Args:
            locale_code: Locale code (e.g., "en-US", "fr-FR")
        """
        self.locale_code = locale_code
        self.settings = LOCALE_SETTINGS.get(locale_code, LOCALE_SETTINGS["en-US"])
    
    def format(self, 
               number: Union[int, float, Decimal],
               precision: Optional[int] = None) -> str:
        """
        Format a number with locale-specific separators.
        
        Args:
            number: Number to format
            precision: Number of decimal places (auto-detect if None)
            
        Returns:
            Formatted number string
        """
        if isinstance(number, int):
            precision = 0
        elif precision is None:
            # Auto-detect decimal places
            precision = len(str(number).split('.')[-1]) if '.' in str(number) else 0
        
        # Format with specified precision
        number_str = f"{float(number):.{precision}f}"
        integer_part, decimal_part = number_str.split('.')
        
        # Add thousands separators
        if len(integer_part) > 3 and self.settings.thousands_separator:
            chars = list(reversed(integer_part))
            for i in range(3, len(chars), 4):
                chars.insert(i, self.settings.thousands_separator)
            integer_part = ''.join(reversed(chars))
        
        # Combine with decimal separator
        if precision > 0:
            return f"{integer_part}{self.settings.decimal_separator}{decimal_part}"
        else:
            return integer_part
    
    def format_percentage(self, 
                         value: Union[int, float, Decimal],
                         precision: int = 1) -> str:
        """
        Format a value as percentage.
        
        Args:
            value: Value to format (0.15 = 15%)
            precision: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        percentage = float(value) * 100
        formatted = self.format(percentage, precision)
        return f"{formatted}%"

# Global formatter instances
_formatters: Dict[str, Dict[str, Any]] = {}

def get_formatter(formatter_type: str, locale_code: str = "en-US") -> Any:
    """
    Get a formatter instance for the specified type and locale.
    
    Args:
        formatter_type: Type of formatter ("currency", "datetime", "number")
        locale_code: Locale code
        
    Returns:
        Formatter instance
    """
    if locale_code not in _formatters:
        _formatters[locale_code] = {}
    
    if formatter_type not in _formatters[locale_code]:
        if formatter_type == "currency":
            _formatters[locale_code][formatter_type] = CurrencyFormatter(locale_code)
        elif formatter_type == "datetime":
            _formatters[locale_code][formatter_type] = DateTimeFormatter(locale_code)
        elif formatter_type == "number":
            _formatters[locale_code][formatter_type] = NumberFormatter(locale_code)
        else:
            raise ValueError(f"Unknown formatter type: {formatter_type}")
    
    return _formatters[locale_code][formatter_type]