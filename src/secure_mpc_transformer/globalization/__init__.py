"""Autonomous globalization framework."""

from .autonomous_global_manager import (
    AutonomousGlobalManager,
    ComplianceFramework,
    ComplianceRule,
    GlobalConfig,
    LocalizationEntry,
    SupportedLocale,
)

__all__ = [
    "AutonomousGlobalManager",
    "GlobalConfig",
    "ComplianceFramework",
    "SupportedLocale",
    "ComplianceRule",
    "LocalizationEntry"
]
