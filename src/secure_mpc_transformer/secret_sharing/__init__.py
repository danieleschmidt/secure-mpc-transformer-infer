"""
Secret sharing components for MPC protocols.
"""

from .engine import SecretSharingEngine
from .schemes import ShamirSecretSharing, AdditiveSharing, ReplicatedSharing

__all__ = ["SecretSharingEngine", "ShamirSecretSharing", "AdditiveSharing", "ReplicatedSharing"]