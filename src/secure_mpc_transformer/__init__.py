"""
Secure MPC Transformer Inference

GPU-accelerated secure multi-party computation for transformer inference.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"

from .models.secure_transformer import SecureTransformer
from .protocols.factory import ProtocolFactory
from .config import SecurityConfig

__all__ = ["SecureTransformer", "ProtocolFactory", "SecurityConfig"]