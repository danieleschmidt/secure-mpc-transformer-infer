"""
Basic imports for Generation 1 - Works with minimal dependencies.
"""

__version__ = "0.3.0-gen1"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"

# Basic components that work without torch/transformers
from .config import SecurityConfig, get_default_config
from .models.basic_transformer import (
    BasicSecureTransformer,
    BasicTransformerConfig,
    BasicTokenizer,
    BasicMPCProtocol
)

__all__ = [
    "SecurityConfig",
    "get_default_config", 
    "BasicSecureTransformer",
    "BasicTransformerConfig",
    "BasicTokenizer",
    "BasicMPCProtocol"
]