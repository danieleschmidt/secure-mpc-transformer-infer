"""Basic error handling for Generation 1 - Simplified version with minimal dependencies."""

import logging
import time
from typing import Any, Dict

# Simple error types for Generation 1
class SecureMPCError(Exception):
    """Basic MPC error for Generation 1."""
    pass

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file if log_file else None
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)