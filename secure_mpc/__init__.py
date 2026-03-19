"""
secure_mpc — MPC primitives for private transformer inference.

Implements additive secret sharing over integers mod p to demonstrate
how transformer inference can be performed without revealing inputs to
the model server (2-party honest-but-curious model).
"""

from .secret_sharing import SecretSharer
from .mpc_linear import MPCLinear
from .mpc_attention import MPCAttention
from .mpc_transformer import MPCTransformer
from .comm_cost import CommCostEstimator

__all__ = [
    "SecretSharer",
    "MPCLinear",
    "MPCAttention",
    "MPCTransformer",
    "CommCostEstimator",
]
