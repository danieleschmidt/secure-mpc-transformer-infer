"""
MPCTransformer: a small transformer built from MPC primitives.

Architecture
────────────
  Embedding   →  [TransformerBlock × n_layers]  →  Classifier head

Each TransformerBlock:
  MPCAttention + residual
  MPCLinear (FFN expand)  + ReLU approx + MPCLinear (FFN contract) + residual
  LayerNorm (public, computed after reconstruction — no MPC overhead)

Classification task
────────────────────
Sequence-level classification: mean-pool the last hidden states,
then pass through a linear head. Used for the accuracy benchmark.

Factory helper
──────────────
`MPCTransformer.random_init(...)` creates random weights for testing.
`MPCTransformer.from_plaintext_weights(...)` wraps existing weight dict.
"""

from __future__ import annotations

import numpy as np
from .secret_sharing import SecretSharer
from .mpc_linear import MPCLinear
from .mpc_attention import MPCAttention, poly_softmax


# ------------------------------------------------------------------
# ReLU approximation for MPC
# ------------------------------------------------------------------

def poly_relu(x: np.ndarray) -> np.ndarray:
    """
    Degree-2 polynomial approximation of ReLU for MPC.

    ReLU(x) ≈ 0.125 * x^2 + 0.5 * x + 0.375
    (Fitted on [-1, 1]; common in CrypTen / SiReLU literature.)

    Costs 1 Beaver round for the x^2 term.
    """
    return np.clip(0.125 * x**2 + 0.5 * x + 0.375, 0, None)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Standard layer norm — applied in plaintext after reconstruction."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ------------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------------

class MPCTransformerBlock:
    """
    One transformer block: attention + FFN, both secure.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_weights: dict[str, np.ndarray],
        ffn_weights: dict[str, np.ndarray],
        sharer: SecretSharer,
    ) -> None:
        self.attn = MPCAttention(d_model, n_heads, attn_weights, sharer=sharer)
        self.ffn1 = MPCLinear(ffn_weights["W1"], ffn_weights.get("b1"), sharer=sharer)
        self.ffn2 = MPCLinear(ffn_weights["W2"], ffn_weights.get("b2"), sharer=sharer)

        self.comm_rounds = 0
        self.comm_elements = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """MPC forward pass."""
        self.comm_rounds = 0
        self.comm_elements = 0

        # Attention + residual
        attn_out = self.attn.forward(x)
        self.comm_rounds += self.attn.comm_rounds
        self.comm_elements += self.attn.comm_elements
        x = layer_norm(x + attn_out)

        # FFN + residual
        h = self.ffn1.forward(x)
        self.comm_rounds += self.ffn1.comm_rounds
        self.comm_elements += self.ffn1.comm_elements

        h = poly_relu(h)
        self.comm_rounds += 1  # 1 Beaver round for poly_relu x^2 term

        h = self.ffn2.forward(h)
        self.comm_rounds += self.ffn2.comm_rounds
        self.comm_elements += self.ffn2.comm_elements

        x = layer_norm(x + h)
        return x

    def plaintext_forward(self, x: np.ndarray) -> np.ndarray:
        """Plaintext forward pass for accuracy comparison."""
        attn_out = self.attn.plaintext_forward(x)
        x = layer_norm(x + attn_out)

        h = self.ffn1.plaintext_forward(x)
        h = np.maximum(0, h)  # True ReLU
        h = self.ffn2.plaintext_forward(h)
        x = layer_norm(x + h)
        return x


# ------------------------------------------------------------------
# Full MPC Transformer
# ------------------------------------------------------------------

class MPCTransformer:
    """
    Small transformer for sequence classification, using MPC primitives.

    Parameters
    ----------
    d_model   : model (embedding) dimension
    n_heads   : attention heads
    d_ff      : feed-forward hidden dimension
    n_layers  : number of transformer blocks
    n_classes : output classes
    weights   : list of per-layer weight dicts (see `random_init`)
    head_weights : classifier head weights {'W': ..., 'b': ...}
    sharer    : SecretSharer instance
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        n_classes: int,
        weights: list[dict],
        head_weights: dict[str, np.ndarray],
        sharer: SecretSharer | None = None,
    ) -> None:
        self.d_model = d_model
        self.n_layers = n_layers
        self.sharer = sharer or SecretSharer()

        self.blocks = [
            MPCTransformerBlock(
                d_model, n_heads, d_ff,
                weights[i]["attn"],
                weights[i]["ffn"],
                sharer=self.sharer,
            )
            for i in range(n_layers)
        ]
        self.head = MPCLinear(
            head_weights["W"], head_weights.get("b"), sharer=self.sharer
        )

        self.comm_rounds: int = 0
        self.comm_elements: int = 0

    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        MPC inference.

        x : (seq_len, d_model) — already embedded input
        Returns: (n_classes,) logits
        """
        self.comm_rounds = 0
        self.comm_elements = 0

        for block in self.blocks:
            x = block.forward(x)
            self.comm_rounds += block.comm_rounds
            self.comm_elements += block.comm_elements

        # Mean pool → classifier head
        pooled = x.mean(axis=0, keepdims=True)  # (1, d_model)
        logits = self.head.forward(pooled)[0]    # (n_classes,)
        self.comm_rounds += self.head.comm_rounds
        self.comm_elements += self.head.comm_elements

        return logits

    def plaintext_forward(self, x: np.ndarray) -> np.ndarray:
        """Plaintext inference for accuracy comparison."""
        for block in self.blocks:
            x = block.plaintext_forward(x)
        pooled = x.mean(axis=0, keepdims=True)
        return self.head.plaintext_forward(pooled)[0]

    def predict(self, x: np.ndarray, mpc: bool = True) -> int:
        """Return the argmax class label."""
        logits = self.forward(x) if mpc else self.plaintext_forward(x)
        return int(np.argmax(logits))

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random_init(
        cls,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64,
        n_layers: int = 2,
        n_classes: int = 4,
        seed: int = 42,
        sharer: SecretSharer | None = None,
    ) -> "MPCTransformer":
        """
        Create a randomly-initialised MPCTransformer for testing/benchmarking.
        """
        rng = np.random.default_rng(seed)
        scale = 0.02  # small init

        def randn(*shape):
            return rng.standard_normal(shape) * scale

        weights = []
        for _ in range(n_layers):
            attn_w = {
                "Wq": randn(d_model, d_model), "bq": randn(d_model),
                "Wk": randn(d_model, d_model), "bk": randn(d_model),
                "Wv": randn(d_model, d_model), "bv": randn(d_model),
                "Wo": randn(d_model, d_model), "bo": randn(d_model),
            }
            ffn_w = {
                "W1": randn(d_ff, d_model), "b1": randn(d_ff),
                "W2": randn(d_model, d_ff), "b2": randn(d_model),
            }
            weights.append({"attn": attn_w, "ffn": ffn_w})

        head_w = {
            "W": randn(n_classes, d_model),
            "b": randn(n_classes),
        }

        return cls(d_model, n_heads, d_ff, n_layers, n_classes, weights, head_w, sharer=sharer)
