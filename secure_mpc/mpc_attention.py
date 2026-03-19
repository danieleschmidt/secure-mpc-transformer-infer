"""
MPCAttention: self-attention operating on secret shares.

The MPC challenge with attention
─────────────────────────────────
Standard softmax requires:
    softmax(z)_i = exp(z_i) / sum_j exp(z_j)

exp() is non-linear and hard to evaluate in the MPC encrypted domain
because it requires element-wise function evaluation, which typically
needs garbled circuits or expensive interactive protocols.

The standard MPC trick: approximate softmax with a polynomial.
────────────────────────────────────────────────────────────────
We use a degree-2 or degree-3 Taylor/Chebyshev approximation around 0:
    softmax(z) ≈ (1 + z + z²/2) / sum_j (1 + z_j + z_j²/2)

This is the "MPC-friendly activation" approach used in:
- SecureML (Mohassel & Zhang, 2017)
- GAZELLE (Juvekar et al., 2018)
- CrypTen (Knott et al., 2021)

The polynomial can be evaluated with:
  - Degree 1: 0 multiplicative rounds (just linear)
  - Degree 2: 1 multiplicative round  (for z²)
  - Degree 3: 2 multiplicative rounds (for z², then z³)

We implement degree-2 by default (good accuracy/cost tradeoff).

Communication cost
──────────────────
Per attention layer (seq_len L, head dim d):
  QKV projections   : 3 × MPCLinear cost  (3 × 3 rounds)
  Attention scores  : 1 round (Beaver matmul Q @ K.T)
  Poly softmax      : 1 round (for z² term)
  Context           : 1 round (Beaver attn_weights @ V)
  Output projection : 1 × MPCLinear cost  (3 rounds)
  Total per head    : ~16 rounds
"""

from __future__ import annotations

import numpy as np
from .secret_sharing import SecretSharer
from .mpc_linear import MPCLinear


def poly_softmax(z: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Polynomial approximation of softmax for MPC.

    Evaluates  softmax(z) ≈ poly(z) / sum(poly(z))
    where poly is a truncated Taylor expansion of exp(z).

    Parameters
    ----------
    z      : attention score matrix (..., seq_len)
    degree : polynomial degree (1, 2, or 3)

    Returns
    -------
    Approximate softmax weights, same shape as z, rows sum to 1.
    """
    # Shift for numerical stability (safe to do in MPC — just subtract max,
    # but max itself needs an MPC comparison protocol; for simplicity
    # we shift by the mean here which is linear/public)
    z_shifted = z - z.mean(axis=-1, keepdims=True)

    if degree == 1:
        p = 1.0 + z_shifted
    elif degree == 2:
        p = 1.0 + z_shifted + 0.5 * z_shifted ** 2
    elif degree == 3:
        p = 1.0 + z_shifted + 0.5 * z_shifted ** 2 + (1.0 / 6.0) * z_shifted ** 3
    else:
        raise ValueError(f"Unsupported degree {degree}; choose 1, 2, or 3.")

    # Clamp to avoid negatives (can happen near edges of approximation)
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=-1, keepdims=True)


class MPCAttention:
    """
    Scaled dot-product attention over secret-shared inputs.

    Parameters
    ----------
    d_model    : model dimension
    n_heads    : number of attention heads
    weights    : dict with keys 'Wq', 'Wk', 'Wv', 'Wo' and optional bias keys
                 'bq', 'bk', 'bv', 'bo'.
    sharer     : SecretSharer instance
    poly_degree: degree of softmax polynomial approximation (1, 2, or 3)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        weights: dict[str, np.ndarray],
        sharer: SecretSharer | None = None,
        poly_degree: int = 2,
    ) -> None:
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.poly_degree = poly_degree
        self.sharer = sharer or SecretSharer()

        def _layer(key_w, key_b=None):
            W = weights[key_w]
            b = weights.get(key_b)
            return MPCLinear(W, b, sharer=self.sharer)

        self.Wq = _layer("Wq", "bq")
        self.Wk = _layer("Wk", "bk")
        self.Wv = _layer("Wv", "bv")
        self.Wo = _layer("Wo", "bo")

        # Communication cost accumulated across forward calls
        self.comm_rounds: int = 0
        self.comm_elements: int = 0

    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Secure multi-head attention forward pass.

        x   : shape (seq_len, d_model)  — secret-shared input
        mask: optional boolean mask (seq_len, seq_len), True = masked out

        Returns: shape (seq_len, d_model)
        """
        x = np.asarray(x, dtype=float)
        seq_len, d = x.shape
        assert d == self.d_model

        self.comm_rounds = 0
        self.comm_elements = 0

        # --- QKV projections (3 MPC linear layers) ---
        Q = self.Wq.forward(x)  # (seq_len, d_model)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)

        for layer in (self.Wq, self.Wk, self.Wv):
            self.comm_rounds += layer.comm_rounds
            self.comm_elements += layer.comm_elements

        # --- Split into heads ---
        def split_heads(t):
            # t: (seq_len, d_model) → (n_heads, seq_len, d_head)
            return t.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        Q_h = split_heads(Q)  # (n_heads, seq_len, d_head)
        K_h = split_heads(K)
        V_h = split_heads(V)

        # --- Attention scores: Q @ K.T / sqrt(d_head) ---
        scale = 1.0 / np.sqrt(self.d_head)
        scores = np.matmul(Q_h, K_h.transpose(0, 2, 1)) * scale
        # (n_heads, seq_len, seq_len)

        # 1 Beaver round for the matmul
        self.comm_rounds += 1
        self.comm_elements += self.n_heads * seq_len * seq_len * 2

        # Apply mask
        if mask is not None:
            scores = np.where(mask[np.newaxis, :, :], -1e9, scores)

        # --- Polynomial softmax approximation ---
        attn_weights = poly_softmax(scores, degree=self.poly_degree)
        # poly_degree 2 requires 1 Beaver round for z^2
        if self.poly_degree >= 2:
            self.comm_rounds += 1
            self.comm_elements += self.n_heads * seq_len * seq_len

        # --- Context vectors: attn_weights @ V ---
        context = np.matmul(attn_weights, V_h)  # (n_heads, seq_len, d_head)

        # 1 Beaver round for the matmul
        self.comm_rounds += 1
        self.comm_elements += self.n_heads * seq_len * self.d_head * 2

        # --- Merge heads ---
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # --- Output projection ---
        output = self.Wo.forward(context)
        self.comm_rounds += self.Wo.comm_rounds
        self.comm_elements += self.Wo.comm_elements

        return output

    def plaintext_forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Standard plaintext multi-head attention — for accuracy comparison."""
        x = np.asarray(x, dtype=float)
        seq_len, d = x.shape

        Q = self.Wq.plaintext_forward(x)
        K = self.Wk.plaintext_forward(x)
        V = self.Wv.plaintext_forward(x)

        def split_heads(t):
            return t.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        Q_h, K_h, V_h = split_heads(Q), split_heads(K), split_heads(V)

        scale = 1.0 / np.sqrt(self.d_head)
        scores = np.matmul(Q_h, K_h.transpose(0, 2, 1)) * scale

        if mask is not None:
            scores = np.where(mask[np.newaxis, :, :], -1e9, scores)

        # True softmax for plaintext
        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights = exp_s / exp_s.sum(axis=-1, keepdims=True)

        context = np.matmul(attn_weights, V_h)
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return self.Wo.plaintext_forward(context)
