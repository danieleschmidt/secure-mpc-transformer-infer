"""
Communication cost estimator for MPC transformer inference.

In MPC, communication cost is measured in:
  - Rounds: sequential communication steps (limits latency)
  - Field elements: total data transferred across parties (limits bandwidth)

Formula derivations
───────────────────
For a 2-party honest-but-curious model with additive sharing over Z_p:

Linear layer (in_f → out_f, batch size B):
  - Rounds: 3 (share input, Beaver matmul, reconstruct output)
  - Elements: B*in_f + out_f*in_f + B*out_f  (share + Beaver mask + result)

Attention (seq_len L, d_model D, n_heads H):
  - 3 QKV projections  → 3 × linear cost(L, D → D)
  - Score matmul       → 1 round, H*L*L elements
  - Poly softmax(deg2) → 1 round, H*L*L elements
  - Context matmul     → 1 round, H*L*D elements
  - Output projection  → 1 × linear cost(L, D → D)
  Total: ~16 rounds per block

FFN (d_model → d_ff → d_model, seq_len L):
  - 2 linear layers  → 2 × linear cost
  - poly_relu        → 1 round, L*d_ff elements

LayerNorm: 0 rounds (computed locally after reconstruction)

Full transformer (n_layers N, classifier head):
  - N × (attention + FFN) + 1 × head linear
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CostBreakdown:
    """Communication cost breakdown for one component."""
    name: str
    rounds: int
    elements: int

    def __str__(self) -> str:
        return f"{self.name:30s}  rounds={self.rounds:4d}  elements={self.elements:>10,}"


class CommCostEstimator:
    """
    Analytical communication cost estimator.

    Computes rounds and field-element transfers without running the model.

    Parameters
    ----------
    d_model   : model dimension
    n_heads   : attention heads
    d_ff      : FFN hidden dimension
    n_layers  : transformer blocks
    seq_len   : input sequence length
    n_classes : output classes
    poly_deg  : softmax polynomial degree (1, 2, or 3)
    field_bits: bits per field element (default 64, for Z_{2^61-1})
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64,
        n_layers: int = 2,
        seq_len: int = 16,
        n_classes: int = 4,
        poly_deg: int = 2,
        field_bits: int = 64,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.poly_deg = poly_deg
        self.field_bits = field_bits

    def linear_cost(self, batch: int, in_f: int, out_f: int) -> tuple[int, int]:
        """Rounds and elements for one linear layer."""
        rounds = 3
        elements = batch * in_f + out_f * in_f + batch * out_f
        return rounds, elements

    def attention_cost(self) -> CostBreakdown:
        """Cost for one MPCAttention layer."""
        L, D, H = self.seq_len, self.d_model, self.n_heads
        rounds, elements = 0, 0

        # 3 QKV + 1 output projection
        for _ in range(4):
            r, e = self.linear_cost(L, D, D)
            rounds += r
            elements += e

        # Score matmul Q @ K.T
        rounds += 1
        elements += H * L * L * 2  # masked values

        # Poly softmax
        if self.poly_deg >= 2:
            rounds += 1
            elements += H * L * L

        # Context matmul attn @ V
        rounds += 1
        elements += H * L * (D // H) * 2

        return CostBreakdown("Attention", rounds, elements)

    def ffn_cost(self) -> CostBreakdown:
        """Cost for one FFN block."""
        L, D, F = self.seq_len, self.d_model, self.d_ff
        rounds, elements = 0, 0

        # FFN linear 1
        r, e = self.linear_cost(L, D, F)
        rounds += r
        elements += e

        # poly_relu (degree 2 → 1 Beaver round)
        rounds += 1
        elements += L * F

        # FFN linear 2
        r, e = self.linear_cost(L, F, D)
        rounds += r
        elements += e

        return CostBreakdown("FFN", rounds, elements)

    def block_cost(self) -> CostBreakdown:
        """Cost for one transformer block (attention + FFN)."""
        a = self.attention_cost()
        f = self.ffn_cost()
        return CostBreakdown("TransformerBlock", a.rounds + f.rounds, a.elements + f.elements)

    def head_cost(self) -> CostBreakdown:
        """Cost for the classifier head."""
        r, e = self.linear_cost(1, self.d_model, self.n_classes)
        return CostBreakdown("ClassifierHead", r, e)

    def total_cost(self) -> CostBreakdown:
        """Total communication cost for full inference."""
        bc = self.block_cost()
        hc = self.head_cost()
        total_rounds = bc.rounds * self.n_layers + hc.rounds
        total_elements = bc.elements * self.n_layers + hc.elements
        return CostBreakdown("TOTAL", total_rounds, total_elements)

    def bandwidth_bytes(self) -> float:
        """Estimated total bytes transferred (both parties combined)."""
        tc = self.total_cost()
        return tc.elements * self.field_bits / 8

    def report(self) -> str:
        """Human-readable cost report."""
        lines = [
            "=" * 60,
            f"  MPC Transformer Communication Cost Estimate",
            "=" * 60,
            f"  Config: d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_ff={self.d_ff}, n_layers={self.n_layers}",
            f"          seq_len={self.seq_len}, n_classes={self.n_classes}, "
            f"poly_deg={self.poly_deg}",
            "-" * 60,
        ]
        a = self.attention_cost()
        f = self.ffn_cost()
        b = self.block_cost()
        h = self.head_cost()
        t = self.total_cost()

        lines += [
            f"  Per block:",
            f"    {a}",
            f"    {f}",
            f"    {b}",
            f"",
            f"  n_layers={self.n_layers} × block + head:",
            f"    {h}",
            f"    {t}",
            f"",
            f"  Bandwidth: {self.bandwidth_bytes() / 1024:.1f} KB  "
            f"({self.field_bits}-bit field elements)",
            "=" * 60,
        ]
        return "\n".join(lines)
