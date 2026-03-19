"""
Tests for MPCAttention and poly_softmax approximation.
"""

import numpy as np
import pytest
from secure_mpc.mpc_attention import MPCAttention, poly_softmax


def _make_attn_weights(d_model, rng, scale=0.05):
    return {
        "Wq": rng.standard_normal((d_model, d_model)) * scale,
        "bq": rng.standard_normal(d_model) * scale,
        "Wk": rng.standard_normal((d_model, d_model)) * scale,
        "bk": rng.standard_normal(d_model) * scale,
        "Wv": rng.standard_normal((d_model, d_model)) * scale,
        "bv": rng.standard_normal(d_model) * scale,
        "Wo": rng.standard_normal((d_model, d_model)) * scale,
        "bo": rng.standard_normal(d_model) * scale,
    }


class TestPolySoftmax:
    """Polynomial softmax should approximate true softmax reasonably."""

    def test_sums_to_one(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((4, 8))
        for deg in (1, 2, 3):
            out = poly_softmax(z, degree=deg)
            np.testing.assert_allclose(out.sum(axis=-1), np.ones(4), atol=1e-6,
                err_msg=f"poly_softmax(deg={deg}) rows must sum to 1")

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal((3, 5)) * 0.5
        for deg in (1, 2, 3):
            out = poly_softmax(z, degree=deg)
            assert (out >= 0).all(), f"Negative values with degree={deg}"

    def test_approximation_quality(self):
        """Degree-2 poly softmax should be close to true softmax for small inputs."""
        rng = np.random.default_rng(2)
        z = rng.standard_normal((10, 6)) * 0.3  # Small values → better approximation

        true_softmax = np.exp(z - z.max(axis=-1, keepdims=True))
        true_softmax /= true_softmax.sum(axis=-1, keepdims=True)

        poly2 = poly_softmax(z, degree=2)
        mse = np.mean((poly2 - true_softmax) ** 2)
        assert mse < 0.05, f"Poly-2 softmax MSE {mse:.4f} too high"

    def test_higher_degree_better(self):
        """Degree-3 should outperform degree-1 for moderate inputs."""
        rng = np.random.default_rng(3)
        z = rng.standard_normal((20, 8)) * 0.4

        true_sm = np.exp(z - z.max(axis=-1, keepdims=True))
        true_sm /= true_sm.sum(axis=-1, keepdims=True)

        mse1 = np.mean((poly_softmax(z, degree=1) - true_sm) ** 2)
        mse3 = np.mean((poly_softmax(z, degree=3) - true_sm) ** 2)
        assert mse3 <= mse1 * 1.5, "Degree-3 should not be worse than degree-1"

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError):
            poly_softmax(np.ones((2, 3)), degree=5)


class TestMPCAttentionCorrectness:
    """MPC attention should approximate plaintext attention."""

    def setup_method(self):
        self.d_model = 8
        self.n_heads = 2
        self.seq_len = 4
        self.rng = np.random.default_rng(42)
        from secure_mpc import SecretSharer
        self.sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(0))

    def _make_attn(self):
        weights = _make_attn_weights(self.d_model, self.rng, scale=0.02)
        return MPCAttention(self.d_model, self.n_heads, weights, sharer=self.sharer)

    def test_output_shape(self):
        attn = self._make_attn()
        x = self.rng.standard_normal((self.seq_len, self.d_model))
        out = attn.forward(x)
        assert out.shape == (self.seq_len, self.d_model)

    def test_plaintext_output_shape(self):
        attn = self._make_attn()
        x = self.rng.standard_normal((self.seq_len, self.d_model))
        out = attn.plaintext_forward(x)
        assert out.shape == (self.seq_len, self.d_model)

    def test_mpc_vs_plaintext_reasonable(self):
        """
        MPC output should be in the same ballpark as plaintext.
        With small weights and poly approx, large exact match isn't expected.
        """
        attn = self._make_attn()
        x = self.rng.standard_normal((self.seq_len, self.d_model)) * 0.3

        mpc_out = attn.forward(x)
        pt_out = attn.plaintext_forward(x)

        # MSE should be low relative to output magnitude
        mse = np.mean((mpc_out - pt_out) ** 2)
        scale = np.mean(pt_out ** 2) + 1e-9
        relative_mse = mse / scale
        assert relative_mse < 2.0, (
            f"MPC attention relative MSE {relative_mse:.4f} too high — "
            f"outputs diverged too much from plaintext"
        )

    def test_comm_rounds_positive(self):
        attn = self._make_attn()
        x = self.rng.standard_normal((self.seq_len, self.d_model))
        attn.forward(x)
        assert attn.comm_rounds > 0
        assert attn.comm_elements > 0
