"""
Tests for MPCLinear — correctness vs plaintext, communication cost.
"""

import numpy as np
import pytest
from secure_mpc import MPCLinear, SecretSharer


class TestMPCLinearCorrectness:
    """MPC linear should match plaintext linear (within fixed-point tolerance)."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(0))

    def _make_layer(self, in_f, out_f, bias=True):
        W = self.rng.standard_normal((out_f, in_f)) * 0.1
        b = self.rng.standard_normal(out_f) * 0.1 if bias else None
        return MPCLinear(W, b, sharer=self.sharer), W, b

    def test_small_vector(self):
        layer, W, b = self._make_layer(4, 8)
        x = self.rng.standard_normal(4)

        pt = layer.plaintext_forward(x)
        mpc = layer.forward(x)

        np.testing.assert_allclose(mpc, pt, atol=1e-2,
            err_msg="MPC output should match plaintext linear transform")

    def test_batch_input(self):
        layer, _, _ = self._make_layer(8, 16)
        x = self.rng.standard_normal((5, 8))

        pt = layer.plaintext_forward(x)
        mpc = layer.forward(x)

        np.testing.assert_allclose(mpc, pt, atol=1e-2)

    def test_no_bias(self):
        layer, _, _ = self._make_layer(6, 4, bias=False)
        x = self.rng.standard_normal((3, 6))

        pt = layer.plaintext_forward(x)
        mpc = layer.forward(x)

        np.testing.assert_allclose(mpc, pt, atol=1e-2)

    def test_identity_weights(self):
        """W = identity, b = 0: output should equal input."""
        n = 4
        layer = MPCLinear(np.eye(n), np.zeros(n), sharer=self.sharer)
        x = np.array([1.0, 2.0, -1.0, 0.5])

        mpc = layer.forward(x)
        np.testing.assert_allclose(mpc, x, atol=1e-2)

    def test_zero_weights(self):
        """W = 0, b = 0: output should be near zero."""
        layer = MPCLinear(np.zeros((4, 4)), np.zeros(4), sharer=self.sharer)
        x = self.rng.standard_normal(4)
        mpc = layer.forward(x)
        np.testing.assert_allclose(mpc, np.zeros(4), atol=1e-2)

    def test_larger_layer(self):
        """Test a realistic-size layer."""
        layer, _, _ = self._make_layer(64, 128)
        x = self.rng.standard_normal((4, 64)) * 0.5

        pt = layer.plaintext_forward(x)
        mpc = layer.forward(x)

        np.testing.assert_allclose(mpc, pt, atol=0.05,
            err_msg="Larger layer should still be close (higher fixed-point error is ok)")


class TestMPCLinearCommCost:
    """Verify communication round counting."""

    def test_rounds_counted(self):
        rng = np.random.default_rng(1)
        sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(2))
        layer = MPCLinear(
            rng.standard_normal((8, 4)) * 0.1,
            rng.standard_normal(8) * 0.1,
            sharer=sharer,
        )
        x = rng.standard_normal((2, 4))
        layer.forward(x)

        # Should use exactly 3 rounds: share input, Beaver matmul, reconstruct
        assert layer.comm_rounds == 3

    def test_elements_positive(self):
        rng = np.random.default_rng(3)
        sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(4))
        layer = MPCLinear(rng.standard_normal((16, 8)) * 0.1, sharer=sharer)
        layer.forward(rng.standard_normal((3, 8)))
        assert layer.comm_elements > 0
