"""
Tests for MPCTransformer — end-to-end forward pass and comm cost.
"""

import numpy as np
import pytest
from secure_mpc import MPCTransformer, SecretSharer, CommCostEstimator


class TestMPCTransformer:

    def setup_method(self):
        self.rng = np.random.default_rng(77)
        self.sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(0))
        self.model = MPCTransformer.random_init(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_layers=1,
            n_classes=3,
            seed=42,
            sharer=self.sharer,
        )

    def test_forward_shape(self):
        x = self.rng.standard_normal((4, 8))
        logits = self.model.forward(x)
        assert logits.shape == (3,), f"Expected (3,), got {logits.shape}"

    def test_plaintext_forward_shape(self):
        x = self.rng.standard_normal((4, 8))
        logits = self.model.plaintext_forward(x)
        assert logits.shape == (3,)

    def test_predict_returns_valid_class(self):
        x = self.rng.standard_normal((6, 8))
        pred = self.model.predict(x, mpc=True)
        assert 0 <= pred < 3

    def test_predict_plaintext_returns_valid_class(self):
        x = self.rng.standard_normal((6, 8))
        pred = self.model.predict(x, mpc=False)
        assert 0 <= pred < 3

    def test_comm_rounds_tracked(self):
        x = self.rng.standard_normal((4, 8))
        self.model.forward(x)
        assert self.model.comm_rounds > 0
        assert self.model.comm_elements > 0

    def test_deterministic_plaintext(self):
        """Same input → same plaintext output (no randomness in plaintext)."""
        x = self.rng.standard_normal((4, 8))
        out1 = self.model.plaintext_forward(x)
        out2 = self.model.plaintext_forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_mpc_plaintext_agreement_rate(self):
        """Over many samples, MPC and plaintext should agree on class often."""
        model = MPCTransformer.random_init(
            d_model=8, n_heads=2, d_ff=16, n_layers=1, n_classes=4, seed=7,
            sharer=SecretSharer(n_parties=2, rng=np.random.default_rng(1)),
        )
        rng = np.random.default_rng(88)
        n = 20
        agree = 0
        for _ in range(n):
            x = rng.standard_normal((4, 8)) * 0.2
            if model.predict(x, mpc=True) == model.predict(x, mpc=False):
                agree += 1
        rate = agree / n
        # With poly approximations, agreement isn't guaranteed to be 100%,
        # but it should be reasonably high for small inputs
        assert rate >= 0.5, f"MPC/plaintext agreement rate {rate:.2f} too low"


class TestCommCostEstimator:

    def test_total_rounds_positive(self):
        est = CommCostEstimator(d_model=16, n_heads=2, d_ff=32, n_layers=2)
        t = est.total_cost()
        assert t.rounds > 0
        assert t.elements > 0

    def test_more_layers_more_cost(self):
        est2 = CommCostEstimator(d_model=16, n_heads=2, d_ff=32, n_layers=2)
        est4 = CommCostEstimator(d_model=16, n_heads=2, d_ff=32, n_layers=4)
        assert est4.total_cost().rounds > est2.total_cost().rounds
        assert est4.total_cost().elements > est2.total_cost().elements

    def test_higher_poly_deg_more_cost(self):
        est1 = CommCostEstimator(poly_deg=1)
        est2 = CommCostEstimator(poly_deg=2)
        est3 = CommCostEstimator(poly_deg=3)
        assert est2.total_cost().rounds >= est1.total_cost().rounds
        assert est3.total_cost().rounds >= est2.total_cost().rounds

    def test_bandwidth_positive(self):
        est = CommCostEstimator()
        assert est.bandwidth_bytes() > 0

    def test_report_string(self):
        est = CommCostEstimator()
        report = est.report()
        assert "TOTAL" in report
        assert "rounds=" in report
        assert "elements=" in report
