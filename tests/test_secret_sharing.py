"""
Tests for SecretSharer — correctness and privacy properties.
"""

import numpy as np
import pytest
from secure_mpc import SecretSharer


class TestSecretSharingCorrectness:
    """Share/reconstruct should return the original value."""

    def setup_method(self):
        self.sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(0))

    def test_scalar_roundtrip(self):
        x = 3.14159
        shares = self.sharer.share(x)
        assert len(shares) == 2
        result = self.sharer.reconstruct(shares)
        assert abs(float(result) - x) < 1e-2  # fixed-point tolerance

    def test_array_roundtrip(self):
        x = np.array([1.0, -2.5, 0.5, 3.75])
        shares = self.sharer.share(x)
        result = self.sharer.reconstruct(shares)
        np.testing.assert_allclose(result, x, atol=1e-2)

    def test_matrix_roundtrip(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((4, 8)) * 2.0
        shares = self.sharer.share(x)
        result = self.sharer.reconstruct(shares)
        np.testing.assert_allclose(result, x, atol=1e-2)

    def test_zero_roundtrip(self):
        x = np.zeros((3, 3))
        shares = self.sharer.share(x)
        result = self.sharer.reconstruct(shares)
        np.testing.assert_allclose(result, x, atol=1e-2)

    def test_negative_values(self):
        x = np.array([-1.0, -0.5, -3.75])
        shares = self.sharer.share(x)
        result = self.sharer.reconstruct(shares)
        np.testing.assert_allclose(result, x, atol=1e-2)

    def test_three_party_sharing(self):
        sharer3 = SecretSharer(n_parties=3, rng=np.random.default_rng(1))
        x = np.array([1.5, -0.25, 2.0])
        shares = sharer3.share(x)
        assert len(shares) == 3
        result = sharer3.reconstruct(shares)
        np.testing.assert_allclose(result, x, atol=1e-2)

    def test_wrong_share_count_raises(self):
        x = np.array([1.0, 2.0])
        shares = self.sharer.share(x)
        with pytest.raises(ValueError):
            self.sharer.reconstruct(shares[:1])  # Missing a share


class TestSecretSharingPrivacy:
    """Each individual share should be statistically independent of x."""

    def setup_method(self):
        self.sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(99))

    def test_share0_looks_random(self):
        """
        If we fix x but generate many share pairs, share[0] should be
        approximately uniform over Z_p (information-theoretic security).
        We check that it doesn't cluster near x.
        """
        x = 1.0
        n = 200
        share0_values = []
        for _ in range(n):
            shares = self.sharer.share(x)
            share0_values.append(int(shares[0].flat[0]))

        # Share[0] should span a large range (it's uniform in [0, p))
        r = max(share0_values) - min(share0_values)
        p = self.sharer.p
        # With 200 samples from [0, p), range should cover >> 1% of p
        assert r > p * 0.01, (
            f"Share[0] range {r} too small — shares not random enough."
        )

    def test_single_share_does_not_reveal_x(self):
        """
        share[0] alone should give no information about x.
        Two different x values should have indistinguishable share[0] distributions.
        """
        rng_a = np.random.default_rng(10)
        rng_b = np.random.default_rng(11)
        sharer_a = SecretSharer(n_parties=2, rng=rng_a)
        sharer_b = SecretSharer(n_parties=2, rng=rng_b)

        x1, x2 = 0.1, 9.9
        s1 = [int(sharer_a.share(x1)[0].flat[0]) for _ in range(50)]
        s2 = [int(sharer_b.share(x2)[0].flat[0]) for _ in range(50)]

        # Both sets of shares should span similar ranges
        # (no statistical test can distinguish which came from x1 vs x2)
        range1 = max(s1) - min(s1)
        range2 = max(s2) - min(s2)
        p = sharer_a.p
        # Both should cover a large portion of Z_p
        assert range1 > p * 0.001
        assert range2 > p * 0.001

    def test_shares_sum_to_x_mod_p(self):
        """Verify additive invariant: sum of shares ≡ x (mod p)."""
        sharer = SecretSharer(n_parties=3, rng=np.random.default_rng(7))
        x = np.array([2.5, -1.25])
        shares = sharer.share(x)
        p = sharer.p
        scale = sharer.scale

        for i in range(len(x)):
            share_sum = sum(int(s.flat[i]) for s in shares) % p
            x_int = round(x[i] * scale) % p
            assert share_sum == x_int, (
                f"Sum of shares {share_sum} != x_int {x_int} for x={x[i]}"
            )


class TestSecretSharingArithmetic:
    """Test arithmetic operations on shares."""

    def setup_method(self):
        self.sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(5))

    def test_add_shares(self):
        """[x] + [y] should reconstruct to x + y."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, -1.0, 0.25])
        sx = self.sharer.share(x)
        sy = self.sharer.share(y)
        sz = self.sharer.add_shares(sx, sy)
        result = self.sharer.reconstruct(sz)
        np.testing.assert_allclose(result, x + y, atol=1e-2)

    def test_add_shares_multiple(self):
        """Chained addition: [a] + [b] + [c] ≈ a + b + c."""
        rng = np.random.default_rng(20)
        a = rng.standard_normal(5)
        b = rng.standard_normal(5)
        c = rng.standard_normal(5)

        sa = self.sharer.share(a)
        sb = self.sharer.share(b)
        sc = self.sharer.share(c)

        sabc = self.sharer.add_shares(self.sharer.add_shares(sa, sb), sc)
        result = self.sharer.reconstruct(sabc)
        np.testing.assert_allclose(result, a + b + c, atol=1e-2)
