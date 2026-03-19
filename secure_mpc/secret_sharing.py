"""
Additive secret sharing over integers mod p.

Security model: honest-but-curious (semi-honest) 2-party.

A value x is split into n shares s_0, s_1, ..., s_{n-1} such that:
    sum(s_i) ≡ x  (mod p)

Any single share leaks nothing about x (information-theoretically
secure when shares are chosen uniformly at random).

We work with fixed-point integers: floats are scaled by SCALE, operated
on as integers mod p, then unscaled at reconstruction.

Implementation note: we use numpy object arrays to hold arbitrary-precision
Python ints, enabling exact arithmetic in Z_p without integer overflow.
"""

from __future__ import annotations

import numpy as np

# Mersenne prime 2^61 - 1 as the ring modulus
DEFAULT_P = (1 << 61) - 1

# Fixed-point scale: 1.0 → SCALE integer units (~4 decimal digits precision)
DEFAULT_SCALE = 1 << 16  # 65536


class SecretSharer:
    """
    Additive secret sharing over Z_p.

    Parameters
    ----------
    n_parties : int
        Number of shares to produce (default 2 for 2-party MPC).
    p : int
        Prime modulus for the ring Z_p.
    scale : int
        Fixed-point scale factor.
    rng : np.random.Generator | None
        Random number generator (for reproducibility in tests).
    """

    def __init__(
        self,
        n_parties: int = 2,
        p: int = DEFAULT_P,
        scale: int = DEFAULT_SCALE,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_parties < 2:
            raise ValueError("Need at least 2 parties for secret sharing.")
        self.n_parties = n_parties
        self.p = p
        self.scale = scale
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Internal helpers — all work with Python int object arrays
    # ------------------------------------------------------------------

    def _to_ring(self, x: np.ndarray) -> np.ndarray:
        """
        Convert float ndarray → Z_p integer object array.
        shape is preserved; each element is int in [0, p).
        """
        p, scale = self.p, self.scale
        # Flatten, convert, reshape — avoids npiter issues with object arrays
        flat = np.round(x.astype(float).ravel() * scale).astype(int)
        out = np.empty(flat.shape, dtype=object)
        for i, v in enumerate(flat):
            out[i] = int(v) % p
        return out.reshape(x.shape)

    def _rand_ring(self, shape: tuple) -> np.ndarray:
        """Generate object array of uniform random ints in Z_p."""
        p = self.p
        # Use two 32-bit random ints to cover the 61-bit range
        # Maximum needed: ~2^61, so two 31-bit parts suffice
        hi = self.rng.integers(0, 1 << 31, size=shape, dtype=np.int64)
        lo = self.rng.integers(0, 1 << 30, size=shape, dtype=np.int64)
        hi_flat = hi.ravel()
        lo_flat = lo.ravel()
        out = np.empty(hi_flat.shape, dtype=object)
        for i in range(len(hi_flat)):
            out[i] = (int(hi_flat[i]) << 30 | int(lo_flat[i])) % p
        return out.reshape(shape)

    def _ring_to_float(self, arr: np.ndarray) -> np.ndarray:
        """Convert Z_p integer object array → float, interpreting > p//2 as negative."""
        p, half_p, scale = self.p, self.p // 2, self.scale
        flat = arr.ravel()
        out = np.empty(len(flat), dtype=float)
        for i, v in enumerate(flat):
            iv = int(v) % p
            if iv > half_p:
                iv -= p
            out[i] = iv / scale
        return out.reshape(arr.shape)

    def _mod_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise (a + b) % p for object arrays of same shape."""
        p = self.p
        flat_a = a.ravel()
        flat_b = b.ravel()
        out = np.empty(len(flat_a), dtype=object)
        for i in range(len(flat_a)):
            out[i] = (int(flat_a[i]) + int(flat_b[i])) % p
        return out.reshape(a.shape)

    def _mod_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise (a - b) % p for object arrays of same shape."""
        p = self.p
        flat_a = a.ravel()
        flat_b = b.ravel()
        out = np.empty(len(flat_a), dtype=object)
        for i in range(len(flat_a)):
            out[i] = (int(flat_a[i]) - int(flat_b[i])) % p
        return out.reshape(a.shape)

    # ------------------------------------------------------------------
    # Core share / reconstruct
    # ------------------------------------------------------------------

    def share(self, x: np.ndarray | float | int) -> list[np.ndarray]:
        """
        Secret-share x into n_parties additive shares.

        Returns a list of integer object-arrays. All arithmetic mod p.
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        x_int = self._to_ring(x_arr)       # Z_p elements
        shape = x_int.shape

        # n-1 uniform random shares
        random_shares = [self._rand_ring(shape) for _ in range(self.n_parties - 1)]

        # Last share = x - sum(random_shares)  mod p
        s_sum = np.zeros(shape, dtype=object)  # starts as 0-filled object array
        for i in range(s_sum.size):
            s_sum.flat[i] = 0
        for s in random_shares:
            s_sum = self._mod_add(s_sum, s)

        last_share = self._mod_sub(x_int, s_sum)

        return random_shares + [last_share]

    def reconstruct(self, shares: list[np.ndarray]) -> np.ndarray:
        """Reconstruct the original float value from all shares."""
        if len(shares) != self.n_parties:
            raise ValueError(f"Expected {self.n_parties} shares, got {len(shares)}.")

        total = np.empty(shares[0].shape, dtype=object)
        for i in range(total.size):
            total.flat[i] = 0

        for s in shares:
            total = self._mod_add(total, s)

        return self._ring_to_float(total)

    # ------------------------------------------------------------------
    # Arithmetic on shares
    # ------------------------------------------------------------------

    def add_shares(
        self, shares_a: list[np.ndarray], shares_b: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Add two secret-shared values (no communication, 0 rounds).
        Each party adds their own shares locally.
        """
        return [self._mod_add(sa, sb) for sa, sb in zip(shares_a, shares_b)]

    def negate_shares(self, shares: list[np.ndarray]) -> list[np.ndarray]:
        """Negate a secret-shared value (no communication)."""
        p = self.p
        result = []
        for s in shares:
            flat = s.ravel()
            neg = np.empty(len(flat), dtype=object)
            for i, v in enumerate(flat):
                neg[i] = (p - int(v)) % p
            result.append(neg.reshape(s.shape))
        return result

    # ------------------------------------------------------------------
    # Ring matrix multiply (used by MPCLinear)
    # ------------------------------------------------------------------

    def ring_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiply in Z_p with fixed-point truncation.

        a : (m, k) object array (ring elements, fixed-point * scale)
        b : (k, n) object array (ring elements, fixed-point * scale)

        Result = (a @ b) // scale  (mod p)

        Strategy: convert ring elements to signed integers (interpreting
        values > p//2 as negative) before multiplying, then re-encode.
        This avoids modular wrap-around errors in the dot product accumulation.
        """
        p = self.p
        half_p = p // 2
        scale = self.scale
        m, k = a.shape
        _, n = b.shape

        def to_signed(v: int) -> int:
            v = int(v) % p
            return v - p if v > half_p else v

        result = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                s = 0
                for l in range(k):
                    s += to_signed(a[i, l]) * to_signed(b[l, j])
                result[i, j] = (s // scale) % p
        return result
