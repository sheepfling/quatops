"""Concrete products in familiar scalar/vector notation for both algebras."""

import numpy as np

from quatops.hamilton import scalar as hamilton
from quatops.jpl import scalar as jpl


class TestConventionProducts:
    def test_hamilton_product(self):
        # (1 + 2i + 3j + 4k) * (5 + 6i + 7j + 8k)
        lhs = hamilton.from_wxyz((1, 2, 3, 4))
        rhs = hamilton.from_wxyz((5, 6, 7, 8))

        # Hamilton: i*j = k, j*k = i, k*i = j.
        np.testing.assert_allclose(
            hamilton.to_wxyz(hamilton.multiply(lhs, rhs)),
            (-60, 12, 30, 24),
            rtol=0,
            atol=0,
        )

    def test_jpl_product(self):
        # The same written components, interpreted in JPL's negative-cross algebra.
        lhs = jpl.from_wxyz((1, 2, 3, 4))
        rhs = jpl.from_wxyz((5, 6, 7, 8))

        # JPL: i*j = -k, j*k = -i, k*i = -j.
        np.testing.assert_allclose(
            jpl.to_wxyz(jpl.multiply(lhs, rhs)),
            (-60, 20, 14, 32),
            rtol=0,
            atol=0,
        )
