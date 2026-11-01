"""Regress a scalar/NumPy difference in diagonal rotation coefficients."""

import math
import unittest
from fractions import Fraction

from quatops import scalar

try:
    import numpy as np

    from quatops.numpy.hamilton import wxyz as qnp
except ImportError:
    np = None
    qnp = None


@unittest.skipIf(np is None, 'NumPy is not installed')
class RotationCoefficientOrderTests(unittest.TestCase):
    def test_near_cancelling_diagonal_coefficients(self):
        # These binary inputs have exact, nonzero tiny diagonal coefficients.
        # Adding the tiny square to one before subtracting one erases it.
        epsilon = 1e-8
        q = np.array([[1., 1., epsilon, 0.], [1., 1., 0., epsilon]])
        v = np.array([[0., 1e308, 0.], [0., 0., 1e308]])
        q_before, v_before = q.copy(), v.copy()
        q.flags.writeable = False
        v.flags.writeable = False
        f = Fraction.from_float(epsilon)
        expected = float(Fraction.from_float(1e308) * f * f / (2 + f*f))
        previous = np.seterr(all='raise')
        try:
            actual = qnp.rotate(q, v)
            for i, axis in ((0, 1), (1, 2)):
                with self.subTest(i=i):
                    self.assertTrue(math.isclose(actual[i, axis], expected, rel_tol=1e-15))
                    self.assertTrue(math.isclose(actual[i, axis], scalar.rotate(q[i], v[i])[axis], rel_tol=1e-15))
            self.assertEqual(np.geterr(), {'divide': 'raise', 'over': 'raise',
                                         'under': 'raise', 'invalid': 'raise'})
            np.testing.assert_array_equal(q, q_before)
            np.testing.assert_array_equal(v, v_before)
        finally:
            np.seterr(**previous)


if __name__ == '__main__':
    unittest.main()
