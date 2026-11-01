"""Mixed-scale rotations checked against exact rational arithmetic."""

import math
import unittest
from fractions import Fraction

from quatops import dataclass, namedtuple, scalar

try:
    import numpy as np

    from quatops import numpy as array
except ImportError:
    np = None
    array = None


def exact_rotate(q, v):
    """Return correctly rounded float components of q*(0,v)*q^-1.

    Fraction.from_float preserves the actual binary inputs, including subnormals.
    """
    w, x, y, z = (Fraction.from_float(float(component)) for component in q)
    vx, vy, vz = (Fraction.from_float(float(component)) for component in v)
    d = w*w + x*x + y*y + z*z
    rows = (
        (w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)),
        (2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)),
        (2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z),
    )
    return tuple(float(sum(c*t for c, t in zip(row, (vx, vy, vz))) / d) for row in rows)


def backends():
    modules = [scalar, namedtuple, dataclass]
    if array is not None:
        modules.append(array)
    return modules


class RotationPrecisionTests(unittest.TestCase):
    def assertClose(self, actual, expected, rel_tol=3e-15, abs_tol=0.0):
        for value, reference in zip(actual, expected):
            self.assertTrue(math.isclose(value, reference, rel_tol=rel_tol, abs_tol=abs_tol),
                            (tuple(actual), tuple(expected)))

    def test_identity_sign_and_permutation_preserve_mixed_scales(self):
        tiny = math.nextafter(0.0, 1.0)
        v = (1e308, 1e-308, tiny)
        # Identity, half-turn around x, quarter-turn around z, cyclic permutation.
        quaternions = ((1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1), (1, 1, 1, 1))
        for backend in backends():
            for q in quaternions:
                with self.subTest(backend=backend.__name__, q=q):
                    expected = exact_rotate(q, v)
                    actual = tuple(backend.rotate(q, v))
                    self.assertEqual(actual, expected)

    def test_generic_rotation_matches_exact_rational_reference(self):
        # Positive and negative finite outputs on unrelated magnitude scales.
        cases = (
            ((1, 2, 3, 4), (8e307, -1e308, 1e-200)),
            ((-2, -1, 0, 1), (1.5e308, 1.5e308, 1.5e308)),
            ((2, -3, 5, 7), (1.25e-200, -4.5e-199, 8e-201)),
        )
        for backend in backends():
            for q, v in cases:
                with self.subTest(backend=backend.__name__, q=q):
                    self.assertClose(backend.rotate(q, v), exact_rotate(q, v))

    def test_large_cancellation_preserves_small_residual(self):
        # Exact first coordinate is -1e-308 after cancellation of 1e308 terms.
        # All other coordinates remain representable, too.
        q = (-2, -1, 0, 1)
        v = (1e308, -1e308, 3e-308)
        expected = exact_rotate(q, v)
        for backend in backends():
            with self.subTest(backend=backend.__name__):
                self.assertClose(backend.rotate(q, v), expected)

    def test_true_result_overflow_is_reported(self):
        q = (-2, -1, 0, 1)
        v = (1.7e308, -1.7e308, -1.7e308)
        with self.assertRaises(OverflowError):
            exact_rotate(q, v)
        for backend in backends():
            with self.subTest(backend=backend.__name__), self.assertRaises(OverflowError):
                backend.rotate(q, v)

    @unittest.skipIf(np is None, 'NumPy is not installed')
    def test_numpy_broadcast_readonly_inputs_and_error_setting(self):
        q = np.array([[[1., 0, 0, 0]], [[1, 0, 0, 1]]])
        v = np.array([[[1e308, 1e-308, 0], [-1e-308, 1e308, 1e-300]]])
        q_saved, v_saved = q.copy(), v.copy()
        q.flags.writeable = False
        v.flags.writeable = False
        previous = np.seterr(all='raise')
        try:
            before = np.geterr().copy()
            result = array.rotate(q, v)
            self.assertEqual(np.geterr(), before)
            self.assertEqual(result.shape, (2, 2, 3))
            for i in range(2):
                for j in range(2):
                    np.testing.assert_array_equal(result[i, j], exact_rotate(q[i, 0], v[0, j]))
            np.testing.assert_array_equal(q, q_saved)
            np.testing.assert_array_equal(v, v_saved)
        finally:
            np.seterr(**previous)


if __name__ == '__main__':
    unittest.main()
