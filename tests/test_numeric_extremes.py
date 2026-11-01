"""Regression tests for extreme NumPy input magnitudes and error handling."""

import math
import unittest

try:
    import numpy as np

    from quatops.numpy.hamilton import wxyz as qnp
except ImportError:
    np = None
    qnp = None


@unittest.skipIf(np is None, "NumPy is not installed")
class NumpyExtremeTests(unittest.TestCase):
    def test_scaled_norm_handles_subnormals_zero_and_overflow(self):
        tiny = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            self.assertEqual(qnp.norm([tiny] * 4), 2 * tiny)
            self.assertEqual(qnp.norm([0, 0, 0, 0]), 0)
            result = qnp.norm([[tiny] * 4, [0, 0, 0, 0], [1e300] * 4])
            np.testing.assert_array_equal(result[:2], [2 * tiny, 0.0])
            self.assertTrue(math.isclose(result[2], 2e300, rel_tol=1e-15))
            with self.assertRaises(OverflowError):
                qnp.norm([1e308] * 4)

    def test_inverse_handles_extreme_scales(self):
        tiny = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            for scale in (1e-308, 1e308):
                inverse = qnp.inverse([scale] * 4)
                self.assertTrue(all(math.isfinite(float(x)) for x in inverse))
                self.assertEqual(tuple(inverse), (.25 / scale, -.25 / scale,
                                                   -.25 / scale, -.25 / scale))
            with self.assertRaises(OverflowError):
                qnp.inverse([tiny, 0, 0, 0])

    def test_rotation_handles_extreme_vector_and_quaternion_scale(self):
        tiny = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            for scale in (tiny, 1e-300, 1e300):
                q = [scale] * 4
                for v in ([tiny, 0, 0], [1e308, 1e308, 1e308], [0, 0, 0]):
                    rotated = qnp.rotate(q, v)
                    expected = [v[2], v[0], v[1]]
                    np.testing.assert_array_equal(rotated, expected)
            with self.assertRaises(OverflowError):
                qnp.rotate([1, 0, 0, math.tan(math.pi / 8)], [1.4e308, -1.4e308, 0])


if __name__ == "__main__":
    unittest.main()
