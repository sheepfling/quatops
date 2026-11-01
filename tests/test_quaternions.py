"""Behavioral tests shared by the pure-Python and optional array interfaces."""

import math
import unittest

from quatops import scalar

try:
    import numpy as np

    from quatops import numpy as qnp
except ImportError:
    np = None
    qnp = None

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


class ScalarTests(unittest.TestCase):
    def assertTupleClose(self, actual, expected, **kwargs):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isclose(a, b, **kwargs), (actual, expected))

    def test_hamilton_basis_and_order(self):
        i, j, k = (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        minus_one = (-1, 0, 0, 0)
        for q in (i, j, k):
            self.assertEqual(scalar.multiply(q, q), minus_one)
        self.assertEqual(scalar.multiply(i, j), k)
        self.assertEqual(scalar.multiply(j, i), (0, 0, 0, -1))
        self.assertEqual(scalar.multiply(scalar.multiply(i, j), k), minus_one)

    def test_layout_and_magnitude_are_explicit(self):
        q = (2, 3, 4, 5)
        self.assertEqual(scalar.from_xyzw((3, 4, 5, 2)), q)
        self.assertEqual(scalar.to_xyzw(q), (3, 4, 5, 2))
        self.assertEqual(scalar.conjugate(q), (2, -3, -4, -5))
        self.assertEqual(scalar.multiply(q, (1, 0, 0, 0)), q)
        self.assertAlmostEqual(scalar.norm(q), math.sqrt(54))
        self.assertAlmostEqual(scalar.norm(scalar.normalize(q)), 1.0)
        self.assertEqual(q, (2, 3, 4, 5))

    def test_inverse_conjugate_and_rotation(self):
        q = (1, 2, -3, 4)
        r = (2, -1, 3, 2)
        self.assertTupleClose(scalar.multiply(q, scalar.inverse(q)), (1, 0, 0, 0), abs_tol=1e-15)
        self.assertTupleClose(scalar.conjugate(scalar.multiply(q, r)),
                              scalar.multiply(scalar.conjugate(r), scalar.conjugate(q)), abs_tol=1e-15)
        # Non-unit q represents the same active rotation; no normalization of q is required.
        h = math.sqrt(0.5)
        z = (2*h, 0, 0, 2*h)
        self.assertTupleClose(scalar.rotate(z, (1, 0, 0)), (0, 1, 0), abs_tol=1e-15)
        self.assertTupleClose(scalar.rotate(tuple(-v for v in z), (1, 0, 0)), (0, 1, 0), abs_tol=1e-15)
        x = (h, h, 0, 0)
        self.assertTupleClose(scalar.rotate(scalar.multiply(z, x), (0, 1, 0)),
                              scalar.rotate(z, scalar.rotate(x, (0, 1, 0))), abs_tol=1e-15)

    def test_extreme_norm_and_normalization(self):
        for scale in (1e-300, 1e300):
            q = (scale, -scale, scale, -scale)
            self.assertTrue(math.isclose(scalar.norm(q), 2*scale, rel_tol=1e-15))
            self.assertTupleClose(scalar.normalize(q), (0.5, -0.5, 0.5, -0.5), abs_tol=1e-15)
            self.assertTupleClose(scalar.multiply(q, scalar.inverse(q)), (1, 0, 0, 0), abs_tol=1e-15)

    def test_invalid_input_and_zero_norm(self):
        for fn in (scalar.from_xyzw, scalar.to_xyzw, scalar.conjugate, scalar.norm,
                   scalar.normalize, scalar.inverse):
            with self.subTest(fn=fn.__name__):
                for bad in ((1, 2, 3), (1, 2, 3, 4, 5), (1, 2, float('nan'), 4),
                            (1, 2, float('inf'), 4), (1, 2, '3', 4)):
                    with self.assertRaises(ValueError):
                        fn(bad)
        for fn in (scalar.normalize, scalar.inverse):
            with self.assertRaises(ValueError):
                fn((0, 0, 0, 0))
        with self.assertRaises(ValueError):
            scalar.rotate((0, 0, 0, 0), (1, 2, 3))
        with self.assertRaises(ValueError):
            scalar.rotate((1, 0, 0, 0), (1, 2))
        with self.assertRaises(ValueError):
            scalar.multiply((1, 0, 0, 0), (1, 0, 0))

    def test_overflow_raises(self):
        with self.assertRaises(OverflowError):
            scalar.multiply((1e308, 0, 0, 0), (1e308, 0, 0, 0))


@unittest.skipIf(np is None, 'NumPy is not installed')
class ArrayTests(unittest.TestCase):
    def test_broadcast_and_scalar_parity(self):
        qs = np.array([[[1, 2, 3, 4]], [[2, -1, 2, -3]]])
        rs = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [2, 3, -1, 1]]])
        vectors = np.array([[[1, 2, 3], [2, -4, 1], [0, 1, 0]]])
        saved = (qs.copy(), rs.copy(), vectors.copy())
        result = qnp.multiply(qs, rs)
        rotated = qnp.rotate(qs, vectors)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertEqual(rotated.shape, (2, 3, 3))
        self.assertEqual(result.dtype, np.float64)
        for i in range(2):
            for j in range(3):
                np.testing.assert_allclose(result[i, j], scalar.multiply(qs[i, 0], rs[0, j]), atol=1e-14)
                np.testing.assert_allclose(rotated[i, j], scalar.rotate(qs[i, 0], vectors[0, j]), atol=1e-14)
        for original, copy in zip((qs, rs, vectors), saved):
            np.testing.assert_array_equal(original, copy)

    def test_all_functions_match_scalar(self):
        q = (2, 3, -4, 5)
        for name in ('from_xyzw', 'to_xyzw', 'conjugate', 'norm', 'normalize', 'inverse'):
            with self.subTest(name=name):
                np.testing.assert_allclose(getattr(qnp, name)(q), getattr(scalar, name)(q), rtol=1e-14)
        np.testing.assert_allclose(qnp.multiply(q, q), scalar.multiply(q, q))
        np.testing.assert_allclose(qnp.rotate(q, (1, 2, 3)), scalar.rotate(q, (1, 2, 3)))

    def test_extremes_and_empty_batches(self):
        q = np.array([[1e-300, -1e-300, 1e-300, -1e-300],
                      [1e300, -1e300, 1e300, -1e300]])
        np.testing.assert_allclose(qnp.norm(q), [2e-300, 2e300], rtol=1e-15, atol=0)
        np.testing.assert_allclose(qnp.normalize(q), [[.5, -.5, .5, -.5]]*2)
        np.testing.assert_allclose(qnp.multiply(q, qnp.inverse(q)), [[1, 0, 0, 0]]*2, atol=1e-15)
        empty = np.empty((0, 4))
        self.assertEqual(qnp.normalize(empty).shape, (0, 4))
        self.assertEqual(qnp.norm(empty).shape, (0,))
        self.assertEqual(qnp.rotate(empty, np.empty((0, 3))).shape, (0, 3))

    def test_invalid_shapes_nonfinite_and_zero_batch(self):
        for fn in (qnp.from_xyzw, qnp.to_xyzw, qnp.conjugate, qnp.norm, qnp.normalize, qnp.inverse):
            with self.subTest(fn=fn.__name__):
                for bad in ([1, 2, 3], [[1, 2, 3, 4, 5]], [1, 2, float('nan'), 4],
                            [1, 2, float('inf'), 4], [1, 2, '3', 4]):
                    with self.assertRaises(ValueError):
                        fn(bad)
        batch = [[1, 0, 0, 0], [0, 0, 0, 0]]
        for fn in (qnp.normalize, qnp.inverse):
            with self.assertRaises(ValueError):
                fn(batch)
        with self.assertRaises(ValueError):
            qnp.rotate(batch, [1, 0, 0])
        with self.assertRaises(ValueError):
            qnp.rotate([1, 0, 0, 0], [1, 2])
        with self.assertRaises(ValueError):
            qnp.multiply(np.ones((2, 4)), np.ones((3, 4)))
        with self.assertRaises(ValueError):
            qnp.rotate(np.ones((2, 4)), np.ones((3, 3)))

    def test_numpy_error_settings_do_not_change_policy(self):
        old = np.seterr(all='raise')
        try:
            with self.assertRaises(OverflowError):
                qnp.multiply([1e308, 0, 0, 0], [1e308, 0, 0, 0])
            with self.assertRaises(ValueError):
                qnp.normalize([0, 0, 0, 0])
            np.testing.assert_allclose(qnp.normalize([1e-300]*4), [.5]*4)
        finally:
            np.seterr(**old)

    @unittest.skipIf(Rotation is None, 'SciPy is not installed')
    def test_rotation_against_scipy(self):
        rng = np.random.default_rng(106)
        qs = rng.normal(size=(25, 4)) * rng.uniform(.1, 10, size=(25, 1))
        vectors = rng.normal(size=(25, 3))
        # SciPy accepts xyzw and normalizes on construction. Adapt both
        # conventions explicitly before comparing signed quaternion values.
        rotations = Rotation.from_quat(qnp.to_xyzw(qs))
        np.testing.assert_allclose(qnp.normalize(qs), qnp.from_xyzw(rotations.as_quat()),
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(qnp.rotate(qs, vectors), rotations.apply(vectors),
                                   rtol=1e-13, atol=1e-13)
        # A unit inverse represents SciPy's inverse rotation. Scaling by the
        # original norm restores the actual inverse of the non-unit input.
        expected_inverse = qnp.from_xyzw(rotations.inv().as_quat()) / qnp.norm(qs)[:, None]
        np.testing.assert_allclose(qnp.inverse(qs), expected_inverse, rtol=1e-13, atol=1e-13)
        rs = rng.normal(size=(25, 4)) * rng.uniform(.1, 10, size=(25, 1))
        rotations_r = Rotation.from_quat(qnp.to_xyzw(rs))
        expected_product = qnp.from_xyzw((rotations * rotations_r).as_quat())
        actual_product = qnp.normalize(qnp.multiply(qs, rs))
        np.testing.assert_allclose(actual_product, expected_product, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(qnp.rotate(qnp.multiply(qs, rs), vectors),
                                   (rotations * rotations_r).apply(vectors),
                                   rtol=1e-13, atol=1e-13)
        # Verify the scalar implementation against the same independent oracle.
        for q, r, v, normalized, inverse, product, rotated in zip(
                qs, rs, vectors, qnp.from_xyzw(rotations.as_quat()), expected_inverse,
                expected_product, rotations.apply(vectors)):
            np.testing.assert_allclose(scalar.normalize(q), normalized, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(scalar.inverse(q), inverse, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(scalar.normalize(scalar.multiply(q, r)), product,
                                       rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(scalar.rotate(q, v), rotated, rtol=1e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
