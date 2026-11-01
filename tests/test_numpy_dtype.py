"""NumPy dtype conversion and floating-error policy are part of the API."""

import unittest
from decimal import Decimal
from fractions import Fraction

try:
    import numpy as np

    from quatops import numpy as qnp
except ImportError:
    np = None
    qnp = None


@unittest.skipIf(np is None, 'NumPy is not installed')
class NumPyDtypeTests(unittest.TestCase):
    def test_supported_native_dtypes_promote_to_float64(self):
        for dtype in (np.bool_, np.int8, np.uint64, np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                values = np.array([1, 0, 0, 0], dtype=dtype)
                self.assertEqual(qnp.from_xyzw(values).dtype, np.dtype('float64'))
                self.assertEqual(qnp.to_xyzw(values).dtype, np.dtype('float64'))
                self.assertEqual(qnp.conjugate(values).dtype, np.dtype('float64'))
                self.assertEqual(qnp.multiply(values, values).dtype, np.dtype('float64'))
                self.assertEqual(np.asarray(qnp.norm(values)).dtype, np.dtype('float64'))
                self.assertEqual(qnp.normalize(values).dtype, np.dtype('float64'))
                self.assertEqual(qnp.inverse(values).dtype, np.dtype('float64'))
                self.assertEqual(qnp.rotate(values, [1, 0, 0]).dtype, np.dtype('float64'))
        # Float64 conversion is not exact for every integer.
        self.assertEqual(qnp.conjugate(np.array([2**53 + 1, 0, 0, 0], dtype=np.uint64))[0], float(2**53))

    def test_norm_returns_numpy_scalar_for_single_quaternion(self):
        previous = np.seterr(all='raise')
        try:
            for values, expected in (([3, 4, 0, 0], 5.0), ([0, 0, 0, 0], 0.0),
                                     ([5e-324] * 4, 1e-323)):
                with self.subTest(values=values):
                    result = qnp.norm(values)
                    self.assertIs(type(result), np.float64)
                    self.assertEqual(result, expected)
            for shape in ((1, 4), (2, 3, 4), (0, 4), (2, 0, 4)):
                with self.subTest(shape=shape):
                    result = qnp.norm(np.ones(shape))
                    self.assertIsInstance(result, np.ndarray)
                    self.assertEqual(result.shape, shape[:-1])
                    self.assertEqual(result.dtype, np.dtype('float64'))
            with self.assertRaises(ValueError):
                qnp.norm(np.array(1.0))
            with self.assertRaises(OverflowError):
                qnp.norm([1e308] * 4)
            self.assertEqual(np.geterr(), {'divide': 'raise', 'over': 'raise',
                                         'under': 'raise', 'invalid': 'raise'})
        finally:
            np.seterr(**previous)

    def test_complex_object_and_string_dtypes_rejected(self):
        # Even complex values with zero imaginary parts are rejected, never cast.
        for bad in (
            np.array([1, 0, 0, 0], dtype=np.complex128),
            np.array([1, 0, 0, 0], dtype=object),
            np.array([Fraction(1), 0, 0, 0], dtype=object),
            np.array([Decimal(1), 0, 0, 0], dtype=object),
            np.array(['1', '0', '0', '0']),
        ):
            for operation in (qnp.from_xyzw, qnp.to_xyzw, qnp.conjugate,
                              qnp.norm, qnp.normalize, qnp.inverse):
                with self.subTest(dtype=bad.dtype, operation=operation.__name__), self.assertRaises(ValueError):
                    operation(bad)
            with self.assertRaises(ValueError):
                qnp.multiply(bad, [1, 0, 0, 0])
            with self.assertRaises(ValueError):
                qnp.rotate(bad, [1, 0, 0])
        with self.assertRaises(ValueError):
            qnp.rotate([1, 0, 0, 0], np.array([1, 0, 0], dtype=complex))

    def test_numpy_seterr_does_not_override_policy(self):
        previous = np.seterr(all='raise')
        try:
            np.testing.assert_allclose(qnp.normalize([1e-300] * 4), [.5] * 4)
            np.testing.assert_allclose(qnp.inverse([1e-300, 0, 0, 0]), [1e300, 0, 0, 0])
            with self.assertRaises(OverflowError):
                qnp.norm([1e308] * 4)
            with self.assertRaises(OverflowError):
                qnp.multiply([1e308, 0, 0, 0], [1e308, 0, 0, 0])
            with self.assertRaises(ValueError):
                qnp.normalize([0, 0, 0, 0])
            self.assertEqual(np.geterr(), {'divide': 'raise', 'over': 'raise',
                                         'under': 'raise', 'invalid': 'raise'})
        finally:
            np.seterr(**previous)

    def test_read_only_noncontiguous_inputs_are_unchanged(self):
        q = np.array([[1., 2, 3, 4], [5, 6, 7, 8]])[::-1]
        v = np.array([[1., 2, 3], [4, 5, 6]])[::-1]
        q_before, v_before = q.copy(), v.copy()
        q.flags.writeable = False
        v.flags.writeable = False
        operations = (
            (qnp.from_xyzw, (q,)), (qnp.to_xyzw, (q,)),
            (qnp.conjugate, (q,)), (qnp.multiply, (q, q)),
            (qnp.norm, (q,)), (qnp.normalize, (q,)),
            (qnp.inverse, (q,)), (qnp.rotate, (q, v)),
        )
        for operation, args in operations:
            with self.subTest(operation=operation.__name__):
                operation(*args)
                np.testing.assert_array_equal(q, q_before)
                np.testing.assert_array_equal(v, v_before)

    def test_empty_leading_dimensions_broadcast(self):
        self.assertEqual(qnp.multiply(np.ones((0, 1, 4)), np.ones((1, 3, 4))).shape,
                         (0, 3, 4))
        self.assertEqual(qnp.rotate(np.ones((0, 1, 4)), np.ones((1, 3, 3))).shape,
                         (0, 3, 3))
        self.assertEqual(qnp.multiply(np.ones((2, 0, 4)), np.ones((1, 0, 4))).shape,
                         (2, 0, 4))
        self.assertEqual(qnp.rotate(np.ones((2, 0, 4)), np.ones((1, 0, 3))).shape,
                         (2, 0, 3))

    def test_finite_out_of_range_extended_float_is_rejected_on_conversion(self):
        if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
            self.skipTest('longdouble has no larger range than float64')
        values = np.array([np.longdouble('1e400'), 0, 0, 0], dtype=np.longdouble)
        self.assertTrue(np.isfinite(values).all())
        with self.assertRaises(ValueError):
            qnp.norm(values)


if __name__ == '__main__':
    unittest.main()
