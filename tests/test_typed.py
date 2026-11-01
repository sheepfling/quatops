"""Contract and reference checks for the named-field scalar backend."""

import math
import unittest

from quatops import dataclass, namedtuple, scalar

try:
    import numpy as np
    from scipy.spatial.transform import Rotation
except ImportError:
    np = None
    Rotation = None


class BackendTests:
    backend = None
    def assertComponentsClose(self, actual, expected, *, rel_tol=1e-14, abs_tol=1e-14):
        actual, expected = tuple(actual), tuple(expected)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol), (actual, expected))

    def test_named_fields_and_tuple_interoperability(self):
        q = self.backend.Quaternion(w=2, x=3, y=4, z=5)
        self.assertEqual((q.w, q.x, q.y, q.z), (2, 3, 4, 5))
        self.assertEqual(tuple(q), (2, 3, 4, 5))
        with self.assertRaises(AttributeError):
            q.w = 9
        # Values can be used with the existing scalar backend directly.
        self.assertEqual(scalar.conjugate(q), (2, -3, -4, -5))

    def test_layout_conversion_types_and_no_hidden_normalization(self):
        q = self.backend.from_xyzw([3, 4, 5, 2])
        self.assertIs(type(q), self.backend.Quaternion)
        self.assertEqual(tuple(q), (2, 3, 4, 5))
        self.assertEqual(q.w, 2)
        xyzw = self.backend.to_xyzw(q)
        self.assertIs(type(xyzw), tuple)
        self.assertEqual(xyzw, (3, 4, 5, 2))
        self.assertIs(type(self.backend.from_xyzw(xyzw)), self.backend.Quaternion)
        self.assertEqual(self.backend.from_xyzw(xyzw), q)
        self.assertAlmostEqual(self.backend.norm(q), math.sqrt(54))
        self.assertEqual(tuple(self.backend.multiply(q, (1, 0, 0, 0))), tuple(q))
        self.assertAlmostEqual(self.backend.norm(self.backend.normalize(q)), 1.0)

    def test_all_operations_match_scalar_and_return_types(self):
        q = self.backend.Quaternion(2, 3, -4, 5)
        r = self.backend.Quaternion(3, -1, 2, 4)
        for name, args in (
            ('from_xyzw', (q,)), ('conjugate', (q,)), ('multiply', (q, r)),
            ('normalize', (q,)), ('inverse', (q,)),
        ):
            with self.subTest(name=name):
                actual = getattr(self.backend, name)(*args)
                self.assertIs(type(actual), self.backend.Quaternion)
                self.assertTrue(all(isinstance(component, float) for component in actual))
                self.assertComponentsClose(actual, getattr(scalar, name)(*args))
        self.assertIs(type(self.backend.norm(q)), float)
        self.assertEqual(self.backend.norm(q), scalar.norm(q))
        vector = self.backend.rotate(q, (1, 2, 3))
        self.assertIs(type(vector), tuple)
        self.assertComponentsClose(vector, scalar.rotate(q, (1, 2, 3)))

    def test_known_rotations_inverse_and_extremes(self):
        h = math.sqrt(0.5)
        z = self.backend.Quaternion(2*h, 0, 0, 2*h)
        x = self.backend.Quaternion(h, h, 0, 0)
        self.assertComponentsClose(self.backend.rotate(z, (1, 0, 0)), (0, 1, 0))
        self.assertComponentsClose(self.backend.rotate(self.backend.multiply(z, x), (0, 1, 0)),
                                   self.backend.rotate(z, self.backend.rotate(x, (0, 1, 0))))
        self.assertComponentsClose(self.backend.rotate(tuple(-v for v in z), (1, 0, 0)), (0, 1, 0))
        for scale in (1e-300, 1e300):
            q = self.backend.Quaternion(scale, -scale, scale, -scale)
            self.assertTrue(math.isclose(self.backend.norm(q), 2*scale, rel_tol=1e-15))
            self.assertComponentsClose(self.backend.normalize(q), (.5, -.5, .5, -.5))
            self.assertComponentsClose(self.backend.multiply(q, self.backend.inverse(q)), (1, 0, 0, 0))

    def test_invalid_zero_and_overflow(self):
        # NamedTuple construction itself is unchecked; operations validate.
        for name in ('from_xyzw', 'to_xyzw', 'conjugate', 'norm', 'normalize', 'inverse'):
            fn = getattr(self.backend, name)
            for bad in ((1, 2, 3), (1, 2, 3, 4, 5), (1, 2, '3', 4),
                        self.backend.Quaternion(1, 2, float('nan'), 4), (1, 2, float('inf'), 4)):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    fn(bad)
        zero = self.backend.Quaternion(0, 0, 0, 0)
        self.assertEqual(self.backend.norm(zero), 0)
        for fn in (self.backend.normalize, self.backend.inverse):
            with self.assertRaises(ValueError):
                fn(zero)
        with self.assertRaises(ValueError):
            self.backend.rotate(zero, (1, 0, 0))
        with self.assertRaises(ValueError):
            self.backend.rotate((1, 0, 0, 0), (1, 0))
        with self.assertRaises(ValueError):
            self.backend.multiply((1, 0, 0, 0), (1, 0, 0))
        with self.assertRaises(OverflowError):
            self.backend.multiply((1e308, 0, 0, 0), (1e308, 0, 0, 0))

    @unittest.skipIf(Rotation is None, 'SciPy is not installed')
    def test_against_scipy_with_explicit_layout_and_scale(self):
        rng = np.random.default_rng(207)
        for _ in range(25):
            q = self.backend.Quaternion(*(rng.normal(size=4) * rng.uniform(.1, 10)))
            r = self.backend.Quaternion(*(rng.normal(size=4) * rng.uniform(.1, 10)))
            v = rng.normal(size=3)
            # SciPy consumes scalar-last values and normalizes on construction.
            sq = Rotation.from_quat(self.backend.to_xyzw(q))
            sr = Rotation.from_quat(self.backend.to_xyzw(r))
            expected_unit = self.backend.from_xyzw(sq.as_quat())
            expected_inverse = np.asarray(tuple(self.backend.from_xyzw(sq.inv().as_quat()))) / self.backend.norm(q)
            expected_product = self.backend.from_xyzw((sq * sr).as_quat())
            self.assertComponentsClose(self.backend.normalize(q), expected_unit)
            self.assertComponentsClose(self.backend.inverse(q), expected_inverse)
            self.assertComponentsClose(self.backend.normalize(self.backend.multiply(q, r)), expected_product)
            self.assertComponentsClose(self.backend.rotate(q, v), sq.apply(v))
            self.assertComponentsClose(self.backend.rotate(self.backend.multiply(q, r), v), (sq * sr).apply(v))


class NamedTupleTests(BackendTests, unittest.TestCase):
    backend = namedtuple

    def test_is_indexable_tuple(self):
        q = self.backend.Quaternion(1, 2, 3, 4)
        self.assertIsInstance(q, tuple)
        self.assertEqual(q[2], 3)


class DataclassTests(BackendTests, unittest.TestCase):
    backend = dataclass

    def test_dataclass_is_not_a_tuple(self):
        q = self.backend.Quaternion(1, 2, 3, 4)
        self.assertNotIsInstance(q, tuple)


if __name__ == '__main__':
    unittest.main()
