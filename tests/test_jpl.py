"""JPL XYZW negative-cross algebra, active rotations and backend parity."""

import math
import unittest
from dataclasses import FrozenInstanceError

from quatops import dataclass as h_dataclass
from quatops import namedtuple as h_namedtuple
from quatops import scalar as h_scalar
from quatops.jpl import dataclass as j_dataclass
from quatops.jpl import namedtuple as j_namedtuple
from quatops.jpl import scalar as j_scalar

try:
    import numpy as np

    from quatops.numpy.hamilton import wxyz as h_numpy
    from quatops.numpy.jpl import xyzw as j_numpy
except ImportError:
    np = None
    h_numpy = None
    j_numpy = None

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


SCALAR_PAIRS = ((h_scalar, j_scalar), (h_namedtuple, j_namedtuple),
                (h_dataclass, j_dataclass))
PAIRS = SCALAR_PAIRS + (((h_numpy, j_numpy),) if np is not None else ())


def h_to_j(q):
    """Map the active scalar-first Hamilton coordinates to active JPL XYZW."""
    w, x, y, z = q
    return (-x, -y, -z, w)


def j_to_h(q):
    """Map active JPL XYZW coordinates to scalar-first Hamilton coordinates."""
    x, y, z, w = q
    return (w, -x, -y, -z)


class JPLContractTests(unittest.TestCase):
    def assertComponentsClose(self, actual, expected, *, rel_tol=1e-13, abs_tol=1e-14):
        actual, expected = tuple(actual), tuple(expected)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol),
                            (actual, expected))

    def test_negative_cross_product_and_layout(self):
        i, j, k = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)
        identity, negative_identity = (0, 0, 0, 1), (0, 0, 0, -1)
        q = (1, 2, 3, 4)
        for _, backend in PAIRS:
            with self.subTest(backend=backend.__name__):
                self.assertComponentsClose(backend.from_xyzw(q), q)
                self.assertComponentsClose(backend.to_xyzw(q), q)
                for basis in (i, j, k):
                    self.assertComponentsClose(backend.multiply(basis, basis), negative_identity)
                self.assertComponentsClose(backend.multiply(i, j), (0, 0, -1, 0))
                self.assertComponentsClose(backend.multiply(j, i), k)
                self.assertComponentsClose(backend.multiply(q, identity), q)
                self.assertComponentsClose(backend.conjugate(q), (-1, -2, -3, 4))
                self.assertEqual(float(backend.norm(q)), math.sqrt(30))

    def test_layout_reordering_and_physical_convention_conversion_are_distinct(self):
        hq = (4, 1, 2, 3)
        jq = (-1, -2, -3, 4)
        # A scalar-first JPL layout has the same components; it does not
        # negate the vector part. Convention conversion is a separate action.
        j_wxyz = (4, -1, -2, -3)
        for _, backend in PAIRS:
            with self.subTest(backend=backend.__name__):
                self.assertComponentsClose(backend.from_wxyz(j_wxyz), jq)
                self.assertComponentsClose(backend.to_wxyz(jq), j_wxyz)
                self.assertComponentsClose(backend.from_hamilton(hq), jq)
                self.assertComponentsClose(backend.to_hamilton(jq), hq)
                self.assertComponentsClose(backend.from_wxyz(hq), (1, 2, 3, 4))
                self.assertComponentsClose(backend.from_hamilton(hq), (-1, -2, -3, 4))
                self.assertComponentsClose(backend.from_hamilton(backend.to_hamilton(jq)), jq)

    def test_composition_order_noncommuting_active_rotations(self):
        for _, backend in PAIRS:
            with self.subTest(backend=backend.__name__):
                z = backend.from_axis_angle((0, 0, 1, math.pi / 2))
                x = backend.from_axis_angle((1, 0, 0, math.pi / 2))
                self.assertComponentsClose(backend.rotate(z, (1, 0, 0)), (0, 1, 0))
                self.assertComponentsClose(backend.rotate(backend.multiply(z, x), (0, 1, 0)),
                                           (0, 0, 1))
                self.assertComponentsClose(backend.rotate(backend.multiply(x, z), (0, 1, 0)),
                                           (-1, 0, 0))
                self.assertComponentsClose(backend.rotate(backend.multiply(z, x), (0, 1, 0)),
                                           backend.rotate(z, backend.rotate(x, (0, 1, 0))))

    def test_cross_convention_product_inverse_and_rotation(self):
        hq, hr = (1, 2, -3, 4), (2, -1, 3, 2)
        jq, jr, v = h_to_j(hq), h_to_j(hr), (1, -2, 3)
        for hamilton, jpl in PAIRS:
            with self.subTest(backend=jpl.__name__):
                self.assertComponentsClose(jpl.multiply(jq, jr), h_to_j(hamilton.multiply(hq, hr)))
                self.assertComponentsClose(jpl.conjugate(jq), h_to_j(hamilton.conjugate(hq)))
                self.assertComponentsClose(jpl.inverse(jq), h_to_j(hamilton.inverse(hq)))
                self.assertComponentsClose(jpl.normalize(jq), h_to_j(hamilton.normalize(hq)))
                self.assertComponentsClose(jpl.multiply(jq, jpl.inverse(jq)), (0, 0, 0, 1))
                self.assertComponentsClose(jpl.rotate(jq, v), hamilton.rotate(hq, v))
                self.assertComponentsClose(jpl.rotate(tuple(-x for x in jq), v),
                                           hamilton.rotate(hq, v))

    def test_representation_conversions_and_half_turn_tie(self):
        for hamilton, jpl in PAIRS:
            with self.subTest(backend=jpl.__name__):
                for aa in ((0, 0, 2, 0), (0, 0, 2, math.pi/2),
                           (-3, 0, 0, math.pi), (1, 2, 3, -1.7)):
                    jq = jpl.from_axis_angle(aa)
                    hq = hamilton.from_axis_angle(aa)
                    self.assertComponentsClose(jq, h_to_j(hq))
                    self.assertComponentsClose(jpl.to_axis_angle(jq), hamilton.to_axis_angle(hq))
                    self.assertComponentsClose(jpl.to_rotvec(jq), hamilton.to_rotvec(hq))
                    self.assertComponentsClose(jpl.from_rotvec(jpl.to_rotvec(jq)), jq)
                    self.assertComponentsClose(jpl.from_axis_angle(jpl.to_axis_angle(jq)), jq)
                self.assertComponentsClose(jpl.to_axis_angle((0, 0, 0, -1)), (1, 0, 0, 0))
                self.assertComponentsClose(jpl.to_rotvec((0, 0, 0, 1)), (0, 0, 0))
                for q in ((1, 0, 0, 0), (-1, 0, 0, 0), (2, 0, 0, -0.0)):
                    self.assertComponentsClose(jpl.to_axis_angle(q), (1, 0, 0, math.pi))
                    self.assertComponentsClose(jpl.to_rotvec(q), (math.pi, 0, 0))
                self.assertComponentsClose(jpl.from_axis_angle((1, 0, 0, math.pi)), (-1, 0, 0, 0))
                self.assertComponentsClose(jpl.from_rotvec((math.pi, 0, 0)), (-1, 0, 0, 0))

    def test_near_pi_tiny_angles_and_canonical_positive_zeros(self):
        tiny = math.nextafter(0., 1.)
        cases = ((0., 0., -1., 1e-300), (tiny, 0., 0., 2.))
        for hamilton, jpl in PAIRS:
            with self.subTest(backend=jpl.__name__):
                below = tuple(jpl.from_axis_angle((0, 0, 1, math.nextafter(math.pi, 0))))
                above = tuple(jpl.from_axis_angle((0, 0, 1, math.nextafter(math.pi, math.inf))))
                self.assertGreater(below[3], 0.)
                self.assertLess(below[2], 0.)
                self.assertGreater(above[3], 0.)
                self.assertGreater(above[2], 0.)
                self.assertComponentsClose(jpl.to_axis_angle((tiny, -1e308, 0, -0.)),
                                           (0, -1, 0, math.pi))
                # A nonzero w must not be mistaken for an exact pi tie when
                # an extremely small ratio rounds the angle to float pi.
                axis = tuple(jpl.to_axis_angle((1e308, 0, 0, tiny)))
                self.assertLess(axis[0], 0.)
                self.assertEqual(axis[3], math.pi)
                self.assertEqual(tuple(jpl.to_rotvec((tiny, 0, 0, 2.))), (-tiny, 0, 0))
                self.assertEqual(tuple(jpl.from_rotvec((tiny, 0, 0))), (0, 0, 0, 1))
                for q in cases:
                    self.assertComponentsClose(jpl.to_rotvec(q), hamilton.to_rotvec(j_to_h(q)))
                for fn, val in ((jpl.to_axis_angle, (tiny, -1e308, 0, -0.)),
                                (jpl.to_rotvec, (5e-17, -1e307, 0, 1e308)),
                                (jpl.from_axis_angle, (1, -0., 0, -math.pi)),
                                (jpl.from_rotvec, (-0., 0, -math.pi))):
                    for component in fn(val):
                        if component == 0:
                            self.assertEqual(math.copysign(1., float(component)), 1.)

    def test_native_named_types_and_unvalidated_construction(self):
        for backend in (j_namedtuple, j_dataclass):
            with self.subTest(backend=backend.__name__):
                q = backend.Quaternion(1, 2, 3, 4)
                self.assertEqual((q.x, q.y, q.z, q.w), (1, 2, 3, 4))
                self.assertEqual(tuple(q), (1, 2, 3, 4))
                self.assertIs(type(backend.from_axis_angle((1, 0, 0, 0))), backend.Quaternion)
                self.assertIs(type(backend.to_axis_angle((0, 0, 0, 1))), backend.AxisAngle)
                self.assertIs(type(backend.to_rotvec((0, 0, 0, 1))), backend.RotVec)
                with self.assertRaises((AttributeError, FrozenInstanceError)):
                    q.x = 2

    def test_invalid_and_extreme_inputs(self):
        tiny, huge = 1e-300, 1e300
        for _, backend in PAIRS:
            with self.subTest(backend=backend.__name__):
                zero = (0, 0, 0, 0)
                for fn in (backend.normalize, backend.inverse, backend.to_axis_angle, backend.to_rotvec):
                    with self.assertRaises(ValueError):
                        fn(zero)
                with self.assertRaises(ValueError):
                    backend.rotate(zero, (1, 2, 3))
                with self.assertRaises(ValueError):
                    backend.from_axis_angle((0, 0, 0, 0))
                for bad in ((1, 2, 3), (1, 2, 3, float('nan')),
                            (float('inf'), 2, 3, 4)):
                    for fn in (backend.from_xyzw, backend.to_xyzw, backend.conjugate,
                               backend.norm, backend.normalize, backend.inverse,
                               backend.to_axis_angle, backend.to_rotvec):
                        with self.assertRaises(ValueError):
                            fn(bad)
                for scale in (tiny, huge):
                    q = (scale, -scale, scale, -scale)
                    self.assertTrue(math.isclose(float(backend.norm(q)), 2*scale, rel_tol=1e-15))
                    self.assertComponentsClose(backend.multiply(q, backend.inverse(q)), (0, 0, 0, 1))


@unittest.skipIf(np is None, 'NumPy is not installed')
class JPLNumPyTests(unittest.TestCase):
    def test_broadcast_and_per_element_scalar_parity(self):
        qs = np.array([[[1, 2, 3, 4]], [[2, -1, 2, -3]]], dtype=float)
        rs = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [2, 3, -1, 1]]], dtype=float)
        vs = np.array([[[1, 2, 3], [2, -4, 1], [0, 1, 0]]], dtype=float)
        saved = [a.copy() for a in (qs, rs, vs)]
        for item in (qs, rs, vs):
            item.flags.writeable = False
        old = np.seterr(all='raise')
        try:
            before = np.geterr().copy()
            products = j_numpy.multiply(qs, rs)
            rotated = j_numpy.rotate(qs, vs)
            self.assertEqual(products.shape, (2, 3, 4))
            self.assertEqual(rotated.shape, (2, 3, 3))
            self.assertEqual(products.dtype, np.float64)
            for i in range(2):
                for j in range(3):
                    np.testing.assert_allclose(products[i, j], j_scalar.multiply(qs[i, 0], rs[0, j]), atol=1e-14)
                    np.testing.assert_allclose(rotated[i, j], j_scalar.rotate(qs[i, 0], vs[0, j]), atol=1e-14)
            for a, b in zip((qs, rs, vs), saved):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(np.geterr(), before)
        finally:
            np.seterr(**old)

    def test_explicit_cross_convention_batch(self):
        h = np.array([[[1, 2, 3, 4]], [[-5, 6, -7, 8]]], dtype=float)
        saved = h.copy()
        h.flags.writeable = False
        j = j_numpy.from_hamilton(h)
        self.assertEqual(j.shape, (2, 1, 4))
        np.testing.assert_array_equal(j, [[[-2, -3, -4, 1]], [[-6, 7, -8, -5]]])
        np.testing.assert_array_equal(j_numpy.to_hamilton(j), h)
        np.testing.assert_array_equal(j_numpy.from_wxyz(j_numpy.to_wxyz(j)), j)
        np.testing.assert_array_equal(h, saved)
        self.assertEqual(j_numpy.to_hamilton(np.empty((0, 2, 4))).shape, (0, 2, 4))

    def test_conversions_batch_empty_and_validation(self):
        axes = np.array([[[0, 0, 3, math.pi/2]], [[1, 2, 3, math.pi]]], dtype=float)
        qs = j_numpy.from_axis_angle(axes)
        self.assertEqual(qs.shape, (2, 1, 4))
        self.assertEqual(j_numpy.to_axis_angle(qs).shape, (2, 1, 4))
        self.assertEqual(j_numpy.to_rotvec(qs).shape, (2, 1, 3))
        self.assertEqual(j_numpy.from_rotvec(np.empty((0, 2, 3))).shape, (0, 2, 4))
        self.assertEqual(j_numpy.to_axis_angle(np.empty((0, 2, 4))).shape, (0, 2, 4))
        with self.assertRaises(ValueError):
            j_numpy.rotate(np.ones((2, 4)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            j_numpy.multiply(np.ones((2, 4)), np.ones((3, 4)))
        with self.assertRaises(ValueError):
            j_numpy.to_rotvec([[0, 0, 0, 1], [0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            j_numpy.from_axis_angle([[1, 0, 0, 0], [0, 0, 0, 1]])
        with self.assertRaises(ValueError):
            j_numpy.multiply([0, 0, 0, 1], [0, 0, 0, complex(1, 0)])

    @unittest.skipIf(Rotation is None, 'SciPy is not installed')
    def test_scipy_rotation_and_composition(self):
        rng = np.random.default_rng(20261030)
        # SciPy uses active Hamilton XYZW. JPL's vector part has the opposite
        # sign, so explicitly translate without relying on newer keyword options.
        jqs = rng.normal(size=(40, 4)) * rng.uniform(0.1, 10, size=(40, 1))
        jrs = rng.normal(size=(40, 4)) * rng.uniform(0.1, 10, size=(40, 1))
        vectors = rng.normal(size=(40, 3))
        to_scipy = lambda a: np.concatenate((-a[..., :3], a[..., 3:]), axis=-1)
        qs = Rotation.from_quat(to_scipy(jqs))
        rs = Rotation.from_quat(to_scipy(jrs))
        np.testing.assert_allclose(j_numpy.rotate(jqs, vectors), qs.apply(vectors),
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(j_numpy.rotate(j_numpy.multiply(jqs, jrs), vectors),
                                   (qs * rs).apply(vectors), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(j_numpy.to_rotvec(jqs), qs.as_rotvec(), rtol=1e-13, atol=1e-13)
        for q, vector, rotation in zip(jqs, vectors, qs):
            for _, backend in SCALAR_PAIRS:
                np.testing.assert_allclose(backend.rotate(q, vector), rotation.apply(vector),
                                           rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(tuple(backend.to_rotvec(q)), rotation.as_rotvec(),
                                           rtol=1e-13, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
