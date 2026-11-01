"""Shared representation conversion and independent SciPy reference tests."""

import math
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction

from quatops import dataclass, namedtuple, scalar

try:
    import numpy as np

    from quatops import numpy as qnumpy
except ImportError:
    np = None
    qnumpy = None

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


SCALAR_BACKENDS = (scalar, namedtuple, dataclass)
ALL_BACKENDS = SCALAR_BACKENDS + ((qnumpy,) if qnumpy is not None else ())


class ConversionConformanceTests(unittest.TestCase):
    def assertComponentsClose(self, actual, expected, *, rel_tol=1e-13, abs_tol=1e-14):
        a, b = tuple(actual), tuple(expected)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=abs_tol), (a, b))

    def test_known_identity_quarter_turn_and_explicit_axis_normalization(self):
        h = math.sqrt(0.5)
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                for q in ((1, 0, 0, 0), (-1, 0, 0, 0)):
                    self.assertComponentsClose(backend.to_axis_angle(q), (1, 0, 0, 0))
                    self.assertComponentsClose(backend.to_rotvec(q), (0, 0, 0))
                self.assertComponentsClose(backend.from_rotvec((0, 0, 0)), (1, 0, 0, 0))
                self.assertComponentsClose(backend.from_axis_angle((0, 0, 10, 0)), (1, 0, 0, 0))
                for axis_scale in (math.nextafter(0.0, 1.0), 1e-300, 1, 1e300):
                    q = backend.from_axis_angle((0, 0, axis_scale, math.pi/2))
                    self.assertComponentsClose(q, (h, 0, 0, h))
                    self.assertComponentsClose(backend.to_axis_angle(q), (0, 0, 1, math.pi/2))
                    self.assertComponentsClose(backend.to_rotvec(q), (0, 0, math.pi/2))
                self.assertComponentsClose(backend.from_rotvec((0, 0, math.pi/2)), (h, 0, 0, h))

    def test_canonical_half_turn_and_sign_equivalence(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                # Exactly w == 0: the first nonzero axis component is positive.
                for q in ((0, -1, 0, 0), (-0., 1, -0., 0), (0, -2, 0, 0)):
                    self.assertComponentsClose(backend.to_axis_angle(q), (1, 0, 0, math.pi))
                    self.assertComponentsClose(backend.to_rotvec(q), (math.pi, 0, 0))
                self.assertComponentsClose(backend.to_axis_angle((0, 0, -1, 0)), (0, 1, 0, math.pi))
                # Nonunit q and negative q describe the same rotation.
                a = backend.to_axis_angle((2, -3, 4, -5))
                b = backend.to_axis_angle((-2, 3, -4, 5))
                self.assertComponentsClose(a, b)
                self.assertComponentsClose(backend.to_rotvec((2, -3, 4, -5)),
                                           backend.to_rotvec((-2, 3, -4, 5)))

    def test_float_pi_tie_and_no_near_pi_snap(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                for aa in ((-1, 0, 0, math.pi), (1, 0, 0, -math.pi)):
                    q = tuple(backend.from_axis_angle(aa))
                    self.assertEqual(q, (0., 1., 0., 0.))
                self.assertComponentsClose(backend.from_rotvec((-math.pi, 0, 0)), (0, 1, 0, 0))
                below = tuple(backend.from_axis_angle((0, 0, 1, math.nextafter(math.pi, 0))))
                above = tuple(backend.from_axis_angle((0, 0, 1, math.nextafter(math.pi, math.inf))))
                self.assertGreater(below[0], 0.0)
                self.assertGreater(below[3], 0.0)
                self.assertGreater(above[0], 0.0)
                self.assertLess(above[3], 0.0)
                below_negative = tuple(backend.from_axis_angle((0, 0, -1, math.nextafter(math.pi, 0))))
                above_negative = tuple(backend.from_axis_angle((0, 0, -1, math.nextafter(math.pi, math.inf))))
                self.assertLess(below_negative[3], 0.0)
                self.assertGreater(above_negative[3], 0.0)
                # w is strictly positive in the original input, even if its
                # ratio to the vector underflows and the angle rounds to pi.
                tiny = math.nextafter(0.0, 1.0)
                axis = tuple(backend.to_axis_angle((tiny, -1e308, 0, 0)))
                self.assertLess(axis[0], 0.0)
                self.assertEqual(axis[3], math.pi)
                self.assertEqual(axis, tuple(backend.to_axis_angle((-tiny, 1e308, 0, 0))))
                self.assertEqual(tuple(backend.to_axis_angle((tiny, -1, 0, 0))), (-1, 0, 0, math.pi))
                # Exact signed-zero w uses the first nonzero *original* vector
                # component to choose sign even if it later underflows.
                self.assertEqual(tuple(backend.to_axis_angle((-0., -tiny, 1e308, 0))),
                                 (0, -1, 0, math.pi))

    def test_norm_overflow_and_nonunit_large_axis(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                with self.assertRaises(OverflowError):
                    backend.from_rotvec((1.7e308, 1.7e308, 0))
                h = math.sqrt(.5)
                q = backend.from_axis_angle((1.7e308, 1.7e308, 0, math.pi/2))
                self.assertComponentsClose(q, (h, .5, .5, 0))

    def test_large_finite_angle_uses_direct_trigonometric_reduction(self):
        for angle in (1e20, 1e308):
            w, x = math.cos(angle / 2), math.sin(angle / 2)
            if w < 0 or (w == 0 and x < 0):
                w, x = -w, -x
            for backend in ALL_BACKENDS:
                with self.subTest(backend=backend.__name__, angle=angle):
                    self.assertComponentsClose(backend.from_axis_angle((1, 0, 0, angle)), (w, x, 0, 0))
                    self.assertComponentsClose(backend.from_rotvec((angle, 0, 0)), (w, x, 0, 0))

    def test_small_angles_without_identity_threshold(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                aa = tuple(backend.to_axis_angle((1, 1e-300, 0, 0)))
                rv = tuple(backend.to_rotvec((1, 1e-300, 0, 0)))
                self.assertEqual(tuple(aa[:3]), (1, 0, 0))
                self.assertTrue(math.isclose(aa[3], 2e-300, rel_tol=1e-14, abs_tol=0))
                self.assertTrue(math.isclose(rv[0], 2e-300, rel_tol=1e-14, abs_tol=0))
                self.assertEqual(rv[1:], (0, 0))
                q = tuple(backend.from_rotvec((1e-300, 0, 0)))
                self.assertTrue(math.isclose(q[1], 0.5e-300, rel_tol=1e-14, abs_tol=0))

    def test_subnormal_angle_rounding_and_identity_canonicalization(self):
        tiny = math.nextafter(0.0, 1.0)
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                self.assertEqual(tuple(backend.to_axis_angle((2, tiny, 0, 0))), (1, 0, 0, tiny))
                self.assertEqual(tuple(backend.to_rotvec((2, tiny, 0, 0))), (tiny, 0, 0))
                self.assertEqual(tuple(backend.to_axis_angle((1e308, 0, tiny, 0))), (1, 0, 0, 0))
                self.assertEqual(tuple(backend.to_rotvec((1e308, 0, tiny, 0))), (0, 0, 0))
                self.assertEqual(tuple(backend.from_rotvec((tiny, 0, 0))), (1, 0, 0, 0))
                self.assertComponentsClose(backend.to_axis_angle((tiny, 0, 0, tiny)),
                                           (0, 0, 1, math.pi/2))

    def test_pathological_small_angles_against_rational_and_decimal_oracles(self):
        # The correction to 2*v/w is vastly below the smallest subnormal here;
        # compute the ratio exactly rather than rounding v/w to zero first.
        tiny = math.nextafter(0.0, 1.0)
        rational_first_order = Fraction(2) * Fraction.from_float(tiny) / Fraction(2)
        rational_angle = float(rational_first_order)
        self.assertEqual(rational_angle, tiny)
        # Use actual binary input as Decimal, keeping terms well beyond the
        # float64 rounding threshold. These oracles do not call SciPy.
        input_component = 1e-40
        with localcontext() as context:
            context.prec = 120
            t = Decimal.from_float(input_component)
            decimal_angle = float(2 * (t - t**3/3 + t**5/5))
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                aa = backend.to_axis_angle((2, tiny, 0, 0))
                rv = backend.to_rotvec((2, tiny, 0, 0))
                self.assertEqual(tuple(aa), (1, 0, 0, rational_angle))
                self.assertEqual(tuple(rv), (rational_angle, 0, 0))
                aa = backend.to_axis_angle((1, input_component, 0, 0))
                rv = backend.to_rotvec((1, input_component, 0, 0))
                self.assertTrue(math.isclose(tuple(aa)[3], decimal_angle, rel_tol=1e-15, abs_tol=0))
                self.assertTrue(math.isclose(next(iter(rv)), decimal_angle, rel_tol=1e-15, abs_tol=0))

    def test_mixed_subnormal_axis_angle_and_rotvec_rounding(self):
        tiny = math.nextafter(0.0, 1.0)
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                # For w=4 the angle is ~sqrt(2)/2 ulp; for w=5 it is
                # ~2*sqrt(2)/5 ulp. Each rotvec component rounds to zero.
                for w in (4., 5.):
                    q = (w, tiny, tiny, 0.)
                    self.assertEqual(tuple(backend.to_axis_angle(q))[3], tiny)
                    self.assertEqual(tuple(backend.to_rotvec(q)), (0., 0., 0.))
                # Exact angle is ~sqrt(2) ulp, which rounds to one ulp;
                # each corresponding rotvec component rounds to one ulp.
                q = (2., tiny, tiny, 0.)
                self.assertEqual(tuple(backend.to_axis_angle(q))[3], tiny)
                self.assertEqual(tuple(backend.to_rotvec(q)), (tiny, tiny, 0.))

    def test_invalid_lengths_nonfinite_and_zero_inputs(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend.__name__):
                for method in (backend.to_axis_angle, backend.to_rotvec):
                    for bad in ((0, 0, 0, 0), (1, 0, 0), (1, 0, float('nan'), 0),
                                (1, float('inf'), 0, 0), (1, 0, '3', 0)):
                        with self.assertRaises(ValueError):
                            method(bad)
                for bad in ((0, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0),
                            (1, 0, 0, float('nan')), (1, float('inf'), 0, 1)):
                    with self.assertRaises(ValueError):
                        backend.from_axis_angle(bad)
                for bad in ((1, 2), (1, 2, 3, 4), (1, float('nan'), 0),
                            (1, 0, float('inf')), (1, 0, '3')):
                    with self.assertRaises(ValueError):
                        backend.from_rotvec(bad)

    def test_named_representation_types_are_immutable_and_constructor_unvalidated(self):
        for backend in (namedtuple, dataclass):
            with self.subTest(backend=backend.__name__):
                aa = backend.AxisAngle(0, 0, 2, math.pi/2)
                rv = backend.RotVec(0, 0, math.pi/2)
                self.assertEqual((aa.ax, aa.ay, aa.az, aa.angle), tuple(aa))
                self.assertEqual((rv.x, rv.y, rv.z), tuple(rv))
                with self.assertRaises((AttributeError, FrozenInstanceError)):
                    aa.ax = 1
                with self.assertRaises((AttributeError, FrozenInstanceError)):
                    rv.x = 1
                self.assertIs(type(backend.from_axis_angle(aa)), backend.Quaternion)
                self.assertIs(type(backend.from_rotvec(rv)), backend.Quaternion)
                self.assertIs(type(backend.to_axis_angle((1, 0, 0, 0))), backend.AxisAngle)
                self.assertIs(type(backend.to_rotvec((1, 0, 0, 0))), backend.RotVec)
                self.assertEqual(backend.AxisAngle(0, 0, 0, 0).ax, 0)
                with self.assertRaises(ValueError):
                    backend.from_axis_angle(backend.AxisAngle(0, 0, 0, 0))

    def test_sign_equivalent_roundtrip_of_random_fixed_inputs(self):
        for backend in ALL_BACKENDS:
            for original in ((1, 2, 3, 4), (-2, 5, -3, 1), (1e-300, 0, 1e-300, 0),
                             (1e300, -1e300, 1e300, -1e300)):
                with self.subTest(backend=backend.__name__, original=original):
                    expected = backend.normalize(original)
                    for result in (backend.from_axis_angle(backend.to_axis_angle(original)),
                                   backend.from_rotvec(backend.to_rotvec(original))):
                        dot = sum(x*y for x, y in zip(expected, result))
                        self.assertTrue(math.isclose(abs(float(dot)), 1., rel_tol=1e-13, abs_tol=1e-13))


@unittest.skipIf(np is None, 'NumPy is not installed')
class NumPyConversionTests(unittest.TestCase):
    def test_shapes_empty_noncontiguous_readonly_and_no_mutation(self):
        raw = np.array([[1, 9, 0, 9, 0, 9, 0, 9], [2, 9, 0, 9, 0, 9, 2, 9]], dtype=float)
        quaternions = raw[:, ::2]
        self.assertFalse(quaternions.flags.c_contiguous)
        saved = raw.copy()
        quaternions.flags.writeable = False
        for fn, width in ((qnumpy.to_axis_angle, 4), (qnumpy.to_rotvec, 3)):
            result = fn(quaternions)
            self.assertEqual(result.shape, (2, width))
            self.assertEqual(result.dtype, np.float64)
        axes_raw = np.array([[1, 9, 0, 9, 0, 9, .5, 9], [0, 9, 2, 9, 0, 9, 1, 9]])
        axes = axes_raw[:, ::2]
        axes_saved = axes_raw.copy()
        axes.flags.writeable = False
        rotvec_raw = np.array([[0, 9, 0, 9, 0, 9], [0, 9, 1, 9, 0, 9]])
        rotvec = rotvec_raw[:, ::2]
        rotvec_saved = rotvec_raw.copy()
        rotvec.flags.writeable = False
        self.assertEqual(qnumpy.from_axis_angle(axes).shape, (2, 4))
        self.assertEqual(qnumpy.from_rotvec(rotvec).shape, (2, 4))
        np.testing.assert_array_equal(raw, saved)
        np.testing.assert_array_equal(axes_raw, axes_saved)
        np.testing.assert_array_equal(rotvec_raw, rotvec_saved)
        for fn, inp, expected in (
            (qnumpy.to_axis_angle, np.empty((0, 2, 4)), (0, 2, 4)),
            (qnumpy.to_rotvec, np.empty((0, 2, 4)), (0, 2, 3)),
            (qnumpy.from_axis_angle, np.empty((0, 2, 4)), (0, 2, 4)),
            (qnumpy.from_rotvec, np.empty((0, 2, 3)), (0, 2, 4)),
        ):
            self.assertEqual(fn(inp).shape, expected)

    def test_zero_stride_and_negative_stride_views_preserve_bytes(self):
        bases = (
            (qnumpy.to_axis_angle, np.array([[1., 2, 3, 4], [5, 6, 7, 8]])),
            (qnumpy.to_rotvec, np.array([[1., 2, 3, 4], [5, 6, 7, 8]])),
            (qnumpy.from_axis_angle, np.array([[1., 0, 0, 1], [0, 1, 0, 2]])),
            (qnumpy.from_rotvec, np.array([[1., 0, 0], [0, 1, 0]])),
        )
        old = np.seterr(all='raise')
        try:
            before = np.geterr().copy()
            for fn, base in bases:
                reversed_view = base[::-1]
                reversed_view.flags.writeable = False
                broadcast = np.broadcast_to(base[0], (3, 1, base.shape[-1]))
                self.assertLess(reversed_view.strides[0], 0)
                self.assertEqual(broadcast.strides[0], 0)
                saved = base.tobytes()
                self.assertEqual(fn(reversed_view).shape[0], 2)
                self.assertEqual(fn(broadcast).shape[:2], (3, 1))
                self.assertEqual(base.tobytes(), saved)
            invalids = (
                (qnumpy.to_axis_angle, np.array([[1., 0, 0, 0], [0, 0, 0, 0]])),
                (qnumpy.to_rotvec, np.array([[1., 0, 0, 0], [0, 0, 0, 0]])),
                (qnumpy.from_axis_angle, np.array([[1., 0, 0, 0], [0, 0, 0, 1]])),
                (qnumpy.from_rotvec, np.array([[1., 0, 0], [float('nan'), 0, 0]])),
            )
            for fn, base in invalids:
                view = base[::-1]
                view.flags.writeable = False
                saved = base.tobytes()
                with self.assertRaises(ValueError):
                    fn(view)
                self.assertEqual(base.tobytes(), saved)
            self.assertEqual(np.geterr(), before)
        finally:
            np.seterr(**old)

    def test_rounded_pi_norm_can_differ_by_one_ulp_between_backends(self):
        r = (3.1415675208820746, 0.012566337104064343, 0)
        self.assertLessEqual(abs(math.hypot(*r) - math.pi), math.ulp(math.pi))
        q_scalar = scalar.from_rotvec(r)
        q_array = qnumpy.from_rotvec(r)
        # Implementations compute the norm independently; their rounded
        # norms and scalar components may differ by one ULP around pi.
        # Compare represented rotations, not raw quaternion components.
        dot = sum(float(x)*float(y) for x, y in zip(q_scalar, q_array))
        self.assertTrue(math.isclose(abs(dot), 1., rel_tol=1e-14, abs_tol=1e-14))

    def test_batch_shape_dtype_validation_and_seterr(self):
        old = np.seterr(all='raise')
        try:
            before = np.geterr().copy()
            aa = np.array([[[1, 0, 0, 0]], [[0, 0, 1, 1]]])
            q = qnumpy.from_axis_angle(aa)
            self.assertEqual(q.shape, (2, 1, 4))
            self.assertEqual(q.dtype, np.float64)
            self.assertEqual(qnumpy.to_rotvec(q).shape, (2, 1, 3))
            vectors = np.array([[[1., 0, 0], [0, 1, 0], [0, 0, 1]]])
            self.assertEqual(qnumpy.rotate(q, vectors).shape, (2, 3, 3))
            with self.assertRaises(ValueError):
                qnumpy.from_axis_angle([[1, 0, 0, 0], [0, 0, 0, 1]])
            with self.assertRaises(ValueError):
                qnumpy.to_rotvec([[1, 0, 0, 0], [0, 0, 0, 0]])
            for fn, values in ((qnumpy.from_axis_angle, [1, 0, 0, 1]),
                               (qnumpy.from_rotvec, [0, 0, 1]),
                               (qnumpy.to_axis_angle, [1, 0, 0, 0]),
                               (qnumpy.to_rotvec, [1, 0, 0, 0])):
                with self.assertRaises(ValueError):
                    fn(np.asarray(values, dtype=complex))
                with self.assertRaises(ValueError):
                    fn(np.asarray(values, dtype=object))
            self.assertEqual(np.geterr(), before)
        finally:
            np.seterr(**old)


@unittest.skipIf(np is None or Rotation is None, 'SciPy and NumPy are not installed')
class SciPyConversionReferenceTests(unittest.TestCase):
    def test_seeded_quaternion_rotation_equivalence(self):
        rng = np.random.default_rng(713)
        for q, vector in zip(rng.normal(size=(40, 4)), rng.normal(size=(40, 3))):
            q = q * rng.uniform(.05, 10)
            reference = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            expected_rv = reference.as_rotvec()
            for backend in ALL_BACKENDS:
                with self.subTest(backend=backend.__name__):
                    actual_rv = np.asarray(tuple(backend.to_rotvec(q)))
                    np.testing.assert_allclose(actual_rv, expected_rv, rtol=1e-12, atol=1e-13)
                    for reconstructed in (backend.from_rotvec(actual_rv),
                                          backend.from_axis_angle(backend.to_axis_angle(q))):
                        xyzw = backend.to_xyzw(reconstructed)
                        np.testing.assert_allclose(Rotation.from_quat(xyzw).apply(vector),
                                                   reference.apply(vector), rtol=1e-12, atol=1e-13)

    def test_axis_input_normalization_and_angles_against_scipy(self):
        rng = np.random.default_rng(714)
        for axis, angle in zip(rng.normal(size=(40, 3)), rng.uniform(-3, 3, size=40)):
            axis *= rng.uniform(.1, 20)
            expected = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle)
            aa = (*axis, angle)
            for backend in ALL_BACKENDS:
                with self.subTest(backend=backend.__name__):
                    actual_q = backend.from_axis_angle(aa)
                    actual = Rotation.from_quat(backend.to_xyzw(actual_q))
                    self.assertLess((actual.inv() * expected).magnitude(), 1e-12)


if __name__ == '__main__':
    unittest.main()
