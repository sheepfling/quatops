"""JPL XYZW negative-cross algebra, active rotations and backend parity."""

import pytest
import math
from dataclasses import FrozenInstanceError

from quatops import dataclass as h_dataclass
from quatops import namedtuple as h_namedtuple
from quatops import scalar as h_scalar
from quatops.jpl import dataclass as j_dataclass
from quatops.jpl import namedtuple as j_namedtuple
from quatops.jpl import scalar as j_scalar

import numpy as np

from quatops.numpy.hamilton import wxyz as h_numpy
from quatops.numpy.jpl import xyzw as j_numpy

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


SCALAR_PAIRS = ((h_scalar, j_scalar), (h_namedtuple, j_namedtuple),
                (h_dataclass, j_dataclass))
PAIRS = SCALAR_PAIRS + ((h_numpy, j_numpy),)


def h_to_j(q):
    """Map the active scalar-first Hamilton coordinates to active JPL XYZW."""
    w, x, y, z = q
    return (-x, -y, -z, w)


def j_to_h(q):
    """Map active JPL XYZW coordinates to scalar-first Hamilton coordinates."""
    x, y, z, w = q
    return (w, -x, -y, -z)


class JPLContractTests:
    def assert_components_close(self, actual, expected, *, rel_tol=1e-13, abs_tol=1e-14):
        actual, expected = tuple(actual), tuple(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(float(a), float(b), rtol=rel_tol, atol=abs_tol)

    def test_negative_cross_product_and_layout(self):
        i, j, k = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)
        identity, negative_identity = (0, 0, 0, 1), (0, 0, 0, -1)
        q = (1, 2, 3, 4)
        for _, backend in PAIRS:
            self.assert_components_close(backend.from_xyzw(q), q)
            self.assert_components_close(backend.to_xyzw(q), q)
            for basis in (i, j, k):
                self.assert_components_close(backend.multiply(basis, basis), negative_identity)
            self.assert_components_close(backend.multiply(i, j), (0, 0, -1, 0))
            self.assert_components_close(backend.multiply(j, i), k)
            self.assert_components_close(backend.multiply(q, identity), q)
            self.assert_components_close(backend.conjugate(q), (-1, -2, -3, 4))
            assert float(backend.norm(q)) == math.sqrt(30)

    def test_layout_reordering_and_physical_convention_conversion_are_distinct(self):
        hq = (4, 1, 2, 3)
        jq = (-1, -2, -3, 4)
        # A scalar-first JPL layout has the same components; it does not
        # negate the vector part. Convention conversion is a separate action.
        j_wxyz = (4, -1, -2, -3)
        for _, backend in PAIRS:
            self.assert_components_close(backend.from_wxyz(j_wxyz), jq)
            self.assert_components_close(backend.to_wxyz(jq), j_wxyz)
            self.assert_components_close(backend.from_hamilton(hq), jq)
            self.assert_components_close(backend.to_hamilton(jq), hq)
            self.assert_components_close(backend.from_wxyz(hq), (1, 2, 3, 4))
            self.assert_components_close(backend.from_hamilton(hq), (-1, -2, -3, 4))
            self.assert_components_close(backend.from_hamilton(backend.to_hamilton(jq)), jq)

    def test_composition_order_noncommuting_active_rotations(self):
        for _, backend in PAIRS:
            z = backend.from_axis_angle((0, 0, 1, math.pi / 2))
            x = backend.from_axis_angle((1, 0, 0, math.pi / 2))
            self.assert_components_close(backend.rotate(z, (1, 0, 0)), (0, 1, 0))
            self.assert_components_close(backend.rotate(backend.multiply(z, x), (0, 1, 0)),
                                       (0, 0, 1))
            self.assert_components_close(backend.rotate(backend.multiply(x, z), (0, 1, 0)),
                                       (-1, 0, 0))
            self.assert_components_close(backend.rotate(backend.multiply(z, x), (0, 1, 0)),
                                       backend.rotate(z, backend.rotate(x, (0, 1, 0))))

    def test_cross_convention_product_inverse_and_rotation(self):
        hq, hr = (1, 2, -3, 4), (2, -1, 3, 2)
        jq, jr, v = h_to_j(hq), h_to_j(hr), (1, -2, 3)
        for hamilton, jpl in PAIRS:
            self.assert_components_close(jpl.multiply(jq, jr), h_to_j(hamilton.multiply(hq, hr)))
            self.assert_components_close(jpl.conjugate(jq), h_to_j(hamilton.conjugate(hq)))
            self.assert_components_close(jpl.inverse(jq), h_to_j(hamilton.inverse(hq)))
            self.assert_components_close(jpl.normalize(jq), h_to_j(hamilton.normalize(hq)))
            self.assert_components_close(jpl.multiply(jq, jpl.inverse(jq)), (0, 0, 0, 1))
            self.assert_components_close(jpl.rotate(jq, v), hamilton.rotate(hq, v))
            self.assert_components_close(jpl.rotate(tuple(-x for x in jq), v),
                                       hamilton.rotate(hq, v))

    def test_representation_conversions_and_half_turn_tie(self):
        for hamilton, jpl in PAIRS:
            for aa in ((0, 0, 2, 0), (0, 0, 2, math.pi/2),
                       (-3, 0, 0, math.pi), (1, 2, 3, -1.7)):
                jq = jpl.from_axis_angle(aa)
                hq = hamilton.from_axis_angle(aa)
                self.assert_components_close(jq, h_to_j(hq))
                self.assert_components_close(jpl.to_axis_angle(jq), hamilton.to_axis_angle(hq))
                self.assert_components_close(jpl.to_rotvec(jq), hamilton.to_rotvec(hq))
                self.assert_components_close(jpl.from_rotvec(jpl.to_rotvec(jq)), jq)
                self.assert_components_close(jpl.from_axis_angle(jpl.to_axis_angle(jq)), jq)
            self.assert_components_close(jpl.to_axis_angle((0, 0, 0, -1)), (1, 0, 0, 0))
            self.assert_components_close(jpl.to_rotvec((0, 0, 0, 1)), (0, 0, 0))
            for q in ((1, 0, 0, 0), (-1, 0, 0, 0), (2, 0, 0, -0.0)):
                self.assert_components_close(jpl.to_axis_angle(q), (1, 0, 0, math.pi))
                self.assert_components_close(jpl.to_rotvec(q), (math.pi, 0, 0))
            self.assert_components_close(jpl.from_axis_angle((1, 0, 0, math.pi)), (-1, 0, 0, 0))
            self.assert_components_close(jpl.from_rotvec((math.pi, 0, 0)), (-1, 0, 0, 0))

    def test_near_pi_tiny_angles_and_canonical_positive_zeros(self):
        tiny = math.nextafter(0., 1.)
        cases = ((0., 0., -1., 1e-300), (tiny, 0., 0., 2.))
        for hamilton, jpl in PAIRS:
            below = tuple(jpl.from_axis_angle((0, 0, 1, math.nextafter(math.pi, 0))))
            above = tuple(jpl.from_axis_angle((0, 0, 1, math.nextafter(math.pi, math.inf))))
            assert below[3] > 0.
            assert below[2] < 0.
            assert above[3] > 0.
            assert above[2] > 0.
            self.assert_components_close(jpl.to_axis_angle((tiny, -1e308, 0, -0.)),
                                       (0, -1, 0, math.pi))
            # A nonzero w must not be mistaken for an exact pi tie when
            # an extremely small ratio rounds the angle to float pi.
            axis = tuple(jpl.to_axis_angle((1e308, 0, 0, tiny)))
            assert axis[0] < 0.
            assert axis[3] == math.pi
            assert tuple(jpl.to_rotvec((tiny, 0, 0, 2.))) == (-tiny, 0, 0)
            assert tuple(jpl.from_rotvec((tiny, 0, 0))) == (0, 0, 0, 1)
            for q in cases:
                self.assert_components_close(jpl.to_rotvec(q), hamilton.to_rotvec(j_to_h(q)))
            for fn, val in ((jpl.to_axis_angle, (tiny, -1e308, 0, -0.)),
                            (jpl.to_rotvec, (5e-17, -1e307, 0, 1e308)),
                            (jpl.from_axis_angle, (1, -0., 0, -math.pi)),
                            (jpl.from_rotvec, (-0., 0, -math.pi))):
                for component in fn(val):
                    if component == 0:
                        assert math.copysign(1., float(component)) == 1.

    def test_native_named_types_and_unvalidated_construction(self):
        for backend in (j_namedtuple, j_dataclass):
            q = backend.Quaternion(1, 2, 3, 4)
            assert (q.x, q.y, q.z, q.w) == (1, 2, 3, 4)
            assert tuple(q) == (1, 2, 3, 4)
            assert type(backend.from_axis_angle((1, 0, 0, 0))) is backend.Quaternion
            assert type(backend.to_axis_angle((0, 0, 0, 1))) is backend.AxisAngle
            assert type(backend.to_rotvec((0, 0, 0, 1))) is backend.RotVec
            with pytest.raises((AttributeError, FrozenInstanceError)):
                q.x = 2

    def test_invalid_and_extreme_inputs(self):
        tiny, huge = 1e-300, 1e300
        for _, backend in PAIRS:
            zero = (0, 0, 0, 0)
            for fn in (backend.normalize, backend.inverse, backend.to_axis_angle, backend.to_rotvec):
                with pytest.raises(ValueError):
                    fn(zero)
            with pytest.raises(ValueError):
                backend.rotate(zero, (1, 2, 3))
            with pytest.raises(ValueError):
                backend.from_axis_angle((0, 0, 0, 0))
            for bad in ((1, 2, 3), (1, 2, 3, float('nan')),
                        (float('inf'), 2, 3, 4)):
                for fn in (backend.from_xyzw, backend.to_xyzw, backend.conjugate,
                           backend.norm, backend.normalize, backend.inverse,
                           backend.to_axis_angle, backend.to_rotvec):
                    with pytest.raises(ValueError):
                        fn(bad)
            for scale in (tiny, huge):
                q = (scale, -scale, scale, -scale)
                np.testing.assert_allclose(float(backend.norm(q)), 2*scale, rtol=1e-15, atol=0)
                self.assert_components_close(backend.multiply(q, backend.inverse(q)), (0, 0, 0, 1))


class JPLNumPyTests:
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
            assert products.shape == (2, 3, 4)
            assert rotated.shape == (2, 3, 3)
            assert products.dtype == np.float64
            for i in range(2):
                for j in range(3):
                    np.testing.assert_allclose(products[i, j], j_scalar.multiply(qs[i, 0], rs[0, j]), atol=1e-14)
                    np.testing.assert_allclose(rotated[i, j], j_scalar.rotate(qs[i, 0], vs[0, j]), atol=1e-14)
            for a, b in zip((qs, rs, vs), saved):
                np.testing.assert_array_equal(a, b)
            assert np.geterr() == before
        finally:
            np.seterr(**old)

    def test_explicit_cross_convention_batch(self):
        h = np.array([[[1, 2, 3, 4]], [[-5, 6, -7, 8]]], dtype=float)
        saved = h.copy()
        h.flags.writeable = False
        j = j_numpy.from_hamilton(h)
        assert j.shape == (2, 1, 4)
        np.testing.assert_array_equal(j, [[[-2, -3, -4, 1]], [[-6, 7, -8, -5]]])
        np.testing.assert_array_equal(j_numpy.to_hamilton(j), h)
        np.testing.assert_array_equal(j_numpy.from_wxyz(j_numpy.to_wxyz(j)), j)
        np.testing.assert_array_equal(h, saved)
        assert j_numpy.to_hamilton(np.empty((0, 2, 4))).shape == (0, 2, 4)

    def test_conversions_batch_empty_and_validation(self):
        axes = np.array([[[0, 0, 3, math.pi/2]], [[1, 2, 3, math.pi]]], dtype=float)
        qs = j_numpy.from_axis_angle(axes)
        assert qs.shape == (2, 1, 4)
        assert j_numpy.to_axis_angle(qs).shape == (2, 1, 4)
        assert j_numpy.to_rotvec(qs).shape == (2, 1, 3)
        assert j_numpy.from_rotvec(np.empty((0, 2, 3))).shape == (0, 2, 4)
        assert j_numpy.to_axis_angle(np.empty((0, 2, 4))).shape == (0, 2, 4)
        with pytest.raises(ValueError):
            j_numpy.rotate(np.ones((2, 4)), np.ones((3, 3)))
        with pytest.raises(ValueError):
            j_numpy.multiply(np.ones((2, 4)), np.ones((3, 4)))
        with pytest.raises(ValueError):
            j_numpy.to_rotvec([[0, 0, 0, 1], [0, 0, 0, 0]])
        with pytest.raises(ValueError):
            j_numpy.from_axis_angle([[1, 0, 0, 0], [0, 0, 0, 1]])
        with pytest.raises(ValueError):
            j_numpy.multiply([0, 0, 0, 1], [0, 0, 0, complex(1, 0)])

    @pytest.mark.skipif(Rotation is None, reason='SciPy is not installed')
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
