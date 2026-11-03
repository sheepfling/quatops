"""Cross-convention and component-order conformance of NumPy kernels."""

import importlib
import inspect
import math
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from quatops import scalar as h_scalar

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


class NumPyLayoutsTests:
    @classmethod
    def setup_class(cls):
        # An available NumPy must not silently hide an unavailable backend.
        cls.backends = (
            ("hamilton", "wxyz", importlib.import_module("quatops.hamilton.numpy_wxyz")),
            ("hamilton", "xyzw", importlib.import_module("quatops.hamilton.numpy_xyzw")),
            ("jpl", "wxyz", importlib.import_module("quatops.jpl.numpy_wxyz")),
            ("jpl", "xyzw", importlib.import_module("quatops.jpl.numpy_xyzw")),
        )

    @staticmethod
    def from_h_physical(hq, convention, order):
        """Map Hamilton wxyz coordinates independently of library helpers."""
        q = np.asarray(hq, dtype=np.float64).copy()
        if convention == "jpl":
            q[..., 1:] *= -1
        if order == "xyzw":
            q = q[..., [1, 2, 3, 0]]
        return q

    @staticmethod
    def to_h_physical(q, convention, order):
        q = np.asarray(q, dtype=np.float64)
        if order == "xyzw":
            q = q[..., [3, 0, 1, 2]]
        else:
            q = q.copy()
        if convention == "jpl":
            q[..., 1:] *= -1
        return q

    def test_explicit_selection_and_legacy_aliases_have_no_runtime_selector(self):
        for convention, order, backend in self.backends:
            assert tuple(inspect.signature(backend.multiply).parameters) == ("lhs", "rhs")
            assert tuple(inspect.signature(backend.rotate).parameters) == ("q", "v")
        h_legacy = importlib.import_module("quatops.numpy.hamilton.wxyz")
        j_legacy = importlib.import_module("quatops.numpy.jpl.xyzw")
        h_default, j_default = self.backends[0][2], self.backends[3][2]
        for name in (
            "add", "subtract", "negate", "scale", "multiply", "divide",
            "inverse", "rotate", "from_axis_angle", "to_rotvec",
        ):
            assert getattr(h_default, name) is getattr(h_legacy, name)
            assert getattr(j_default, name) is getattr(j_legacy, name)

    def test_layout_reorder_is_separate_from_convention(self):
        wxyz = np.array([[4, 1, 2, 3], [-2, 5, -1, 3]], dtype=np.int64)
        xyzw = wxyz[..., [1, 2, 3, 0]]
        original = wxyz.copy()
        wxyz.flags.writeable = False
        for convention, order, backend in self.backends:
            expected = wxyz if order == "wxyz" else xyzw
            a, b = backend.from_wxyz(wxyz), backend.from_xyzw(xyzw)
            np.testing.assert_array_equal(a, expected)
            np.testing.assert_array_equal(b, expected)
            np.testing.assert_array_equal(backend.to_wxyz(a), wxyz)
            np.testing.assert_array_equal(backend.to_xyzw(a), xyzw)
            assert a.dtype == np.float64
            if convention == "jpl":
                physical = backend.from_hamilton(wxyz)
                wanted = self.from_h_physical(wxyz, convention, order)
                np.testing.assert_array_equal(physical, wanted)
                np.testing.assert_array_equal(backend.to_hamilton(physical), wxyz)
                assert not np.array_equal(physical, a)
            np.testing.assert_array_equal(wxyz, original)

    def test_algebra_rotation_composition_and_inverse(self):
        hq, hr = (1., 2., -3., 4.), (2., -1., 3., 2.)
        v = (1., -2., 3.)
        expected_product = h_scalar.multiply(hq, hr)
        expected_rotate = h_scalar.rotate(hq, v)
        expected_inverse = h_scalar.inverse(hq)
        for convention, order, backend in self.backends:
            q = self.from_h_physical(hq, convention, order)
            r = self.from_h_physical(hr, convention, order)
            identity = self.from_h_physical((1, 0, 0, 0), convention, order)
            np.testing.assert_allclose(self.to_h_physical(backend.multiply(q, r), convention, order),
                                       expected_product, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(self.to_h_physical(backend.inverse(q), convention, order),
                                       expected_inverse, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(backend.multiply(q, backend.inverse(q)), identity, atol=1e-14)
            np.testing.assert_allclose(backend.rotate(q, v), expected_rotate, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(backend.rotate(-q, v), expected_rotate, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(float(backend.norm(q)), math.sqrt(30), rtol=1e-15, atol=0)
            assert isinstance(backend.norm(q), np.float64)
            z = backend.from_axis_angle((0, 0, 1, math.pi / 2))
            x = backend.from_axis_angle((1, 0, 0, math.pi / 2))
            np.testing.assert_allclose(backend.rotate(z, (1, 0, 0)), (0, 1, 0), atol=1e-14)
            np.testing.assert_allclose(backend.rotate(backend.multiply(z, x), (0, 1, 0)),
                                       (0, 0, 1), atol=1e-14)
            np.testing.assert_allclose(backend.rotate(backend.multiply(x, z), (0, 1, 0)),
                                       (-1, 0, 0), atol=1e-14)

    def test_axis_angle_rotvec_pi_sign_subnormal_and_zeros(self):
        tiny = math.nextafter(0., 1.)
        hhalf = (0., -1., 0., 0.)
        for convention, order, backend in self.backends:
            identity = self.from_h_physical((1, 0, 0, 0), convention, order)
            half = self.from_h_physical(hhalf, convention, order)
            np.testing.assert_array_equal(backend.to_axis_angle(identity), (1, 0, 0, 0))
            np.testing.assert_array_equal(backend.to_rotvec(identity), (0, 0, 0))
            np.testing.assert_allclose(backend.to_axis_angle(half), (1, 0, 0, math.pi), atol=0)
            np.testing.assert_array_equal(backend.to_axis_angle(half), backend.to_axis_angle(-half))
            reconstructed = backend.from_axis_angle((1, 0, 0, math.pi))
            expected = self.from_h_physical((0, 1, 0, 0), convention, order)
            np.testing.assert_array_equal(reconstructed, expected)
            np.testing.assert_array_equal(backend.from_rotvec((math.pi, 0, 0)), expected)
            aa = (1, -2, 3, 0.7)
            q = backend.from_axis_angle(aa)
            np.testing.assert_allclose(backend.from_rotvec(backend.to_rotvec(q)), q, atol=1e-14)
            tiny_q = self.from_h_physical((2., tiny, 0., 0.), convention, order)
            np.testing.assert_array_equal(backend.to_rotvec(tiny_q), (tiny, 0, 0))
            for output in (backend.to_axis_angle(tiny_q), backend.to_rotvec(tiny_q), reconstructed):
                assert not np.signbit(output[output == 0]).any()

    def test_broadcast_views_empty_error_state_and_invalid_inputs(self):
        hqs = np.array([[[1., 2, -3, 4]], [[2., -1, 2, -3]]])
        hrs = np.array([[[1., 0, 0, 0], [0, 1, 0, 0], [2, 3, -1, 1]]])
        vectors = np.array([[[1., 2, 3], [2, -4, 1], [0, 1, 0]]])
        for convention, order, backend in self.backends:
            q = self.from_h_physical(hqs, convention, order)[::-1]
            r = self.from_h_physical(hrs, convention, order)
            v = vectors[:, ::-1]
            original = (q.tobytes(), r.tobytes(), v.tobytes())
            for a in (q, r, v):
                a.flags.writeable = False
            old = np.seterr(all="raise")
            try:
                state = np.geterr().copy()
                product = backend.multiply(q, r)
                result = backend.rotate(q, v)
                assert product.shape == (2, 3, 4)
                assert result.shape == (2, 3, 3)
                for i in range(2):
                    for j in range(3):
                        hq = self.to_h_physical(q[i, 0], convention, order)
                        hr = self.to_h_physical(r[0, j], convention, order)
                        np.testing.assert_allclose(self.to_h_physical(product[i, j], convention, order),
                                                   h_scalar.multiply(hq, hr), atol=1e-14)
                        np.testing.assert_allclose(result[i, j], h_scalar.rotate(hq, v[0, j]), atol=1e-14)
                assert backend.multiply(np.empty((0, 1, 4)), r).shape == (0, 3, 4)
                assert backend.rotate(np.empty((0, 1, 4)), v).shape == (0, 3, 3)
                zero_stride = np.broadcast_to(q[0, 0], (2, 3, 4))
                assert zero_stride.strides[0] == 0
                assert backend.rotate(zero_stride, v).shape == (2, 3, 3)
                np.testing.assert_allclose(backend.rotate(zero_stride, v)[0],
                                           backend.rotate(q[0, 0], v)[0], atol=1e-14)
                assert backend.to_axis_angle(np.empty((0, 2, 4))).shape == (0, 2, 4)
                assert backend.from_rotvec(np.empty((0, 2, 3))).shape == (0, 2, 4)
                assert (q.tobytes(), r.tobytes(), v.tobytes()) == original
                assert np.geterr() == state
                for fn, value in ((backend.to_rotvec, [[1, 2, 3, 4], [0, 0, 0, 0]]),
                                  (backend.from_axis_angle, [[1, 0, 0, 0], [0, 0, 0, 0]]),
                                  (backend.norm, [1, 2, float("nan"), 0]),
                                  (backend.from_wxyz, [1, 2, 3]),
                                  (backend.from_xyzw, [1, 2, 3, complex(0)])):
                    with pytest.raises(ValueError):
                        fn(value)
                with pytest.raises(ValueError):
                    backend.multiply(np.ones((2, 4)), np.ones((3, 4)))
                with pytest.raises(ValueError):
                    backend.rotate(np.ones((2, 4)), np.ones((3, 3)))
                assert np.geterr() == state
            finally:
                np.seterr(**old)

    def test_hot_paths_do_not_call_other_modules_or_conversion_helpers(self):
        # Capture the callables, then forbid convention or layout dispatch.
        for convention, order, backend in self.backends:
            multiply, rotate = backend.multiply, backend.rotate
            q = self.from_h_physical((1., 0, 0, 1.), convention, order)
            identity = self.from_h_physical((1., 0, 0, 0), convention, order)
            with ExitStack() as stack:
                others = [entry[2] for entry in self.backends]
                others.extend(importlib.import_module(name) for name in (
                    "quatops.numpy.hamilton.wxyz",
                    "quatops.numpy.hamilton.xyzw",
                    "quatops.numpy.jpl.wxyz",
                    "quatops.numpy.jpl.xyzw",
                ))
                for other in dict.fromkeys(others):
                    for name in ("from_wxyz", "from_xyzw", "to_wxyz", "to_xyzw",
                                 "from_hamilton", "to_hamilton"):
                        if hasattr(other, name):
                            stack.enter_context(patch.object(other, name, side_effect=AssertionError(name)))
                    if other is not backend:
                        for name in ("multiply", "rotate"):
                            stack.enter_context(patch.object(other, name, side_effect=AssertionError(name)))
                np.testing.assert_allclose(multiply(q, identity), q, atol=0)
                np.testing.assert_allclose(rotate(q, (1, 0, 0)), (0, 1, 0), atol=1e-14)

    @pytest.mark.skipif(Rotation is None, reason="SciPy is not installed")
    def test_scipy_development_oracle_for_all_four_modules(self):
        rng = np.random.default_rng(417)
        hqs = rng.normal(size=(20, 1, 4)) * rng.uniform(.2, 4, size=(20, 1, 1))
        hrs = rng.normal(size=(1, 3, 4)) * rng.uniform(.2, 4, size=(1, 3, 1))
        v = rng.normal(size=(1, 3, 3))
        for convention, order, backend in self.backends:
            qs = self.from_h_physical(hqs, convention, order)
            rs = self.from_h_physical(hrs, convention, order)
            actual = backend.rotate(backend.multiply(qs, rs), v)
            for i in range(20):
                for j in range(3):
                    hq, hr = hqs[i, 0], hrs[0, j]
                    left = Rotation.from_quat(hq[[1, 2, 3, 0]])
                    right = Rotation.from_quat(hr[[1, 2, 3, 0]])
                    np.testing.assert_allclose(actual[i, j], (left * right).apply(v[0, j]),
                                               rtol=1e-13, atol=1e-13)
            rotation_vectors = rng.normal(size=(12, 3))
            q = backend.from_rotvec(rotation_vectors)
            expected = Rotation.from_rotvec(rotation_vectors)
            vec = rng.normal(size=(12, 3))
            np.testing.assert_allclose(backend.rotate(q, vec), expected.apply(vec), rtol=1e-13, atol=1e-13)
