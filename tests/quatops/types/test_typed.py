"""Contract and reference checks for the named-field scalar backend."""

import math

import numpy as np
import pytest

from quatops import dataclass, namedtuple, scalar

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


class BackendAssertions:
    backend = None
    def assert_components_close(self, actual, expected, *, rel_tol=1e-14, abs_tol=1e-14):
        actual, expected = tuple(actual), tuple(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=rel_tol, atol=abs_tol)

    def test_named_fields_and_tuple_interoperability(self):
        q = self.backend.Quaternion(w=2, x=3, y=4, z=5)
        assert (q.w, q.x, q.y, q.z) == (2, 3, 4, 5)
        assert tuple(q) == (2, 3, 4, 5)
        with pytest.raises(AttributeError):
            q.w = 9
        # Values can be used with the existing scalar backend directly.
        assert scalar.conjugate(q) == (2, -3, -4, -5)

    def test_layout_conversion_types_and_no_hidden_normalization(self):
        q = self.backend.from_xyzw([3, 4, 5, 2])
        assert type(q) is self.backend.Quaternion
        assert tuple(q) == (2, 3, 4, 5)
        assert q.w == 2
        xyzw = self.backend.to_xyzw(q)
        assert type(xyzw) is tuple
        assert xyzw == (3, 4, 5, 2)
        assert type(self.backend.from_xyzw(xyzw)) is self.backend.Quaternion
        assert self.backend.from_xyzw(xyzw) == q
        np.testing.assert_allclose(self.backend.norm(q), math.sqrt(54), rtol=0, atol=5e-8)
        assert tuple(self.backend.multiply(q, (1, 0, 0, 0))) == tuple(q)
        np.testing.assert_allclose(self.backend.norm(self.backend.normalize(q)), 1.0, rtol=0, atol=5e-8)

    def test_all_operations_match_scalar_and_return_types(self):
        q = self.backend.Quaternion(2, 3, -4, 5)
        r = self.backend.Quaternion(3, -1, 2, 4)
        for name, args in (
            ('from_xyzw', (q,)), ('conjugate', (q,)), ('multiply', (q, r)),
            ('normalize', (q,)), ('inverse', (q,)),
        ):
            actual = getattr(self.backend, name)(*args)
            assert type(actual) is self.backend.Quaternion
            assert all(isinstance(component, float) for component in actual)
            self.assert_components_close(actual, getattr(scalar, name)(*args))
        assert type(self.backend.norm(q)) is float
        assert self.backend.norm(q) == scalar.norm(q)
        vector = self.backend.rotate(q, (1, 2, 3))
        assert type(vector) is tuple
        self.assert_components_close(vector, scalar.rotate(q, (1, 2, 3)))

    def test_known_rotations_inverse_and_extremes(self):
        h = math.sqrt(0.5)
        z = self.backend.Quaternion(2*h, 0, 0, 2*h)
        x = self.backend.Quaternion(h, h, 0, 0)
        self.assert_components_close(self.backend.rotate(z, (1, 0, 0)), (0, 1, 0))
        self.assert_components_close(self.backend.rotate(self.backend.multiply(z, x), (0, 1, 0)),
                                   self.backend.rotate(z, self.backend.rotate(x, (0, 1, 0))))
        self.assert_components_close(self.backend.rotate(tuple(-v for v in z), (1, 0, 0)), (0, 1, 0))
        for scale in (1e-300, 1e300):
            q = self.backend.Quaternion(scale, -scale, scale, -scale)
            np.testing.assert_allclose(self.backend.norm(q), 2*scale, rtol=1e-15, atol=0)
            self.assert_components_close(self.backend.normalize(q), (.5, -.5, .5, -.5))
            self.assert_components_close(self.backend.multiply(q, self.backend.inverse(q)), (1, 0, 0, 0))

    def test_invalid_zero_and_overflow(self):
        # NamedTuple construction itself is unchecked; operations validate.
        for name in ('from_xyzw', 'to_xyzw', 'conjugate', 'norm', 'normalize', 'inverse'):
            fn = getattr(self.backend, name)
            for bad in ((1, 2, 3), (1, 2, 3, 4, 5), (1, 2, '3', 4),
                        self.backend.Quaternion(1, 2, float('nan'), 4), (1, 2, float('inf'), 4)):
                with pytest.raises(ValueError):
                    fn(bad)
        zero = self.backend.Quaternion(0, 0, 0, 0)
        assert self.backend.norm(zero) == 0
        for fn in (self.backend.normalize, self.backend.inverse):
            with pytest.raises(ValueError):
                fn(zero)
        with pytest.raises(ValueError):
            self.backend.rotate(zero, (1, 0, 0))
        with pytest.raises(ValueError):
            self.backend.rotate((1, 0, 0, 0), (1, 0))
        with pytest.raises(ValueError):
            self.backend.multiply((1, 0, 0, 0), (1, 0, 0))
        with pytest.raises(OverflowError):
            self.backend.multiply((1e308, 0, 0, 0), (1e308, 0, 0, 0))

    @pytest.mark.skipif(Rotation is None, reason='SciPy is not installed')
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
            self.assert_components_close(self.backend.normalize(q), expected_unit)
            self.assert_components_close(self.backend.inverse(q), expected_inverse)
            self.assert_components_close(self.backend.normalize(self.backend.multiply(q, r)), expected_product)
            self.assert_components_close(self.backend.rotate(q, v), sq.apply(v))
            self.assert_components_close(self.backend.rotate(self.backend.multiply(q, r), v), (sq * sr).apply(v))


class NamedTupleTests(BackendAssertions):
    backend = namedtuple

    def test_is_indexable_tuple(self):
        q = self.backend.Quaternion(1, 2, 3, 4)
        assert isinstance(q, tuple)
        assert q[2] == 3


class DataclassTests(BackendAssertions):
    backend = dataclass

    def test_dataclass_is_not_a_tuple(self):
        q = self.backend.Quaternion(1, 2, 3, 4)
        assert not isinstance(q, tuple)

    def test_inverse_reuses_one_reciprocal_and_matches_scalar_extremes(self):
        for q in (
            (2.1, -3.4, 0.7, 1.2),
            (1e308, -4e307, 2e307, 1e307),
            (1e-308, 2e-308, -1e-308, 0.0),
        ):
            self.assert_components_close(
                dataclass.inverse(q), scalar.inverse(q), rel_tol=1e-15, abs_tol=0.0
            )
