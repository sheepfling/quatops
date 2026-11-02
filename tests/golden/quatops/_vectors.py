"""Fixed, hand-computable cases shared by tests mirroring public backends."""

import math

import pytest


class QuaternionGoldens:
    backend = None
    convention = None
    order = None
    representation = None

    def _ordered(self, w, x, y, z):
        return (w, x, y, z) if self.order == "wxyz" else (x, y, z, w)

    def _assert_values(self, actual, expected, *, abs_tol=1e-14):
        values = tuple(float(component) for component in actual)
        assert len(values) == len(expected)
        for got, want in zip(values, expected):
            assert got == pytest.approx(want, abs=abs_tol)

    def _assert_quaternion(self, value, expected):
        if self.representation == "tuple":
            assert type(value) is tuple
        elif self.representation in ("namedtuple", "dataclass"):
            assert isinstance(value, self.backend.Quaternion)
        else:
            import numpy as np
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.dtype("float64")
        self._assert_values(value, expected)

    def test_layout_and_magnitude_are_explicit(self):
        wxyz, xyzw = (1.0, 2.0, 2.0, 4.0), (2.0, 2.0, 4.0, 1.0)
        q = self.backend.from_wxyz(wxyz)
        self._assert_quaternion(q, self._ordered(*wxyz))
        self._assert_values(self.backend.to_wxyz(q), wxyz)
        self._assert_quaternion(self.backend.from_xyzw(xyzw), self._ordered(*wxyz))
        self._assert_values(self.backend.to_xyzw(q), xyzw)
        assert float(self.backend.norm(q)) == pytest.approx(5.0)

    def test_conjugate_normalize_inverse_and_product(self):
        q = self.backend.from_wxyz((1.0, 2.0, 2.0, 4.0))
        self._assert_quaternion(self.backend.conjugate(q), self._ordered(1, -2, -2, -4))
        self._assert_quaternion(self.backend.normalize(q), self._ordered(.2, .4, .4, .8))
        self._assert_quaternion(self.backend.inverse(q), self._ordered(.04, -.08, -.08, -.16))
        self._assert_values(self.backend.multiply(q, self.backend.inverse(q)), self._ordered(1, 0, 0, 0))
        first, second = [0.] * 4, [0.] * 4
        i, j, k = ((1, 2, 3) if self.order == "wxyz" else (0, 1, 2))
        first[i], second[j] = 1., 1.
        expected = [0.] * 4
        expected[k] = 1. if self.convention == "hamilton" else -1.
        self._assert_values(self.backend.multiply(first, second), expected)

    def test_rotation_vector_axis_angle_and_active_rotation(self):
        half = math.sqrt(.5)
        z = half if self.convention == "hamilton" else -half
        expected_q = self._ordered(half, 0, 0, z)
        q = self.backend.from_axis_angle((0., 0., 1., math.pi / 2))
        self._assert_quaternion(q, expected_q)
        self._assert_values(self.backend.to_axis_angle(q), (0, 0, 1, math.pi / 2))
        from_rotvec = self.backend.from_rotvec((0, 0, math.pi / 2))
        self._assert_quaternion(from_rotvec, expected_q)
        self._assert_values(self.backend.to_rotvec(q), (0, 0, math.pi / 2))
        self._assert_values(self.backend.rotate(q, (1, 0, 0)), (0, 1, 0))

    def test_jpl_hamilton_conversion_when_exported(self):
        if self.convention != "jpl":
            pytest.skip("cross-convention conversion is exported by JPL modules")
        expected = self._ordered(1, -2, -2, -4)
        q = self.backend.from_hamilton((1, 2, 2, 4))
        self._assert_quaternion(q, expected)
        self._assert_values(self.backend.to_hamilton(q), (1, 2, 2, 4))
