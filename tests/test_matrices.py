"""Matrix representations for both quaternion algebras and NumPy layouts."""

import pytest
import math

import numpy as np

from quatops import scalar as hamilton
from quatops.jpl import scalar as jpl_scalar
from quatops import dataclass as hamilton_dataclass
from quatops import namedtuple as hamilton_namedtuple
from quatops.jpl import dataclass as jpl_dataclass
from quatops.jpl import namedtuple as jpl_namedtuple
from quatops.numpy.hamilton import wxyz as h_numpy_wxyz
from quatops.numpy.hamilton import xyzw as h_numpy_xyzw
from quatops.numpy.jpl import wxyz as j_numpy_wxyz
from quatops.numpy.jpl import xyzw as j_numpy_xyzw


SCALAR_BACKENDS = (("Hamilton", hamilton, (1.0, 2.0, -3.0, 4.0),
                    (2.0, -1.0, 0.5, 3.0)),
                   ("JPL", jpl_scalar, (2.0, -3.0, 4.0, 1.0),
                    (-1.0, 0.5, 3.0, 2.0)))
NUMPY_BACKENDS = (
    ("Hamilton wxyz", h_numpy_wxyz, False, 1),
    ("Hamilton xyzw", h_numpy_xyzw, True, 1),
    ("JPL wxyz", j_numpy_wxyz, False, -1),
    ("JPL xyzw", j_numpy_xyzw, True, -1),
)


class MatrixTests:
    def check_backend(self, name, backend, q, r):
        left = np.asarray(backend.to_left_matrix(q))
        right = np.asarray(backend.to_right_matrix(r))
        product = np.asarray(tuple(backend.multiply(q, r)))
        np.testing.assert_allclose(left @ np.asarray(tuple(r)), product, rtol=1e-14, atol=1e-14,
                                   err_msg=f"left product: {name}")
        np.testing.assert_allclose(right @ np.asarray(tuple(q)), product, rtol=1e-14, atol=1e-14,
                                   err_msg=f"right product: {name}")
        left_r = np.asarray(backend.to_left_matrix(r))
        right_q = np.asarray(backend.to_right_matrix(q))
        np.testing.assert_allclose(left @ left_r,
                                   backend.to_left_matrix(backend.multiply(q, r)),
                                   rtol=1e-14, atol=1e-14,
                                   err_msg=f"left composition: {name}")
        np.testing.assert_allclose(right @ right_q,
                                   backend.to_right_matrix(backend.multiply(q, r)),
                                   rtol=1e-14, atol=1e-14,
                                   err_msg=f"right composition: {name}")
        np.testing.assert_allclose(left @ right_q, right_q @ left,
                                   rtol=1e-14, atol=1e-14,
                                   err_msg=f"left/right commutation: {name}")
        np.testing.assert_allclose(backend.to_rotation_matrix(q) @ np.array((1., 2., 3.)),
                                   backend.rotate(q, (1., 2., 3.)), rtol=1e-14, atol=1e-14,
                                   err_msg=f"active rotation: {name}")

    def test_scalar_matrix_identities(self):
        for name, backend, q, r in SCALAR_BACKENDS:
            self.check_backend(name, backend, q, r)

    def test_typed_values_and_explicit_module_wrappers(self):
        cases = (
            ("Hamilton dataclass", hamilton_dataclass,
             hamilton_dataclass.Quaternion(1., 2., -3., 4.),
             hamilton_dataclass.Quaternion(2., -1., .5, 3.)),
            ("Hamilton NamedTuple", hamilton_namedtuple,
             hamilton_namedtuple.Quaternion(1., 2., -3., 4.),
             hamilton_namedtuple.Quaternion(2., -1., .5, 3.)),
            ("JPL dataclass", jpl_dataclass,
             jpl_dataclass.Quaternion(2., -3., 4., 1.),
             jpl_dataclass.Quaternion(-1., .5, 3., 2.)),
            ("JPL NamedTuple", jpl_namedtuple,
             jpl_namedtuple.Quaternion(2., -3., 4., 1.),
             jpl_namedtuple.Quaternion(-1., .5, 3., 2.)),
        )
        for name, backend, q, r in cases:
            self.check_backend(name, backend, q, r)

    def test_numpy_matrix_identities_in_each_layout(self):
        for name, backend, scalar_last, _epsilon in NUMPY_BACKENDS:
            q = (2., -3., 4., 1.) if "JPL" in name else (1., 2., -3., 4.)
            r = (-1., .5, 3., 2.) if "JPL" in name else (2., -1., .5, 3.)
            if scalar_last:
                q, r = q[1:] + q[:1], r[1:] + r[:1]
            self.check_backend(name, backend, np.array(q), np.array(r))

    def test_jpl_and_hamilton_active_quarter_turn(self):
        s = math.sqrt(.5)
        expected = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        for name, backend, scalar_last, epsilon in NUMPY_BACKENDS:
            q = (0., 0., -s, s) if epsilon < 0 else (s, 0., 0., s)
            if scalar_last and epsilon > 0:
                q = q[1:] + q[:1]
            elif not scalar_last and epsilon < 0:
                q = q[3:] + q[:3]
            np.testing.assert_allclose(backend.to_rotation_matrix(q), expected,
                                       atol=2e-15)
            np.testing.assert_allclose(backend.rotate(q, (1., 0., 0.)), (0., 1., 0.),
                                       atol=2e-15)

    def test_numpy_batch_shapes_and_layout_invariance(self):
        values = np.array(((1., 0., 0., 0.), (math.sqrt(.5), 0., 0., math.sqrt(.5))))
        reference = h_numpy_wxyz.to_rotation_matrix(values)
        for name, backend, scalar_last, _epsilon in NUMPY_BACKENDS:
            q = values.copy()
            if _epsilon < 0:
                q[:, 1:] *= -1
            if scalar_last:
                q = q[:, (1, 2, 3, 0)]
            rotation = backend.to_rotation_matrix(q)
            assert rotation.shape == (2, 3, 3)
            assert backend.to_left_matrix(q).shape == (2, 4, 4)
            assert backend.to_right_matrix(q).shape == (2, 4, 4)
            np.testing.assert_allclose(rotation, reference, atol=2e-15)
            assert backend.to_rotation_matrix(np.empty((0, 4))).shape == (0, 3, 3)

    def test_rotation_matrix_rejects_zero(self):
        for name, backend, scalar_last, _epsilon in NUMPY_BACKENDS:
            with pytest.raises(ValueError):
                backend.to_rotation_matrix(np.zeros(4))
