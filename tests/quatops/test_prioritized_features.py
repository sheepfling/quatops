"""Behavior and numerical conventions for the prioritized quaternion APIs."""

import math

import numpy as np
import pytest

from quatops.hamilton import dataclass as hd
from quatops.hamilton import namedtuple as hn
from quatops.hamilton import scalar as h
from quatops.jpl import dataclass as jd
from quatops.jpl import namedtuple as jn
from quatops.jpl import scalar as j
from quatops.numpy.hamilton import wxyz as hw
from quatops.numpy.hamilton import xyzw as hx
from quatops.numpy.jpl import wxyz as jw
from quatops.numpy.jpl import xyzw as jx

SCALARS = (h, j, hd, hn, jd, jn)
TYPED = (hd, hn, jd, jn)
ARRAYS = (hw, hx, jw, jx)


@pytest.mark.parametrize("backend", SCALARS + ARRAYS)
def test_powers_roots_and_zero_rules(backend):
    q = backend.from_wxyz((1.0, 0.2, -0.3, 0.4))
    squared = backend.power(q, 2)
    if backend in TYPED:
        assert isinstance(squared, backend.Quaternion)
    expected = backend.multiply(q, q)
    np.testing.assert_allclose(backend.to_wxyz(squared), backend.to_wxyz(expected), atol=1e-14)
    cubed = backend.power(q, 3)
    expected_cubed = backend.multiply(expected, q)
    np.testing.assert_allclose(backend.to_wxyz(cubed), backend.to_wxyz(expected_cubed), atol=2e-14)
    squared_inverse = backend.power(q, -2)
    np.testing.assert_allclose(backend.to_wxyz(squared_inverse),
                               backend.to_wxyz(backend.inverse(expected)), atol=2e-14)
    root = backend.sqrt(q)
    np.testing.assert_allclose(backend.to_wxyz(backend.multiply(root, root)),
                               backend.to_wxyz(q), atol=2e-14)
    np.testing.assert_allclose(backend.to_wxyz(backend.power(q, 0)), (1, 0, 0, 0), atol=0)
    np.testing.assert_allclose(backend.to_wxyz(backend.power((0, 0, 0, 0), 2)), (0, 0, 0, 0))
    huge_integer = 10**400
    identity = backend.power((1, 0, 0, 0), huge_integer)
    np.testing.assert_allclose(backend.to_wxyz(identity), (1, 0, 0, 0), atol=0)
    negative_real_root = backend.sqrt(backend.from_wxyz((-1, 0, 0, 0)))
    np.testing.assert_allclose(backend.to_wxyz(negative_real_root), (0, 1, 0, 0), atol=1e-14)
    with pytest.raises(ValueError):
        backend.power((0, 0, 0, 0), -1)


@pytest.mark.parametrize("backend", SCALARS + ARRAYS)
def test_hyperbolic_functions_and_identities(backend):
    q = backend.from_wxyz((0.3, 0.2, -0.4, 0.1))
    w, x, y, z = backend.to_wxyz(q)
    radius = math.hypot(x, y, z)
    sinc = 1.0 if radius == 0 else math.sin(radius) / radius
    expected_sinh = (math.sinh(w) * math.cos(radius),
                     *(v * math.cosh(w) * sinc for v in (x, y, z)))
    expected_cosh = (math.cosh(w) * math.cos(radius),
                     *(v * math.sinh(w) * sinc for v in (x, y, z)))
    np.testing.assert_allclose(backend.to_wxyz(backend.sinh(q)), expected_sinh, atol=2e-14)
    np.testing.assert_allclose(backend.to_wxyz(backend.cosh(q)), expected_cosh, atol=2e-14)
    ratio = backend.divide(backend.sinh(q), backend.cosh(q))
    np.testing.assert_allclose(backend.to_wxyz(backend.tanh(q)), backend.to_wxyz(ratio), atol=2e-14)


@pytest.mark.parametrize("backend", SCALARS + ARRAYS)
def test_slerp_uses_shortest_path_and_supports_extrapolation(backend):
    q0 = backend.from_rotation_vector((0, 0, 0))
    q1 = backend.from_rotation_vector((0, 0, math.pi / 2))
    half = backend.slerp(q0, q1, 0.5)
    np.testing.assert_allclose(backend.rotate(half, (1, 0, 0)),
                               (math.sqrt(.5), math.sqrt(.5), 0), atol=2e-14)
    same = backend.slerp(q1, tuple(-x for x in q1), 0.5)
    np.testing.assert_allclose(backend.to_rotation_matrix(same),
                               backend.to_rotation_matrix(q1), atol=2e-14)
    np.testing.assert_allclose(backend.to_rotation_matrix(backend.slerp(q0, q1, 0.0)),
                               backend.to_rotation_matrix(q0), atol=2e-14)
    np.testing.assert_allclose(backend.to_rotation_matrix(backend.slerp(q0, q1, 1.0)),
                               backend.to_rotation_matrix(q1), atol=2e-14)
    outside = backend.slerp(q0, q1, 1.5)
    np.testing.assert_allclose(tuple(backend.to_rotation_vector(outside)), (0, 0, 3 * math.pi / 4),
                               atol=2e-14)
    with pytest.raises(ValueError):
        backend.slerp(q1, tuple(-x for x in q1), .5, shortest=False)
    with pytest.raises(ValueError):
        backend.slerp(q0, q1, .5, shortest=1)


@pytest.mark.parametrize("backend", SCALARS + ARRAYS)
def test_so3_jacobians_invert_each_other_and_are_stable_at_zero(backend):
    for vector in ((0, 0, 0), (1e-10, -2e-10, 3e-10), (.2, -.4, .7)):
        left = np.asarray(backend.so3_left_jacobian(vector))
        right = np.asarray(backend.so3_right_jacobian(vector))
        inverse = np.asarray(backend.so3_left_jacobian_inverse(vector))
        inverse_right = np.asarray(backend.so3_right_jacobian_inverse(vector))
        np.testing.assert_allclose(left @ inverse, np.eye(3), atol=2e-14)
        np.testing.assert_allclose(right @ inverse_right, np.eye(3), atol=2e-14)
        np.testing.assert_allclose(right, backend.so3_left_jacobian(tuple(-x for x in vector)), atol=2e-14)
    vector = np.asarray((.2, -.4, .7))
    theta = np.linalg.norm(vector)
    x, y, z = vector
    cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    reference = (np.eye(3) + ((1 - math.cos(theta)) / theta**2) * cross
                 + ((theta - math.sin(theta)) / theta**3) * (cross @ cross))
    np.testing.assert_allclose(backend.so3_left_jacobian(vector), reference, atol=2e-14)
    with pytest.raises(ValueError):
        backend.so3_left_jacobian_inverse((0, 0, 2 * math.pi))


@pytest.mark.parametrize("backend", SCALARS + ARRAYS)
def test_angular_velocity_integration_respects_frame(backend):
    half = math.sqrt(.5)
    attitude = backend.from_wxyz((half, 0, 0, half))
    omega = (math.pi / 2, 0, 0)
    body = backend.integrate_angular_velocity(attitude, omega, 1.0, frame="body")
    world = backend.integrate_angular_velocity(attitude, omega, 1.0, frame="world")
    expected_body = backend.multiply(attitude, backend.from_rotation_vector((math.pi / 2, 0, 0)))
    expected_world = backend.multiply(backend.from_rotation_vector((math.pi / 2, 0, 0)), attitude)
    np.testing.assert_allclose(backend.to_rotation_matrix(body), backend.to_rotation_matrix(expected_body),
                               atol=2e-14)
    np.testing.assert_allclose(backend.to_rotation_matrix(world), backend.to_rotation_matrix(expected_world),
                               atol=2e-14)
    assert not np.allclose(backend.to_rotation_matrix(body), backend.to_rotation_matrix(world))
    with pytest.raises(ValueError):
        backend.integrate_angular_velocity(attitude, omega, 1.0, frame="camera")


@pytest.mark.parametrize("backend", SCALARS + ARRAYS)
def test_skew_matrix_has_the_cross_product_orientation(backend):
    vector = (1.0, -2.0, 3.0)
    expected = np.array(((0.0, -3.0, -2.0), (3.0, 0.0, -1.0), (2.0, 1.0, 0.0)))
    np.testing.assert_array_equal(backend.skew_matrix(vector), expected)


@pytest.mark.parametrize("backend", ARRAYS)
def test_numpy_integer_power_checks_overflow_without_changing_numpy_policy(backend):
    previous = np.seterr(over="raise", invalid="raise")
    try:
        with pytest.raises(OverflowError):
            backend.power((1e308, 0.0, 0.0, 0.0), 2)
        assert np.geterr()["over"] == "raise"
        assert np.geterr()["invalid"] == "raise"
    finally:
        np.seterr(**previous)


def test_numpy_batching_for_interpolation_jacobians_and_integration():
    q0 = np.array(((1., 0, 0, 0), (1., 0, 0, 0)))
    q1 = np.array(((1., 0, 0, 0), (math.sqrt(.5), 0, 0, math.sqrt(.5))))
    vectors = np.array(((0., 0, 0), (0., 0, math.pi / 2)))
    for backend, scalar_last, jpl in ((hw, False, False), (hx, True, False),
                                      (jw, False, True), (jx, True, True)):
        a, b = q0.copy(), q1.copy()
        if jpl:
            a[:, 1:] *= -1
            b[:, 1:] *= -1
        if scalar_last:
            a, b = a[:, (1, 2, 3, 0)], b[:, (1, 2, 3, 0)]
        assert backend.slerp(a, b, .5).shape == (2, 4)
        assert backend.so3_left_jacobian(vectors).shape == (2, 3, 3)
        assert backend.integrate_angular_velocity(a, vectors, .25).shape == (2, 4)
        empty = np.empty((0, 4))
        assert backend.slerp(empty, empty, .5).shape == (0, 4)
        assert backend.so3_right_jacobian(np.empty((0, 3))).shape == (0, 3, 3)
