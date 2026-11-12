"""Numerical coverage for newly compiled NumPy batch operations."""

from importlib.util import find_spec

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.hamilton import wxyz as hamilton_wxyz
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw

NUMBA_AVAILABLE = find_spec("numba") is not None
COUNT = 128
BACKENDS = (
    (hamilton_wxyz, hamilton_scalar, "wxyz"),
    (hamilton_xyzw, hamilton_scalar, "xyzw"),
    (jpl_wxyz, jpl_scalar, "wxyz"),
    (jpl_xyzw, jpl_scalar, "xyzw"),
)


def _scalar_order(values, scalar, layout):
    if scalar is hamilton_scalar and layout == "xyzw":
        return values[..., (3, 0, 1, 2)]
    if scalar is jpl_scalar and layout == "wxyz":
        return values[..., (1, 2, 3, 0)]
    return values


def _array_order(values, scalar, layout):
    if scalar is hamilton_scalar and layout == "xyzw":
        return values[..., (1, 2, 3, 0)]
    if scalar is jpl_scalar and layout == "wxyz":
        return values[..., (3, 0, 1, 2)]
    return values


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_axis_angle_paths_match_scalar(backend, scalar, layout):
    rng = np.random.default_rng(401)
    axis_angles = np.column_stack(
        (rng.normal(size=(COUNT, 3)), rng.uniform(-np.pi, np.pi, COUNT))
    )
    axis_angles[0] = (1.0, 0.0, 0.0, 0.0)
    axis_angles[1] = (0.0, 1.0, 0.0, np.pi)
    axis_angles[2] = (1e308, 1e308, 0.0, -np.pi)
    axis_angles[3] = (1.0, 0.0, 0.0, np.nextafter(0.0, 1.0))

    actual = backend.from_axis_angle(axis_angles)
    expected = np.asarray(
        [scalar.from_axis_angle(tuple(row)) for row in axis_angles]
    )
    assert_allclose(actual, _array_order(expected, scalar, layout), rtol=4e-14, atol=4e-14)

    scalar_quaternions = _scalar_order(actual, scalar, layout)
    expected_axis_angles = np.asarray(
        [scalar.to_axis_angle(tuple(row)) for row in scalar_quaternions]
    )
    assert_allclose(
        backend.to_axis_angle(actual), expected_axis_angles,
        rtol=4e-13, atol=1e-300,
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_analytic_paths_match_scalar(backend, scalar, layout):
    rng = np.random.default_rng(402)
    scalar_values = rng.normal(scale=0.35, size=(COUNT, 4))
    scalar_values[0] = (-2.0, 0.0, 0.0, 0.0)
    scalar_values[1] = (0.5, 1e-300, 0.0, 0.0)
    array_values = _array_order(scalar_values, scalar, layout)
    operations = (
        "exp", "log", "sin", "cos", "sinh", "cosh", "tan", "tanh",
        "asin", "acos", "atan", "asinh", "acosh", "atanh", "expm1",
        "log1p", "log2", "log10",
    )
    for operation in operations:
        actual = getattr(backend, operation)(array_values)
        expected = np.asarray(
            [getattr(scalar, operation)(tuple(row)) for row in scalar_values]
        )
        assert_allclose(
            actual,
            _array_order(expected, scalar, layout),
            rtol=1e-11,
            atol=1e-11,
            err_msg=f"{backend.__name__}.{operation}",
        )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_log_keeps_finite_result_when_norm_exceeds_float64(
    backend, scalar, layout
):
    for magnitude in (1e308, 1e-300):
        scalar_values = np.full((COUNT, 4), magnitude)
        array_values = _array_order(scalar_values, scalar, layout)
        actual = backend.log(array_values)
        expected = np.asarray([scalar.log(tuple(row)) for row in scalar_values])
        assert_allclose(
            actual,
            _array_order(expected, scalar, layout),
            rtol=2e-14,
            atol=2e-14,
        )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_power_paths_match_scalar_and_keep_zero_rules(backend, scalar, layout):
    rng = np.random.default_rng(403)
    scalar_values = rng.normal(scale=0.4, size=(COUNT, 4))
    scalar_values[0] = (-2.0, 0.0, 0.0, 0.0)
    array_values = _array_order(scalar_values, scalar, layout)
    for exponent in (3, -2, 1.25, -0.75):
        actual = backend.power(array_values, exponent)
        expected = np.asarray(
            [scalar.power(tuple(row), exponent) for row in scalar_values]
        )
        assert_allclose(
            actual, _array_order(expected, scalar, layout),
            rtol=8e-12, atol=8e-12,
        )

    scalar_exponents = rng.normal(scale=0.2, size=(COUNT, 4))
    scalar_exponents[0] = 0.0
    array_exponents = _array_order(scalar_exponents, scalar, layout)
    actual = backend.power_quaternion(array_values, array_exponents)
    expected = np.asarray(
        [
            scalar.power_quaternion(tuple(value), tuple(exponent))
            for value, exponent in zip(scalar_values, scalar_exponents)
        ]
    )
    assert_allclose(
        actual, _array_order(expected, scalar, layout),
        rtol=1e-11, atol=1e-11,
    )

    zero = np.zeros((COUNT, 4))
    assert_allclose(backend.power(zero, 2), zero)
    scalar_identity = (
        (1.0, 0.0, 0.0, 0.0)
        if scalar is hamilton_scalar
        else (0.0, 0.0, 0.0, 1.0)
    )
    identity = _array_order(np.asarray(scalar_identity), scalar, layout)
    assert_allclose(backend.power(zero, 0), np.tile(identity, (COUNT, 1)))
    with pytest.raises(ValueError, match="negative power"):
        backend.power(zero, -1)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_skew_and_so3_jacobians_match_scalar(backend, scalar, layout):
    rng = np.random.default_rng(404)
    vectors = rng.normal(scale=0.4, size=(COUNT, 3))
    vectors[0] = 0.0
    vectors[1] = (1e-10, -2e-10, 3e-10)
    assert_allclose(
        backend.skew_matrix(vectors),
        np.asarray([scalar.skew_matrix(tuple(row)) for row in vectors]),
        rtol=3e-14,
        atol=3e-14,
    )
    for operation in (
        "so3_left_jacobian", "so3_right_jacobian",
        "so3_left_jacobian_inverse", "so3_right_jacobian_inverse",
    ):
        actual = getattr(backend, operation)(vectors)
        expected = np.asarray(
            [getattr(scalar, operation)(tuple(row)) for row in vectors]
        )
        assert_allclose(actual, expected, rtol=5e-12, atol=5e-12)
    singular = vectors.copy()
    singular[0] = (2.0 * np.pi, 0.0, 0.0)
    with pytest.raises(ValueError, match="singular"):
        backend.so3_left_jacobian_inverse(singular)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_batch_estimators_match_scalar(backend, scalar, layout):
    rng = np.random.default_rng(405)
    scalar_vectors = rng.normal(scale=0.15, size=(COUNT, 6, 3))
    native_samples = backend.from_rotvec(scalar_vectors)
    scalar_samples = _scalar_order(native_samples, scalar, layout)
    weights = rng.random((COUNT, 6))

    for method in ("markley", "intrinsic"):
        actual = backend.rotation_mean(native_samples, weights, method=method)
        expected = np.asarray(
            [
                scalar.rotation_mean(group, row, method=method)
                for group, row in zip(scalar_samples, weights)
            ]
        )
        assert_allclose(
            actual, _array_order(expected, scalar, layout),
            rtol=5e-11, atol=5e-11,
        )

    for frame in ("body", "world"):
        actual = backend.rotation_covariance(
            native_samples, weights=weights, frame=frame
        )
        expected = np.asarray(
            [
                scalar.rotation_covariance(group, weights=row, frame=frame)
                for group, row in zip(scalar_samples, weights)
            ]
        )
        assert_allclose(actual, expected, rtol=5e-11, atol=5e-11)

    source = np.broadcast_to(np.eye(3), (COUNT, 3, 3)).copy()
    rotations = backend.from_rotvec(rng.normal(scale=0.25, size=(COUNT, 3)))
    target = backend.rotate(rotations[:, None, :], source)
    vector_weights = rng.random((COUNT, 3))
    actual = backend.align_vectors(source, target, weights=vector_weights)
    expected = np.asarray(
        [
            scalar.align_vectors(left, right, row)
            for left, right, row in zip(source, target, vector_weights)
        ]
    )
    assert_allclose(
        actual, _array_order(expected, scalar, layout),
        rtol=6e-11, atol=6e-11,
    )

    sequence = np.broadcast_to(native_samples[:, :4, :], (COUNT, 4, 4)).copy()
    sequence[:, 1::2, :] *= -1.0
    scalar_index = 3 if layout == "xyzw" else 0
    vector_index = 0 if layout == "xyzw" else 1
    sequence[0, 0, :] = 0.0
    sequence[0, 0, scalar_index] = 1.0
    sequence[0, 1, :] = 0.0
    sequence[0, 1, vector_index] = 1.0
    actual = backend.make_sign_continuous(sequence)
    scalar_sequence = _scalar_order(sequence, scalar, layout)
    expected = np.asarray(
        [scalar.make_sign_continuous(group) for group in scalar_sequence]
    )
    assert_allclose(actual, _array_order(expected, scalar, layout), atol=0.0)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_numba_fused_quotient_and_distance_paths_match_scalar(
    backend, scalar, layout
):
    rng = np.random.default_rng(406)
    batch_size = 8192
    scalar_lhs = rng.normal(scale=0.4, size=(batch_size, 4))
    scalar_rhs = rng.normal(scale=0.4, size=(batch_size, 4))
    scalar_lhs[0] *= 1e300
    scalar_rhs[0] *= 1e300
    scalar_lhs[1] *= 1e-300
    scalar_rhs[1] *= 1e-300
    lhs = _array_order(scalar_lhs, scalar, layout)
    rhs = _array_order(scalar_rhs, scalar, layout)

    for operation in ("left_divide", "right_divide"):
        actual = getattr(backend, operation)(lhs, rhs)
        expected = np.asarray(
            [
                getattr(scalar, operation)(tuple(left), tuple(right))
                for left, right in zip(scalar_lhs, scalar_rhs)
            ]
        )
        assert_allclose(
            actual,
            _array_order(expected, scalar, layout),
            rtol=2e-11,
            atol=2e-11,
            err_msg=f"{backend.__name__}.{operation}",
        )

    actual_distance = backend.rotation_distance(lhs, rhs)
    expected_distance = np.asarray(
        [
            scalar.rotation_distance(tuple(left), tuple(right))
            for left, right in zip(scalar_lhs, scalar_rhs)
        ]
    )
    assert_allclose(actual_distance, expected_distance, rtol=2e-12, atol=2e-12)
    tolerance = 0.8
    assert_array_equal(
        backend.rotations_equivalent(lhs, rhs, atol=tolerance),
        expected_distance <= tolerance,
    )

    zero = np.zeros_like(lhs)
    with pytest.raises(ValueError, match="zero quaternion"):
        backend.left_divide(zero, rhs)
    with pytest.raises(ValueError, match="zero quaternion"):
        backend.right_divide(lhs, zero)
    with pytest.raises(ValueError, match="zero quaternion"):
        backend.rotation_distance(zero, rhs)
