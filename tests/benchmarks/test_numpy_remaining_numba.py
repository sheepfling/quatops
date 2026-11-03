"""Benchmark the remaining optional Numba batch paths against fallbacks.

Run with ``.venv/bin/python -m pytest
tests/benchmarks/test_numpy_remaining_numba.py -m benchmark --benchmark-only
--benchmark-group-by=group --benchmark-columns=median``.
"""

import numpy as np
import pytest
import quaternion
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops import _numpy
from quatops.numpy.hamilton import xyzw

pytestmark = pytest.mark.benchmark

RNG = np.random.default_rng(203)
COUNT = 10_000
ANALYTIC_COUNT = 2_048
ESTIMATOR_GROUPS = 256
ESTIMATOR_SAMPLES = 8

ROTATION_VECTORS = RNG.uniform(-1.0, 1.0, (COUNT, 3))
QUATERNIONS = xyzw.from_rotvec(ROTATION_VECTORS)
AXES = RNG.normal(size=(COUNT, 3))
AXES /= np.linalg.norm(AXES, axis=-1, keepdims=True)
ANGLES = RNG.uniform(-np.pi, np.pi, COUNT)
AXIS_ANGLES = np.column_stack((AXES, ANGLES))
AXIS_ANGLE_ROTATION_VECTORS = AXES * ANGLES[:, None]
AXIS_ANGLE_ROTATIONS = Rotation.from_rotvec(AXIS_ANGLE_ROTATION_VECTORS)
QUATERNION_ROTATIONS = Rotation.from_quat(QUATERNIONS)
NUMPY_QUATERNIONS = quaternion.as_quat_array(
    np.column_stack((QUATERNIONS[:, 3], QUATERNIONS[:, :3]))
)

ANALYTIC_VALUES = RNG.normal(scale=0.35, size=(ANALYTIC_COUNT, 4))
ANALYTIC_QUATERNIONS = xyzw.from_wxyz(ANALYTIC_VALUES)
ANALYTIC_NUMPY_QUATERNIONS = quaternion.as_quat_array(
    np.column_stack(
        (ANALYTIC_QUATERNIONS[:, 3], ANALYTIC_QUATERNIONS[:, :3])
    )
)
ANALYTIC_OPERATIONS = (
    "exp", "log", "sin", "cos", "sinh", "cosh", "tan", "tanh",
    "asin", "acos", "atan", "asinh", "acosh", "atanh", "expm1",
    "log1p", "log2", "log10",
)

POWER_BASES = xyzw.from_rotvec(RNG.normal(scale=0.4, size=(ANALYTIC_COUNT, 3)))
POWER_EXPONENTS = xyzw.from_rotvec(
    RNG.normal(scale=0.15, size=(ANALYTIC_COUNT, 3))
)
JACOBIAN_VECTORS = RNG.normal(scale=0.5, size=(COUNT, 3))

SAMPLE_ROTATION_VECTORS = RNG.normal(
    scale=0.15,
    size=(ESTIMATOR_GROUPS, ESTIMATOR_SAMPLES, 3),
)
SAMPLE_QUATERNIONS = xyzw.from_rotvec(SAMPLE_ROTATION_VECTORS)
SAMPLE_WEIGHTS = RNG.random((ESTIMATOR_GROUPS, ESTIMATOR_SAMPLES))
ALIGN_SOURCE = np.broadcast_to(
    np.eye(3), (ESTIMATOR_GROUPS, 3, 3)
).copy()
ALIGN_QUATERNIONS = xyzw.from_rotvec(
    RNG.normal(scale=0.3, size=(ESTIMATOR_GROUPS, 3))
)
ALIGN_TARGET = xyzw.rotate(ALIGN_QUATERNIONS[:, None, :], ALIGN_SOURCE)
ALIGN_WEIGHTS = RNG.random((ESTIMATOR_GROUPS, 3))
SEQUENCES = np.broadcast_to(
    SAMPLE_QUATERNIONS[:, :ESTIMATOR_SAMPLES, :],
    (ESTIMATOR_GROUPS, ESTIMATOR_SAMPLES, 4),
).copy()
SEQUENCES[:, 1::2, :] *= -1.0


def _array_from_numpy_quaternion(values):
    coefficients = quaternion.as_float_array(values)
    return coefficients[:, (1, 2, 3, 0)]


def _checked_numpy_quaternion(function):
    if not np.all(np.isfinite(QUATERNIONS)):
        raise ValueError("quaternion inputs must be finite")
    result = _array_from_numpy_quaternion(function(NUMPY_QUATERNIONS))
    if not np.all(np.isfinite(result)):
        raise OverflowError("quaternion arithmetic produced a non-finite result")
    return result


def _axis_angle_suite():
    return xyzw.from_axis_angle(AXIS_ANGLES), xyzw.to_axis_angle(QUATERNIONS)


def _analytic_suite():
    return tuple(
        getattr(xyzw, operation)(ANALYTIC_QUATERNIONS)
        for operation in ANALYTIC_OPERATIONS
    )


def _power_suite():
    return (
        xyzw.power(POWER_BASES, 3),
        xyzw.power(POWER_BASES, 1.25),
        xyzw.power_quaternion(POWER_BASES, POWER_EXPONENTS),
    )


def _jacobian_suite():
    return (
        xyzw.skew_matrix(JACOBIAN_VECTORS),
        xyzw.so3_left_jacobian(JACOBIAN_VECTORS),
        xyzw.so3_right_jacobian(JACOBIAN_VECTORS),
        xyzw.so3_left_jacobian_inverse(JACOBIAN_VECTORS),
        xyzw.so3_right_jacobian_inverse(JACOBIAN_VECTORS),
    )


def _estimator_and_sequence_suite():
    return (
        xyzw.make_sign_continuous(SEQUENCES),
        xyzw.rotation_mean(SAMPLE_QUATERNIONS, SAMPLE_WEIGHTS),
        xyzw.rotation_mean(
            SAMPLE_QUATERNIONS, SAMPLE_WEIGHTS, method="intrinsic"
        ),
        xyzw.rotation_covariance(SAMPLE_QUATERNIONS, weights=SAMPLE_WEIGHTS),
        xyzw.align_vectors(ALIGN_SOURCE, ALIGN_TARGET, ALIGN_WEIGHTS),
    )


def _assert_suite_equal(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        assert_allclose(actual_value, expected_value, rtol=2e-11, atol=2e-12)


def _without_numba(monkeypatch):
    monkeypatch.setattr(_numpy, "_numba_rotation_kernel", lambda _operation: None)


def _run_numpy_fallback(monkeypatch, function):
    with monkeypatch.context() as patch:
        _without_numba(patch)
        return function()


def test_axis_angle_numba_vs_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy axis-angle conversions (10k)"
    expected = _run_numpy_fallback(monkeypatch, _axis_angle_suite)
    _assert_suite_equal(_axis_angle_suite(), expected)
    benchmark(_axis_angle_suite)


def test_axis_angle_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy axis-angle conversions (10k)"
    expected = _axis_angle_suite()
    with monkeypatch.context() as patch:
        _without_numba(patch)
        fallback = _axis_angle_suite()
    _assert_suite_equal(fallback, expected)
    _without_numba(monkeypatch)
    benchmark(_axis_angle_suite)


def test_axis_angle_from_scipy_and_numpy_quaternion(benchmark):
    benchmark.group = "NumPy from axis-angle (10k)"
    expected = AXIS_ANGLE_ROTATIONS.as_quat()
    actual = xyzw.from_axis_angle(AXIS_ANGLES)
    assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    benchmark(xyzw.from_axis_angle, AXIS_ANGLES)


def test_axis_angle_from_numpy_quaternion(benchmark):
    benchmark.group = "NumPy from axis-angle (10k)"
    benchmark(
        lambda: _array_from_numpy_quaternion(
            quaternion.from_rotation_vector(AXIS_ANGLE_ROTATION_VECTORS)
        )
    )


def test_axis_angle_from_scipy(benchmark):
    benchmark.group = "NumPy from axis-angle (10k)"
    benchmark(Rotation.from_rotvec, AXIS_ANGLE_ROTATION_VECTORS)


def test_axis_angle_to_scipy(benchmark):
    benchmark.group = "NumPy to axis-angle (10k)"
    expected = QUATERNION_ROTATIONS.as_rotvec()
    actual = xyzw.to_axis_angle(QUATERNIONS)
    assert_allclose(actual[:, :3] * actual[:, 3, None], ROTATION_VECTORS,
                    rtol=3e-13, atol=3e-14)
    assert_allclose(expected, ROTATION_VECTORS, rtol=3e-13, atol=3e-14)
    benchmark(xyzw.to_axis_angle, QUATERNIONS)


def test_axis_angle_to_scipy_prebuilt(benchmark):
    benchmark.group = "NumPy to axis-angle (10k)"
    benchmark(QUATERNION_ROTATIONS.as_rotvec)


def test_analytic_numba_vs_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy quaternion analytics (18 ops, 2k)"
    expected = _run_numpy_fallback(monkeypatch, _analytic_suite)
    _assert_suite_equal(_analytic_suite(), expected)
    benchmark(_analytic_suite)


def test_analytic_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy quaternion analytics (18 ops, 2k)"
    expected = _analytic_suite()
    with monkeypatch.context() as patch:
        _without_numba(patch)
        fallback = _analytic_suite()
    _assert_suite_equal(fallback, expected)
    _without_numba(monkeypatch)
    benchmark(_analytic_suite)


def test_exp_numba_vs_numpy_quaternion(benchmark):
    benchmark.group = "NumPy quaternion exp (10k)"
    expected = _array_from_numpy_quaternion(np.exp(NUMPY_QUATERNIONS))
    actual = xyzw.exp(QUATERNIONS)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    benchmark(xyzw.exp, QUATERNIONS)


def test_exp_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy quaternion exp (10k)"
    expected = xyzw.exp(QUATERNIONS)
    _without_numba(monkeypatch)
    assert_allclose(xyzw.exp(QUATERNIONS), expected, rtol=2e-13, atol=2e-14)
    benchmark(xyzw.exp, QUATERNIONS)


def test_exp_numpy_quaternion(benchmark):
    benchmark.group = "NumPy quaternion exp (10k)"
    benchmark(lambda: _array_from_numpy_quaternion(np.exp(NUMPY_QUATERNIONS)))


def test_exp_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy quaternion exp (10k)"
    benchmark(_checked_numpy_quaternion, np.exp)


def test_log_numba_vs_numpy_quaternion(benchmark):
    benchmark.group = "NumPy quaternion log (10k)"
    expected = _array_from_numpy_quaternion(np.log(NUMPY_QUATERNIONS))
    actual = xyzw.log(QUATERNIONS)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    benchmark(xyzw.log, QUATERNIONS)


def test_log_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy quaternion log (10k)"
    expected = xyzw.log(QUATERNIONS)
    _without_numba(monkeypatch)
    assert_allclose(xyzw.log(QUATERNIONS), expected, rtol=2e-13, atol=2e-14)
    benchmark(xyzw.log, QUATERNIONS)


def test_log_numpy_quaternion(benchmark):
    benchmark.group = "NumPy quaternion log (10k)"
    benchmark(lambda: _array_from_numpy_quaternion(np.log(NUMPY_QUATERNIONS)))


def test_log_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy quaternion log (10k)"
    benchmark(_checked_numpy_quaternion, np.log)


def test_power_numba_vs_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy quaternion powers (3 ops, 2k)"
    expected = _run_numpy_fallback(monkeypatch, _power_suite)
    _assert_suite_equal(_power_suite(), expected)
    benchmark(_power_suite)


def test_power_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy quaternion powers (3 ops, 2k)"
    expected = _power_suite()
    with monkeypatch.context() as patch:
        _without_numba(patch)
        fallback = _power_suite()
    _assert_suite_equal(fallback, expected)
    _without_numba(monkeypatch)
    benchmark(_power_suite)


def test_jacobian_numba_vs_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy skew and SO(3) Jacobians (10k)"
    expected = _run_numpy_fallback(monkeypatch, _jacobian_suite)
    _assert_suite_equal(_jacobian_suite(), expected)
    benchmark(_jacobian_suite)


def test_jacobian_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy skew and SO(3) Jacobians (10k)"
    expected = _jacobian_suite()
    with monkeypatch.context() as patch:
        _without_numba(patch)
        fallback = _jacobian_suite()
    _assert_suite_equal(fallback, expected)
    _without_numba(monkeypatch)
    benchmark(_jacobian_suite)


def test_estimators_numba_vs_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy sequence and estimators (256x8)"
    expected = _run_numpy_fallback(monkeypatch, _estimator_and_sequence_suite)
    _assert_suite_equal(_estimator_and_sequence_suite(), expected)
    benchmark(_estimator_and_sequence_suite)


def test_estimators_numpy_fallback(monkeypatch, benchmark):
    benchmark.group = "NumPy sequence and estimators (256x8)"
    expected = _estimator_and_sequence_suite()
    with monkeypatch.context() as patch:
        _without_numba(patch)
        fallback = _estimator_and_sequence_suite()
    _assert_suite_equal(fallback, expected)
    _without_numba(monkeypatch)
    benchmark(_estimator_and_sequence_suite)
