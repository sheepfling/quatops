"""Benchmark the remaining scalar-map and fused NumPy candidates.

Run with ``.venv/bin/python -m pytest
tests/benchmarks/numpy/test_numpy_remaining_candidates.py -m benchmark
--benchmark-only --benchmark-group-by=group --benchmark-columns=median``.
Correctness is checked before timing; measurements have no pass/fail threshold.
"""

import numpy as np
import pytest
import quaternion
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from quatops import _numpy as numpy_internal
from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.hamilton import xyzw as hamilton_numpy
from quatops.numpy.jpl import xyzw as jpl_numpy

pytestmark = pytest.mark.benchmark

BATCH_SIZE = 10_000
RNG = np.random.default_rng(918)
AXES = RNG.normal(size=(BATCH_SIZE, 3))
AXES /= np.linalg.norm(AXES, axis=-1, keepdims=True)
HAM_QUATERNIONS = hamilton_numpy.from_rotvec(
    RNG.normal(scale=0.4, size=(BATCH_SIZE, 3))
)
HAM_OTHER = hamilton_numpy.from_rotvec(
    RNG.normal(scale=0.4, size=(BATCH_SIZE, 3))
)
JPL_QUATERNIONS = jpl_numpy.from_rotvec(
    RNG.normal(scale=0.4, size=(BATCH_SIZE, 3))
)
JPL_OTHER = jpl_numpy.from_rotvec(RNG.normal(scale=0.4, size=(BATCH_SIZE, 3)))
HAM_CONTROL_A = hamilton_numpy.from_rotvec(RNG.normal(scale=0.4, size=(BATCH_SIZE, 3)))
HAM_CONTROL_B = hamilton_numpy.from_rotvec(RNG.normal(scale=0.4, size=(BATCH_SIZE, 3)))
JPL_CONTROL_A = jpl_numpy.from_rotvec(RNG.normal(scale=0.4, size=(BATCH_SIZE, 3)))
JPL_CONTROL_B = jpl_numpy.from_rotvec(RNG.normal(scale=0.4, size=(BATCH_SIZE, 3)))
HAM_DELTAS = RNG.normal(scale=0.2, size=(BATCH_SIZE, 3))
JPL_DELTAS = RNG.normal(scale=0.2, size=(BATCH_SIZE, 3))
HAM_SCIPY = Rotation.from_quat(HAM_QUATERNIONS)
HAM_OTHER_SCIPY = Rotation.from_quat(HAM_OTHER)
HAM_NUMPY_QUATERNION = quaternion.as_quat_array(
    hamilton_numpy.to_wxyz(HAM_QUATERNIONS)
)
HAM_OTHER_NUMPY_QUATERNION = quaternion.as_quat_array(
    hamilton_numpy.to_wxyz(HAM_OTHER)
)
ANGLE_TOLERANCE = 0.45


def _hamilton_scalar_swing_twist():
    values = np.asarray(
        [
            hamilton_scalar.swing_twist(
                (q[3], q[0], q[1], q[2]), tuple(axis)
            )
            for q, axis in zip(HAM_QUATERNIONS, AXES)
        ]
    )
    return values[..., (1, 2, 3, 0)]


def _jpl_scalar_swing_twist():
    return np.asarray(
        [
            jpl_scalar.swing_twist(tuple(q), tuple(axis))
            for q, axis in zip(JPL_QUATERNIONS, AXES)
        ]
    )


def _hamilton_scalar_squad():
    result = []
    for q0, q1, a, b in zip(
        HAM_QUATERNIONS, HAM_OTHER, HAM_CONTROL_A, HAM_CONTROL_B
    ):
        scalar_result = hamilton_scalar.squad(
            _hamilton_wxyz(q0), _hamilton_wxyz(q1),
            _hamilton_wxyz(a), _hamilton_wxyz(b), 0.37
        )
        result.append((scalar_result[1], scalar_result[2], scalar_result[3], scalar_result[0]))
    return np.asarray(result)


def _jpl_scalar_squad():
    return np.asarray(
        [
            jpl_scalar.squad(tuple(q0), tuple(q1), tuple(a), tuple(b), 0.37)
            for q0, q1, a, b in zip(
                JPL_QUATERNIONS, JPL_OTHER, JPL_CONTROL_A, JPL_CONTROL_B
            )
        ]
    )


def _hamilton_wxyz(value):
    return (value[3], value[0], value[1], value[2])


def _hamilton_scalar_boxplus():
    return np.asarray(
        [
            (value[1], value[2], value[3], value[0])
            for value in (
                hamilton_scalar.boxplus(
                    _hamilton_wxyz(quaternion), tuple(delta)
                )
                for quaternion, delta in zip(HAM_QUATERNIONS, HAM_DELTAS)
            )
        ]
    )


def _hamilton_scalar_boxminus():
    return np.asarray(
        [
            hamilton_scalar.boxminus(
                _hamilton_wxyz(quaternion), _hamilton_wxyz(reference)
            )
            for quaternion, reference in zip(HAM_OTHER, HAM_QUATERNIONS)
        ]
    )


def _jpl_scalar_boxplus():
    return np.asarray(
        [
            jpl_scalar.boxplus(tuple(quaternion), tuple(delta))
            for quaternion, delta in zip(JPL_QUATERNIONS, JPL_DELTAS)
        ]
    )


def _jpl_scalar_boxminus():
    return np.asarray(
        [
            jpl_scalar.boxminus(tuple(quaternion), tuple(reference))
            for quaternion, reference in zip(JPL_OTHER, JPL_QUATERNIONS)
        ]
    )


_BOX_CASES = (
    (
        "Hamilton",
        hamilton_numpy,
        HAM_QUATERNIONS,
        HAM_OTHER,
        HAM_DELTAS,
        _hamilton_scalar_boxplus,
        _hamilton_scalar_boxminus,
    ),
    (
        "JPL",
        jpl_numpy,
        JPL_QUATERNIONS,
        JPL_OTHER,
        JPL_DELTAS,
        _jpl_scalar_boxplus,
        _jpl_scalar_boxminus,
    ),
)


def _assert_swing_twist_equal(actual, expected):
    assert_allclose(actual[0], expected[:, 0], rtol=3e-13, atol=3e-13)
    assert_allclose(actual[1], expected[:, 1], rtol=3e-13, atol=3e-13)


def _assert_rotation_result(actual, expected):
    assert_allclose(
        Rotation.from_quat(actual).as_matrix(),
        expected.as_matrix(),
        rtol=3e-13,
        atol=3e-13,
    )


def test_hamilton_swing_twist_numpy(benchmark):
    expected = _hamilton_scalar_swing_twist()
    actual = hamilton_numpy.swing_twist(HAM_QUATERNIONS, AXES)
    _assert_swing_twist_equal(actual, expected)
    benchmark.group = "swing_twist (Hamilton, 10k)"
    benchmark(hamilton_numpy.swing_twist, HAM_QUATERNIONS, AXES)


def test_hamilton_swing_twist_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(
        numpy_internal, "_numba_rotation_kernel", lambda _name: None
    )
    monkeypatch.setattr(numpy_internal, "_numba_multiply_kernel", lambda *_args: None)
    expected = _hamilton_scalar_swing_twist()
    actual = hamilton_numpy.swing_twist(HAM_QUATERNIONS, AXES)
    _assert_swing_twist_equal(actual, expected)
    benchmark.group = "swing_twist (Hamilton, 10k)"
    benchmark(hamilton_numpy.swing_twist, HAM_QUATERNIONS, AXES)


def test_hamilton_swing_twist_scalar_loop(benchmark):
    expected = _hamilton_scalar_swing_twist()
    actual = hamilton_numpy.swing_twist(HAM_QUATERNIONS, AXES)
    _assert_swing_twist_equal(actual, expected)
    benchmark.group = "swing_twist (Hamilton, 10k)"
    benchmark(_hamilton_scalar_swing_twist)


def test_jpl_swing_twist_numpy(benchmark):
    expected = _jpl_scalar_swing_twist()
    actual = jpl_numpy.swing_twist(JPL_QUATERNIONS, AXES)
    _assert_swing_twist_equal(actual, expected)
    benchmark.group = "swing_twist (JPL, 10k)"
    benchmark(jpl_numpy.swing_twist, JPL_QUATERNIONS, AXES)


def test_jpl_swing_twist_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(
        numpy_internal, "_numba_rotation_kernel", lambda _name: None
    )
    monkeypatch.setattr(numpy_internal, "_numba_multiply_kernel", lambda *_args: None)
    expected = _jpl_scalar_swing_twist()
    actual = jpl_numpy.swing_twist(JPL_QUATERNIONS, AXES)
    _assert_swing_twist_equal(actual, expected)
    benchmark.group = "swing_twist (JPL, 10k)"
    benchmark(jpl_numpy.swing_twist, JPL_QUATERNIONS, AXES)


def test_jpl_swing_twist_scalar_loop(benchmark):
    expected = _jpl_scalar_swing_twist()
    actual = jpl_numpy.swing_twist(JPL_QUATERNIONS, AXES)
    _assert_swing_twist_equal(actual, expected)
    benchmark.group = "swing_twist (JPL, 10k)"
    benchmark(_jpl_scalar_swing_twist)


def test_hamilton_squad_numba_primitives(benchmark):
    expected = _hamilton_scalar_squad()
    actual = hamilton_numpy.squad(
        HAM_QUATERNIONS, HAM_OTHER, HAM_CONTROL_A, HAM_CONTROL_B, 0.37
    )
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    benchmark.group = "SQUAD (Hamilton, 10k)"
    benchmark(
        hamilton_numpy.squad,
        HAM_QUATERNIONS,
        HAM_OTHER,
        HAM_CONTROL_A,
        HAM_CONTROL_B,
        0.37,
    )


def test_hamilton_squad_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(
        numpy_internal, "_numba_rotation_kernel", lambda _name: None
    )
    expected = _hamilton_scalar_squad()
    actual = hamilton_numpy.squad(
        HAM_QUATERNIONS, HAM_OTHER, HAM_CONTROL_A, HAM_CONTROL_B, 0.37
    )
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    benchmark.group = "SQUAD (Hamilton, 10k)"
    benchmark(
        hamilton_numpy.squad,
        HAM_QUATERNIONS,
        HAM_OTHER,
        HAM_CONTROL_A,
        HAM_CONTROL_B,
        0.37,
    )


def test_hamilton_squad_scalar_loop(benchmark):
    benchmark.group = "SQUAD (Hamilton, 10k)"
    benchmark(_hamilton_scalar_squad)


def test_jpl_squad_numba_primitives(benchmark):
    expected = _jpl_scalar_squad()
    actual = jpl_numpy.squad(
        JPL_QUATERNIONS, JPL_OTHER, JPL_CONTROL_A, JPL_CONTROL_B, 0.37
    )
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    benchmark.group = "SQUAD (JPL, 10k)"
    benchmark(
        jpl_numpy.squad,
        JPL_QUATERNIONS,
        JPL_OTHER,
        JPL_CONTROL_A,
        JPL_CONTROL_B,
        0.37,
    )


def test_jpl_squad_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(
        numpy_internal, "_numba_rotation_kernel", lambda _name: None
    )
    expected = _jpl_scalar_squad()
    actual = jpl_numpy.squad(
        JPL_QUATERNIONS, JPL_OTHER, JPL_CONTROL_A, JPL_CONTROL_B, 0.37
    )
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    benchmark.group = "SQUAD (JPL, 10k)"
    benchmark(
        jpl_numpy.squad,
        JPL_QUATERNIONS,
        JPL_OTHER,
        JPL_CONTROL_A,
        JPL_CONTROL_B,
        0.37,
    )


def test_jpl_squad_scalar_loop(benchmark):
    benchmark.group = "SQUAD (JPL, 10k)"
    benchmark(_jpl_scalar_squad)


@pytest.mark.parametrize("operation", ("boxplus", "boxminus"))
@pytest.mark.parametrize(
    (
        "convention", "backend", "quaternions", "other", "deltas",
        "scalar_boxplus", "scalar_boxminus",
    ),
    _BOX_CASES,
    ids=("Hamilton", "JPL"),
)
def test_box_operation_numba_primitives(
    benchmark,
    operation,
    convention,
    backend,
    quaternions,
    other,
    deltas,
    scalar_boxplus,
    scalar_boxminus,
):
    if operation == "boxplus":
        args, expected = (quaternions, deltas), scalar_boxplus()
    else:
        args, expected = (other, quaternions), scalar_boxminus()
    function = getattr(backend, operation)
    actual = function(*args)
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    benchmark.group = f"{operation} ({convention}, 10k)"
    benchmark(function, *args)


@pytest.mark.parametrize("operation", ("boxplus", "boxminus"))
@pytest.mark.parametrize(
    (
        "convention", "backend", "quaternions", "other", "deltas",
        "scalar_boxplus", "scalar_boxminus",
    ),
    _BOX_CASES,
    ids=("Hamilton", "JPL"),
)
def test_box_operation_numpy_fallback(
    benchmark,
    monkeypatch,
    operation,
    convention,
    backend,
    quaternions,
    other,
    deltas,
    scalar_boxplus,
    scalar_boxminus,
):
    monkeypatch.setattr(
        numpy_internal, "_numba_rotation_kernel", lambda _name: None
    )
    monkeypatch.setattr(numpy_internal, "_numba_multiply_kernel", lambda *_args: None)
    if operation == "boxplus":
        args, expected = (quaternions, deltas), scalar_boxplus()
    else:
        args, expected = (other, quaternions), scalar_boxminus()
    function = getattr(backend, operation)
    actual = function(*args)
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    benchmark.group = f"{operation} ({convention}, 10k)"
    benchmark(function, *args)


@pytest.mark.parametrize("operation", ("boxplus", "boxminus"))
@pytest.mark.parametrize(
    ("convention", "scalar_boxplus", "scalar_boxminus"),
    (
        ("Hamilton", _hamilton_scalar_boxplus, _hamilton_scalar_boxminus),
        ("JPL", _jpl_scalar_boxplus, _jpl_scalar_boxminus),
    ),
    ids=("Hamilton", "JPL"),
)
def test_box_operation_scalar_loop(
    benchmark,
    operation,
    convention,
    scalar_boxplus,
    scalar_boxminus,
):
    scalar_operation = scalar_boxplus if operation == "boxplus" else scalar_boxminus
    benchmark.group = f"{operation} ({convention}, 10k)"
    benchmark(scalar_operation)


def test_left_divide_quatops(benchmark):
    actual = hamilton_numpy.left_divide(HAM_QUATERNIONS, HAM_OTHER)
    scipy_expected = HAM_SCIPY.inv() * HAM_OTHER_SCIPY
    _assert_rotation_result(actual, scipy_expected)
    quaternion_expected = (
        1.0 / HAM_NUMPY_QUATERNION * HAM_OTHER_NUMPY_QUATERNION
    )
    assert_allclose(
        hamilton_numpy.to_wxyz(actual),
        quaternion.as_float_array(quaternion_expected),
        rtol=3e-13,
        atol=3e-13,
    )
    benchmark.group = "left_divide (10k unit quaternions)"
    benchmark(hamilton_numpy.left_divide, HAM_QUATERNIONS, HAM_OTHER)


def test_left_divide_numpy_quaternion(benchmark):
    benchmark.group = "left_divide (10k unit quaternions)"
    benchmark(
        lambda: 1.0 / HAM_NUMPY_QUATERNION * HAM_OTHER_NUMPY_QUATERNION
    )


def test_left_divide_scipy(benchmark):
    benchmark.group = "left_divide (10k unit quaternions)"
    benchmark(lambda: HAM_SCIPY.inv() * HAM_OTHER_SCIPY)


def test_right_divide_quatops(benchmark):
    actual = hamilton_numpy.right_divide(HAM_QUATERNIONS, HAM_OTHER)
    scipy_expected = HAM_SCIPY * HAM_OTHER_SCIPY.inv()
    _assert_rotation_result(actual, scipy_expected)
    quaternion_expected = HAM_NUMPY_QUATERNION / HAM_OTHER_NUMPY_QUATERNION
    assert_allclose(
        hamilton_numpy.to_wxyz(actual),
        quaternion.as_float_array(quaternion_expected),
        rtol=3e-13,
        atol=3e-13,
    )
    benchmark.group = "right_divide (10k unit quaternions)"
    benchmark(hamilton_numpy.right_divide, HAM_QUATERNIONS, HAM_OTHER)


def test_right_divide_numpy_quaternion(benchmark):
    benchmark.group = "right_divide (10k unit quaternions)"
    benchmark(HAM_NUMPY_QUATERNION.__truediv__, HAM_OTHER_NUMPY_QUATERNION)


def test_right_divide_scipy(benchmark):
    benchmark.group = "right_divide (10k unit quaternions)"
    benchmark(lambda: HAM_SCIPY * HAM_OTHER_SCIPY.inv())


def test_rotation_distance_quatops(benchmark):
    expected = quaternion.rotation_intrinsic_distance(
        HAM_NUMPY_QUATERNION, HAM_OTHER_NUMPY_QUATERNION
    )
    actual = hamilton_numpy.rotation_distance(HAM_QUATERNIONS, HAM_OTHER)
    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    assert_allclose(actual, (HAM_SCIPY.inv() * HAM_OTHER_SCIPY).magnitude())
    benchmark.group = "rotation_distance (10k)"
    benchmark(hamilton_numpy.rotation_distance, HAM_QUATERNIONS, HAM_OTHER)


def test_rotation_distance_numpy_quaternion(benchmark):
    benchmark.group = "rotation_distance (10k)"
    benchmark(
        quaternion.rotation_intrinsic_distance,
        HAM_NUMPY_QUATERNION,
        HAM_OTHER_NUMPY_QUATERNION,
    )


def test_rotation_distance_scipy(benchmark):
    benchmark.group = "rotation_distance (10k)"
    benchmark(lambda: (HAM_SCIPY.inv() * HAM_OTHER_SCIPY).magnitude())


def test_rotations_equivalent_quatops(benchmark):
    distances = quaternion.rotation_intrinsic_distance(
        HAM_NUMPY_QUATERNION, HAM_OTHER_NUMPY_QUATERNION
    )
    expected = distances <= ANGLE_TOLERANCE
    actual = hamilton_numpy.rotations_equivalent(
        HAM_QUATERNIONS, HAM_OTHER, atol=ANGLE_TOLERANCE
    )
    assert_array_equal(actual, expected)
    benchmark.group = "rotations_equivalent (10k)"
    benchmark(
        hamilton_numpy.rotations_equivalent,
        HAM_QUATERNIONS,
        HAM_OTHER,
        ANGLE_TOLERANCE,
    )


def test_rotations_equivalent_numpy_quaternion(benchmark):
    benchmark.group = "rotations_equivalent (10k)"
    benchmark(
        lambda: quaternion.rotation_intrinsic_distance(
            HAM_NUMPY_QUATERNION, HAM_OTHER_NUMPY_QUATERNION
        )
        <= ANGLE_TOLERANCE
    )


def test_rotations_equivalent_scipy(benchmark):
    benchmark.group = "rotations_equivalent (10k)"
    benchmark(
        lambda: (HAM_SCIPY.inv() * HAM_OTHER_SCIPY).magnitude()
        <= ANGLE_TOLERANCE
    )
