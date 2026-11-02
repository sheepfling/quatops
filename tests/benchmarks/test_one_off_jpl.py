"""One-off JPL operation costs by value representation.

Hot cases reuse quaternion values. Cold cases create their values inside each
timed call. NumPy uses one-element arrays. SciPy receives the equivalent
Hamilton rotation. No timing thresholds.
"""

import operator

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import dataclass as jpl_dataclass
from quatops.jpl import namedtuple as jpl_namedtuple
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.jpl import xyzw as jpl_numpy

pytestmark = pytest.mark.benchmark

JPL_LHS = jpl_scalar.normalize((0.91, 0.11, 0.31, -0.20))
JPL_RHS = jpl_scalar.normalize((0.97, -0.14, 0.08, 0.17))
JPL_XYZW_LHS = jpl_scalar.to_xyzw(JPL_LHS)
JPL_XYZW_RHS = jpl_scalar.to_xyzw(JPL_RHS)
VECTOR = (1.0, -2.0, 3.0)

NAMED_LHS = jpl_namedtuple.Quaternion(*JPL_LHS)
NAMED_RHS = jpl_namedtuple.Quaternion(*JPL_RHS)
DATA_LHS = jpl_dataclass.Quaternion(*JPL_LHS)
DATA_RHS = jpl_dataclass.Quaternion(*JPL_RHS)
ARRAY_LHS = np.array([JPL_XYZW_LHS])
ARRAY_RHS = np.array([JPL_XYZW_RHS])
SCIPY_LHS = Rotation.from_quat(hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_LHS)))
SCIPY_RHS = Rotation.from_quat(hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_RHS)))


def _tuple_multiply_cold():
    lhs = (JPL_LHS[0], JPL_LHS[1], JPL_LHS[2], JPL_LHS[3])
    rhs = (JPL_RHS[0], JPL_RHS[1], JPL_RHS[2], JPL_RHS[3])
    return jpl_scalar.multiply(lhs, rhs)


def _namedtuple_multiply_cold():
    lhs = jpl_namedtuple.Quaternion(*JPL_LHS)
    rhs = jpl_namedtuple.Quaternion(*JPL_RHS)
    return jpl_namedtuple.multiply(lhs, rhs)


def _dataclass_multiply_cold():
    lhs = jpl_dataclass.Quaternion(*JPL_LHS)
    rhs = jpl_dataclass.Quaternion(*JPL_RHS)
    return jpl_dataclass.multiply(lhs, rhs)


def _numpy_multiply_cold():
    lhs = np.array([JPL_XYZW_LHS])
    rhs = np.array([JPL_XYZW_RHS])
    return jpl_numpy.multiply(lhs, rhs)


def _scipy_multiply_cold():
    lhs = Rotation.from_quat(hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_LHS)))
    rhs = Rotation.from_quat(hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_RHS)))
    return operator.mul(lhs, rhs)


def _tuple_rotate_cold():
    q = (JPL_LHS[0], JPL_LHS[1], JPL_LHS[2], JPL_LHS[3])
    return jpl_scalar.rotate(q, VECTOR)


def _namedtuple_rotate_cold():
    q = jpl_namedtuple.Quaternion(*JPL_LHS)
    return jpl_namedtuple.rotate(q, VECTOR)


def _dataclass_rotate_cold():
    q = jpl_dataclass.Quaternion(*JPL_LHS)
    return jpl_dataclass.rotate(q, VECTOR)


def _numpy_rotate_cold():
    q = np.array([JPL_XYZW_LHS])
    vector = np.array([VECTOR])
    return jpl_numpy.rotate(q, vector)


def _scipy_rotate_cold():
    q = Rotation.from_quat(hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_LHS)))
    return q.apply(VECTOR)


def test_jpl_hot_multiply_tuple(benchmark):
    benchmark.group = "jpl hot multiply"
    result = jpl_scalar.multiply(JPL_LHS, JPL_RHS)
    assert_allclose(
        jpl_scalar.to_rotation_matrix(result), (SCIPY_LHS * SCIPY_RHS).as_matrix()
    )
    benchmark(jpl_scalar.multiply, JPL_LHS, JPL_RHS)


def test_jpl_hot_multiply_namedtuple(benchmark):
    benchmark.group = "jpl hot multiply"
    result = jpl_namedtuple.multiply(NAMED_LHS, NAMED_RHS)
    assert_allclose(
        jpl_namedtuple.to_rotation_matrix(result), (SCIPY_LHS * SCIPY_RHS).as_matrix()
    )
    benchmark(jpl_namedtuple.multiply, NAMED_LHS, NAMED_RHS)


def test_jpl_hot_multiply_dataclass(benchmark):
    benchmark.group = "jpl hot multiply"
    result = jpl_dataclass.multiply(DATA_LHS, DATA_RHS)
    assert_allclose(
        jpl_dataclass.to_rotation_matrix(result), (SCIPY_LHS * SCIPY_RHS).as_matrix()
    )
    benchmark(jpl_dataclass.multiply, DATA_LHS, DATA_RHS)


def test_jpl_hot_multiply_numpy_one(benchmark):
    benchmark.group = "jpl hot multiply"
    result = jpl_numpy.multiply(ARRAY_LHS, ARRAY_RHS)
    assert_allclose(
        jpl_numpy.to_rotation_matrix(result)[0], (SCIPY_LHS * SCIPY_RHS).as_matrix()
    )
    benchmark(jpl_numpy.multiply, ARRAY_LHS, ARRAY_RHS)


def test_jpl_hot_multiply_scipy(benchmark):
    benchmark.group = "jpl hot multiply"
    benchmark(operator.mul, SCIPY_LHS, SCIPY_RHS)


def test_jpl_cold_multiply_tuple(benchmark):
    benchmark.group = "jpl cold multiply"
    assert_allclose(
        jpl_scalar.to_rotation_matrix(_tuple_multiply_cold()),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_tuple_multiply_cold)


def test_jpl_cold_multiply_namedtuple(benchmark):
    benchmark.group = "jpl cold multiply"
    assert_allclose(
        jpl_namedtuple.to_rotation_matrix(_namedtuple_multiply_cold()),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_namedtuple_multiply_cold)


def test_jpl_cold_multiply_dataclass(benchmark):
    benchmark.group = "jpl cold multiply"
    assert_allclose(
        jpl_dataclass.to_rotation_matrix(_dataclass_multiply_cold()),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_dataclass_multiply_cold)


def test_jpl_cold_multiply_numpy_one(benchmark):
    benchmark.group = "jpl cold multiply"
    result = _numpy_multiply_cold()
    assert_allclose(
        jpl_numpy.to_rotation_matrix(result)[0], (SCIPY_LHS * SCIPY_RHS).as_matrix()
    )
    benchmark(_numpy_multiply_cold)


def test_jpl_cold_multiply_scipy(benchmark):
    benchmark.group = "jpl cold multiply"
    benchmark(_scipy_multiply_cold)


def test_jpl_hot_rotate_tuple(benchmark):
    benchmark.group = "jpl hot rotate"
    assert_allclose(jpl_scalar.rotate(JPL_LHS, VECTOR), SCIPY_LHS.apply(VECTOR))
    benchmark(jpl_scalar.rotate, JPL_LHS, VECTOR)


def test_jpl_hot_rotate_namedtuple(benchmark):
    benchmark.group = "jpl hot rotate"
    assert_allclose(jpl_namedtuple.rotate(NAMED_LHS, VECTOR), SCIPY_LHS.apply(VECTOR))
    benchmark(jpl_namedtuple.rotate, NAMED_LHS, VECTOR)


def test_jpl_hot_rotate_dataclass(benchmark):
    benchmark.group = "jpl hot rotate"
    assert_allclose(jpl_dataclass.rotate(DATA_LHS, VECTOR), SCIPY_LHS.apply(VECTOR))
    benchmark(jpl_dataclass.rotate, DATA_LHS, VECTOR)


def test_jpl_hot_rotate_numpy_one(benchmark):
    benchmark.group = "jpl hot rotate"
    assert_allclose(
        jpl_numpy.rotate(ARRAY_LHS, np.array([VECTOR]))[0], SCIPY_LHS.apply(VECTOR)
    )
    benchmark(jpl_numpy.rotate, ARRAY_LHS, np.array([VECTOR]))


def test_jpl_hot_rotate_scipy(benchmark):
    benchmark.group = "jpl hot rotate"
    benchmark(SCIPY_LHS.apply, VECTOR)


def test_jpl_cold_rotate_tuple(benchmark):
    benchmark.group = "jpl cold rotate"
    assert_allclose(_tuple_rotate_cold(), SCIPY_LHS.apply(VECTOR))
    benchmark(_tuple_rotate_cold)


def test_jpl_cold_rotate_namedtuple(benchmark):
    benchmark.group = "jpl cold rotate"
    assert_allclose(_namedtuple_rotate_cold(), SCIPY_LHS.apply(VECTOR))
    benchmark(_namedtuple_rotate_cold)


def test_jpl_cold_rotate_dataclass(benchmark):
    benchmark.group = "jpl cold rotate"
    assert_allclose(_dataclass_rotate_cold(), SCIPY_LHS.apply(VECTOR))
    benchmark(_dataclass_rotate_cold)


def test_jpl_cold_rotate_numpy_one(benchmark):
    benchmark.group = "jpl cold rotate"
    assert_allclose(_numpy_rotate_cold()[0], SCIPY_LHS.apply(VECTOR))
    benchmark(_numpy_rotate_cold)


def test_jpl_cold_rotate_scipy(benchmark):
    benchmark.group = "jpl cold rotate"
    benchmark(_scipy_rotate_cold)
