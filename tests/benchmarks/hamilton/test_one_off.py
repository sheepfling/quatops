"""One-off Hamilton operation costs by value representation.

Hot cases reuse quaternion values. Cold cases create both quaternion values
inside each timed call. NumPy uses one-element arrays. No timing thresholds.
"""

import operator

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops.hamilton import dataclass as hamilton_dataclass
from quatops.hamilton import namedtuple as hamilton_namedtuple
from quatops.hamilton import scalar as hamilton_scalar
from quatops.numpy.hamilton import xyzw as hamilton_numpy

pytestmark = pytest.mark.benchmark

HAM_LHS = hamilton_scalar.normalize((0.91, 0.11, 0.31, -0.20))
HAM_RHS = hamilton_scalar.normalize((0.97, -0.14, 0.08, 0.17))
HAM_XYZW_LHS = hamilton_scalar.to_xyzw(HAM_LHS)
HAM_XYZW_RHS = hamilton_scalar.to_xyzw(HAM_RHS)
VECTOR = (1.0, -2.0, 3.0)

NAMED_LHS = hamilton_namedtuple.Quaternion(*HAM_LHS)
NAMED_RHS = hamilton_namedtuple.Quaternion(*HAM_RHS)
DATA_LHS = hamilton_dataclass.Quaternion(*HAM_LHS)
DATA_RHS = hamilton_dataclass.Quaternion(*HAM_RHS)
ARRAY_LHS = np.array([HAM_XYZW_LHS])
ARRAY_RHS = np.array([HAM_XYZW_RHS])
SCIPY_LHS = Rotation.from_quat(HAM_XYZW_LHS)
SCIPY_RHS = Rotation.from_quat(HAM_XYZW_RHS)


def _tuple_multiply_cold():
    lhs = (HAM_LHS[0], HAM_LHS[1], HAM_LHS[2], HAM_LHS[3])
    rhs = (HAM_RHS[0], HAM_RHS[1], HAM_RHS[2], HAM_RHS[3])
    return hamilton_scalar.multiply(lhs, rhs)


def _namedtuple_multiply_cold():
    lhs = hamilton_namedtuple.Quaternion(*HAM_LHS)
    rhs = hamilton_namedtuple.Quaternion(*HAM_RHS)
    return hamilton_namedtuple.multiply(lhs, rhs)


def _dataclass_multiply_cold():
    lhs = hamilton_dataclass.Quaternion(*HAM_LHS)
    rhs = hamilton_dataclass.Quaternion(*HAM_RHS)
    return hamilton_dataclass.multiply(lhs, rhs)


def _numpy_multiply_cold():
    lhs = np.array([HAM_XYZW_LHS])
    rhs = np.array([HAM_XYZW_RHS])
    return hamilton_numpy.multiply(lhs, rhs)


def _scipy_multiply_cold():
    lhs = Rotation.from_quat(HAM_XYZW_LHS)
    rhs = Rotation.from_quat(HAM_XYZW_RHS)
    return operator.mul(lhs, rhs)


def _tuple_rotate_cold():
    q = (HAM_LHS[0], HAM_LHS[1], HAM_LHS[2], HAM_LHS[3])
    return hamilton_scalar.rotate(q, VECTOR)


def _namedtuple_rotate_cold():
    q = hamilton_namedtuple.Quaternion(*HAM_LHS)
    return hamilton_namedtuple.rotate(q, VECTOR)


def _dataclass_rotate_cold():
    q = hamilton_dataclass.Quaternion(*HAM_LHS)
    return hamilton_dataclass.rotate(q, VECTOR)


def _numpy_rotate_cold():
    q = np.array([HAM_XYZW_LHS])
    vector = np.array([VECTOR])
    return hamilton_numpy.rotate(q, vector)


def _scipy_rotate_cold():
    q = Rotation.from_quat(HAM_XYZW_LHS)
    return q.apply(VECTOR)


def test_hot_multiply_tuple(benchmark):
    benchmark.group = "hot multiply"
    result = hamilton_scalar.multiply(HAM_LHS, HAM_RHS)
    assert_allclose(
        hamilton_scalar.to_rotation_matrix(result),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(hamilton_scalar.multiply, HAM_LHS, HAM_RHS)


def test_hot_multiply_namedtuple(benchmark):
    benchmark.group = "hot multiply"
    result = hamilton_namedtuple.multiply(NAMED_LHS, NAMED_RHS)
    assert_allclose(
        hamilton_namedtuple.to_rotation_matrix(result),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(hamilton_namedtuple.multiply, NAMED_LHS, NAMED_RHS)


def test_hot_multiply_dataclass(benchmark):
    benchmark.group = "hot multiply"
    result = hamilton_dataclass.multiply(DATA_LHS, DATA_RHS)
    assert_allclose(
        hamilton_dataclass.to_rotation_matrix(result),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(hamilton_dataclass.multiply, DATA_LHS, DATA_RHS)


def test_hot_multiply_numpy_one(benchmark):
    benchmark.group = "hot multiply"
    result = hamilton_numpy.multiply(ARRAY_LHS, ARRAY_RHS)
    assert_allclose(
        hamilton_numpy.to_rotation_matrix(result)[0],
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(hamilton_numpy.multiply, ARRAY_LHS, ARRAY_RHS)


def test_hot_multiply_scipy(benchmark):
    benchmark.group = "hot multiply"
    benchmark(operator.mul, SCIPY_LHS, SCIPY_RHS)


def test_cold_multiply_tuple(benchmark):
    benchmark.group = "cold multiply"
    assert_allclose(
        hamilton_scalar.to_rotation_matrix(_tuple_multiply_cold()),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_tuple_multiply_cold)


def test_cold_multiply_namedtuple(benchmark):
    benchmark.group = "cold multiply"
    assert_allclose(
        hamilton_namedtuple.to_rotation_matrix(_namedtuple_multiply_cold()),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_namedtuple_multiply_cold)


def test_cold_multiply_dataclass(benchmark):
    benchmark.group = "cold multiply"
    assert_allclose(
        hamilton_dataclass.to_rotation_matrix(_dataclass_multiply_cold()),
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_dataclass_multiply_cold)


def test_cold_multiply_numpy_one(benchmark):
    benchmark.group = "cold multiply"
    result = _numpy_multiply_cold()
    assert_allclose(
        hamilton_numpy.to_rotation_matrix(result)[0],
        (SCIPY_LHS * SCIPY_RHS).as_matrix(),
    )
    benchmark(_numpy_multiply_cold)


def test_cold_multiply_scipy(benchmark):
    benchmark.group = "cold multiply"
    benchmark(_scipy_multiply_cold)


def test_hot_rotate_tuple(benchmark):
    benchmark.group = "hot rotate"
    assert_allclose(hamilton_scalar.rotate(HAM_LHS, VECTOR), SCIPY_LHS.apply(VECTOR))
    benchmark(hamilton_scalar.rotate, HAM_LHS, VECTOR)


def test_hot_rotate_namedtuple(benchmark):
    benchmark.group = "hot rotate"
    assert_allclose(
        hamilton_namedtuple.rotate(NAMED_LHS, VECTOR), SCIPY_LHS.apply(VECTOR)
    )
    benchmark(hamilton_namedtuple.rotate, NAMED_LHS, VECTOR)


def test_hot_rotate_dataclass(benchmark):
    benchmark.group = "hot rotate"
    assert_allclose(
        hamilton_dataclass.rotate(DATA_LHS, VECTOR), SCIPY_LHS.apply(VECTOR)
    )
    benchmark(hamilton_dataclass.rotate, DATA_LHS, VECTOR)


def test_hot_rotate_numpy_one(benchmark):
    benchmark.group = "hot rotate"
    assert_allclose(
        hamilton_numpy.rotate(ARRAY_LHS, np.array([VECTOR]))[0],
        SCIPY_LHS.apply(VECTOR),
    )
    benchmark(hamilton_numpy.rotate, ARRAY_LHS, np.array([VECTOR]))


def test_hot_rotate_scipy(benchmark):
    benchmark.group = "hot rotate"
    benchmark(SCIPY_LHS.apply, VECTOR)


def test_cold_rotate_tuple(benchmark):
    benchmark.group = "cold rotate"
    assert_allclose(_tuple_rotate_cold(), SCIPY_LHS.apply(VECTOR))
    benchmark(_tuple_rotate_cold)


def test_cold_rotate_namedtuple(benchmark):
    benchmark.group = "cold rotate"
    assert_allclose(_namedtuple_rotate_cold(), SCIPY_LHS.apply(VECTOR))
    benchmark(_namedtuple_rotate_cold)


def test_cold_rotate_dataclass(benchmark):
    benchmark.group = "cold rotate"
    assert_allclose(_dataclass_rotate_cold(), SCIPY_LHS.apply(VECTOR))
    benchmark(_dataclass_rotate_cold)


def test_cold_rotate_numpy_one(benchmark):
    benchmark.group = "cold rotate"
    assert_allclose(_numpy_rotate_cold()[0], SCIPY_LHS.apply(VECTOR))
    benchmark(_numpy_rotate_cold)


def test_cold_rotate_scipy(benchmark):
    benchmark.group = "cold rotate"
    benchmark(_scipy_rotate_cold)
