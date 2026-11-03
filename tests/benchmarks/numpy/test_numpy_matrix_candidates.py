"""Benchmark the optional batched 4x4 quaternion multiplication matrices."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quatops import _numpy
from quatops.numpy.hamilton import xyzw as hamilton
from quatops.numpy.jpl import xyzw as jpl

pytestmark = pytest.mark.benchmark

BATCH_SIZE = 10_000
ROTATION_VECTORS = np.random.default_rng(318).normal(
    scale=0.4, size=(BATCH_SIZE, 3)
)
CASES = (
    ("Hamilton", hamilton, hamilton.from_rotvec(ROTATION_VECTORS)),
    ("JPL", jpl, jpl.from_rotvec(ROTATION_VECTORS)),
)


def _scalar_matrices(operation, quaternions):
    return np.asarray([operation(value) for value in quaternions])


@pytest.mark.parametrize("operation", ("to_left_matrix", "to_right_matrix"))
@pytest.mark.parametrize(
    ("convention", "backend", "quaternions"), CASES, ids=("Hamilton", "JPL")
)
def test_quaternion_multiplication_matrix_numba(
    benchmark, monkeypatch, operation, convention, backend, quaternions
):
    matrix_operation = getattr(backend, operation)
    with monkeypatch.context() as patch:
        patch.setattr(_numpy, "_numba_rotation_kernel", lambda _name: None)
        expected = _scalar_matrices(matrix_operation, quaternions)
    actual = matrix_operation(quaternions)
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    benchmark.group = f"{convention} {operation} (10k)"
    benchmark(matrix_operation, quaternions)


@pytest.mark.parametrize("operation", ("to_left_matrix", "to_right_matrix"))
@pytest.mark.parametrize(
    ("convention", "backend", "quaternions"), CASES, ids=("Hamilton", "JPL")
)
def test_quaternion_multiplication_matrix_numpy_fallback(
    benchmark, monkeypatch, operation, convention, backend, quaternions
):
    monkeypatch.setattr(_numpy, "_numba_rotation_kernel", lambda _name: None)
    matrix_operation = getattr(backend, operation)
    expected = _scalar_matrices(matrix_operation, quaternions)
    actual = matrix_operation(quaternions)
    assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    benchmark.group = f"{convention} {operation} (10k)"
    benchmark(matrix_operation, quaternions)
