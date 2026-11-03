"""Compare QuatOps scalar and batch operations with reference libraries.

Run with ``.venv/bin/python -m pytest -m benchmark --benchmark-only``.
Inputs and reference objects are built before timing; correctness checks run
once per test. With Numba installed, preflight compiles the batch kernels before
timing. Results are steady-state measurements, not pass/fail thresholds.
"""

import operator

import numpy as np
import pytest
import quaternion
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.hamilton import xyzw as hamilton_numpy
from quatops.numpy.jpl import xyzw as jpl_numpy

pytestmark = pytest.mark.benchmark

HAM_LHS = hamilton_scalar.normalize((0.91, 0.11, 0.31, -0.20))
HAM_RHS = hamilton_scalar.normalize((0.97, -0.14, 0.08, 0.17))
VECTOR = (1.0, -2.0, 3.0)
JPL_LHS = jpl_scalar.from_hamilton(HAM_LHS)
JPL_RHS = jpl_scalar.from_hamilton(HAM_RHS)

HAM_BATCH_LHS = np.tile(hamilton_scalar.to_xyzw(HAM_LHS), (10_000, 1))
HAM_BATCH_RHS = np.tile(hamilton_scalar.to_xyzw(HAM_RHS), (10_000, 1))
JPL_BATCH_LHS = np.tile(JPL_LHS, (10_000, 1))
JPL_BATCH_RHS = np.tile(JPL_RHS, (10_000, 1))
VECTORS = np.tile(VECTOR, (10_000, 1))

SCIPY_HAM_LHS = Rotation.from_quat(hamilton_scalar.to_xyzw(HAM_LHS))
SCIPY_HAM_RHS = Rotation.from_quat(hamilton_scalar.to_xyzw(HAM_RHS))
SCIPY_JPL_LHS = Rotation.from_quat(
    hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_LHS))
)
SCIPY_JPL_RHS = Rotation.from_quat(
    hamilton_scalar.to_xyzw(jpl_scalar.to_hamilton(JPL_RHS))
)
SCIPY_HAM_BATCH_LHS = Rotation.from_quat(HAM_BATCH_LHS)
SCIPY_HAM_BATCH_RHS = Rotation.from_quat(HAM_BATCH_RHS)
SCIPY_JPL_BATCH_LHS = Rotation.from_quat(
    hamilton_numpy.from_wxyz(jpl_numpy.to_hamilton(JPL_BATCH_LHS))
)
SCIPY_JPL_BATCH_RHS = Rotation.from_quat(
    hamilton_numpy.from_wxyz(jpl_numpy.to_hamilton(JPL_BATCH_RHS))
)

QUAT_HAM_BATCH_LHS = quaternion.as_quat_array(np.tile(HAM_LHS, (10_000, 1)))
QUAT_HAM_BATCH_RHS = quaternion.as_quat_array(np.tile(HAM_RHS, (10_000, 1)))
QUAT_JPL_BATCH_LHS = quaternion.as_quat_array(
    np.tile(jpl_scalar.to_hamilton(JPL_LHS), (10_000, 1))
)
QUAT_JPL_BATCH_RHS = quaternion.as_quat_array(
    np.tile(jpl_scalar.to_hamilton(JPL_RHS), (10_000, 1))
)


def _scipy_hamilton(q):
    return Rotation.from_quat(hamilton_scalar.to_xyzw(q))


def _scipy_jpl(q):
    hamilton_q = jpl_scalar.to_hamilton(q)
    return Rotation.from_quat(hamilton_scalar.to_xyzw(hamilton_q))


def _quaternion_rotate_batch(q):
    """Use the package's matrix based API for the repeated vector workload."""
    return quaternion.rotate_vectors(q, VECTOR)


def _quaternion_rotate_batch_products(q):
    """Apply pairwise pure-vector products for the batch's repeated vector."""
    vectors = quaternion.from_vector_part(VECTORS)
    return quaternion.as_vector_part(q * vectors * q.conjugate())


def _quaternion_multiply_checked(lhs, rhs):
    """Match QuatOps finite-input and finite-output error behavior."""
    result = operator.mul(lhs, rhs)
    if not np.all(np.isfinite(quaternion.as_float_array(result))):
        if not np.all(np.isfinite(quaternion.as_float_array(lhs))) or not np.all(
            np.isfinite(quaternion.as_float_array(rhs))
        ):
            raise ValueError("quaternion inputs must be finite")
        raise OverflowError("quaternion arithmetic produced a non-finite result")
    return result


def test_hamilton_scalar_multiply_quatops(benchmark):
    benchmark.group = "scalar multiply"
    result = hamilton_scalar.multiply(HAM_LHS, HAM_RHS)
    assert_allclose(
        hamilton_scalar.to_rotation_matrix(result),
        (SCIPY_HAM_LHS * SCIPY_HAM_RHS).as_matrix(),
    )
    benchmark(hamilton_scalar.multiply, HAM_LHS, HAM_RHS)


def test_hamilton_scalar_multiply_scipy(benchmark):
    benchmark.group = "scalar multiply"
    benchmark(lambda: SCIPY_HAM_LHS * SCIPY_HAM_RHS)


def test_jpl_scalar_multiply_quatops(benchmark):
    benchmark.group = "scalar multiply"
    result = jpl_scalar.multiply(JPL_LHS, JPL_RHS)
    assert_allclose(
        jpl_scalar.to_rotation_matrix(result),
        (SCIPY_JPL_LHS * SCIPY_JPL_RHS).as_matrix(),
    )
    benchmark(jpl_scalar.multiply, JPL_LHS, JPL_RHS)


def test_jpl_scalar_multiply_scipy(benchmark):
    benchmark.group = "scalar multiply"
    benchmark(lambda: SCIPY_JPL_LHS * SCIPY_JPL_RHS)


def test_hamilton_scalar_rotate_quatops(benchmark):
    benchmark.group = "scalar rotate"
    assert_allclose(
        hamilton_scalar.rotate(HAM_LHS, VECTOR),
        SCIPY_HAM_LHS.apply(VECTOR),
    )
    benchmark(hamilton_scalar.rotate, HAM_LHS, VECTOR)


def test_hamilton_scalar_rotate_scipy(benchmark):
    benchmark.group = "scalar rotate"
    benchmark(SCIPY_HAM_LHS.apply, VECTOR)


def test_jpl_scalar_rotate_quatops(benchmark):
    benchmark.group = "scalar rotate"
    assert_allclose(jpl_scalar.rotate(JPL_LHS, VECTOR), SCIPY_JPL_LHS.apply(VECTOR))
    benchmark(jpl_scalar.rotate, JPL_LHS, VECTOR)


def test_jpl_scalar_rotate_scipy(benchmark):
    benchmark.group = "scalar rotate"
    benchmark(SCIPY_JPL_LHS.apply, VECTOR)


def test_hamilton_numpy_multiply_10k_quatops(benchmark):
    benchmark.group = "NumPy multiply (10k)"
    result = hamilton_numpy.multiply(HAM_BATCH_LHS, HAM_BATCH_RHS)
    assert_allclose(
        hamilton_numpy.to_rotation_matrix(result),
        (SCIPY_HAM_BATCH_LHS * SCIPY_HAM_BATCH_RHS).as_matrix(),
    )
    assert_allclose(
        hamilton_numpy.to_wxyz(result),
        quaternion.as_float_array(QUAT_HAM_BATCH_LHS * QUAT_HAM_BATCH_RHS),
    )
    benchmark(hamilton_numpy.multiply, HAM_BATCH_LHS, HAM_BATCH_RHS)


def test_hamilton_numpy_multiply_10k_scipy(benchmark):
    benchmark.group = "NumPy multiply (10k)"
    benchmark(lambda: SCIPY_HAM_BATCH_LHS * SCIPY_HAM_BATCH_RHS)


def test_hamilton_numpy_multiply_10k_numpy_quaternion(benchmark):
    benchmark.group = "NumPy multiply (10k)"
    assert_allclose(
        quaternion.as_float_array(QUAT_HAM_BATCH_LHS * QUAT_HAM_BATCH_RHS),
        hamilton_numpy.to_wxyz(hamilton_numpy.multiply(HAM_BATCH_LHS, HAM_BATCH_RHS)),
    )
    benchmark(operator.mul, QUAT_HAM_BATCH_LHS, QUAT_HAM_BATCH_RHS)


def test_hamilton_numpy_multiply_10k_quatops_checked(benchmark):
    benchmark.group = "NumPy multiply (10k, checked)"
    result = hamilton_numpy.multiply(HAM_BATCH_LHS, HAM_BATCH_RHS)
    assert_allclose(
        hamilton_numpy.to_wxyz(result),
        quaternion.as_float_array(QUAT_HAM_BATCH_LHS * QUAT_HAM_BATCH_RHS),
    )
    benchmark(hamilton_numpy.multiply, HAM_BATCH_LHS, HAM_BATCH_RHS)


def test_hamilton_numpy_multiply_10k_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy multiply (10k, checked)"
    expected = hamilton_numpy.to_wxyz(
        hamilton_numpy.multiply(HAM_BATCH_LHS, HAM_BATCH_RHS)
    )
    assert_allclose(
        quaternion.as_float_array(
            _quaternion_multiply_checked(QUAT_HAM_BATCH_LHS, QUAT_HAM_BATCH_RHS)
        ),
        expected,
    )
    benchmark(
        _quaternion_multiply_checked,
        QUAT_HAM_BATCH_LHS,
        QUAT_HAM_BATCH_RHS,
    )


def test_jpl_numpy_multiply_10k_quatops(benchmark):
    benchmark.group = "NumPy multiply (10k)"
    result = jpl_numpy.multiply(JPL_BATCH_LHS, JPL_BATCH_RHS)
    assert_allclose(
        jpl_numpy.to_rotation_matrix(result),
        (SCIPY_JPL_BATCH_LHS * SCIPY_JPL_BATCH_RHS).as_matrix(),
    )
    assert_allclose(
        jpl_numpy.to_hamilton(result),
        quaternion.as_float_array(QUAT_JPL_BATCH_LHS * QUAT_JPL_BATCH_RHS),
    )
    benchmark(jpl_numpy.multiply, JPL_BATCH_LHS, JPL_BATCH_RHS)


def test_jpl_numpy_multiply_10k_scipy(benchmark):
    benchmark.group = "NumPy multiply (10k)"
    benchmark(lambda: SCIPY_JPL_BATCH_LHS * SCIPY_JPL_BATCH_RHS)


def test_jpl_numpy_multiply_10k_numpy_quaternion(benchmark):
    benchmark.group = "NumPy multiply (10k)"
    jpl_result = jpl_numpy.multiply(JPL_BATCH_LHS, JPL_BATCH_RHS)
    assert_allclose(
        quaternion.as_float_array(QUAT_JPL_BATCH_LHS * QUAT_JPL_BATCH_RHS),
        jpl_numpy.to_hamilton(jpl_result),
    )
    benchmark(operator.mul, QUAT_JPL_BATCH_LHS, QUAT_JPL_BATCH_RHS)


def test_jpl_numpy_multiply_10k_quatops_checked(benchmark):
    benchmark.group = "NumPy multiply (10k, checked)"
    result = jpl_numpy.multiply(JPL_BATCH_LHS, JPL_BATCH_RHS)
    assert_allclose(
        jpl_numpy.to_hamilton(result),
        quaternion.as_float_array(QUAT_JPL_BATCH_LHS * QUAT_JPL_BATCH_RHS),
    )
    benchmark(jpl_numpy.multiply, JPL_BATCH_LHS, JPL_BATCH_RHS)


def test_jpl_numpy_multiply_10k_numpy_quaternion_checked(benchmark):
    benchmark.group = "NumPy multiply (10k, checked)"
    expected = jpl_numpy.to_hamilton(jpl_numpy.multiply(JPL_BATCH_LHS, JPL_BATCH_RHS))
    assert_allclose(
        quaternion.as_float_array(
            _quaternion_multiply_checked(QUAT_JPL_BATCH_LHS, QUAT_JPL_BATCH_RHS)
        ),
        expected,
    )
    benchmark(
        _quaternion_multiply_checked,
        QUAT_JPL_BATCH_LHS,
        QUAT_JPL_BATCH_RHS,
    )


def test_hamilton_numpy_rotate_10k_quatops(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    assert_allclose(
        hamilton_numpy.rotate(HAM_BATCH_LHS, VECTORS),
        SCIPY_HAM_BATCH_LHS.apply(VECTORS),
    )
    assert_allclose(
        hamilton_numpy.rotate(HAM_BATCH_LHS, VECTORS),
        _quaternion_rotate_batch(QUAT_HAM_BATCH_LHS),
    )
    benchmark(hamilton_numpy.rotate, HAM_BATCH_LHS, VECTORS)


def test_hamilton_numpy_rotate_10k_scipy(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    benchmark(SCIPY_HAM_BATCH_LHS.apply, VECTORS)


def test_hamilton_numpy_rotate_10k_numpy_quaternion(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    assert_allclose(
        _quaternion_rotate_batch(QUAT_HAM_BATCH_LHS),
        hamilton_numpy.rotate(HAM_BATCH_LHS, VECTORS),
    )
    benchmark(_quaternion_rotate_batch, QUAT_HAM_BATCH_LHS)


def test_hamilton_numpy_rotate_10k_numpy_quaternion_products(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    assert_allclose(
        _quaternion_rotate_batch_products(QUAT_HAM_BATCH_LHS),
        hamilton_numpy.rotate(HAM_BATCH_LHS, VECTORS),
    )
    benchmark(_quaternion_rotate_batch_products, QUAT_HAM_BATCH_LHS)


def test_jpl_numpy_rotate_10k_quatops(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    assert_allclose(
        jpl_numpy.rotate(JPL_BATCH_LHS, VECTORS),
        SCIPY_JPL_BATCH_LHS.apply(VECTORS),
    )
    assert_allclose(
        jpl_numpy.rotate(JPL_BATCH_LHS, VECTORS),
        _quaternion_rotate_batch(QUAT_JPL_BATCH_LHS),
    )
    benchmark(jpl_numpy.rotate, JPL_BATCH_LHS, VECTORS)


def test_jpl_numpy_rotate_10k_scipy(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    benchmark(SCIPY_JPL_BATCH_LHS.apply, VECTORS)


def test_jpl_numpy_rotate_10k_numpy_quaternion(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    assert_allclose(
        _quaternion_rotate_batch(QUAT_JPL_BATCH_LHS),
        jpl_numpy.rotate(JPL_BATCH_LHS, VECTORS),
    )
    benchmark(_quaternion_rotate_batch, QUAT_JPL_BATCH_LHS)


def test_jpl_numpy_rotate_10k_numpy_quaternion_products(benchmark):
    benchmark.group = "NumPy rotate (10k)"
    assert_allclose(
        _quaternion_rotate_batch_products(QUAT_JPL_BATCH_LHS),
        jpl_numpy.rotate(JPL_BATCH_LHS, VECTORS),
    )
    benchmark(_quaternion_rotate_batch_products, QUAT_JPL_BATCH_LHS)
