"""Compare batch rotation-vector conversions with SciPy and numpy-quaternion."""

import numpy as np
import pytest
import quaternion
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops.numpy.hamilton import xyzw

pytestmark = pytest.mark.benchmark

BATCH_SIZE = 10_000
ROTATION_VECTORS = np.random.default_rng(73).uniform(
    -2.5, 2.5, (BATCH_SIZE, 3)
)
SCIPY_ROTATIONS = Rotation.from_rotvec(ROTATION_VECTORS)
QUATERNIONS_XYZW = SCIPY_ROTATIONS.as_quat()
NUMPY_QUATERNIONS = quaternion.as_quat_array(
    np.column_stack((QUATERNIONS_XYZW[:, 3], QUATERNIONS_XYZW[:, :3]))
)
NUMPY_QUATERNION_COEFFICIENTS = quaternion.as_float_array(NUMPY_QUATERNIONS).copy()
FIRST_NONZERO_VECTOR = np.where(
    NUMPY_QUATERNION_COEFFICIENTS[:, 1] != 0.0,
    NUMPY_QUATERNION_COEFFICIENTS[:, 1],
    np.where(
        NUMPY_QUATERNION_COEFFICIENTS[:, 2] != 0.0,
        NUMPY_QUATERNION_COEFFICIENTS[:, 2],
        NUMPY_QUATERNION_COEFFICIENTS[:, 3],
    ),
)
NEGATIVE_CANONICAL = (NUMPY_QUATERNION_COEFFICIENTS[:, 0] < 0.0) | (
    (NUMPY_QUATERNION_COEFFICIENTS[:, 0] == 0.0)
    & (FIRST_NONZERO_VECTOR < 0.0)
)
NUMPY_QUATERNION_COEFFICIENTS[NEGATIVE_CANONICAL] *= -1.0
CANONICAL_NUMPY_QUATERNIONS = quaternion.as_quat_array(
    NUMPY_QUATERNION_COEFFICIENTS
)


def test_numpy_from_rotvec_quatops(benchmark):
    benchmark.group = "NumPy from rotation vector (10k)"
    result = xyzw.from_rotvec(ROTATION_VECTORS)
    assert_allclose(
        xyzw.to_rotation_matrix(result), SCIPY_ROTATIONS.as_matrix(), atol=2e-14
    )
    benchmark(xyzw.from_rotvec, ROTATION_VECTORS)


def test_numpy_from_rotvec_scipy(benchmark):
    benchmark.group = "NumPy from rotation vector (10k)"
    benchmark(Rotation.from_rotvec, ROTATION_VECTORS)


def test_numpy_from_rotvec_numpy_quaternion(benchmark):
    benchmark.group = "NumPy from rotation vector (10k)"
    result = quaternion.from_rotation_vector(ROTATION_VECTORS)
    assert_allclose(
        quaternion.as_rotation_matrix(result), SCIPY_ROTATIONS.as_matrix(),
        atol=2e-14,
    )
    benchmark(quaternion.from_rotation_vector, ROTATION_VECTORS)


def test_numpy_to_rotvec_quatops(benchmark):
    benchmark.group = "NumPy to rotation vector (10k)"
    result = xyzw.to_rotvec(QUATERNIONS_XYZW)
    assert_allclose(
        Rotation.from_rotvec(result).as_matrix(), SCIPY_ROTATIONS.as_matrix(),
        atol=2e-14,
    )
    benchmark(xyzw.to_rotvec, QUATERNIONS_XYZW)


def test_numpy_to_rotvec_scipy_prebuilt(benchmark):
    benchmark.group = "NumPy to rotation vector (10k)"
    benchmark(SCIPY_ROTATIONS.as_rotvec)


def test_numpy_to_rotvec_scipy_end_to_end(benchmark):
    benchmark.group = "NumPy to rotation vector (10k)"
    benchmark(lambda: Rotation.from_quat(QUATERNIONS_XYZW).as_rotvec())


def test_numpy_to_rotvec_numpy_quaternion(benchmark):
    benchmark.group = "NumPy to rotation vector (10k)"
    assert_allclose(
        quaternion.as_rotation_vector(CANONICAL_NUMPY_QUATERNIONS),
        xyzw.to_rotvec(QUATERNIONS_XYZW),
        rtol=2e-13,
        atol=2e-14,
    )
    benchmark(quaternion.as_rotation_vector, CANONICAL_NUMPY_QUATERNIONS)
