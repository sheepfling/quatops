"""Compare batch matrix conversions with SciPy and numpy-quaternion."""

import numpy as np
import pytest
import quaternion
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops.numpy.hamilton import xyzw

pytestmark = pytest.mark.benchmark

BATCH_SIZE = 10_000
MATRICES = Rotation.random(BATCH_SIZE, random_state=24).as_matrix()
QUATERNIONS_XYZW = Rotation.from_matrix(MATRICES).as_quat()
SCIPY_ROTATIONS = Rotation.from_quat(QUATERNIONS_XYZW)
NUMPY_QUATERNIONS = quaternion.as_quat_array(
    np.column_stack((QUATERNIONS_XYZW[:, 3], QUATERNIONS_XYZW[:, :3]))
)


def test_numpy_to_rotation_matrix_quatops(benchmark):
    benchmark.group = "NumPy to rotation matrix (10k)"
    assert_allclose(xyzw.to_rotation_matrix(QUATERNIONS_XYZW), MATRICES, atol=2e-14)
    benchmark(xyzw.to_rotation_matrix, QUATERNIONS_XYZW)


def test_numpy_to_rotation_matrix_scipy_prebuilt(benchmark):
    benchmark.group = "NumPy to rotation matrix (10k)"
    benchmark(SCIPY_ROTATIONS.as_matrix)


def test_numpy_to_rotation_matrix_scipy_end_to_end(benchmark):
    benchmark.group = "NumPy to rotation matrix (10k)"
    benchmark(lambda: Rotation.from_quat(QUATERNIONS_XYZW).as_matrix())


def test_numpy_to_rotation_matrix_numpy_quaternion(benchmark):
    benchmark.group = "NumPy to rotation matrix (10k)"
    assert_allclose(
        quaternion.as_rotation_matrix(NUMPY_QUATERNIONS), MATRICES, atol=2e-14
    )
    benchmark(quaternion.as_rotation_matrix, NUMPY_QUATERNIONS)


def test_numpy_from_rotation_matrix_quatops(benchmark):
    benchmark.group = "NumPy from rotation matrix (10k)"
    result = xyzw.from_rotation_matrix(MATRICES)
    assert_allclose(xyzw.to_rotation_matrix(result), MATRICES, atol=2e-14)
    benchmark(xyzw.from_rotation_matrix, MATRICES)


def test_numpy_from_rotation_matrix_scipy(benchmark):
    benchmark.group = "NumPy from rotation matrix (10k)"
    benchmark(Rotation.from_matrix, MATRICES)


def test_numpy_from_rotation_matrix_numpy_quaternion(benchmark):
    benchmark.group = "NumPy from rotation matrix (10k)"
    result = quaternion.from_rotation_matrix(MATRICES, nonorthogonal=False)
    assert_allclose(quaternion.as_rotation_matrix(result), MATRICES, atol=2e-14)
    benchmark(quaternion.from_rotation_matrix, MATRICES, nonorthogonal=False)
