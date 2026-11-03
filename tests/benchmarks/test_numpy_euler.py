"""Compare compiled batch Euler conversions with SciPy."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from quatops import _numpy
from quatops.numpy.hamilton import xyzw

pytestmark = pytest.mark.benchmark

BATCH_SIZE = 10_000
ANGLES_XYZ = np.random.default_rng(41).uniform(-1.0, 1.0, (BATCH_SIZE, 3))
SCIPY_ROTATIONS = Rotation.from_euler("xyz", ANGLES_XYZ)
QUATERNIONS_XYZW = SCIPY_ROTATIONS.as_quat()
ROTATION_MATRICES = SCIPY_ROTATIONS.as_matrix()


def test_numpy_from_euler_quatops(benchmark):
    benchmark.group = "NumPy from Euler xyz (10k)"
    result = xyzw.from_euler(ANGLES_XYZ, "xyz")
    assert_allclose(xyzw.to_rotation_matrix(result), ROTATION_MATRICES, atol=2e-14)
    benchmark(xyzw.from_euler, ANGLES_XYZ, "xyz")


def test_numpy_from_euler_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(_numpy, "_numba_rotation_kernel", lambda _name: None)
    benchmark.group = "NumPy from Euler xyz (10k)"
    result = xyzw.from_euler(ANGLES_XYZ, "xyz")
    assert_allclose(xyzw.to_rotation_matrix(result), ROTATION_MATRICES, atol=2e-14)
    benchmark(xyzw.from_euler, ANGLES_XYZ, "xyz")


def test_numpy_from_euler_scipy(benchmark):
    benchmark.group = "NumPy from Euler xyz (10k)"
    benchmark(Rotation.from_euler, "xyz", ANGLES_XYZ)


def test_numpy_to_euler_quatops(benchmark):
    benchmark.group = "NumPy to Euler xyz (10k)"
    result = xyzw.to_euler(QUATERNIONS_XYZW, "xyz")
    assert_allclose(
        Rotation.from_euler("xyz", result).as_matrix(), ROTATION_MATRICES,
        atol=2e-14,
    )
    benchmark(xyzw.to_euler, QUATERNIONS_XYZW, "xyz")


def test_numpy_to_euler_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(_numpy, "_numba_rotation_kernel", lambda _name: None)
    benchmark.group = "NumPy to Euler xyz (10k)"
    result = xyzw.to_euler(QUATERNIONS_XYZW, "xyz")
    assert_allclose(
        Rotation.from_euler("xyz", result).as_matrix(), ROTATION_MATRICES,
        atol=2e-14,
    )
    benchmark(xyzw.to_euler, QUATERNIONS_XYZW, "xyz")


def test_numpy_to_euler_scipy_prebuilt(benchmark):
    benchmark.group = "NumPy to Euler xyz (10k)"
    benchmark(SCIPY_ROTATIONS.as_euler, "xyz")


def test_numpy_to_euler_scipy_end_to_end(benchmark):
    benchmark.group = "NumPy to Euler xyz (10k)"
    benchmark(lambda: Rotation.from_quat(QUATERNIONS_XYZW).as_euler("xyz"))
