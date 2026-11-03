"""Keep the vectorized rotation available when optional Numba is absent."""

import numpy as np
from numpy.testing import assert_allclose

from quatops import _numpy
from quatops.hamilton import scalar
from quatops.numpy.hamilton import xyzw


def test_large_batch_uses_pure_numpy_without_numba(monkeypatch):
    monkeypatch.setattr(_numpy, "_NUMBA_CHECKED", True)
    monkeypatch.setattr(_numpy, "_NUMBA_ROTATE", None)
    quaternion = np.tile((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), (128, 1))
    vectors = np.tile((1.0, 0.0, 0.0), (128, 1))

    actual = xyzw.rotate(quaternion, vectors)

    assert_allclose(actual, np.tile((0.0, 1.0, 0.0), (128, 1)), atol=1e-14)


def test_large_batch_rotation_preserves_extreme_finite_values():
    quaternions = np.tile((0.2, 0.3, 0.4, 0.84), (128, 1))
    vectors = np.tile((1.0, -2.0, 3.0), (128, 1))
    quaternions[0] = (1e308, 0.0, 0.0, 1e308)
    vectors[0] = (1e308, 1e308, 1e308)
    quaternions[1] = (1e-308, 0.0, 0.0, 1e-308)
    vectors[1] = (1e308, 1e308, 1e308)
    vectors[2] = (1.4e308, -1.4e308, 0.0)

    actual = xyzw.rotate(quaternions, vectors)
    expected = np.asarray([
        scalar.rotate(scalar.from_xyzw(q), vector)
        for q, vector in zip(quaternions, vectors)
    ])

    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
