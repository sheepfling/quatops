"""Benchmark the optional Numba path for game-engine array conversion."""

from importlib.util import find_spec

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.engines import SignedAxisMap
from quatops.engines import numpy as engine_numpy

pytestmark = pytest.mark.benchmark
NUMBA_AVAILABLE = find_spec("numba") is not None
BATCH_SIZE = 10_000
AXIS_MAP = SignedAxisMap(("y", "-z", "x"))
VALUES = Rotation.random(BATCH_SIZE, random_state=916).as_quat()
EXPECTED = np.asarray(
    [AXIS_MAP.map_hamilton_xyzw(tuple(value)) for value in VALUES]
)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional dependency")
def test_engine_map_numba(benchmark):
    actual = engine_numpy.map_hamilton_xyzw(VALUES, AXIS_MAP)
    np.testing.assert_array_equal(actual, EXPECTED)
    benchmark.group = "engine axis map (Hamilton XYZW, 10k)"
    benchmark(engine_numpy.map_hamilton_xyzw, VALUES, AXIS_MAP)


def test_engine_map_numpy_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(
        engine_numpy, "_numba_rotation_kernel", lambda _operation: None
    )
    actual = engine_numpy.map_hamilton_xyzw(VALUES, AXIS_MAP)
    np.testing.assert_array_equal(actual, EXPECTED)
    benchmark.group = "engine axis map (Hamilton XYZW, 10k)"
    benchmark(engine_numpy.map_hamilton_xyzw, VALUES, AXIS_MAP)


def test_engine_map_scalar_loop(benchmark):
    actual = np.asarray(
        [AXIS_MAP.map_hamilton_xyzw(tuple(value)) for value in VALUES]
    )
    np.testing.assert_array_equal(actual, EXPECTED)
    benchmark.group = "engine axis map (Hamilton XYZW, 10k)"
    benchmark(
        lambda: np.asarray(
            [AXIS_MAP.map_hamilton_xyzw(tuple(value)) for value in VALUES]
        )
    )
