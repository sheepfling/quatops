"""Benchmark the remaining vectorized helper kernels and NumPy fallbacks.

Run with ``.venv/bin/python -m pytest
tests/benchmarks/numpy/test_numpy_helper_candidates.py -m benchmark
--benchmark-only --benchmark-group-by=group --benchmark-columns=median``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quatops import _numpy
from quatops.numpy.hamilton import xyzw as hamilton
from quatops.numpy.jpl import xyzw as jpl

pytestmark = pytest.mark.benchmark

COUNT = 10_000
RNG = np.random.default_rng(2611)

_VALUES = {
    "Hamilton": RNG.normal(size=(COUNT, 4)),
    "JPL": RNG.normal(size=(COUNT, 4)),
}
_OTHER = {
    "Hamilton": RNG.normal(size=(COUNT, 4)),
    "JPL": RNG.normal(size=(COUNT, 4)),
}
_SOURCE = RNG.normal(size=(COUNT, 3))
_TARGET = RNG.normal(size=(COUNT, 3))
_SOURCE[:4] = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1e308, -1e308, 1e308),
    (1e-300, 2e-300, -1e-300),
)
_TARGET[:4] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (-1e308, 1e308, 1e308),
    (-2e-300, 1e-300, 3e-300),
)

_BACKENDS = (("Hamilton", hamilton), ("JPL", jpl))


def _without_numba(monkeypatch):
    monkeypatch.setattr(_numpy, "_numba_rotation_kernel", lambda _name: None)


def _fallback(monkeypatch, function):
    with monkeypatch.context() as patch:
        _without_numba(patch)
        return function()


@pytest.mark.parametrize(("convention", "backend"), _BACKENDS)
def test_canonicalize_numba_matches_numpy(convention, backend, monkeypatch, benchmark):
    values = _VALUES[convention]
    function = lambda: backend.canonicalize(values)
    expected = _fallback(monkeypatch, function)
    actual = function()
    assert_array_equal(actual, expected)
    benchmark.group = f"canonicalize ({convention}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(("convention", "backend"), _BACKENDS)
def test_canonicalize_numpy_fallback_matches_numba(
    convention, backend, monkeypatch, benchmark
):
    values = _VALUES[convention]
    function = lambda: backend.canonicalize(values)
    expected = function()
    _without_numba(monkeypatch)
    actual = function()
    assert_array_equal(actual, expected)
    benchmark.group = f"canonicalize ({convention}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(("convention", "backend"), _BACKENDS)
def test_commutator_numba_matches_numpy(convention, backend, monkeypatch, benchmark):
    lhs, rhs = _VALUES[convention], _OTHER[convention]
    function = lambda: backend.commutator(lhs, rhs)
    expected = _fallback(monkeypatch, function)
    actual = function()
    assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    benchmark.group = f"commutator ({convention}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(("convention", "backend"), _BACKENDS)
def test_commutator_numpy_fallback_matches_numba(
    convention, backend, monkeypatch, benchmark
):
    lhs, rhs = _VALUES[convention], _OTHER[convention]
    function = lambda: backend.commutator(lhs, rhs)
    expected = function()
    _without_numba(monkeypatch)
    actual = function()
    assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    benchmark.group = f"commutator ({convention}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(("convention", "backend"), _BACKENDS)
def test_from_two_vectors_numba_matches_numpy(
    convention, backend, monkeypatch, benchmark
):
    function = lambda: backend.from_two_vectors(_SOURCE, _TARGET)
    expected = _fallback(monkeypatch, function)
    actual = function()
    assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    benchmark.group = f"from_two_vectors ({convention}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(("convention", "backend"), _BACKENDS)
def test_from_two_vectors_numpy_fallback_matches_numba(
    convention, backend, monkeypatch, benchmark
):
    function = lambda: backend.from_two_vectors(_SOURCE, _TARGET)
    expected = function()
    _without_numba(monkeypatch)
    actual = function()
    assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    benchmark.group = f"from_two_vectors ({convention}, 10k)"
    benchmark(function)
