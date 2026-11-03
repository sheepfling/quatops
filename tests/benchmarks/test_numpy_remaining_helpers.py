"""Benchmark optional Numba paths for coefficient and conversion helpers.

Run with ``.venv/bin/python -m pytest
tests/benchmarks/test_numpy_remaining_helpers.py -m benchmark
--benchmark-only --benchmark-group-by=group --benchmark-columns=median``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quatops import _numpy
from quatops.numpy.hamilton import wxyz as hamilton_wxyz
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw

pytestmark = pytest.mark.benchmark

COUNT = 10_000
RNG = np.random.default_rng(2613)
_VALUES_WXYZ = RNG.normal(size=(COUNT, 4))

_BACKENDS = (
    ("Hamilton wxyz", hamilton_wxyz, False, False),
    ("Hamilton xyzw", hamilton_xyzw, True, False),
    ("JPL wxyz", jpl_wxyz, False, True),
    ("JPL xyzw", jpl_xyzw, True, True),
)
_SCALAR_FIRST_BACKENDS = (
    ("Hamilton wxyz", hamilton_wxyz),
    ("JPL wxyz", jpl_wxyz),
)


def _without_numba(monkeypatch):
    monkeypatch.setattr(_numpy, "_numba_rotation_kernel", lambda _name: None)


def _fallback(monkeypatch, function):
    with monkeypatch.context() as patch:
        _without_numba(patch)
        return function()


def _coefficient_suite(backend):
    return backend.conjugate(_VALUES_WXYZ), backend.vector_part(_VALUES_WXYZ)


def _transform_suite(backend, scalar_last, is_jpl):
    outputs = [
        backend.from_wxyz(_VALUES_WXYZ)
        if scalar_last
        else backend.to_xyzw(_VALUES_WXYZ)
    ]
    if is_jpl:
        converted = backend.from_hamilton(_VALUES_WXYZ)
        outputs.append(converted)
        if not scalar_last:
            outputs.append(backend.to_hamilton(converted))
    return tuple(outputs)


def _random_suite(backend):
    return (
        backend.random_quaternion(
            size=COUNT, rng=np.random.default_rng(726)
        ),
        backend.random_rotation(
            size=COUNT, rng=np.random.default_rng(726)
        ),
    )


def _assert_exact_batches(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        assert_array_equal(actual_value, expected_value)


def _assert_random_batches(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        assert_allclose(actual_value, expected_value, rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize(
    ("label", "backend"),
    _SCALAR_FIRST_BACKENDS,
    ids=[case[0] for case in _SCALAR_FIRST_BACKENDS],
)
def test_scalar_first_coefficient_numba_matches_numpy(
    label, backend, monkeypatch, benchmark
):
    function = lambda: _coefficient_suite(backend)
    expected = _fallback(monkeypatch, function)
    _assert_exact_batches(function(), expected)
    benchmark.group = f"conjugate and vector extraction ({label}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(
    ("label", "backend"),
    _SCALAR_FIRST_BACKENDS,
    ids=[case[0] for case in _SCALAR_FIRST_BACKENDS],
)
def test_scalar_first_coefficient_numpy_fallback_matches_numba(
    label, backend, monkeypatch, benchmark
):
    function = lambda: _coefficient_suite(backend)
    expected = function()
    _without_numba(monkeypatch)
    _assert_exact_batches(function(), expected)
    benchmark.group = f"conjugate and vector extraction ({label}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(
    ("label", "backend", "scalar_last", "is_jpl"),
    _BACKENDS,
    ids=[case[0] for case in _BACKENDS],
)
def test_component_transforms_numba_match_numpy(
    label, backend, scalar_last, is_jpl, monkeypatch, benchmark
):
    function = lambda: _transform_suite(backend, scalar_last, is_jpl)
    expected = _fallback(monkeypatch, function)
    _assert_exact_batches(function(), expected)
    benchmark.group = f"accelerated component transforms ({label}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(
    ("label", "backend", "scalar_last", "is_jpl"),
    _BACKENDS,
    ids=[case[0] for case in _BACKENDS],
)
def test_component_transforms_numpy_fallback_matches_numba(
    label, backend, scalar_last, is_jpl, monkeypatch, benchmark
):
    function = lambda: _transform_suite(backend, scalar_last, is_jpl)
    expected = function()
    _without_numba(monkeypatch)
    _assert_exact_batches(function(), expected)
    benchmark.group = f"accelerated component transforms ({label}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(
    ("label", "backend", "scalar_last", "is_jpl"),
    _BACKENDS,
    ids=[case[0] for case in _BACKENDS],
)
def test_random_sampling_numba_matches_numpy(
    label, backend, scalar_last, is_jpl, monkeypatch, benchmark
):
    del scalar_last, is_jpl
    function = lambda: _random_suite(backend)
    expected = _fallback(monkeypatch, function)
    _assert_random_batches(function(), expected)
    benchmark.group = f"random sampling ({label}, 10k)"
    benchmark(function)


@pytest.mark.parametrize(
    ("label", "backend", "scalar_last", "is_jpl"),
    _BACKENDS,
    ids=[case[0] for case in _BACKENDS],
)
def test_random_sampling_numpy_fallback_matches_numba(
    label, backend, scalar_last, is_jpl, monkeypatch, benchmark
):
    del scalar_last, is_jpl
    function = lambda: _random_suite(backend)
    expected = function()
    _without_numba(monkeypatch)
    _assert_random_batches(function(), expected)
    benchmark.group = f"random sampling ({label}, 10k)"
    benchmark(function)
