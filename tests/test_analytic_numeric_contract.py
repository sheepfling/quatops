"""Numerical and singularity boundaries for analytic quaternion functions."""

import math

import numpy as np
import pytest

from tests.oracle_support import BACKENDS, from_wxyz, to_wxyz


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_small_real_expm1_and_log1p_preserve_the_input(backend):
    _, module, _, is_numpy, component_first = backend
    value = math.ldexp(1.0, -54)
    quaternion = from_wxyz(module, (value, 0.0, 0.0, 0.0), is_numpy, component_first)

    exp_result = to_wxyz(module, module.expm1(quaternion), is_numpy, component_first)
    log_result = to_wxyz(module, module.log1p(quaternion), is_numpy, component_first)
    np.testing.assert_array_equal(exp_result, (math.expm1(value), 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(log_result, (math.log1p(value), 0.0, 0.0, 0.0))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_analytic_singularities_and_overflow_are_distinct(backend):
    _, module, _, is_numpy, component_first = backend
    for real in (-1.0, 1.0):
        quaternion = from_wxyz(module, (real, 0.0, 0.0, 0.0), is_numpy, component_first)
        with pytest.raises(ValueError):
            module.atanh(quaternion)
    imaginary_unit = from_wxyz(module, (0.0, 1.0, 0.0, 0.0), is_numpy, component_first)
    with pytest.raises(ValueError):
        module.atan(imaginary_unit)
    too_large = from_wxyz(module, (710.0, 0.0, 0.0, 0.0), is_numpy, component_first)
    with pytest.raises(OverflowError):
        module.exp(too_large)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_invalid_nonfinite_components_are_rejected_before_analytic_operations(backend):
    _, module, _, _, _ = backend
    for value in (float("nan"), float("inf"), -float("inf")):
        for operation in (module.norm, module.inverse, module.exp):
            with pytest.raises(ValueError):
                operation((value, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("backend", [backend for backend in BACKENDS if backend[3]],
                         ids=lambda backend: backend[0])
def test_numpy_analytic_singularity_rejects_the_whole_batch(backend):
    _, module, _, _, component_first = backend
    samples = np.tile((0.25, 0.0, 0.0, 0.0), (128, 1))
    samples[42, 0] = 1.0
    quaternion = from_wxyz(module, samples, True, component_first)
    with np.errstate(all="raise"):
        # Numba and the scalar fallback report this singularity differently.
        with pytest.raises((ValueError, OverflowError)):
            module.atanh(quaternion)
        assert all(value == "raise" for value in np.geterr().values())
