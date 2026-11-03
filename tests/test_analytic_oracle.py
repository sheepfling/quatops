"""Independent complex-slice oracles for analytic quaternion operations."""

import cmath
import math

import numpy as np
import pytest

from tests.oracle_support import BACKENDS, from_wxyz, to_wxyz, vectors_to_trailing

ANALYTIC_FUNCTIONS = (
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "asin", "acos", "atan", "asinh", "acosh", "atanh",
)


def _complex_slice(function, values: np.ndarray, is_jpl: bool) -> np.ndarray:
    """Evaluate f(w + i|v|), then lift its imaginary part along v."""
    result = np.empty_like(values, dtype=float)
    for index, (w, x, y, z) in enumerate(values):
        radius = math.hypot(x, y, z)
        value = function(complex(w, radius))
        if radius:
            direction = np.array((x, y, z)) / radius
        else:
            # The documented real-axis branch uses +i in Hamilton. JPL
            # coefficients map to Hamilton with the opposite vector sign.
            direction = np.array((-1.0 if is_jpl else 1.0, 0.0, 0.0))
        result[index] = (value.real, *(value.imag * direction))
    return result


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_analytic_functions_match_independent_complex_slices(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rng = np.random.default_rng(0xA11A + len(backend_name))
    values = rng.normal(scale=0.42, size=(96, 4))
    values[:, 0] += 0.17
    if is_numpy:
        operands = from_wxyz(module, values, True, component_first)
        for name in ANALYTIC_FUNCTIONS:
            actual = to_wxyz(
                module,
                getattr(module, name)(operands),
                True,
                component_first,
            )
            expected = _complex_slice(getattr(cmath, name), values, is_jpl)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=4e-13,
                atol=4e-13,
                err_msg=f"{backend_name}.{name}",
            )
    else:
        for name in ANALYTIC_FUNCTIONS:
            actual = np.stack(
                [
                    to_wxyz(
                        module,
                        getattr(module, name)(from_wxyz(module, row, False, False)),
                        False,
                        False,
                    )
                    for row in values
                ]
            )
            expected = _complex_slice(getattr(cmath, name), values, is_jpl)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=4e-13,
                atol=4e-13,
                err_msg=f"{backend_name}.{name}",
            )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_analytic_real_axis_branch_and_component_extraction(
    backend_name, module, is_jpl, is_numpy, component_first
):
    values = np.array(((0.35, 0.0, 0.0, 0.0), (-1.5, 0.0, 0.0, 0.0),
                       (1.7, 0.0, 0.0, 0.0)))
    if is_numpy:
        operands = from_wxyz(module, values, True, component_first)
        for name in ("asin", "acos", "atan", "asinh", "acosh", "atanh"):
            actual = to_wxyz(module, getattr(module, name)(operands), True, component_first)
            expected = _complex_slice(getattr(cmath, name), values, is_jpl)
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14,
                                       err_msg=f"{backend_name}.{name} real-axis branch")
        scalar = np.asarray(module.scalar_part(operands))
        vector = vectors_to_trailing(module.vector_part(operands), component_first)
        np.testing.assert_array_equal(scalar, values[:, 0])
        np.testing.assert_array_equal(vector, values[:, 1:])
    else:
        for row in values:
            value = from_wxyz(module, row, False, False)
            np.testing.assert_equal(module.scalar_part(value), row[0])
            np.testing.assert_array_equal(tuple(module.vector_part(value)), row[1:])
            for name in ("asin", "acos", "atan", "asinh", "acosh", "atanh"):
                actual = to_wxyz(module, getattr(module, name)(value), False, False)
                expected = _complex_slice(getattr(cmath, name), row[None, :], is_jpl)[0]
                np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14,
                                           err_msg=f"{backend_name}.{name} real-axis branch")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_log_negative_real_branch_matches_numpy_quaternion(
    backend_name, module, is_jpl, is_numpy, component_first
):
    values = np.array(((-2.0, 0.0, 0.0, 0.0), (-0.25, 0.0, 0.0, 0.0)))
    if is_numpy:
        operands = from_wxyz(module, values, True, component_first)
        actual = to_wxyz(module, module.log(operands), True, component_first)
    else:
        actual = np.stack(
            [to_wxyz(module, module.log(from_wxyz(module, row, False, False)), False, False)
             for row in values]
        )
    # Generic quaternion log keeps its algebraic positive-X branch in both
    # coefficient conventions. numpy-quaternion's JPL coefficient mapping
    # chooses the opposite physical vector axis at this singular real input.
    expected = _complex_slice(cmath.log, values, is_jpl=False)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14,
                               err_msg=f"{backend_name}.log branch")


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_analytic_log_rejects_zero_for_every_representation(
    backend_name, module, is_jpl, is_numpy, component_first
):
    zero = from_wxyz(module, np.zeros(4), is_numpy, component_first)
    with pytest.raises(ValueError, match="zero"):
        module.log(zero)
