"""Numerical boundaries of coefficient algebra across public backends."""

import math
import sys
from fractions import Fraction

import numpy as np
import pytest

from tests.oracle_support import BACKENDS, from_wxyz, to_wxyz


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_scaled_norm_and_inverse_at_binary64_boundaries(backend):
    _, module, _, is_numpy, component_first = backend
    tiny = math.nextafter(0.0, 1.0)
    small = from_wxyz(module, (tiny, 0.0, 0.0, 0.0), is_numpy, component_first)
    maximum = from_wxyz(
        module, (sys.float_info.max, 0.0, 0.0, 0.0), is_numpy, component_first
    )

    with np.errstate(all="raise"):
        assert float(module.norm(small)) == tiny
        assert float(module.norm_squared(small)) == 0.0
        np.testing.assert_array_equal(
            to_wxyz(module, module.normalize(small), is_numpy, component_first),
            (1.0, 0.0, 0.0, 0.0),
        )
        with pytest.raises(OverflowError):
            module.inverse(small)
        with pytest.raises(OverflowError):
            module.add(maximum, maximum)
        assert all(value == "raise" for value in np.geterr().values())


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_quotient_may_overflow_while_forming_inverse(backend):
    _, module, _, is_numpy, component_first = backend
    tiny = math.nextafter(0.0, 1.0)
    quaternion = from_wxyz(module, (tiny, 0.0, 0.0, 0.0), is_numpy, component_first)

    # The exact quotient is identity; a future implementation may compute it
    # without first forming the unrepresentable inverse.
    try:
        result = module.divide(quaternion, quaternion)
    except OverflowError:
        return
    np.testing.assert_array_equal(
        to_wxyz(module, result, is_numpy, component_first), (1.0, 0.0, 0.0, 0.0)
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend[0])
def test_dot_underflow_example_and_overflow_cancellation(backend):
    _, module, _, is_numpy, component_first = backend
    tiny = math.nextafter(0.0, 1.0)
    small_lhs = from_wxyz(module, (tiny, tiny, 0.0, 0.0), is_numpy, component_first)
    small_rhs = from_wxyz(module, (0.5, 0.5, 0.0, 0.0), is_numpy, component_first)
    exact_small = float(2 * Fraction.from_float(tiny) * Fraction.from_float(0.5))
    assert exact_small == tiny
    with np.errstate(all="raise"):
        actual = float(module.dot(small_lhs, small_rhs))
        # Scalar currently preserves this result; array paths can lose both
        # products before summation. Either result permits a future fix.
        assert actual in (0.0, exact_small)
        if not is_numpy:
            assert actual == exact_small

        large_lhs = from_wxyz(
            module, (1e308, 1e308, 1.0, 0.0), is_numpy, component_first
        )
        large_rhs = from_wxyz(
            module, (1e308, -1e308, 1.0, 0.0), is_numpy, component_first
        )
        assert float(module.dot(large_lhs, large_rhs)) == 1.0
        assert all(value == "raise" for value in np.geterr().values())
