"""Keep optional compiled multiplication correct across both conventions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quatops import _numpy
from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.hamilton import wxyz as hamilton_wxyz
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw

COUNT = 128
HAM_LHS = hamilton_scalar.normalize((0.91, 0.11, 0.31, -0.20))
HAM_RHS = hamilton_scalar.normalize((0.97, -0.14, 0.08, 0.17))
JPL_LHS = jpl_scalar.from_hamilton(HAM_LHS)
JPL_RHS = jpl_scalar.from_hamilton(HAM_RHS)


def test_hamilton_wxyz_large_batch_matches_scalar():
    lhs = np.tile(HAM_LHS, (COUNT, 1))
    rhs = np.tile(HAM_RHS, (COUNT, 1))

    actual = hamilton_wxyz.multiply(lhs, rhs)

    assert_allclose(
        actual, np.tile(hamilton_scalar.multiply(HAM_LHS, HAM_RHS), (COUNT, 1))
    )


def test_hamilton_xyzw_large_batch_matches_scalar():
    lhs = np.tile(hamilton_scalar.to_xyzw(HAM_LHS), (COUNT, 1))
    rhs = np.tile(hamilton_scalar.to_xyzw(HAM_RHS), (COUNT, 1))

    actual = hamilton_xyzw.multiply(lhs, rhs)

    expected = np.tile(
        hamilton_scalar.to_xyzw(hamilton_scalar.multiply(HAM_LHS, HAM_RHS)),
        (COUNT, 1),
    )
    assert_allclose(actual, expected)


def test_jpl_wxyz_large_batch_matches_scalar():
    lhs = np.tile(jpl_scalar.to_wxyz(JPL_LHS), (COUNT, 1))
    rhs = np.tile(jpl_scalar.to_wxyz(JPL_RHS), (COUNT, 1))

    actual = jpl_wxyz.multiply(lhs, rhs)

    expected = np.tile(
        jpl_scalar.to_wxyz(jpl_scalar.multiply(JPL_LHS, JPL_RHS)), (COUNT, 1)
    )
    assert_allclose(actual, expected)


def test_jpl_xyzw_large_batch_matches_scalar():
    lhs = np.tile(jpl_scalar.to_xyzw(JPL_LHS), (COUNT, 1))
    rhs = np.tile(jpl_scalar.to_xyzw(JPL_RHS), (COUNT, 1))

    actual = jpl_xyzw.multiply(lhs, rhs)

    expected = np.tile(
        jpl_scalar.to_xyzw(jpl_scalar.multiply(JPL_LHS, JPL_RHS)),
        (COUNT, 1),
    )
    assert_allclose(actual, expected)


def test_large_batch_falls_back_to_numpy_without_numba(monkeypatch):
    monkeypatch.setattr(_numpy, "_NUMBA_MULTIPLY_CHECKED", True)
    monkeypatch.setattr(_numpy, "_NUMBA_MULTIPLY", {})
    lhs = np.tile(hamilton_scalar.to_xyzw(HAM_LHS), (COUNT, 1))
    rhs = np.tile(hamilton_scalar.to_xyzw(HAM_RHS), (COUNT, 1))

    actual = hamilton_xyzw.multiply(lhs, rhs)

    expected = np.tile(
        hamilton_scalar.to_xyzw(hamilton_scalar.multiply(HAM_LHS, HAM_RHS)),
        (COUNT, 1),
    )
    assert_allclose(actual, expected)


@pytest.mark.parametrize("bad_value", (np.nan, np.inf))
def test_large_batch_nonfinite_input_remains_value_error(bad_value):
    lhs = np.tile(hamilton_scalar.to_xyzw(HAM_LHS), (COUNT, 1))
    rhs = np.tile(hamilton_scalar.to_xyzw(HAM_RHS), (COUNT, 1))
    lhs[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        hamilton_xyzw.multiply(lhs, rhs)


def test_large_batch_finite_overflow_remains_overflow_error():
    lhs = np.zeros((COUNT, 4))
    rhs = np.zeros((COUNT, 4))
    lhs[:, 3] = 1e308
    rhs[:, 3] = 1e308

    with pytest.raises(OverflowError, match="non-finite"):
        hamilton_xyzw.multiply(lhs, rhs)
