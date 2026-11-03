"""Check vectorized scalar-mapped helpers across convention and layout APIs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import scalar as jpl_scalar
from quatops.numpy.hamilton import wxyz as hamilton_wxyz
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw

BACKENDS = (
    (hamilton_wxyz, hamilton_scalar, "wxyz"),
    (hamilton_xyzw, hamilton_scalar, "xyzw"),
    (jpl_wxyz, jpl_scalar, "wxyz"),
    (jpl_xyzw, jpl_scalar, "xyzw"),
)


def _scalar_order(values, scalar, layout):
    if scalar is hamilton_scalar and layout == "xyzw":
        return values[..., (3, 0, 1, 2)]
    if scalar is jpl_scalar and layout == "wxyz":
        return values[..., (1, 2, 3, 0)]
    return values


def _array_order(values, scalar, layout):
    if scalar is hamilton_scalar and layout == "xyzw":
        return values[..., (1, 2, 3, 0)]
    if scalar is jpl_scalar and layout == "wxyz":
        return values[..., (3, 0, 1, 2)]
    return values


@pytest.mark.parametrize("backend,scalar,layout", BACKENDS)
def test_vectorized_helpers_match_scalar_with_broadcasting(backend, scalar, layout):
    rng = np.random.default_rng(912)
    q_array = rng.normal(size=(3, 1, 4))
    reference_array = rng.normal(size=(1, 5, 4))
    q_scalar = _scalar_order(q_array, scalar, layout)
    reference_scalar = _scalar_order(reference_array, scalar, layout)
    q_broadcast, reference_broadcast = np.broadcast_arrays(q_scalar, reference_scalar)

    expected_commutator = np.asarray(
        [
            scalar.commutator(tuple(q), tuple(reference))
            for q, reference in zip(
                q_broadcast.reshape((-1, 4)),
                reference_broadcast.reshape((-1, 4)),
            )
        ]
    ).reshape((3, 5, 4))
    expected_aligned = np.asarray(
        [
            scalar.align_sign(tuple(q), tuple(reference))
            for q, reference in zip(
                q_broadcast.reshape((-1, 4)),
                reference_broadcast.reshape((-1, 4)),
            )
        ]
    ).reshape((3, 5, 4))

    assert_allclose(
        backend.commutator(q_array, reference_array),
        _array_order(expected_commutator, scalar, layout),
        rtol=2e-14,
        atol=2e-14,
    )
    assert_allclose(
        backend.align_sign(q_array, reference_array),
        _array_order(expected_aligned, scalar, layout),
        rtol=0.0,
        atol=0.0,
    )

    expected_scalar_part = np.asarray(
        [scalar.scalar_part(tuple(q)) for q in q_scalar.reshape((-1, 4))]
    ).reshape((3, 1))
    expected_vector_part = np.asarray(
        [scalar.vector_part(tuple(q)) for q in q_scalar.reshape((-1, 4))]
    ).reshape((3, 1, 3))
    assert_array_equal(backend.scalar_part(q_array), expected_scalar_part)
    assert_array_equal(backend.vector_part(q_array), expected_vector_part)


@pytest.mark.parametrize("backend,_,_layout", BACKENDS)
def test_vectorized_component_helpers_return_independent_float64_arrays(
    backend, _, _layout
):
    values = np.arange(12, dtype=np.int16).reshape((3, 4))
    scalar = backend.scalar_part(values)
    vector = backend.vector_part(values)

    assert scalar.dtype == np.float64
    assert vector.dtype == np.float64
    assert scalar.shape == (3,)
    assert vector.shape == (3, 3)
    assert not np.shares_memory(scalar, values)
    assert not np.shares_memory(vector, values)


@pytest.mark.parametrize("backend,_,_layout", BACKENDS)
def test_vectorized_helpers_preserve_broadcast_errors(backend, _, _layout):
    with pytest.raises(ValueError, match="broadcastable"):
        backend.commutator(np.zeros((2, 4)), np.zeros((3, 4)))
    with pytest.raises(ValueError, match="broadcastable"):
        backend.align_sign(np.zeros((2, 4)), np.zeros((3, 4)))
