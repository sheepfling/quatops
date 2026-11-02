"""Vectorized signed-axis conversion into game-engine Hamilton XYZW values.

Importing this module requires the optional NumPy dependency. The input
component axis is preserved; samples and batch axes are otherwise untouched.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from .. import _numpy
from .._numpy import (
    _numba_rotation_kernel,
    _warn_numba_failure,
)
from .basis import SignedAxisMap


def _map(
    q: Any,
    axis_map: SignedAxisMap,
    component_axis: int,
    scalar_first: bool,
    jpl: bool,
) -> np.ndarray:
    values = np.asarray(q)
    if values.ndim == 0:
        raise ValueError("q must have a quaternion component axis")
    try:
        axis = operator.index(component_axis)
    except TypeError as exc:
        raise TypeError("component_axis must be an integer") from exc
    if axis < 0:
        axis += values.ndim
    if not 0 <= axis < values.ndim:
        raise ValueError("component_axis is out of range for q")

    moved = np.moveaxis(values, axis, -1)
    if moved.shape[-1] != 4:
        raise ValueError("the quaternion component axis must have length four")
    if not isinstance(axis_map, SignedAxisMap):
        raise TypeError("axis_map must be a SignedAxisMap")

    sample_count = moved.size // 4
    if sample_count >= 128 and values.dtype in (np.dtype("float32"), np.dtype("float64")):
        kernel = _numba_rotation_kernel("engine_axis_map")
        if kernel is not None:
            rows = np.ascontiguousarray(moved.reshape((-1, 4)))
            indices = np.asarray(axis_map._indices, dtype=np.int64)
            signs = np.asarray(
                axis_map._jpl_signs if jpl else axis_map._hamilton_signs,
                dtype=np.int64,
            )
            try:
                result = kernel(rows, indices, signs, scalar_first)
            except Exception as exc:  # noqa: BLE001
                _numpy._NUMBA_ROTATION_KERNELS.pop("engine_axis_map", None)
                _warn_numba_failure("engine axis map", exc)
            else:
                return np.moveaxis(result.reshape(moved.shape), -1, axis)

    output = np.empty_like(moved)
    offset = 1 if scalar_first else 0
    scalar_index = 0 if scalar_first else 3
    signs = axis_map._jpl_signs if jpl else axis_map._hamilton_signs
    for destination_index, (source_index, sign) in enumerate(
        zip(axis_map._indices, signs)
    ):
        source_component = moved[..., offset + source_index]
        if sign > 0:
            output[..., destination_index] = source_component
        else:
            output[..., destination_index] = -source_component
    output[..., 3] = moved[..., scalar_index]
    return np.moveaxis(output, -1, axis)


def map_hamilton_xyzw(q: Any, axis_map: SignedAxisMap,
                      component_axis: int = -1) -> np.ndarray:
    """Map Hamilton XYZW arrays to engine XYZW arrays without normalization."""
    return _map(q, axis_map, component_axis, False, False)


def map_hamilton_wxyz(q: Any, axis_map: SignedAxisMap,
                      component_axis: int = -1) -> np.ndarray:
    """Map Hamilton WXYZ arrays to engine XYZW arrays without normalization."""
    return _map(q, axis_map, component_axis, True, False)


def map_jpl_xyzw(q: Any, axis_map: SignedAxisMap,
                 component_axis: int = -1) -> np.ndarray:
    """Map active JPL XYZW arrays directly to engine Hamilton XYZW."""
    return _map(q, axis_map, component_axis, False, True)


def map_jpl_wxyz(q: Any, axis_map: SignedAxisMap,
                 component_axis: int = -1) -> np.ndarray:
    """Map active JPL WXYZ arrays directly to engine Hamilton XYZW."""
    return _map(q, axis_map, component_axis, True, True)


__all__ = [
    "map_hamilton_wxyz",
    "map_hamilton_xyzw",
    "map_jpl_wxyz",
    "map_jpl_xyzw",
]
