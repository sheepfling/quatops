"""Shared validation and result helpers for NumPy quaternion operations."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np

_LOGGER = logging.getLogger("quatops._numpy")


def _warn_numba_failure(operation: str, error: Exception) -> None:
    """Report a broken optional accelerator without affecting the result."""
    if isinstance(error, ModuleNotFoundError) and error.name == "numba":
        return
    _LOGGER.warning(
        "Numba %s acceleration failed; using the NumPy fallback (%s: %s)",
        operation,
        type(error).__name__,
        error,
    )


class _Scaled(NamedTuple):
    scale: np.ndarray
    unit_scale: np.ndarray
    sum_squares: np.ndarray


def _components(
    value: Any, width: int, name: str, *, validate_finite: bool = True
) -> np.ndarray:
    """Validate a trailing component axis and convert it to float64."""
    try:
        original = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if original.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric array")
    if original.ndim == 0 or original.shape[-1] != width:
        raise ValueError(f"{name} must have shape (..., {width})")
    with np.errstate(all="ignore"):
        array = original.astype(np.float64, copy=False)
    if validate_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite float64 values")
    return array


def _result(value: Any) -> Any:
    if not np.all(np.isfinite(value)):
        raise OverflowError("quaternion arithmetic produced a non-finite result")
    return value


def _scaled(q: np.ndarray) -> _Scaled:
    scale = np.max(np.abs(q), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero quaternion has no normalization or inverse")
    with np.errstate(all="ignore"):
        unit_scale = q / scale
        sum_squares = np.sum(unit_scale * unit_scale, axis=-1, keepdims=True)
    return _Scaled(scale, unit_scale, sum_squares)
