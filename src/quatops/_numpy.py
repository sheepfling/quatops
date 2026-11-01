"""Internal validation and numerical helpers for explicit NumPy backends."""

from typing import Any, NamedTuple

import numpy as np


class _Scaled(NamedTuple):
    scale: np.ndarray
    unit_scale: np.ndarray
    sum_squares: np.ndarray


def _components(value: Any, width: int, name: str) -> np.ndarray:
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
    if not np.all(np.isfinite(array)):
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


def _sum_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Sum three finite products, cancelling large opposing terms first."""
    def opposing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x < 0) & (y > 0)) | ((x > 0) & (y < 0))

    ab = np.where(opposing(a, b), np.minimum(np.abs(a), np.abs(b)), 0.0)
    ac = np.where(opposing(a, c), np.minimum(np.abs(a), np.abs(c)), 0.0)
    bc = np.where(opposing(b, c), np.minimum(np.abs(b), np.abs(c)), 0.0)
    use_ab = (ab >= ac) & (ab >= bc)
    use_ac = ~use_ab & (ac >= bc)
    first = np.where(use_ab | use_ac, a, b)
    second = np.where(use_ab, b, c)
    third = np.where(use_ab, c, np.where(use_ac, b, a))

    s = first + second
    s_part = s - first
    s_error = (first - (s - s_part)) + (second - s_part)
    t = s + third
    t_part = t - s
    t_error = (s - (t - t_part)) + (third - t_part)
    return t + (s_error + t_error)


def _direction(v: np.ndarray, allow_zero: bool = False) -> tuple:
    """Return unit directions, maximum magnitude, and scaled norms."""
    with np.errstate(all="ignore"):
        scale = np.max(np.abs(v), axis=-1, keepdims=True)
        if not allow_zero and np.any(scale == 0):
            raise ValueError("axis must be nonzero")
        parts = np.divide(v, scale, out=np.zeros_like(v), where=scale != 0)
        length = np.sqrt(np.sum(parts * parts, axis=-1, keepdims=True))
        unit = np.divide(parts, length, out=np.zeros_like(parts), where=length != 0)
    return unit, scale, length
