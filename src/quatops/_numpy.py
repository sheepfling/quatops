"""Internal validation and numerical helpers for explicit NumPy backends."""

from __future__ import annotations

import cmath
import logging
from math import (
    acos,
    atan2,
    copysign,
    cos,
    exp,
    frexp,
    hypot,
    isfinite,
    ldexp,
    log,
    pi,
    prod,
    sin,
    sqrt,
)
from numbers import Integral, Real
from typing import Any, Callable, NamedTuple

import numpy as np

_LOGGER = logging.getLogger(__name__)
_NUMBA_PRANGE: Any = range


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


from ._euler_sequence import euler_metadata as _euler_sequence_metadata
from ._euler_types import EulerResult


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


def _factor(value: Any, name: str = "factor") -> float:
    """Validate a finite real scalar used by an operation argument."""
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")  # noqa: TRY004
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _axis_index(value: Any, ndim: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer array axis")  # noqa: TRY004
    axis = int(value)
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for an array with {ndim} dimensions")
    return axis % ndim


def _euler_from_angles(
    angles: Any,
    sequence: str,
    *,
    degrees: bool,
    angle_axis: int,
    component_axis: int,
    algebra_epsilon: int,
    scalar_last: bool,
) -> np.ndarray:
    """Vectorized direct Euler-to-quaternion formulas with configurable axes."""
    original = np.asarray(angles)
    if original.ndim == 0:
        raise ValueError("angles must have a length-three angle axis")
    source_axis = _axis_index(angle_axis, original.ndim, "angle_axis")
    moved = np.moveaxis(original, source_axis, -1)
    values = _components(moved, 3, "angles")
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    intrinsic, i, j, k, ell, epsilon, proper = _euler_sequence_metadata(sequence)
    if not intrinsic:
        values = values[..., ::-1]
    if degrees:
        values = np.deg2rad(values)
    a1, a2, a3 = np.moveaxis(values, -1, 0)
    with np.errstate(all="ignore"):
        if proper:
            u = 0.5 * a1 + 0.5 * a3
            v = 0.5 * a1 - 0.5 * a3
            w = np.cos(0.5 * a2) * np.cos(u)
            qi = np.cos(0.5 * a2) * np.sin(u)
            qj = np.sin(0.5 * a2) * np.cos(v)
            qell = epsilon * np.sin(0.5 * a2) * np.sin(v)
            h = [np.zeros_like(w), np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)]
            h[0], h[i], h[j], h[ell] = w, qi, qj, qell
        else:
            half = values * 0.5
            c1, c2, c3 = np.cos(half[..., 0]), np.cos(half[..., 1]), np.cos(half[..., 2])
            s1, s2, s3 = np.sin(half[..., 0]), np.sin(half[..., 1]), np.sin(half[..., 2])
            w = c1*c2*c3 - epsilon*s1*s2*s3
            qi = s1*c2*c3 + epsilon*c1*s2*s3
            qj = c1*s2*c3 - epsilon*s1*c2*s3
            qk = c1*c2*s3 + epsilon*s1*s2*c3
            h = [w, np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)]
            h[i], h[j], h[k] = qi, qj, qk
        first_is_negative = (
            (h[1] < 0.0)
            | ((h[1] == 0.0) & (h[2] < 0.0))
            | ((h[1] == 0.0) & (h[2] == 0.0) & (h[3] < 0.0))
        )
        flip = (h[0] < 0.0) | ((h[0] == 0.0) & first_is_negative)
        if scalar_last:
            result = np.stack((algebra_epsilon*h[1], algebra_epsilon*h[2],
                               algebra_epsilon*h[3], h[0]), axis=-1)
        else:
            result = np.stack((h[0], algebra_epsilon*h[1],
                               algebra_epsilon*h[2], algebra_epsilon*h[3]), axis=-1)
        np.negative(result, out=result, where=flip[..., None])
        np.copyto(result, 0.0, where=result == 0.0)
        result = np.moveaxis(result, -1, _axis_index(component_axis, result.ndim, "component_axis"))
    return result


def _euler_to_angles(
    q: Any,
    sequence: str,
    *,
    degrees: bool,
    gimbal_tolerance: Any,
    component_axis: int,
    angle_axis: int,
    algebra_epsilon: int,
    scalar_last: bool,
    with_status: bool,
) -> Any:
    """Vectorized, scaled quaternion-to-Euler phase extraction."""
    original = np.asarray(q)
    if original.ndim == 0:
        raise ValueError("quaternion must have a length-four component axis")
    source_axis = _axis_index(component_axis, original.ndim, "component_axis")
    values = _components(np.moveaxis(original, source_axis, -1), 4, "quaternion")
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    tolerance = _factor(gimbal_tolerance, "gimbal_tolerance")
    if tolerance < 0.0 or tolerance >= np.pi / 2.0:
        raise ValueError("gimbal_tolerance must be in [0, pi/2)")
    intrinsic, i, j, k, ell, epsilon, proper = _euler_sequence_metadata(sequence)
    scale = np.max(np.abs(values), axis=-1)
    if np.any(scale == 0.0):
        raise ValueError("quaternion must be nonzero")
    scaled = values / scale[..., None]
    if scalar_last:
        x, y, z, w = np.moveaxis(scaled, -1, 0)
    else:
        w, x, y, z = np.moveaxis(scaled, -1, 0)
    x, y, z = algebra_epsilon*x, algebra_epsilon*y, algebra_epsilon*z
    coefficients = (w, x, y, z)
    if proper:
        a, b = w, coefficients[i]
        c, d = coefficients[j], epsilon*coefficients[ell]
        kappa = 1
    else:
        a = w + coefficients[j]
        b = coefficients[i] + epsilon*coefficients[k]
        c = w - coefficients[j]
        d = coefficients[i] - epsilon*coefficients[k]
        kappa = epsilon
    r_plus, r_minus = np.hypot(a, b), np.hypot(c, d)
    chi = 2.0 * np.arctan2(r_minus, r_plus)
    near_zero = chi <= tolerance
    near_pi = np.pi - chi <= tolerance
    locked = near_zero | near_pi
    # Endpoint branches ignore the ambiguous phase whose vector vanishes.
    u = np.arctan2(b, a)
    v = np.arctan2(d, c)
    if intrinsic:
        alpha = np.where(near_zero, 2.0*u,
                         np.where(near_pi, 2.0*v, u + v))
        gamma = np.where(locked, 0.0, kappa*(u - v))
    else:
        alpha = np.where(near_zero, 2.0*kappa*u,
                         np.where(near_pi, -2.0*kappa*v, kappa*(u - v)))
        gamma = np.where(locked, 0.0, u + v)
    chi = np.where(near_zero, 0.0, np.where(near_pi, np.pi, chi))
    beta = chi if proper else np.pi / 2.0 - chi
    alpha = np.where(alpha >= np.pi, alpha - 2.0*np.pi,
                     np.where(alpha < -np.pi, alpha + 2.0*np.pi, alpha))
    gamma = np.where(gamma >= np.pi, gamma - 2.0*np.pi,
                     np.where(gamma < -np.pi, gamma + 2.0*np.pi, gamma))
    angles = np.stack((alpha, beta, gamma), axis=-1)
    if degrees:
        angles = np.rad2deg(angles)
    output_axis = _axis_index(angle_axis, angles.ndim, "angle_axis")
    angles = np.moveaxis(angles, -1, output_axis)
    if not with_status:
        return angles
    return EulerResult(angles, locked)


def _binary(
    lhs: Any,
    rhs: Any,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply a component-wise binary operation with leading-axis broadcasting."""
    left = _components(lhs, 4, "lhs")
    right = _components(rhs, 4, "rhs")
    left, right = np.broadcast_arrays(left, right)
    with np.errstate(all="ignore"):
        result = operation(left, right)
    return _result(result)


def _scale(q: Any, factor: Any) -> np.ndarray:
    """Scale quaternion components by one finite real scalar."""
    array = _components(q, 4, "quaternion")
    with np.errstate(all="ignore"):
        result = array * _factor(factor)
    return _result(result)


def _algebraic(q: Any, scalar_last: bool, operation: str) -> np.ndarray:
    """Apply convention-neutral quaternion functions in either storage order."""
    a = _components(q, 4, "quaternion")
    operation_codes = {
        "exp": 0,
        "log": 1,
        "sin": 2,
        "cos": 3,
        "sinh": 4,
        "cosh": 5,
    }
    if a.size // 4 >= 128 and operation in operation_codes:
        kernel = _numba_rotation_kernel("algebraic")
        if kernel is not None:
            rows = np.ascontiguousarray(a.reshape((-1, 4)))
            try:
                result, status = kernel(rows, operation_codes[operation], scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("algebraic", None)
                _warn_numba_failure(f"quaternion {operation}", exc)
            else:
                if status == 1:
                    raise ValueError("zero quaternion has no logarithm")
                if status == 0:
                    return _result(result.reshape(a.shape))
    if scalar_last:
        x, y, z, w = np.moveaxis(a, -1, 0)
    else:
        w, x, y, z = np.moveaxis(a, -1, 0)
    vector = np.stack((x, y, z), axis=-1)
    with np.errstate(all="ignore"):
        radius = np.hypot(np.hypot(x, y), z)
        if operation == "exp":
            scalar = np.exp(w) * np.cos(radius)
            ratio = np.divide(np.sin(radius), radius, out=np.ones_like(radius), where=radius != 0)
            out_vector = vector * (np.exp(w) * ratio)[..., None]
        elif operation in ("sin", "cos"):
            hyperbolic = np.cosh(radius)
            ratio = np.divide(np.sinh(radius), radius, out=np.ones_like(radius), where=radius != 0)
            if operation == "sin":
                scalar = np.sin(w) * hyperbolic
                out_vector = vector * (np.cos(w) * ratio)[..., None]
            else:
                scalar = np.cos(w) * hyperbolic
                out_vector = vector * (-np.sin(w) * ratio)[..., None]
        elif operation in ("sinh", "cosh"):
            ratio = np.divide(np.sin(radius), radius, out=np.ones_like(radius), where=radius != 0)
            if operation == "sinh":
                scalar = np.sinh(w) * np.cos(radius)
                out_vector = vector * (np.cosh(w) * ratio)[..., None]
            else:
                scalar = np.cosh(w) * np.cos(radius)
                out_vector = vector * (np.sinh(w) * ratio)[..., None]
        elif operation == "log":
            scale = np.max(np.abs(a), axis=-1)
            if np.any(scale == 0):
                raise ValueError("zero quaternion has no logarithm")
            scaled_w = w / scale
            scaled_vector = vector / scale[..., None]
            scaled_norm = np.sqrt(scaled_w * scaled_w + np.sum(scaled_vector * scaled_vector, axis=-1))
            vector_norm = np.sqrt(np.sum(scaled_vector * scaled_vector, axis=-1))
            scalar = np.log(scale) + np.log(scaled_norm)
            angle = np.arctan2(vector_norm, scaled_w)
            direction = np.divide(scaled_vector, vector_norm[..., None],
                                  out=np.zeros_like(scaled_vector), where=vector_norm[..., None] != 0)
            out_vector = direction * angle[..., None]
            negative_real = (vector_norm == 0) & (scaled_w < 0)
            out_vector[..., 0] = np.where(negative_real, np.pi, out_vector[..., 0])
        else:
            raise ValueError(f"unknown quaternion function: {operation}")
        result = np.concatenate((out_vector, scalar[..., None]), axis=-1) if scalar_last else np.concatenate((scalar[..., None], out_vector), axis=-1)
    return _result(result)


_ANALYTIC_NUMBA_CODES = {
    "tan": 6,
    "tanh": 7,
    "asin": 8,
    "acos": 9,
    "atan": 10,
    "asinh": 11,
    "acosh": 12,
    "atanh": 13,
    "expm1": 14,
    "log1p": 15,
    "log2": 16,
    "log10": 17,
}


def _analytic_samples(
    function: Callable[..., Any],
    q: Any,
    operation: str,
    scalar_last: bool,
    function_scalar_last: bool,
) -> np.ndarray:
    """Use one fused Numba pass for large quaternion analytic batches."""
    values = _components(q, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("algebraic")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, status = kernel(
                    rows, _ANALYTIC_NUMBA_CODES[operation], scalar_last
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("algebraic", None)
                _warn_numba_failure(f"quaternion {operation}", exc)
            else:
                if status == 1:
                    raise ValueError("zero quaternion has no logarithm")
                if status == 0:
                    return _result(result.reshape(values.shape))
    return _map_scalar_ordered(
        function,
        q,
        component_arguments={0: (4, "quaternion")},
        result_shape=(4,),
        array_scalar_last=scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _power_quaternion_samples(
    function: Callable[..., Any],
    quaternions: Any,
    exponents: Any,
    epsilon: int,
    scalar_last: bool,
    function_scalar_last: bool,
) -> np.ndarray:
    """Use a fused principal quaternion power for large broadcast batches."""
    bases, powers = _components(quaternions, 4, "quaternion"), _components(exponents, 4, "exponent")
    try:
        bases, powers = np.broadcast_arrays(bases, powers)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    shape = bases.shape[:-1]
    if prod(shape) >= 128:
        kernel = _numba_rotation_kernel("power_quaternion")
        if kernel is not None:
            base_rows = np.ascontiguousarray(bases.reshape((-1, 4)))
            power_rows = np.ascontiguousarray(powers.reshape((-1, 4)))
            try:
                result, status = kernel(
                    base_rows, power_rows, epsilon, scalar_last
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("power_quaternion", None)
                _warn_numba_failure("quaternion-valued power", exc)
            else:
                if status == 1:
                    raise ValueError(
                        "zero quaternion cannot have a nonzero quaternion power"
                    )
                return _result(result.reshape(shape + (4,)))
    return _map_scalar_ordered(
        function,
        quaternions,
        exponents,
        component_arguments={0: (4, "quaternion"), 1: (4, "exponent")},
        result_shape=(4,),
        array_scalar_last=scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _from_axis_angle_samples(
    axis_angles: Any, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Convert physical axis-angle rows with an optional fused batch loop."""
    values = _components(axis_angles, 4, "axis-angle")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("from_axis_angle")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, status = kernel(rows, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("from_axis_angle", None)
                _warn_numba_failure("from_axis_angle", exc)
            else:
                if status == 1:
                    raise ValueError("axis must be nonzero")
                return _result(result.reshape(values.shape[:-1] + (4,)))

    axis, _, _ = _direction(values[..., :3])
    angle = values[..., 3]
    with np.errstate(all="ignore"):
        half = angle * 0.5
        scalar = np.cos(half)
        scalar = np.where(np.abs(angle) == np.pi, 0.0, scalar)
        vector = epsilon * axis * np.sin(half)[..., None]
        result = np.concatenate((vector, scalar[..., None]), axis=-1) if scalar_last else np.concatenate((scalar[..., None], vector), axis=-1)
    return _result(_canonicalize(result, scalar_last, epsilon))


def _to_axis_angle_samples(
    quaternions: Any, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Extract canonical physical axis-angle rows with a compiled large path."""
    values = _components(quaternions, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("to_axis_angle")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, status = kernel(rows, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("to_axis_angle", None)
                _warn_numba_failure("to_axis_angle", exc)
            else:
                if status == 1:
                    raise ValueError("zero quaternion has no rotation")
                return _result(result.reshape(values.shape))

    if np.any(np.all(values == 0.0, axis=-1)):
        raise ValueError("zero quaternion has no rotation")
    canonical = _canonicalize(values, scalar_last, epsilon)
    if scalar_last:
        raw_vector, w = canonical[..., :3], canonical[..., 3]
    else:
        w, raw_vector = canonical[..., 0], canonical[..., 1:]
    vector = epsilon * raw_vector
    axis, vector_scale, length = _direction(vector, allow_zero=True)
    scale = vector_scale[..., 0]
    magnitude = length[..., 0]
    with np.errstate(all="ignore"):
        base = np.maximum(w, scale)
        angle = 2.0 * np.arctan2((scale / base) * magnitude, w / base)
        tiny = (w > 0.0) & (scale <= w * 1e-8)
        vector_mantissa, vector_exponent = np.frexp(scale)
        w_mantissa, w_exponent = np.frexp(w)
        small_angle = np.ldexp(
            (2.0 * vector_mantissa / w_mantissa) * magnitude,
            vector_exponent - w_exponent,
        )
        angle = np.where(tiny, small_angle, angle)
        axis = np.where((angle == 0.0)[..., None], (1.0, 0.0, 0.0), axis)
        axis = np.where(axis == 0.0, 0.0, axis)
        angle = np.where(angle == 0.0, 0.0, angle)
        result = np.concatenate((axis, angle[..., None]), axis=-1)
    return _result(result)


def _quaternion_product(lhs: np.ndarray, rhs: np.ndarray, epsilon: int,
                        scalar_last: bool) -> np.ndarray:
    """Multiply equal-shaped coefficient arrays in one quaternion convention."""
    with np.errstate(all="ignore"):
        if scalar_last:
            ax, ay, az, aw = np.moveaxis(lhs, -1, 0)
            bx, by, bz, bw = np.moveaxis(rhs, -1, 0)
            scalar = aw*bw - ax*bx - ay*by - az*bz
            x = aw*bx + bw*ax + epsilon*(ay*bz - az*by)
            y = aw*by + bw*ay + epsilon*(az*bx - ax*bz)
            z = aw*bz + bw*az + epsilon*(ax*by - ay*bx)
            result = np.stack((x, y, z, scalar), axis=-1)
        else:
            aw, ax, ay, az = np.moveaxis(lhs, -1, 0)
            bw, bx, by, bz = np.moveaxis(rhs, -1, 0)
            scalar = aw*bw - ax*bx - ay*by - az*bz
            x = aw*bx + bw*ax + epsilon*(ay*bz - az*by)
            y = aw*by + bw*ay + epsilon*(az*bx - ax*bz)
            z = aw*bz + bw*az + epsilon*(ax*by - ay*bx)
            result = np.stack((scalar, x, y, z), axis=-1)
    return _result(result)


def _multiply_hamilton_wxyz(lhs: np.ndarray, rhs: np.ndarray, out: np.ndarray) -> None:
    w, x, y, z = lhs[0], lhs[1], lhs[2], lhs[3]
    a, b, c, d = rhs[0], rhs[1], rhs[2], rhs[3]
    out[0] = w*a - x*b - y*c - z*d
    out[1] = w*b + x*a + y*d - z*c
    out[2] = w*c - x*d + y*a + z*b
    out[3] = w*d + x*c - y*b + z*a


def _multiply_jpl_wxyz(lhs: np.ndarray, rhs: np.ndarray, out: np.ndarray) -> None:
    w, x, y, z = lhs[0], lhs[1], lhs[2], lhs[3]
    a, b, c, d = rhs[0], rhs[1], rhs[2], rhs[3]
    out[0] = w*a - x*b - y*c - z*d
    out[1] = w*b + x*a - y*d + z*c
    out[2] = w*c + x*d + y*a - z*b
    out[3] = w*d - x*c + y*b + z*a


def _multiply_hamilton_xyzw(lhs: np.ndarray, rhs: np.ndarray, out: np.ndarray) -> None:
    x, y, z, w = lhs[0], lhs[1], lhs[2], lhs[3]
    b, c, d, a = rhs[0], rhs[1], rhs[2], rhs[3]
    out[0] = w*b + x*a + y*d - z*c
    out[1] = w*c + y*a + z*b - x*d
    out[2] = w*d + z*a + x*c - y*b
    out[3] = w*a - x*b - y*c - z*d


def _multiply_jpl_xyzw(lhs: np.ndarray, rhs: np.ndarray, out: np.ndarray) -> None:
    x, y, z, w = lhs[0], lhs[1], lhs[2], lhs[3]
    b, c, d, a = rhs[0], rhs[1], rhs[2], rhs[3]
    out[0] = w*b + x*a - y*d + z*c
    out[1] = w*c + y*a - z*b + x*d
    out[2] = w*d + z*a - x*c + y*b
    out[3] = w*a - x*b - y*c - z*d


_NUMBA_MULTIPLY_IMPLEMENTATIONS = {
    (False, 1): _multiply_hamilton_wxyz,
    (False, -1): _multiply_jpl_wxyz,
    (True, 1): _multiply_hamilton_xyzw,
    (True, -1): _multiply_jpl_xyzw,
}
_NUMBA_MULTIPLY: dict[tuple[bool, int], Any] = {}
_NUMBA_MULTIPLY_CHECKED = False


def _numba_multiply_kernel(scalar_last: bool, epsilon: int) -> Any:
    """Lazily load a compiled, convention-specific quaternion ufunc."""
    global _NUMBA_MULTIPLY_CHECKED
    if not _NUMBA_MULTIPLY_CHECKED:
        _NUMBA_MULTIPLY_CHECKED = True
        try:
            from numba import guvectorize
            kernels = {
                key: guvectorize(
                    ["void(float64[:], float64[:], float64[:])"],
                    "(n),(n)->(n)",
                    target="cpu",
                    nopython=True,
                    cache=True,
                    fastmath=True,
                )(implementation)
                for key, implementation in _NUMBA_MULTIPLY_IMPLEMENTATIONS.items()
            }
        except Exception as exc:  # noqa: BLE001
            # Numba is optional; incompatible or partial installations must not
            # make a NumPy operation fail. Leave the cache empty to disable it.
            _NUMBA_MULTIPLY.clear()
            _warn_numba_failure("multiplication", exc)
            return None
        _NUMBA_MULTIPLY.update(kernels)
    return _NUMBA_MULTIPLY.get((scalar_last, epsilon))


def _multiply_numpy(
    lhs: np.ndarray, rhs: np.ndarray, scalar_last: bool, epsilon: int
) -> np.ndarray:
    """Use the original NumPy expression order for scalar and fallback calls."""
    lhs, rhs = np.broadcast_arrays(lhs, rhs)
    with np.errstate(all="ignore"):
        if scalar_last:
            x, y, z, w = np.moveaxis(lhs, -1, 0)
            b, c, d, a = np.moveaxis(rhs, -1, 0)
            result = np.stack((
                w*b + x*a + epsilon*y*d - epsilon*z*c,
                w*c + y*a + epsilon*z*b - epsilon*x*d,
                w*d + z*a + epsilon*x*c - epsilon*y*b,
                w*a - x*b - y*c - z*d,
            ), axis=-1)
        else:
            w, x, y, z = np.moveaxis(lhs, -1, 0)
            a, b, c, d = np.moveaxis(rhs, -1, 0)
            scalar = w*a - x*b - y*c - z*d
            qx = w*b + x*a + epsilon*y*d - epsilon*z*c
            qy = w*c + y*a + epsilon*z*b - epsilon*x*d
            qz = w*d + z*a + epsilon*x*c - epsilon*y*b
            result = np.stack((scalar, qx, qy, qz), axis=-1)
    return _result(result)


def _multiply(lhs: Any, rhs: Any, scalar_last: bool, epsilon: int) -> np.ndarray:
    """Multiply validated-shape inputs, using a compiled loop for large batches."""
    a = _components(lhs, 4, "lhs", validate_finite=False)
    b = _components(rhs, 4, "rhs", validate_finite=False)
    shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1]) + (4,)
    if prod(shape[:-1]) >= 128:
        kernel = _numba_multiply_kernel(scalar_last, epsilon)
        if kernel is not None:
            try:
                with np.errstate(all="ignore"):
                    result = kernel(a, b)
            except Exception as exc:  # noqa: BLE001
                # First-call JIT compilation can fail even after Numba imports.
                _NUMBA_MULTIPLY.clear()
                _warn_numba_failure("multiplication", exc)
            else:
                if not np.all(np.isfinite(result)):
                    _components(a, 4, "lhs")
                    _components(b, 4, "rhs")
                    raise OverflowError(
                        "quaternion arithmetic produced a non-finite result"
                    )
                return result
    a = _components(a, 4, "lhs")
    b = _components(b, 4, "rhs")
    return _multiply_numpy(a, b, scalar_last, epsilon)


def _inverse_coefficients(q: np.ndarray, scalar_last: bool) -> np.ndarray:
    """Invert each nonzero quaternion in a coefficient array."""
    scaled = _scaled(q)
    conjugate = scaled.unit_scale.copy()
    if scalar_last:
        conjugate[..., :3] *= -1.0
    else:
        conjugate[..., 1:] *= -1.0
    with np.errstate(all="ignore"):
        result = (conjugate / scaled.sum_squares) / scaled.scale
    return _result(result)


def _power(q: Any, exponent: Any, scalar_last: bool, epsilon: int = 1) -> np.ndarray:
    """Apply integer or principal real powers with explicit zero-base rules."""
    a = _components(q, 4, "quaternion")
    zeros = np.all(a == 0.0, axis=-1)
    if isinstance(exponent, Integral):
        n = int(exponent)
        if n == 0:
            out = np.zeros_like(a)
            out[..., 3 if scalar_last else 0] = 1.0
            return out
        if a.size // 4 >= 128 and -(1 << 63) < n < (1 << 63):
            kernel = _numba_rotation_kernel("power_integer")
            if kernel is not None:
                rows = np.ascontiguousarray(a.reshape((-1, 4)))
                try:
                    result, status = kernel(rows, n, epsilon, scalar_last)
                except Exception as exc:  # noqa: BLE001
                    _NUMBA_ROTATION_KERNELS.pop("power_integer", None)
                    _warn_numba_failure("integer quaternion power", exc)
                else:
                    if status == 1:
                        raise ValueError("zero quaternion cannot have a negative power")
                    return _result(result.reshape(a.shape))
        if n < 0:
            if np.any(zeros):
                raise ValueError("zero quaternion cannot have a negative power")
            base = _inverse_coefficients(a, scalar_last)
            n = -n
        else:
            base = a.copy()
        result = np.zeros_like(a)
        result[..., 3 if scalar_last else 0] = 1.0
        while n:
            if n & 1:
                result = _quaternion_product(result, base, epsilon, scalar_last)
            n >>= 1
            if n:
                base = _quaternion_product(base, base, epsilon, scalar_last)
        return _result(result)

    value = _factor(exponent, "exponent")
    if value == 0.0:
        out = np.zeros_like(a)
        out[..., 3 if scalar_last else 0] = 1.0
        return out
    if a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("power_real")
        if kernel is not None:
            rows = np.ascontiguousarray(a.reshape((-1, 4)))
            try:
                result, status = kernel(rows, value, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("power_real", None)
                _warn_numba_failure("real quaternion power", exc)
            else:
                if status == 1:
                    raise ValueError("zero quaternion cannot have a negative power")
                return _result(result.reshape(a.shape))
    if value < 0.0 and np.any(zeros):
        raise ValueError("zero quaternion cannot have a negative power")
    safe = a.copy()
    if value > 0.0 and np.any(zeros):
        safe[zeros] = 1.0
    with np.errstate(all="ignore"):
        logarithm = _algebraic(safe, scalar_last, "log")
        powered = _algebraic(logarithm * value, scalar_last, "exp")
        if value > 0.0:
            powered[zeros] = 0.0
    return _result(powered)


def _slerp(q0: Any, q1: Any, t: Any, scalar_last: bool, shortest: bool = True) -> np.ndarray:
    """Broadcast and spherically interpolate unit quaternion coefficients."""
    amount = _factor(t, "t")
    a, b = _components(q0, 4, "q0"), _components(q1, 4, "q1")
    a, b = np.broadcast_arrays(a, b)
    sa, sb = _scaled(a), _scaled(b)
    with np.errstate(all="ignore"):
        a = sa.unit_scale / np.sqrt(sa.sum_squares)
        b = sb.unit_scale / np.sqrt(sb.sum_squares)
        dot = np.sum(a * b, axis=-1, keepdims=True)
        if shortest:
            flip = dot < 0.0
            b = np.where(flip, -b, b)
            dot = np.where(flip, -dot, dot)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        sine = np.sin(theta)
        if np.any((np.abs(sine) <= 1e-15) & (dot < 0.0)):
            raise ValueError("interpolation path is not unique for antipodal endpoints")
        denominator = np.where(np.abs(sine) <= 1e-15, 1.0, sine)
        wa = np.sin((1.0 - amount) * theta) / denominator
        wb = np.sin(amount * theta) / denominator
        spherical = wa * a + wb * b
        linear = (1.0 - amount) * a + amount * b
        blended = np.where(dot > 1.0 - 1e-12, linear, spherical)
        scaled = _scaled(blended)
        result = scaled.unit_scale / np.sqrt(scaled.sum_squares)
    return _result(result)


def _skew(v: Any) -> np.ndarray:
    vector = _components(v, 3, "vector")
    if vector.size // 3 >= 128:
        kernel = _numba_rotation_kernel("skew_matrix")
        if kernel is not None:
            rows = np.ascontiguousarray(vector.reshape((-1, 3)))
            try:
                result = kernel(rows)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("skew_matrix", None)
                _warn_numba_failure("skew matrix", exc)
            else:
                return _result(result.reshape(vector.shape[:-1] + (3, 3)))
    x, y, z = np.moveaxis(vector, -1, 0)
    zero = np.zeros_like(x)
    return np.stack((np.stack((zero, -z, y), axis=-1),
                     np.stack((z, zero, -x), axis=-1),
                     np.stack((-y, x, zero), axis=-1)), axis=-2)


def _so3_jacobian(r: Any, side: int, inverse_matrix: bool = False) -> np.ndarray:
    vector = _components(r, 3, "rotation vector")
    if vector.size // 3 >= 128:
        kernel = _numba_rotation_kernel("so3_jacobian")
        if kernel is not None:
            rows = np.ascontiguousarray(vector.reshape((-1, 3)))
            try:
                result, status = kernel(rows, side, inverse_matrix)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("so3_jacobian", None)
                _warn_numba_failure("SO(3) Jacobian", exc)
            else:
                if status == 1:
                    raise ValueError(
                        "SO(3) Jacobian is singular at a nonzero 2*pi multiple"
                    )
                if status == 2:
                    raise OverflowError(
                        "rotation-vector norm is too large for a float"
                    )
                return _result(result.reshape(vector.shape[:-1] + (3, 3)))
    x, y, z = np.moveaxis(vector, -1, 0)
    with np.errstate(all="ignore"):
        theta = np.hypot(np.hypot(x, y), z)
        if not np.all(np.isfinite(theta)):
            raise OverflowError("rotation-vector norm is too large for a float")
        small = theta < 1e-4
        t2 = theta * theta
        axis = np.divide(vector, theta[..., None], out=np.zeros_like(vector),
                         where=theta[..., None] != 0.0)
        k = _skew(axis)
        k2 = np.matmul(k, k)
        if inverse_matrix:
            half = theta * 0.5
            sin_half = np.sin(half)
            singular = (~small) & (np.abs(sin_half) <= 1e-15)
            if np.any(singular):
                raise ValueError("SO(3) Jacobian is singular at a nonzero 2*pi multiple")
            c = np.divide(1.0 - half * np.cos(half) / np.where(small, 1.0, sin_half),
                          np.where(small, 1.0, np.ones_like(theta)))
            c_small = t2 * (1.0/12.0 + t2/720.0 + t2*t2/30240.0)
            c = np.where(small, c_small, c)
            a = -0.5 * side * theta
            b = c
        else:
            a_large = np.divide(1.0 - np.cos(theta), theta,
                                out=np.zeros_like(theta), where=theta != 0.0)
            b_large = 1.0 - np.divide(np.sin(theta), theta,
                                      out=np.ones_like(theta), where=theta != 0.0)
            a_small = theta * (0.5 - t2/24.0 + t2*t2/720.0)
            b_small = t2 * (1.0/6.0 - t2/120.0 + t2*t2/5040.0)
            a = side * np.where(small, a_small, a_large)
            b = np.where(small, b_small, b_large)
        identity = np.broadcast_to(np.eye(3), theta.shape + (3, 3))
        result = identity + a[..., None, None] * k + b[..., None, None] * k2
    return _result(result)


def _scaled(q: np.ndarray) -> _Scaled:
    scale = np.max(np.abs(q), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero quaternion has no normalization or inverse")
    with np.errstate(all="ignore"):
        unit_scale = q / scale
        sum_squares = np.sum(unit_scale * unit_scale, axis=-1, keepdims=True)
    return _Scaled(scale, unit_scale, sum_squares)


def _normalize_samples(q: Any) -> np.ndarray:
    """Normalize quaternion arrays with a compiled loop for large batches."""
    values = _components(q, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("normalize")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, valid = kernel(rows)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("normalize", None)
                _warn_numba_failure("normalization", exc)
            else:
                if valid:
                    return _result(result.reshape(values.shape))
    scaled = _scaled(values)
    with np.errstate(all="ignore"):
        return _result(scaled.unit_scale / np.sqrt(scaled.sum_squares))


def _inverse_samples(q: Any, scalar_last: bool) -> np.ndarray:
    """Invert quaternion arrays with a compiled loop for large batches."""
    values = _components(q, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("inverse")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, valid = kernel(rows, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("inverse", None)
                _warn_numba_failure("inverse", exc)
            else:
                if valid:
                    return _result(result.reshape(values.shape))
    return _inverse_coefficients(values, scalar_last)


def _norm_samples(q: Any) -> Any:
    """Return stable quaternion norms, using a compiled loop for large batches."""
    values = _components(q, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("norm")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, valid = kernel(rows)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("norm", None)
                _warn_numba_failure("norm", exc)
            else:
                if valid:
                    shaped = result.reshape(values.shape[:-1])
                    return _result(shaped[()] if shaped.ndim == 0 else shaped)
    with np.errstate(all="ignore"):
        scale = np.max(np.abs(values), axis=-1, keepdims=True)
        scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
        result = (
            scale * np.sqrt(np.sum(scaled * scaled, axis=-1, keepdims=True))
        )[..., 0]
    return _result(result[()] if result.ndim == 0 else result)


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


def _rotate_numba_impl(
    q: np.ndarray, v: np.ndarray, scalar_last: bool, epsilon: int
) -> tuple[np.ndarray, bool]:
    """Compiled, range-scaled per-quaternion kernel for validated batches."""
    result = np.empty((q.shape[0], 3), dtype=np.float64)
    valid = True
    for index in range(q.shape[0]):
        if scalar_last:
            x, y, z, w = q[index, 0], q[index, 1], q[index, 2], q[index, 3]
        else:
            w, x, y, z = q[index, 0], q[index, 1], q[index, 2], q[index, 3]
        vx, vy, vz = v[index, 0], v[index, 1], v[index, 2]
        quaternion_scale = max(abs(x), abs(y), abs(z), abs(w))
        if quaternion_scale == 0.0:
            valid = False
            result[index, 0] = np.nan
            result[index, 1] = np.nan
            result[index, 2] = np.nan
            continue
        x /= quaternion_scale
        y /= quaternion_scale
        z /= quaternion_scale
        w /= quaternion_scale
        vector_scale = max(abs(vx), abs(vy), abs(vz))
        if vector_scale == 0.0:
            result[index, 0] = 0.0
            result[index, 1] = 0.0
            result[index, 2] = 0.0
            continue
        vx /= vector_scale
        vy /= vector_scale
        vz /= vector_scale
        first_x = y*vz - z*vy
        first_y = z*vx - x*vz
        first_z = x*vy - y*vx
        second_x = y*first_z - z*first_y
        second_y = z*first_x - x*first_z
        second_z = x*first_y - y*first_x
        factor = 2.0 / (w*w + x*x + y*y + z*z)
        result[index, 0] = (vx + factor * (epsilon*w*first_x + second_x)) * vector_scale
        result[index, 1] = (vy + factor * (epsilon*w*first_y + second_y)) * vector_scale
        result[index, 2] = (vz + factor * (epsilon*w*first_z + second_z)) * vector_scale
    return result, valid


_NUMBA_ROTATE: Any = None
_NUMBA_CHECKED = False


def _numba_rotate_kernel() -> Any:
    """Lazily load Numba so it remains an optional, non-import-time dependency."""
    global _NUMBA_CHECKED, _NUMBA_ROTATE
    if not _NUMBA_CHECKED:
        _NUMBA_CHECKED = True
        try:
            from numba import njit
            _NUMBA_ROTATE = njit(cache=True, nogil=True)(_rotate_numba_impl)
        except Exception as exc:  # noqa: BLE001
            # Keep the pure NumPy path usable with a broken optional runtime.
            _NUMBA_ROTATE = None
            _warn_numba_failure("rotation", exc)
    return _NUMBA_ROTATE


def _rotate(q: Any, v: Any, scalar_last: bool, epsilon: int) -> np.ndarray:
    """Rotate with a scaled cross-product kernel and a compensated fallback."""
    global _NUMBA_ROTATE
    coefficients = _components(q, 4, "quaternion")
    vectors = _components(v, 3, "vector")
    shape = np.broadcast_shapes(coefficients.shape[:-1], vectors.shape[:-1])
    if prod(shape) >= 128:
        kernel = _numba_rotate_kernel()
        if kernel is not None:
            rows = np.ascontiguousarray(
                np.broadcast_to(coefficients, shape + (4,)).reshape((-1, 4))
            )
            vector_rows = np.ascontiguousarray(
                np.broadcast_to(vectors, shape + (3,)).reshape((-1, 3))
            )
            with np.errstate(all="ignore"):
                try:
                    result, valid = kernel(rows, vector_rows, scalar_last, epsilon)
                except Exception as exc:  # noqa: BLE001
                    # Disable a dispatcher that cannot compile or execute here.
                    _NUMBA_ROTATE = None
                    _warn_numba_failure("rotation", exc)
                else:
                    if not valid:
                        raise ValueError(
                            "zero quaternion has no normalization or inverse"
                        )
                    return _result(result.reshape(shape + (3,)))

    coefficient_scale = np.max(np.abs(coefficients), axis=-1, keepdims=True)
    if np.any(coefficient_scale == 0.0):
        raise ValueError("zero quaternion has no normalization or inverse")
    vector_max = np.max(np.abs(vectors), initial=0.0)

    # Most rotation quaternions are unit length. Avoid scaling and dividing
    # their coefficients when both the quaternion and vector ranges are safe.
    raw_coefficients = (
        np.all((coefficient_scale >= 0.5) & (coefficient_scale <= 2.0))
        and vector_max <= np.finfo(np.float64).max / 256.0
    )
    if raw_coefficients:
        unit = coefficients
        with np.errstate(all="ignore"):
            denominator = np.sum(coefficients * coefficients, axis=-1, keepdims=True)
    else:
        scaled_state = _scaled(coefficients)
        unit = scaled_state.unit_scale
        denominator = scaled_state.sum_squares
    shape = np.broadcast_shapes(unit.shape[:-1], vectors.shape[:-1])
    unit = np.broadcast_to(unit, shape + (4,))
    vectors = np.broadcast_to(vectors, shape + (3,))
    denominator = np.broadcast_to(denominator, shape + (1,))

    # The cross-product form is faster for ordinary vectors. Keep the matrix
    # path for near-limit values, where a cross product could overflow before
    # cancellation even though the rotated result is finite.
    limit = np.finfo(np.float64).max / 16.0
    if not raw_coefficients and vector_max > limit:
        if scalar_last:
            x, y, z, w = np.moveaxis(unit, -1, 0)
        else:
            w, x, y, z = np.moveaxis(unit, -1, 0)
        vx, vy, vz = np.moveaxis(vectors, -1, 0)
        sums = denominator[..., 0]
        with np.errstate(all="ignore"):
            ww, xx, yy, zz = w*w, x*x, y*y, z*z
            m00 = np.clip(((ww + xx) - yy - zz) / sums, -1.0, 1.0)
            m01 = np.clip((2 * (x*y - epsilon*w*z)) / sums, -1.0, 1.0)
            m02 = np.clip((2 * (x*z + epsilon*w*y)) / sums, -1.0, 1.0)
            m10 = np.clip((2 * (x*y + epsilon*w*z)) / sums, -1.0, 1.0)
            m11 = np.clip(((ww - xx) + yy - zz) / sums, -1.0, 1.0)
            m12 = np.clip((2 * (y*z - epsilon*w*x)) / sums, -1.0, 1.0)
            m20 = np.clip((2 * (x*z - epsilon*w*y)) / sums, -1.0, 1.0)
            m21 = np.clip((2 * (y*z + epsilon*w*x)) / sums, -1.0, 1.0)
            m22 = np.clip(((ww - xx) - yy + zz) / sums, -1.0, 1.0)
            result = np.stack((
                _sum_three(m00*vx, m01*vy, m02*vz),
                _sum_three(m10*vx, m11*vy, m12*vz),
                _sum_three(m20*vx, m21*vy, m22*vz),
            ), axis=-1)
        return _result(result)

    if scalar_last:
        x, y, z, w = np.moveaxis(unit, -1, 0)
    else:
        w, x, y, z = np.moveaxis(unit, -1, 0)
    vx, vy, vz = np.moveaxis(vectors, -1, 0)
    with np.errstate(all="ignore"):
        first_x = y*vz - z*vy
        first_y = z*vx - x*vz
        first_z = x*vy - y*vx
        second_x = y*first_z - z*first_y
        second_y = z*first_x - x*first_z
        second_z = x*first_y - y*first_x
        factor = 2.0 / denominator[..., 0]
        result = np.stack((
            vx + factor * (epsilon*w*first_x + second_x),
            vy + factor * (epsilon*w*first_y + second_y),
            vz + factor * (epsilon*w*first_z + second_z),
        ), axis=-1)
    # Bounds above keep every intermediate finite in this fast path.
    return result


def _matrix_components(q: Any, epsilon: int, scalar_last: bool, kind: str) -> np.ndarray:
    """Build a batched quaternion matrix in the requested algebra and layout."""
    use_numba = False
    if kind == "rotation":
        a = _components(q, 4, "quaternion", validate_finite=False)
        use_numba = a.size // 4 >= 128
    else:
        a = _components(q, 4, "quaternion")
    if use_numba:
        kernel = _numba_rotation_kernel("to_rotation_matrix")
        if kernel is not None:
            rows = np.ascontiguousarray(a.reshape((-1, 4)))
            with np.errstate(all="ignore"):
                try:
                    result, finite, nonzero = kernel(rows, epsilon, scalar_last)
                except Exception as exc:  # noqa: BLE001
                    _NUMBA_ROTATION_KERNELS.pop("to_rotation_matrix", None)
                    _warn_numba_failure("to_rotation_matrix", exc)
                else:
                    if not finite:
                        raise ValueError(
                            "quaternion must contain finite float64 values"
                        )
                    if not nonzero:
                        raise ValueError("zero quaternion has no normalization or inverse")
                    return result.reshape(a.shape[:-1] + (3, 3))
    if kind == "rotation" and not np.all(np.isfinite(a)):
        raise ValueError("quaternion must contain finite float64 values")
    if scalar_last:
        x, y, z, w = np.moveaxis(a, -1, 0)
    else:
        w, x, y, z = np.moveaxis(a, -1, 0)
    if kind == "rotation":
        scaled = _scaled(a).unit_scale
        if scalar_last:
            x, y, z, w = np.moveaxis(scaled, -1, 0)
        else:
            w, x, y, z = np.moveaxis(scaled, -1, 0)
        d = w*w + x*x + y*y + z*z
        rotation_rows = (
            ((w*w+x*x-y*y-z*z)/d, 2*(x*y-epsilon*w*z)/d, 2*(x*z+epsilon*w*y)/d),
            (2*(x*y+epsilon*w*z)/d, (w*w-x*x+y*y-z*z)/d, 2*(y*z-epsilon*w*x)/d),
            (2*(x*z-epsilon*w*y)/d, 2*(y*z+epsilon*w*x)/d, (w*w-x*x-y*y+z*z)/d),
        )
        out = np.stack([np.stack(row, axis=-1) for row in rotation_rows], axis=-2)
    else:
        cross = epsilon if kind == "left" else -epsilon
        if scalar_last:
            matrix_rows = ((w,-cross*z,cross*y,x),(cross*z,w,-cross*x,y),
                           (-cross*y,cross*x,w,z),(-x,-y,-z,w))
        else:
            matrix_rows = ((w,-x,-y,-z),(x,w,-cross*z,cross*y),
                           (y,cross*z,w,-cross*x),(z,-cross*y,cross*x,w))
        out = np.stack([np.stack(row, axis=-1) for row in matrix_rows], axis=-2)
    return _result(out)


def _matrix_to_quaternion_numba_impl(
    matrices: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert contiguous active rotation matrices in one loop."""
    count = matrices.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    status = np.zeros(count, dtype=np.uint8)
    margin = 128.0 * 2.220446049250313e-16 * (1.0 + tolerance) ** 3
    for index in range(count):
        matrix = matrices[index]
        maximum = 0.0
        finite = True
        for row in range(3):
            for column in range(3):
                component = matrix[row, column]
                if not np.isfinite(component):
                    finite = False
                    continue
                value = abs(component)
                maximum = max(maximum, value)
        if not finite:
            status[index] = 3
            continue
        if maximum > 1.0 + tolerance:
            status[index] = 1
            continue

        invalid = False
        uncertain = False
        for column_a in range(3):
            for column_b in range(3):
                total = (
                    matrix[0, column_a] * matrix[0, column_b]
                    + matrix[1, column_a] * matrix[1, column_b]
                    + matrix[2, column_a] * matrix[2, column_b]
                )
                target = 1.0 if column_a == column_b else 0.0
                error = abs(total - target)
                if error > tolerance + margin:
                    invalid = True
                elif error >= tolerance - margin:
                    uncertain = True
        determinant = (
            matrix[0, 0]
            * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
            - matrix[0, 1]
            * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
            + matrix[0, 2]
            * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
        )
        determinant_error = abs(determinant - 1.0)
        if determinant_error > tolerance + margin:
            invalid = True
        elif determinant_error >= tolerance - margin:
            uncertain = True
        if invalid:
            status[index] = 1
            continue
        if uncertain:
            status[index] = 2

        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] >= matrix[1, 1] and matrix[0, 0] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            )
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            )
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            )
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

        magnitude = max(abs(w), abs(x), abs(y), abs(z))
        w /= magnitude
        x /= magnitude
        y /= magnitude
        z /= magnitude
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        w /= norm
        x /= norm
        y /= norm
        z /= norm
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w = -w
            x = -x
            y = -y
            z = -z
        result[index, 0] = w
        result[index, 1] = x
        result[index, 2] = y
        result[index, 3] = z
    return result, status


def _rotation_matrix_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool, bool]:
    """Build active rotation matrices without NumPy's temporary arrays."""
    count = quaternions.shape[0]
    result = np.empty((count, 3, 3), dtype=np.float64)
    finite = True
    nonzero = True
    for index in range(count):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0], quaternions[index, 1],
                quaternions[index, 2], quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0], quaternions[index, 1],
                quaternions[index, 2], quaternions[index, 3],
            )
        if not (
            np.isfinite(w) and np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
        ):
            finite = False
            result[index, :, :] = np.nan
            continue
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            nonzero = False
            result[index, :, :] = np.nan
            continue
        w /= scale
        x /= scale
        y /= scale
        z /= scale
        denominator = w * w + x * x + y * y + z * z
        result[index, 0, 0] = (w*w + x*x - y*y - z*z) / denominator
        result[index, 0, 1] = 2.0 * (x*y - epsilon*w*z) / denominator
        result[index, 0, 2] = 2.0 * (x*z + epsilon*w*y) / denominator
        result[index, 1, 0] = 2.0 * (x*y + epsilon*w*z) / denominator
        result[index, 1, 1] = (w*w - x*x + y*y - z*z) / denominator
        result[index, 1, 2] = 2.0 * (y*z - epsilon*w*x) / denominator
        result[index, 2, 0] = 2.0 * (x*z - epsilon*w*y) / denominator
        result[index, 2, 1] = 2.0 * (y*z + epsilon*w*x) / denominator
        result[index, 2, 2] = (w*w - x*x - y*y + z*z) / denominator
    return result, finite, nonzero


def _from_rotvec_numba_impl(
    vectors: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Construct canonical quaternions directly in the requested layout."""
    count = vectors.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    overflow = False
    for index in range(count):
        rx, ry, rz = vectors[index, 0], vectors[index, 1], vectors[index, 2]
        scale = max(abs(rx), abs(ry), abs(rz))
        if scale == 0.0:
            if scalar_last:
                result[index, 0] = 0.0
                result[index, 1] = 0.0
                result[index, 2] = 0.0
                result[index, 3] = 1.0
            else:
                result[index, 0] = 1.0
                result[index, 1] = 0.0
                result[index, 2] = 0.0
                result[index, 3] = 0.0
            continue

        if 1e-150 < scale < 1e150:
            # Typical finite rotations can use the direct norm and factor.
            # The bounds keep the sum of squares away from underflow/overflow;
            # extreme magnitudes use the scaled calculation below.
            angle = sqrt(rx*rx + ry*ry + rz*rz)
            if not isfinite(angle):
                overflow = True
                result[index, :] = np.nan
                continue
            half_angle = 0.5 * angle
            w = cos(half_angle)
            if abs(angle) == np.pi:
                w = 0.0
            factor = sin(half_angle) / angle
            x, y, z = rx * factor, ry * factor, rz * factor
        else:
            sx, sy, sz = rx / scale, ry / scale, rz / scale
            direction_norm = sqrt(sx*sx + sy*sy + sz*sz)
            angle = scale * direction_norm
            if not isfinite(angle):
                overflow = True
                result[index, :] = np.nan
                continue
            half_angle = 0.5 * angle
            w = cos(half_angle)
            if abs(angle) == np.pi:
                w = 0.0
            sine = sin(half_angle)
            x = (sx / direction_norm) * sine
            y = (sy / direction_norm) * sine
            z = (sz / direction_norm) * sine
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w, x, y, z = -w, -x, -y, -z
        x *= epsilon
        y *= epsilon
        z *= epsilon
        if scalar_last:
            result[index, 0] = 0.0 if x == 0.0 else x
            result[index, 1] = 0.0 if y == 0.0 else y
            result[index, 2] = 0.0 if z == 0.0 else z
            result[index, 3] = 0.0 if w == 0.0 else w
        else:
            result[index, 0] = 0.0 if w == 0.0 else w
            result[index, 1] = 0.0 if x == 0.0 else x
            result[index, 2] = 0.0 if y == 0.0 else y
            result[index, 3] = 0.0 if z == 0.0 else z
    return result, overflow


def _to_rotvec_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Extract canonical physical rotation vectors with stable tiny limits."""
    count = quaternions.shape[0]
    result = np.empty((count, 3), dtype=np.float64)
    valid = True
    for index in range(count):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0], quaternions[index, 1],
                quaternions[index, 2], quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0], quaternions[index, 1],
                quaternions[index, 2], quaternions[index, 3],
            )
        if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
            valid = False
            result[index, :] = np.nan
            continue

        first = x if x != 0.0 else (y if y != 0.0 else z)
        flip = w < 0.0 or (w == 0.0 and (
            (epsilon > 0 and first < 0.0) or (epsilon < 0 and first > 0.0)
        ))
        if flip:
            w, x, y, z = -w, -x, -y, -z
        if w == 0.0:
            w = 0.0
        if x == 0.0:
            x = 0.0
        if y == 0.0:
            y = 0.0
        if z == 0.0:
            z = 0.0

        vx, vy, vz = epsilon * x, epsilon * y, epsilon * z
        vector_scale = max(abs(vx), abs(vy), abs(vz))
        if 1e-150 < vector_scale < 1e150:
            # For ordinary vector magnitudes, the direct norm avoids the
            # scaling divisions needed by the extreme-value fallback below.
            vector_norm = sqrt(vx*vx + vy*vy + vz*vz)
            angle = 2.0 * atan2(vector_norm, w)
            tiny = w > 0.0 and vector_scale <= w * 1e-8
            if tiny:
                out_x, out_y, out_z = (
                    (2.0 * vx) / w,
                    (2.0 * vy) / w,
                    (2.0 * vz) / w,
                )
            else:
                factor = angle / vector_norm
                out_x, out_y, out_z = vx * factor, vy * factor, vz * factor
        else:
            if vector_scale == 0.0:
                length = 0.0
                axis_x, axis_y, axis_z = 1.0, 0.0, 0.0
            else:
                sx, sy, sz = vx / vector_scale, vy / vector_scale, vz / vector_scale
                length = sqrt(sx*sx + sy*sy + sz*sz)
                axis_x, axis_y, axis_z = sx / length, sy / length, sz / length

            scale = max(w, vector_scale)
            angle = 2.0 * atan2((vector_scale / scale) * length, w / scale)
            tiny = w > 0.0 and vector_scale <= w * 1e-8
            if tiny:
                vector_mantissa, vector_exponent = frexp(vector_scale)
                scalar_mantissa, scalar_exponent = frexp(w)
                angle = ldexp(
                    (2.0 * vector_mantissa / scalar_mantissa) * length,
                    vector_exponent - scalar_exponent,
                )

            if angle == 0.0:
                angle = 0.0
            if tiny:
                out_x, out_y, out_z = (
                    (2.0 * vx) / w,
                    (2.0 * vy) / w,
                    (2.0 * vz) / w,
                )
            else:
                out_x = axis_x * angle
                out_y = axis_y * angle
                out_z = axis_z * angle
        result[index, 0] = 0.0 if out_x == 0.0 else out_x
        result[index, 1] = 0.0 if out_y == 0.0 else out_y
        result[index, 2] = 0.0 if out_z == 0.0 else out_z
    return result, valid


def _integrate_angular_velocity_numba_impl(
    attitudes: np.ndarray,
    rates: np.ndarray,
    times: np.ndarray,
    world_frame: bool,
    midpoint: bool,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, int]:
    """Integrate each sampled trajectory using physical Hamilton components."""
    count, samples = rates.shape[:2]
    result = np.empty((count, 4), dtype=np.float64)
    status = 0
    for row in range(count):
        if scalar_last:
            x, y, z, w = (
                attitudes[row, 0], attitudes[row, 1],
                attitudes[row, 2], attitudes[row, 3],
            )
        else:
            w, x, y, z = (
                attitudes[row, 0], attitudes[row, 1],
                attitudes[row, 2], attitudes[row, 3],
            )
        if epsilon < 0:
            x, y, z = -x, -y, -z

        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            status = 1
            break
        norm = sqrt(
            (w / scale) * (w / scale)
            + (x / scale) * (x / scale)
            + (y / scale) * (y / scale)
            + (z / scale) * (z / scale)
        )
        w, x, y, z = (w / scale) / norm, (x / scale) / norm, (y / scale) / norm, (z / scale) / norm

        for sample in range(samples - 1):
            dt = times[row, sample + 1] - times[row, sample]
            if not isfinite(dt):
                status = 3
                break
            if dt <= 0.0:
                status = 2
                break

            if midpoint:
                rx = (rates[row, sample, 0] + rates[row, sample + 1, 0]) * 0.5
                ry = (rates[row, sample, 1] + rates[row, sample + 1, 1]) * 0.5
                rz = (rates[row, sample, 2] + rates[row, sample + 1, 2]) * 0.5
                if not isfinite(rx) or not isfinite(ry) or not isfinite(rz):
                    status = 4
                    break
            else:
                rx, ry, rz = (
                    rates[row, sample, 0],
                    rates[row, sample, 1],
                    rates[row, sample, 2],
                )
            rx, ry, rz = rx * dt, ry * dt, rz * dt
            if not isfinite(rx) or not isfinite(ry) or not isfinite(rz):
                status = 5
                break

            vector_scale = max(abs(rx), abs(ry), abs(rz))
            if vector_scale == 0.0:
                dw, dx, dy, dz = 1.0, 0.0, 0.0, 0.0
            else:
                sx, sy, sz = rx / vector_scale, ry / vector_scale, rz / vector_scale
                vector_norm = sqrt(sx*sx + sy*sy + sz*sz)
                angle = vector_scale * vector_norm
                if not isfinite(angle):
                    status = 6
                    break
                half_angle = 0.5 * angle
                dw = cos(half_angle)
                if angle == np.pi:
                    dw = 0.0
                sine = sin(half_angle)
                dx = (sx / vector_norm) * sine
                dy = (sy / vector_norm) * sine
                dz = (sz / vector_norm) * sine
                first = dx if dx != 0.0 else (dy if dy != 0.0 else dz)
                if dw < 0.0 or (dw == 0.0 and first < 0.0):
                    dw, dx, dy, dz = -dw, -dx, -dy, -dz

            if world_frame:
                nw = dw*w - dx*x - dy*y - dz*z
                nx = dw*x + dx*w + dy*z - dz*y
                ny = dw*y - dx*z + dy*w + dz*x
                nz = dw*z + dx*y - dy*x + dz*w
            else:
                nw = w*dw - x*dx - y*dy - z*dz
                nx = w*dx + dw*x + y*dz - z*dy
                ny = w*dy + dw*y + z*dx - x*dz
                nz = w*dz + dw*z + x*dy - y*dx

            product_scale = max(abs(nw), abs(nx), abs(ny), abs(nz))
            product_norm = sqrt(
                (nw / product_scale) * (nw / product_scale)
                + (nx / product_scale) * (nx / product_scale)
                + (ny / product_scale) * (ny / product_scale)
                + (nz / product_scale) * (nz / product_scale)
            )
            w, x, y, z = (
                (nw / product_scale) / product_norm,
                (nx / product_scale) / product_norm,
                (ny / product_scale) / product_norm,
                (nz / product_scale) / product_norm,
            )

        if status != 0:
            break
        if epsilon < 0:
            x, y, z = -x, -y, -z
        if scalar_last:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = x, y, z, w
        else:
            result[row, 0], result[row, 1], result[row, 2], result[row, 3] = w, x, y, z
    return result, status


def _nlerp_numba_impl(
    first_values: np.ndarray,
    second_values: np.ndarray,
    amount: float,
    shortest: bool,
) -> tuple[np.ndarray, bool]:
    """Normalize linear quaternion blends without per-sample Python calls."""
    count = first_values.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        a0, a1, a2, a3 = (
            first_values[index, 0], first_values[index, 1],
            first_values[index, 2], first_values[index, 3],
        )
        b0, b1, b2, b3 = (
            second_values[index, 0], second_values[index, 1],
            second_values[index, 2], second_values[index, 3],
        )
        scale_a = max(abs(a0), abs(a1), abs(a2), abs(a3))
        scale_b = max(abs(b0), abs(b1), abs(b2), abs(b3))
        if scale_a == 0.0 or scale_b == 0.0:
            valid = False
            break
        norm_a = sqrt(
            (a0/scale_a)*(a0/scale_a) + (a1/scale_a)*(a1/scale_a)
            + (a2/scale_a)*(a2/scale_a) + (a3/scale_a)*(a3/scale_a)
        )
        norm_b = sqrt(
            (b0/scale_b)*(b0/scale_b) + (b1/scale_b)*(b1/scale_b)
            + (b2/scale_b)*(b2/scale_b) + (b3/scale_b)*(b3/scale_b)
        )
        a0, a1, a2, a3 = (
            (a0/scale_a)/norm_a, (a1/scale_a)/norm_a,
            (a2/scale_a)/norm_a, (a3/scale_a)/norm_a,
        )
        b0, b1, b2, b3 = (
            (b0/scale_b)/norm_b, (b1/scale_b)/norm_b,
            (b2/scale_b)/norm_b, (b3/scale_b)/norm_b,
        )
        dot = a0*b0 + a1*b1 + a2*b2 + a3*b3
        if shortest and dot < 0.0:
            b0, b1, b2, b3 = -b0, -b1, -b2, -b3
        c0 = (1.0 - amount)*a0 + amount*b0
        c1 = (1.0 - amount)*a1 + amount*b1
        c2 = (1.0 - amount)*a2 + amount*b2
        c3 = (1.0 - amount)*a3 + amount*b3
        scale_c = max(abs(c0), abs(c1), abs(c2), abs(c3))
        if scale_c == 0.0 or not isfinite(scale_c):
            valid = False
            break
        norm_c = sqrt(
            (c0/scale_c)*(c0/scale_c) + (c1/scale_c)*(c1/scale_c)
            + (c2/scale_c)*(c2/scale_c) + (c3/scale_c)*(c3/scale_c)
        )
        result[index, 0] = (c0/scale_c)/norm_c
        result[index, 1] = (c1/scale_c)/norm_c
        result[index, 2] = (c2/scale_c)/norm_c
        result[index, 3] = (c3/scale_c)/norm_c
        for component in range(4):
            if not isfinite(result[index, component]):
                valid = False
    return result, valid


def _slerp_numba_impl(
    first_values: np.ndarray,
    second_values: np.ndarray,
    amount: float,
    shortest: bool,
) -> tuple[np.ndarray, bool]:
    """Spherically interpolate rows using a fused, stable compiled loop."""
    count = first_values.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        a0, a1, a2, a3 = (
            first_values[index, 0], first_values[index, 1],
            first_values[index, 2], first_values[index, 3],
        )
        b0, b1, b2, b3 = (
            second_values[index, 0], second_values[index, 1],
            second_values[index, 2], second_values[index, 3],
        )
        scale_a = max(abs(a0), abs(a1), abs(a2), abs(a3))
        scale_b = max(abs(b0), abs(b1), abs(b2), abs(b3))
        if scale_a == 0.0 or scale_b == 0.0:
            valid = False
            break
        norm_a = sqrt(
            (a0/scale_a)*(a0/scale_a) + (a1/scale_a)*(a1/scale_a)
            + (a2/scale_a)*(a2/scale_a) + (a3/scale_a)*(a3/scale_a)
        )
        norm_b = sqrt(
            (b0/scale_b)*(b0/scale_b) + (b1/scale_b)*(b1/scale_b)
            + (b2/scale_b)*(b2/scale_b) + (b3/scale_b)*(b3/scale_b)
        )
        a0, a1, a2, a3 = (
            (a0/scale_a)/norm_a, (a1/scale_a)/norm_a,
            (a2/scale_a)/norm_a, (a3/scale_a)/norm_a,
        )
        b0, b1, b2, b3 = (
            (b0/scale_b)/norm_b, (b1/scale_b)/norm_b,
            (b2/scale_b)/norm_b, (b3/scale_b)/norm_b,
        )
        dot = a0*b0 + a1*b1 + a2*b2 + a3*b3
        if shortest and dot < 0.0:
            b0, b1, b2, b3 = -b0, -b1, -b2, -b3
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        theta = acos(dot)
        sine = sin(theta)
        if abs(sine) <= 1e-15 and dot < 0.0:
            valid = False
            break
        denominator = 1.0 if abs(sine) <= 1e-15 else sine
        weight_a = sin((1.0 - amount) * theta) / denominator
        weight_b = sin(amount * theta) / denominator
        s0 = weight_a*a0 + weight_b*b0
        s1 = weight_a*a1 + weight_b*b1
        s2 = weight_a*a2 + weight_b*b2
        s3 = weight_a*a3 + weight_b*b3
        if dot > 1.0 - 1e-12:
            s0 = (1.0 - amount)*a0 + amount*b0
            s1 = (1.0 - amount)*a1 + amount*b1
            s2 = (1.0 - amount)*a2 + amount*b2
            s3 = (1.0 - amount)*a3 + amount*b3
        scale_s = max(abs(s0), abs(s1), abs(s2), abs(s3))
        if scale_s == 0.0 or not isfinite(scale_s):
            valid = False
            break
        norm_s = sqrt(
            (s0/scale_s)*(s0/scale_s) + (s1/scale_s)*(s1/scale_s)
            + (s2/scale_s)*(s2/scale_s) + (s3/scale_s)*(s3/scale_s)
        )
        result[index, 0] = (s0/scale_s)/norm_s
        result[index, 1] = (s1/scale_s)/norm_s
        result[index, 2] = (s2/scale_s)/norm_s
        result[index, 3] = (s3/scale_s)/norm_s
        for component in range(4):
            if not isfinite(result[index, component]):
                valid = False
    return result, valid


def _from_mrp_numba_impl(
    mrps: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Convert physical MRP vectors to canonical convention-specific q values."""
    count = mrps.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    for index in range(count):
        x, y, z = mrps[index, 0], mrps[index, 1], mrps[index, 2]
        scale = max(abs(x), abs(y), abs(z))
        if scale == 0.0:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        elif scale <= 1.0:
            norm2 = x*x + y*y + z*z
            denominator = 1.0 + norm2
            w = (1.0 - norm2) / denominator
            x, y, z = 2.0*x/denominator, 2.0*y/denominator, 2.0*z/denominator
        else:
            sx, sy, sz = x/scale, y/scale, z/scale
            inverse_scale = 1.0 / scale
            norm2 = sx*sx + sy*sy + sz*sz
            denominator = inverse_scale*inverse_scale + norm2
            w = (inverse_scale*inverse_scale - norm2) / denominator
            x = (2.0 * inverse_scale * sx) / denominator
            y = (2.0 * inverse_scale * sy) / denominator
            z = (2.0 * inverse_scale * sz) / denominator

        scale_q = max(abs(w), abs(x), abs(y), abs(z))
        norm_q = sqrt(
            (w/scale_q)*(w/scale_q) + (x/scale_q)*(x/scale_q)
            + (y/scale_q)*(y/scale_q) + (z/scale_q)*(z/scale_q)
        )
        w, x, y, z = (
            (w/scale_q)/norm_q, (x/scale_q)/norm_q,
            (y/scale_q)/norm_q, (z/scale_q)/norm_q,
        )
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w, x, y, z = -w, -x, -y, -z
        x, y, z = epsilon*x, epsilon*y, epsilon*z
        if scalar_last:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = x, y, z, w
        else:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = w, x, y, z
    return result


def _to_mrp_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Convert arbitrary nonzero quaternion magnitudes to principal MRPs."""
    count = quaternions.shape[0]
    result = np.empty((count, 3), dtype=np.float64)
    valid = True
    for index in range(count):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0], quaternions[index, 1],
                quaternions[index, 2], quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0], quaternions[index, 1],
                quaternions[index, 2], quaternions[index, 3],
            )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        norm = sqrt(
            (w/scale)*(w/scale) + (x/scale)*(x/scale)
            + (y/scale)*(y/scale) + (z/scale)*(z/scale)
        )
        w, x, y, z = (
            (w/scale)/norm, (x/scale)/norm,
            (y/scale)/norm, (z/scale)/norm,
        )
        x, y, z = epsilon*x, epsilon*y, epsilon*z
        first = x if x != 0.0 else (y if y != 0.0 else z)
        if w < 0.0 or (w == 0.0 and first < 0.0):
            w, x, y, z = -w, -x, -y, -z
        denominator = 1.0 + w
        if denominator == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        result[index, 0] = 0.0 if x == 0.0 else x / denominator
        result[index, 1] = 0.0 if y == 0.0 else y / denominator
        result[index, 2] = 0.0 if z == 0.0 else z / denominator
    return result, valid


def _normalize_numba_impl(
    quaternions: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize rows with scaled arithmetic and no intermediate arrays."""
    count = quaternions.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        w, x, y, z = (
            quaternions[index, 0], quaternions[index, 1],
            quaternions[index, 2], quaternions[index, 3],
        )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        if scale == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        norm = sqrt(
            (w/scale)*(w/scale) + (x/scale)*(x/scale)
            + (y/scale)*(y/scale) + (z/scale)*(z/scale)
        )
        result[index, 0] = (w/scale)/norm
        result[index, 1] = (x/scale)/norm
        result[index, 2] = (y/scale)/norm
        result[index, 3] = (z/scale)/norm
    return result, valid


def _inverse_numba_impl(
    quaternions: np.ndarray, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Invert rows while preserving the vectorized API's overflow behavior."""
    count = quaternions.shape[0]
    result = np.empty((count, 4), dtype=np.float64)
    valid = True
    for index in range(count):
        values = (
            quaternions[index, 0], quaternions[index, 1],
            quaternions[index, 2], quaternions[index, 3],
        )
        scale = max(abs(values[0]), abs(values[1]), abs(values[2]), abs(values[3]))
        if scale == 0.0:
            valid = False
            result[index, :] = np.nan
            continue
        w, x, y, z = values
        parts = (w/scale, x/scale, y/scale, z/scale)
        denominator = (
            parts[0]*parts[0] + parts[1]*parts[1]
            + parts[2]*parts[2] + parts[3]*parts[3]
        )
        if scalar_last:
            inverse_values = (
                (-parts[0]/denominator)/scale,
                (-parts[1]/denominator)/scale,
                (-parts[2]/denominator)/scale,
                (parts[3]/denominator)/scale,
            )
        else:
            inverse_values = (
                (parts[0]/denominator)/scale,
                (-parts[1]/denominator)/scale,
                (-parts[2]/denominator)/scale,
                (-parts[3]/denominator)/scale,
            )
        for component in range(4):
            result[index, component] = inverse_values[component]
            if not isfinite(inverse_values[component]):
                valid = False
    return result, valid


def _norm_numba_impl(quaternions: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute scaled quaternion norms in one compiled batch loop."""
    count = quaternions.shape[0]
    result = np.empty(count, dtype=np.float64)
    valid = True
    for index in range(count):
        w, x, y, z = (
            quaternions[index, 0], quaternions[index, 1],
            quaternions[index, 2], quaternions[index, 3],
        )
        scale = max(abs(w), abs(x), abs(y), abs(z))
        length = sqrt(
            (w/scale)*(w/scale) + (x/scale)*(x/scale)
            + (y/scale)*(y/scale) + (z/scale)*(z/scale)
        ) if scale != 0.0 else 0.0
        result[index] = scale * length
        if not isfinite(result[index]):
            valid = False
    return result, valid


def _dot_numba_impl(
    left: np.ndarray,
    right: np.ndarray,
    result: np.ndarray,
    exact: np.ndarray,
) -> None:
    """Compute a fast dot and flag products that need exact scalar arithmetic."""
    tiny = 2.2250738585072014e-308
    p0 = left[0] * right[0]
    p1 = left[1] * right[1]
    p2 = left[2] * right[2]
    p3 = left[3] * right[3]
    result[0] = (p0 + p1) + (p2 + p3)
    exact[0] = (
        not isfinite(result[0])
        or result[0] == 0.0
        or (p0 != 0.0 and abs(p0) < tiny)
        or (p1 != 0.0 and abs(p1) < tiny)
        or (p2 != 0.0 and abs(p2) < tiny)
        or (p3 != 0.0 and abs(p3) < tiny)
    )


def _analytic_numba_impl(
    quaternions: np.ndarray, operation: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Apply scalar-slice quaternion analytic functions in one batch loop."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x = quaternions[index, 0]
            y = quaternions[index, 1]
            z = quaternions[index, 2]
            w = quaternions[index, 3]
        else:
            w = quaternions[index, 0]
            x = quaternions[index, 1]
            y = quaternions[index, 2]
            z = quaternions[index, 3]
        if operation == 0:
            radius_squared = x * x + y * y + z * z
            if 1e-300 < radius_squared < 1e300:
                radius = sqrt(radius_squared)
            elif x == 0.0 and y == 0.0 and z == 0.0:
                radius = 0.0
            else:
                radius = hypot(hypot(x, y), z)
            magnitude = exp(w)
            scalar = magnitude * cos(radius)
            if radius == 0.0:
                vector_x = vector_y = vector_z = 0.0
            else:
                factor = magnitude * sin(radius) / radius
                vector_x, vector_y, vector_z = factor * x, factor * y, factor * z
            if scalar_last:
                result[index, 0] = vector_x
                result[index, 1] = vector_y
                result[index, 2] = vector_z
                result[index, 3] = scalar
            else:
                result[index, 0] = scalar
                result[index, 1] = vector_x
                result[index, 2] = vector_y
                result[index, 3] = vector_z
            continue
        if operation == 1:
            scale = max(abs(w), abs(x), abs(y), abs(z))
            if scale == 0.0:
                return result, 1
            if 1e-150 < scale < 1e150:
                radius = sqrt(x * x + y * y + z * z)
                norm_squared = w * w + x * x + y * y + z * z
                scalar = 0.5 * log(norm_squared)
            else:
                radius = hypot(hypot(x, y), z)
                norm_scale = max(abs(w), radius)
                scalar = log(norm_scale) + log(
                    hypot(w / norm_scale, radius / norm_scale)
                )
            if radius == 0.0:
                vector_x = pi if w < 0.0 else 0.0
                vector_y = vector_z = 0.0
            elif radius < 1e-150:
                angle = atan2(radius, w)
                vector_x = (x / radius) * angle
                vector_y = (y / radius) * angle
                vector_z = (z / radius) * angle
            else:
                factor = atan2(radius, w) / radius
                vector_x, vector_y, vector_z = factor * x, factor * y, factor * z
            if scalar_last:
                result[index, 0] = vector_x
                result[index, 1] = vector_y
                result[index, 2] = vector_z
                result[index, 3] = scalar
            else:
                result[index, 0] = scalar
                result[index, 1] = vector_x
                result[index, 2] = vector_y
                result[index, 3] = vector_z
            continue
        radius = hypot(hypot(x, y), z)
        value = complex(w, radius)
        if operation == 2:
            transformed = cmath.sin(value)
        elif operation == 3:
            transformed = cmath.cos(value)
        elif operation == 4:
            transformed = cmath.sinh(value)
        elif operation == 5:
            transformed = cmath.cosh(value)
        elif operation == 6:
            transformed = cmath.tan(value)
        elif operation == 7:
            transformed = cmath.tanh(value)
        elif operation == 8:
            transformed = cmath.asin(value)
        elif operation == 9:
            transformed = cmath.acos(value)
        elif operation == 10:
            transformed = cmath.atan(value)
        elif operation == 11:
            transformed = cmath.asinh(value)
        elif operation == 12:
            transformed = cmath.acosh(value)
        elif operation == 13:
            transformed = cmath.atanh(value)
        elif operation == 14:
            if abs(value) >= 0.5:
                transformed = cmath.exp(value) - 1.0
            else:
                term = value
                transformed = value
                for power in range(2, 48):
                    term = term * value / power
                    transformed += term
                    if abs(term) <= 2e-17 * max(1.0, abs(transformed)):
                        break
        elif operation == 15:
            if abs(value) >= 0.25:
                shifted = complex(1.0 + w, radius)
                if shifted.real == 0.0 and shifted.imag == 0.0:
                    return result, 1
                transformed = cmath.log(shifted)
                if radius == 0.0 and shifted.real < 0.0:
                    transformed = complex(log(-shifted.real), pi)
            else:
                term = value
                transformed = value
                for power in range(2, 96):
                    term = term * value
                    factor = 1.0 / power if power % 2 == 1 else -1.0 / power
                    transformed += factor * term
                    if abs(term) <= 2e-17 * max(1.0, abs(transformed)):
                        break
        elif operation == 16:
            if w == 0.0 and radius == 0.0:
                return result, 1
            transformed = cmath.log(value) / log(2.0)
        elif operation == 17:
            if w == 0.0 and radius == 0.0:
                return result, 1
            transformed = cmath.log(value) / log(10.0)
        else:
            return result, 2

        if radius == 0.0:
            vector_x = transformed.imag
            vector_y = 0.0
            vector_z = 0.0
        else:
            factor = transformed.imag / radius
            vector_x = factor * x
            vector_y = factor * y
            vector_z = factor * z
        if scalar_last:
            result[index, 0] = vector_x
            result[index, 1] = vector_y
            result[index, 2] = vector_z
            result[index, 3] = transformed.real
        else:
            result[index, 0] = transformed.real
            result[index, 1] = vector_x
            result[index, 2] = vector_y
            result[index, 3] = vector_z
    return result, 0


def _quaternion_product_row_numba(
    lhs: np.ndarray, rhs: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Multiply two coefficient rows in either convention and storage order."""
    result = np.empty(4, dtype=np.float64)
    if scalar_last:
        ax, ay, az, aw = lhs[0], lhs[1], lhs[2], lhs[3]
        bx, by, bz, bw = rhs[0], rhs[1], rhs[2], rhs[3]
    else:
        aw, ax, ay, az = lhs[0], lhs[1], lhs[2], lhs[3]
        bw, bx, by, bz = rhs[0], rhs[1], rhs[2], rhs[3]
    scalar = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + bw*ax + epsilon*(ay*bz - az*by)
    y = aw*by + bw*ay + epsilon*(az*bx - ax*bz)
    z = aw*bz + bw*az + epsilon*(ax*by - ay*bx)
    if scalar_last:
        result[0], result[1], result[2], result[3] = x, y, z, scalar
    else:
        result[0], result[1], result[2], result[3] = scalar, x, y, z
    return result


def _inverse_row_numba(
    value: np.ndarray, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Invert a quaternion row using scaled coefficients."""
    scale = max(abs(value[0]), abs(value[1]), abs(value[2]), abs(value[3]))
    result = np.empty(4, dtype=np.float64)
    if scale == 0.0:
        return result, False
    total = 0.0
    for component in range(4):
        normalized = value[component] / scale
        total += normalized * normalized
    for component in range(4):
        sign = -1.0 if (component < 3 if scalar_last else component > 0) else 1.0
        result[component] = ((value[component] / scale) * sign / total) / scale
    return result, True


def _divide_numba_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    left: bool,
) -> tuple[np.ndarray, int]:
    """Fuse ordinary-range quaternion division over a batch."""
    result = np.empty_like(lhs)
    denominator_values = lhs if left else rhs
    numerator_values = rhs if left else lhs
    cross_sign = -epsilon if left else epsilon
    for index in range(lhs.shape[0]):
        denominator = denominator_values[index]
        scale = max(
            abs(denominator[0]),
            abs(denominator[1]),
            abs(denominator[2]),
            abs(denominator[3]),
        )
        if scale == 0.0:
            return result, 1
        if scale < 1e-150 or scale > 1e150:
            return result, 2
    for index in range(lhs.shape[0]):
        denominator = denominator_values[index]
        numerator = numerator_values[index]
        if scalar_last:
            denominator_x, denominator_y, denominator_z, denominator_w = (
                denominator[0], denominator[1], denominator[2], denominator[3]
            )
            numerator_x, numerator_y, numerator_z, numerator_w = (
                numerator[0], numerator[1], numerator[2], numerator[3]
            )
        else:
            denominator_w, denominator_x, denominator_y, denominator_z = (
                denominator[0], denominator[1], denominator[2], denominator[3]
            )
            numerator_w, numerator_x, numerator_y, numerator_z = (
                numerator[0], numerator[1], numerator[2], numerator[3]
            )
        denominator_norm_squared = (
            (denominator_w * denominator_w + denominator_x * denominator_x)
            + denominator_y * denominator_y
        ) + denominator_z * denominator_z
        inverse_norm_squared = 1.0 / denominator_norm_squared
        scalar = (
            (denominator_w * numerator_w + denominator_x * numerator_x)
            + denominator_y * numerator_y
        ) + denominator_z * numerator_z
        result_w = scalar * inverse_norm_squared
        result_x = (
            denominator_w * numerator_x
            - numerator_w * denominator_x
            + cross_sign * (denominator_y * numerator_z - denominator_z * numerator_y)
        ) * inverse_norm_squared
        result_y = (
            denominator_w * numerator_y
            - numerator_w * denominator_y
            + cross_sign * (denominator_z * numerator_x - denominator_x * numerator_z)
        ) * inverse_norm_squared
        result_z = (
            denominator_w * numerator_z
            - numerator_w * denominator_z
            + cross_sign * (denominator_x * numerator_y - denominator_y * numerator_x)
        ) * inverse_norm_squared
        if scalar_last:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                result_x, result_y, result_z, result_w
            )
        else:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = (
                result_w, result_x, result_y, result_z
            )
    return result, 0


def _log_row_numba(
    value: np.ndarray, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Compute the principal quaternion logarithm for one validated row."""
    if scalar_last:
        x, y, z, w = value[0], value[1], value[2], value[3]
    else:
        w, x, y, z = value[0], value[1], value[2], value[3]
    scale = max(abs(w), abs(x), abs(y), abs(z))
    result = np.zeros(4, dtype=np.float64)
    if scale == 0.0:
        return result, False
    sw, sx, sy, sz = w / scale, x / scale, y / scale, z / scale
    vector_norm = sqrt(sx*sx + sy*sy + sz*sz)
    norm_scaled = sqrt(sw*sw + vector_norm*vector_norm)
    real = log(scale) + log(norm_scaled)
    if vector_norm == 0.0:
        vx = pi if sw < 0.0 else 0.0
        vy, vz = 0.0, 0.0
    else:
        angle = atan2(vector_norm, sw)
        factor = angle / vector_norm
        vx, vy, vz = sx*factor, sy*factor, sz*factor
    if scalar_last:
        result[0], result[1], result[2], result[3] = vx, vy, vz, real
    else:
        result[0], result[1], result[2], result[3] = real, vx, vy, vz
    return result, True


def _exp_row_numba(value: np.ndarray, scalar_last: bool) -> np.ndarray:
    """Compute the quaternion exponential for one coefficient row."""
    if scalar_last:
        x, y, z, w = value[0], value[1], value[2], value[3]
    else:
        w, x, y, z = value[0], value[1], value[2], value[3]
    radius = hypot(hypot(x, y), z)
    magnitude = exp(w)
    scalar = magnitude * cos(radius)
    if radius == 0.0:
        vx, vy, vz = 0.0, 0.0, 0.0
    else:
        factor = magnitude * sin(radius) / radius
        vx, vy, vz = factor*x, factor*y, factor*z
    result = np.empty(4, dtype=np.float64)
    if scalar_last:
        result[0], result[1], result[2], result[3] = vx, vy, vz, scalar
    else:
        result[0], result[1], result[2], result[3] = scalar, vx, vy, vz
    return result


def _power_integer_numba_impl(
    quaternions: np.ndarray, exponent: int, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Apply integer powers by squaring in one compiled batch loop."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        n = exponent
        base = quaternions[index].copy()
        if n < 0:
            base, valid = _inverse_row_numba(base, scalar_last)
            if not valid:
                return result, 1
            n = -n
        accumulated = np.zeros(4, dtype=np.float64)
        accumulated[3 if scalar_last else 0] = 1.0
        while n:
            if n & 1:
                accumulated = _quaternion_product_row_numba(
                    accumulated, base, epsilon, scalar_last
                )
            n >>= 1
            if n:
                base = _quaternion_product_row_numba(base, base, epsilon, scalar_last)
        result[index] = accumulated
    return result, 0


def _power_real_numba_impl(
    quaternions: np.ndarray, exponent: float, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Apply principal real powers through the stable complex slice formula."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x, y, z, w = quaternions[index, 0], quaternions[index, 1], quaternions[index, 2], quaternions[index, 3]
        else:
            w, x, y, z = quaternions[index, 0], quaternions[index, 1], quaternions[index, 2], quaternions[index, 3]
        radius = hypot(hypot(x, y), z)
        if w == 0.0 and radius == 0.0:
            if exponent < 0.0:
                return result, 1
            if exponent == 0.0:
                scalar, vx, vy, vz = 1.0, 0.0, 0.0, 0.0
            else:
                scalar, vx, vy, vz = 0.0, 0.0, 0.0, 0.0
        elif exponent == 0.0:
            scalar, vx, vy, vz = 1.0, 0.0, 0.0, 0.0
        else:
            transformed = cmath.exp(exponent * cmath.log(complex(w, radius)))
            if radius == 0.0:
                vx, vy, vz = transformed.imag, 0.0, 0.0
            else:
                factor = transformed.imag / radius
                vx, vy, vz = factor*x, factor*y, factor*z
            scalar = transformed.real
        if scalar_last:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = vx, vy, vz, scalar
        else:
            result[index, 0], result[index, 1], result[index, 2], result[index, 3] = scalar, vx, vy, vz
    return result, 0


def _power_quaternion_numba_impl(
    quaternions: np.ndarray,
    exponents: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, int]:
    """Apply quaternion-valued powers with one fused batch pass."""
    result = np.empty_like(quaternions)
    for index in range(quaternions.shape[0]):
        zero_exponent = True
        for component in range(4):
            if exponents[index, component] != 0.0:
                zero_exponent = False
                break
        if zero_exponent:
            result[index] = 0.0
            result[index, 3 if scalar_last else 0] = 1.0
            continue
        logarithm, valid = _log_row_numba(quaternions[index], scalar_last)
        if not valid:
            return result, 1
        product = _quaternion_product_row_numba(
            exponents[index], logarithm, epsilon, scalar_last
        )
        result[index] = _exp_row_numba(product, scalar_last)
    return result, 0


def _from_axis_angle_numba_impl(
    axis_angles: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Convert physical axis-angle rows to convention-native quaternions."""
    result = np.empty((axis_angles.shape[0], 4), dtype=np.float64)
    for index in range(axis_angles.shape[0]):
        x, y, z, angle = (
            axis_angles[index, 0],
            axis_angles[index, 1],
            axis_angles[index, 2],
            axis_angles[index, 3],
        )
        scale = max(abs(x), abs(y), abs(z))
        if scale == 0.0:
            return result, 1
        x, y, z = x / scale, y / scale, z / scale
        length = sqrt(x * x + y * y + z * z)
        x, y, z = x / length, y / length, z / length
        half = angle * 0.5
        scalar = 0.0 if abs(angle) == pi else cos(half)
        vector_scale = sin(half) * epsilon
        x, y, z = x * vector_scale, y * vector_scale, z * vector_scale
        first = x if x != 0.0 else y if y != 0.0 else z
        flip = scalar < 0.0 or (scalar == 0.0 and (first < 0.0 if epsilon > 0 else first > 0.0))
        if flip:
            scalar, x, y, z = -scalar, -x, -y, -z
        if scalar_last:
            result[index, 0] = 0.0 if x == 0.0 else x
            result[index, 1] = 0.0 if y == 0.0 else y
            result[index, 2] = 0.0 if z == 0.0 else z
            result[index, 3] = 0.0 if scalar == 0.0 else scalar
        else:
            result[index, 0] = 0.0 if scalar == 0.0 else scalar
            result[index, 1] = 0.0 if x == 0.0 else x
            result[index, 2] = 0.0 if y == 0.0 else y
            result[index, 3] = 0.0 if z == 0.0 else z
    return result, 0


def _to_axis_angle_numba_impl(
    quaternions: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, int]:
    """Extract canonical physical axis-angle rows from quaternion data."""
    result = np.empty((quaternions.shape[0], 4), dtype=np.float64)
    for index in range(quaternions.shape[0]):
        if scalar_last:
            x, y, z, w = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        else:
            w, x, y, z = (
                quaternions[index, 0],
                quaternions[index, 1],
                quaternions[index, 2],
                quaternions[index, 3],
            )
        if w == 0.0 and x == 0.0 and y == 0.0 and z == 0.0:
            return result, 1
        first = x if x != 0.0 else y if y != 0.0 else z
        flip = w < 0.0 or (w == 0.0 and (first < 0.0 if epsilon > 0 else first > 0.0))
        if flip:
            w, x, y, z = -w, -x, -y, -z
        x, y, z = epsilon * x, epsilon * y, epsilon * z
        scale = max(abs(x), abs(y), abs(z))
        if scale == 0.0:
            axis_x, axis_y, axis_z, angle = 1.0, 0.0, 0.0, 0.0
        else:
            sx, sy, sz = x / scale, y / scale, z / scale
            length = sqrt(sx * sx + sy * sy + sz * sz)
            axis_x, axis_y, axis_z = sx / length, sy / length, sz / length
            if w > 0.0 and scale <= w * 1e-8:
                vm, ve = frexp(scale)
                wm, we = frexp(w)
                angle = ldexp((2.0 * (vm / wm)) * length, ve - we)
            else:
                base = max(w, scale)
                angle = 2.0 * atan2((scale / base) * length, w / base)
            if angle == 0.0:
                axis_x, axis_y, axis_z = 1.0, 0.0, 0.0
        result[index, 0] = 0.0 if axis_x == 0.0 else axis_x
        result[index, 1] = 0.0 if axis_y == 0.0 else axis_y
        result[index, 2] = 0.0 if axis_z == 0.0 else axis_z
        result[index, 3] = 0.0 if angle == 0.0 else angle
    return result, 0


def _so3_jacobian_numba_impl(
    vectors: np.ndarray, side: int, inverse_matrix: bool
) -> tuple[np.ndarray, int]:
    """Build SO(3) Jacobians in a fused compiled batch loop."""
    result = np.empty((vectors.shape[0], 3, 3), dtype=np.float64)
    for index in range(vectors.shape[0]):
        x, y, z = vectors[index, 0], vectors[index, 1], vectors[index, 2]
        theta = hypot(hypot(x, y), z)
        if not isfinite(theta):
            return result, 2
        if theta == 0.0:
            axis_x, axis_y, axis_z = 0.0, 0.0, 0.0
        else:
            axis_x, axis_y, axis_z = x / theta, y / theta, z / theta
        theta2 = theta * theta
        small = theta < 1e-4
        if inverse_matrix:
            if small:
                theta4 = theta2 * theta2
                a = -0.5 * side * theta
                b = theta2 * (1.0 / 12.0 + theta2 / 720.0 + theta4 / 30240.0)
            else:
                half = theta * 0.5
                sine_half = sin(half)
                if abs(sine_half) <= 1e-15:
                    return result, 1
                a = -0.5 * side * theta
                b = 1.0 - half * cos(half) / sine_half
        elif small:
            theta4 = theta2 * theta2
            a = side * theta * (0.5 - theta2 / 24.0 + theta4 / 720.0)
            b = theta2 * (1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0)
        else:
            a = side * (1.0 - cos(theta)) / theta
            b = 1.0 - sin(theta) / theta

        k = np.empty((3, 3), dtype=np.float64)
        k[0, 0], k[0, 1], k[0, 2] = 0.0, -axis_z, axis_y
        k[1, 0], k[1, 1], k[1, 2] = axis_z, 0.0, -axis_x
        k[2, 0], k[2, 1], k[2, 2] = -axis_y, axis_x, 0.0
        for row in range(3):
            for column in range(3):
                squared = 0.0
                for inner in range(3):
                    squared += k[row, inner] * k[inner, column]
                result[index, row, column] = (
                    (1.0 if row == column else 0.0)
                    + a * k[row, column]
                    + b * squared
                )
    return result, 0


def _skew_numba_impl(vectors: np.ndarray) -> np.ndarray:
    """Build skew matrices in a fused compiled batch loop."""
    result = np.zeros((vectors.shape[0], 3, 3), dtype=np.float64)
    for index in range(vectors.shape[0]):
        x, y, z = vectors[index, 0], vectors[index, 1], vectors[index, 2]
        result[index, 0, 1] = -z
        result[index, 0, 2] = y
        result[index, 1, 0] = z
        result[index, 1, 2] = -x
        result[index, 2, 0] = -y
        result[index, 2, 1] = x
    return result


def _make_sign_continuous_numba_impl(
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make quaternion sequences sign-continuous and flag uncertain dots."""
    result = np.empty_like(sequences)
    ambiguous = np.zeros(sequences.shape[0], dtype=np.bool_)
    for sequence in range(sequences.shape[0]):
        count = sequences.shape[1]
        if count == 0:
            continue
        for component in range(4):
            result[sequence, 0, component] = sequences[sequence, 0, component]
        for index in range(1, count):
            left_scale = 0.0
            right_scale = 0.0
            for component in range(4):
                left_scale = max(left_scale, abs(result[sequence, index - 1, component]))
                right_scale = max(right_scale, abs(sequences[sequence, index, component]))
            dot = 0.0
            magnitude = 0.0
            if left_scale != 0.0 and right_scale != 0.0:
                for component in range(4):
                    term = (
                        result[sequence, index - 1, component] / left_scale
                    ) * (sequences[sequence, index, component] / right_scale)
                    dot += term
                    magnitude += abs(term)
            if abs(dot) <= 8.881784197001252e-16 * magnitude:
                ambiguous[sequence] = True
            flip = dot < 0.0
            for component in range(4):
                value = sequences[sequence, index, component]
                result[sequence, index, component] = -value if flip else value
    return result, ambiguous


def _jacobi_largest_numba(
    matrix: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Find the dominant eigenvector of a 4x4 symmetric matrix."""
    values = matrix.copy()
    vectors = np.eye(4, dtype=np.float64)
    for _ in range(1600):
        p, q = 0, 1
        off = abs(values[p, q])
        for row in range(4):
            for column in range(row + 1, 4):
                candidate = abs(values[row, column])
                if candidate > off:
                    p, q, off = row, column, candidate
        diagonal_scale = 1.0
        for index in range(4):
            diagonal_scale = max(diagonal_scale, abs(values[index, index]))
        if off <= 2e-16 * diagonal_scale:
            break
        tau = (values[q, q] - values[p, p]) / (2.0 * values[p, q])
        t = copysign(1.0, tau) / (abs(tau) + hypot(1.0, tau))
        if tau == 0.0:
            t = 1.0
        c = 1.0 / hypot(1.0, t)
        s = t * c
        app, aqq = values[p, p], values[q, q]
        values[p, p] = app - t * values[p, q]
        values[q, q] = aqq + t * values[p, q]
        values[p, q] = values[q, p] = 0.0
        for index in range(4):
            if index != p and index != q:
                akp, akq = values[index, p], values[index, q]
                values[index, p] = values[p, index] = c * akp - s * akq
                values[index, q] = values[q, index] = s * akp + c * akq
            vkp, vkq = vectors[index, p], vectors[index, q]
            vectors[index, p] = c * vkp - s * vkq
            vectors[index, q] = s * vkp + c * vkq
    largest_index = 0
    runner_up = -np.inf
    largest = -np.inf
    for index in range(4):
        value = values[index, index]
        if value > largest:
            runner_up = largest
            largest = value
            largest_index = index
        elif value > runner_up:
            runner_up = value
    eigenvector = np.empty(4, dtype=np.float64)
    for index in range(4):
        eigenvector[index] = vectors[index, largest_index]
    return eigenvector, largest, runner_up


def _normalize_quaternion_row_numba(
    value: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize a quaternion row using scaled coefficients."""
    scale = max(abs(value[0]), abs(value[1]), abs(value[2]), abs(value[3]))
    result = np.empty(4, dtype=np.float64)
    if scale == 0.0:
        return result, False
    total = 0.0
    for component in range(4):
        scaled = value[component] / scale
        total += scaled * scaled
    length = sqrt(total)
    for component in range(4):
        result[component] = (value[component] / scale) / length
    return result, True


def _rotation_distance_row_numba(
    q0: np.ndarray,
    q1: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[float, bool]:
    """Return one shortest angle from scaled conjugate-product coefficients."""
    scale0 = max(abs(q0[0]), abs(q0[1]), abs(q0[2]), abs(q0[3]))
    scale1 = max(abs(q1[0]), abs(q1[1]), abs(q1[2]), abs(q1[3]))
    if scale0 == 0.0 or scale1 == 0.0:
        return 0.0, False
    if scalar_last:
        x0, y0, z0, w0 = q0[0] / scale0, q0[1] / scale0, q0[2] / scale0, q0[3] / scale0
        x1, y1, z1, w1 = q1[0] / scale1, q1[1] / scale1, q1[2] / scale1, q1[3] / scale1
    else:
        w0, x0, y0, z0 = q0[0] / scale0, q0[1] / scale0, q0[2] / scale0, q0[3] / scale0
        w1, x1, y1, z1 = q1[0] / scale1, q1[1] / scale1, q1[2] / scale1, q1[3] / scale1
        w1, x1, y1, z1 = q1[0] / scale1, q1[1] / scale1, q1[2] / scale1, q1[3] / scale1
    scalar = abs(((w0*w1 + x0*x1) + y0*y1) + z0*z1)
    vector_x = w0*x1 - w1*x0 - epsilon*(y0*z1 - z0*y1)
    vector_y = w0*y1 - w1*y0 - epsilon*(z0*x1 - x0*z1)
    vector_z = w0*z1 - w1*z0 - epsilon*(x0*y1 - y0*x1)
    vector_norm = hypot(hypot(vector_x, vector_y), vector_z)
    return 2.0 * atan2(vector_norm, scalar), True


def _rotation_distance_numba_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, bool]:
    """Return all distances from a batch of quaternion row pairs."""
    result = np.empty(lhs.shape[0], dtype=np.float64)
    for index in range(lhs.shape[0]):
        result[index], valid = _rotation_distance_row_numba(
            lhs[index], rhs[index], epsilon, scalar_last
        )
        if not valid:
            return result, False
    return result, True


def _rotation_distance_numba_parallel_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute shortest angles in parallel for large quaternion batches."""
    result = np.empty(lhs.shape[0], dtype=np.float64)
    valid = np.ones(lhs.shape[0], dtype=np.bool_)
    for index in _NUMBA_PRANGE(lhs.shape[0]):
        scale0 = max(
            abs(lhs[index, 0]),
            abs(lhs[index, 1]),
            abs(lhs[index, 2]),
            abs(lhs[index, 3]),
        )
        scale1 = max(
            abs(rhs[index, 0]),
            abs(rhs[index, 1]),
            abs(rhs[index, 2]),
            abs(rhs[index, 3]),
        )
        if scale0 == 0.0 or scale1 == 0.0:
            result[index] = 0.0
            valid[index] = False
            continue
        if scalar_last:
            x0, y0, z0, w0 = (
                lhs[index, 0] / scale0,
                lhs[index, 1] / scale0,
                lhs[index, 2] / scale0,
                lhs[index, 3] / scale0,
            )
            x1, y1, z1, w1 = (
                rhs[index, 0] / scale1,
                rhs[index, 1] / scale1,
                rhs[index, 2] / scale1,
                rhs[index, 3] / scale1,
            )
        else:
            w0, x0, y0, z0 = (
                lhs[index, 0] / scale0,
                lhs[index, 1] / scale0,
                lhs[index, 2] / scale0,
                lhs[index, 3] / scale0,
            )
            w1, x1, y1, z1 = (
                rhs[index, 0] / scale1,
                rhs[index, 1] / scale1,
                rhs[index, 2] / scale1,
                rhs[index, 3] / scale1,
            )
        scalar = abs(((w0 * w1 + x0 * x1) + y0 * y1) + z0 * z1)
        vector_x = w0 * x1 - w1 * x0 - epsilon * (y0 * z1 - z0 * y1)
        vector_y = w0 * y1 - w1 * y0 - epsilon * (z0 * x1 - x0 * z1)
        vector_z = w0 * z1 - w1 * z0 - epsilon * (x0 * y1 - y0 * x1)
        vector_norm = hypot(hypot(vector_x, vector_y), vector_z)
        result[index] = 2.0 * atan2(vector_norm, scalar)
    return result, valid


def _rotations_equivalent_numba_impl(
    lhs: np.ndarray,
    rhs: np.ndarray,
    tolerance: float,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, bool]:
    """Compare a batch of shortest distances against a tolerance."""
    result = np.empty(lhs.shape[0], dtype=np.bool_)
    for index in range(lhs.shape[0]):
        distance, valid = _rotation_distance_row_numba(
            lhs[index], rhs[index], epsilon, scalar_last
        )
        if not valid:
            return result, False
        result[index] = distance <= tolerance
    return result, True


def _canonical_quaternion_row_numba(
    value: np.ndarray, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Choose the public deterministic sign for one quaternion row."""
    result = value.copy()
    if scalar_last:
        scalar = result[3]
        vector_start = 0
    else:
        scalar = result[0]
        vector_start = 1
    first = 0.0
    for offset in range(3):
        component = result[vector_start + offset]
        if component != 0.0:
            first = component
            break
    flip = scalar < 0.0 or (
        scalar == 0.0
        and ((first < 0.0) if epsilon > 0 else (first > 0.0))
    )
    for component in range(4):
        if flip:
            result[component] = -result[component]
        if result[component] == 0.0:
            result[component] = 0.0
    return result


def _to_rotvec_row_numba(
    value: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Convert a nonzero quaternion row to a canonical physical rotation vector."""
    canonical = _canonical_quaternion_row_numba(value, epsilon, scalar_last)
    if scalar_last:
        x, y, z, w = canonical[0], canonical[1], canonical[2], canonical[3]
    else:
        w, x, y, z = canonical[0], canonical[1], canonical[2], canonical[3]
    x, y, z = epsilon * x, epsilon * y, epsilon * z
    scale = max(abs(x), abs(y), abs(z))
    result = np.zeros(3, dtype=np.float64)
    if scale == 0.0:
        return result, w != 0.0
    sx, sy, sz = x / scale, y / scale, z / scale
    vector_norm = sqrt(sx*sx + sy*sy + sz*sz)
    if w > 0.0 and scale < w * 1e-8:
        wm, we = frexp(w)
        for index, component in enumerate((x, y, z)):
            if component != 0.0:
                mantissa, exponent = frexp(component)
                result[index] = ldexp(2.0 * (mantissa / wm), exponent - we)
    else:
        base = max(w, scale)
        angle = 2.0 * atan2((scale / base) * vector_norm, w / base)
        result[0] = (sx / vector_norm) * angle
        result[1] = (sy / vector_norm) * angle
        result[2] = (sz / vector_norm) * angle
    for component in range(3):
        if result[component] == 0.0:
            result[component] = 0.0
    return result, True


def _from_rotvec_row_numba(
    value: np.ndarray, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, bool]:
    """Convert one physical rotation vector to a native quaternion row."""
    result = np.zeros(4, dtype=np.float64)
    scale = max(abs(value[0]), abs(value[1]), abs(value[2]))
    if scale == 0.0:
        result[3 if scalar_last else 0] = 1.0
        return result, True
    sx, sy, sz = value[0] / scale, value[1] / scale, value[2] / scale
    scaled_norm = sqrt(sx*sx + sy*sy + sz*sz)
    theta = scale * scaled_norm
    if not isfinite(theta):
        return result, False
    half = theta * 0.5
    vector_scale = sin(half) / scaled_norm * epsilon
    scalar = cos(half)
    if scalar_last:
        result[0] = sx * vector_scale
        result[1] = sy * vector_scale
        result[2] = sz * vector_scale
        result[3] = scalar
    else:
        result[0] = scalar
        result[1] = sx * vector_scale
        result[2] = sy * vector_scale
        result[3] = sz * vector_scale
    return _canonical_quaternion_row_numba(result, epsilon, scalar_last), True


def _rotation_mean_one_numba(
    samples: np.ndarray,
    weights: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    intrinsic: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    """Compute one Markley or intrinsic weighted rotation mean."""
    count = samples.shape[0]
    normalized = np.empty_like(samples)
    scatter = np.zeros((4, 4), dtype=np.float64)
    for sample_index in range(count):
        value, valid = _normalize_quaternion_row_numba(samples[sample_index])
        if not valid:
            return np.zeros(4, dtype=np.float64), 1
        normalized[sample_index] = value
        for row in range(4):
            for column in range(4):
                scatter[row, column] += weights[sample_index] * value[row] * value[column]
    eigenvector, _, _ = _jacobi_largest_numba(scatter)
    mean, valid = _normalize_quaternion_row_numba(eigenvector)
    if not valid:
        return np.zeros(4, dtype=np.float64), 1
    mean = _canonical_quaternion_row_numba(mean, epsilon, scalar_last)
    if not intrinsic:
        return mean, 0
    for _ in range(max_iterations):
        inverse, valid = _inverse_row_numba(mean, scalar_last)
        if not valid:
            return mean, 1
        delta = np.zeros(3, dtype=np.float64)
        for sample_index in range(count):
            relative = _quaternion_product_row_numba(
                inverse, normalized[sample_index], epsilon, scalar_last
            )
            error, valid = _to_rotvec_row_numba(relative, epsilon, scalar_last)
            if not valid:
                return mean, 1
            for component in range(3):
                delta[component] += weights[sample_index] * error[component]
        magnitude = hypot(hypot(delta[0], delta[1]), delta[2])
        if magnitude <= tolerance:
            return mean, 0
        increment, valid = _from_rotvec_row_numba(delta, epsilon, scalar_last)
        if not valid:
            return mean, 3
        mean = _quaternion_product_row_numba(mean, increment, epsilon, scalar_last)
        mean, valid = _normalize_quaternion_row_numba(mean)
        if not valid:
            return mean, 1
        mean = _canonical_quaternion_row_numba(mean, epsilon, scalar_last)
    return mean, 2


def _rotation_mean_numba_impl(
    samples: np.ndarray,
    weights: np.ndarray,
    epsilon: int,
    scalar_last: bool,
    intrinsic: bool,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run weighted rotation means for a batch of sample groups."""
    result = np.empty((samples.shape[0], 4), dtype=np.float64)
    statuses = np.zeros(samples.shape[0], dtype=np.int64)
    for group in range(samples.shape[0]):
        result[group], statuses[group] = _rotation_mean_one_numba(
            samples[group], weights[group], epsilon, scalar_last, intrinsic,
            tolerance, max_iterations,
        )
    return result, statuses


def _rotation_covariance_numba_impl(
    samples: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    has_mean: bool,
    world_frame: bool,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted tangent covariance matrices for batched samples."""
    result = np.zeros((samples.shape[0], 3, 3), dtype=np.float64)
    statuses = np.zeros(samples.shape[0], dtype=np.int64)
    for group in range(samples.shape[0]):
        normalized = np.empty_like(samples[group])
        for sample_index in range(samples.shape[1]):
            normalized[sample_index], valid = _normalize_quaternion_row_numba(
                samples[group, sample_index]
            )
            if not valid:
                statuses[group] = 1
                break
        if statuses[group] != 0:
            continue
        if has_mean:
            center, valid = _normalize_quaternion_row_numba(means[group])
            if not valid:
                statuses[group] = 1
                continue
        else:
            center, status = _rotation_mean_one_numba(
                normalized, weights[group], epsilon, scalar_last, False, 1e-12, 100
            )
            if status != 0:
                statuses[group] = status
                continue
        inverse, valid = _inverse_row_numba(center, scalar_last)
        if not valid:
            statuses[group] = 1
            continue
        for sample_index in range(samples.shape[1]):
            if world_frame:
                relative = _quaternion_product_row_numba(
                    normalized[sample_index], inverse, epsilon, scalar_last
                )
            else:
                relative = _quaternion_product_row_numba(
                    inverse, normalized[sample_index], epsilon, scalar_last
                )
            error, valid = _to_rotvec_row_numba(relative, epsilon, scalar_last)
            if not valid:
                statuses[group] = 1
                break
            for row in range(3):
                for column in range(3):
                    result[group, row, column] += (
                        weights[group, sample_index] * error[row] * error[column]
                    )
    return result, statuses


def _align_vectors_numba_impl(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    epsilon: int,
    scalar_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve batched Wahba problems through Davenport's eigenproblem."""
    result = np.empty((sources.shape[0], 4), dtype=np.float64)
    statuses = np.zeros(sources.shape[0], dtype=np.int64)
    for group in range(sources.shape[0]):
        source_scale = 0.0
        target_scale = 0.0
        for sample in range(sources.shape[1]):
            for component in range(3):
                source_scale = max(source_scale, abs(sources[group, sample, component]))
                target_scale = max(target_scale, abs(targets[group, sample, component]))
        if source_scale == 0.0 or target_scale == 0.0:
            statuses[group] = 1
            continue
        b = np.zeros((3, 3), dtype=np.float64)
        for sample in range(sources.shape[1]):
            for row in range(3):
                source = sources[group, sample, row] / source_scale
                for column in range(3):
                    target = targets[group, sample, column] / target_scale
                    b[row, column] += weights[group, sample] * source * target
        sigma = b[0, 0] + b[1, 1] + b[2, 2]
        k = np.zeros((4, 4), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                k[row, column] = b[row, column] + b[column, row]
                if row == column:
                    k[row, column] -= sigma
        k[0, 3] = k[3, 0] = b[1, 2] - b[2, 1]
        k[1, 3] = k[3, 1] = b[2, 0] - b[0, 2]
        k[2, 3] = k[3, 2] = b[0, 1] - b[1, 0]
        k[3, 3] = sigma
        matrix_scale = 0.0
        for row in range(4):
            for column in range(4):
                matrix_scale = max(matrix_scale, abs(k[row, column]))
        if matrix_scale == 0.0:
            statuses[group] = 2
            continue
        eigenvector, largest, runner_up = _jacobi_largest_numba(k / matrix_scale)
        if largest - runner_up <= 1e-12 * max(1.0, abs(largest)):
            statuses[group] = 2
            continue
        quaternion = np.zeros(4, dtype=np.float64)
        if scalar_last:
            quaternion[0] = epsilon * eigenvector[0]
            quaternion[1] = epsilon * eigenvector[1]
            quaternion[2] = epsilon * eigenvector[2]
            quaternion[3] = eigenvector[3]
        else:
            quaternion[0] = eigenvector[3]
            quaternion[1] = epsilon * eigenvector[0]
            quaternion[2] = epsilon * eigenvector[1]
            quaternion[3] = epsilon * eigenvector[2]
        quaternion, valid = _normalize_quaternion_row_numba(quaternion)
        if not valid:
            statuses[group] = 2
            continue
        result[group] = _canonical_quaternion_row_numba(quaternion, epsilon, scalar_last)
    return result, statuses


def _engine_axis_map_numba_impl(
    values: np.ndarray,
    indices: np.ndarray,
    signs: np.ndarray,
    scalar_first: bool,
) -> np.ndarray:
    """Reorder and sign-flip batched quaternion vectors into engine XYZW."""
    result = np.empty_like(values)
    offset = 1 if scalar_first else 0
    scalar_index = 0 if scalar_first else 3
    for sample in range(values.shape[0]):
        for destination in range(3):
            source = values[sample, offset + indices[destination]]
            result[sample, destination] = (
                source if signs[destination] > 0 else -source
            )
        result[sample, 3] = values[sample, scalar_index]
    return result


_NUMBA_ROTATION_KERNEL_NAMES = frozenset(
    {
        "algebraic",
        "from_axis_angle",
        "from_rotation_matrix",
        "to_rotation_matrix",
        "from_rotvec",
        "to_rotvec",
        "integrate_angular_velocity",
        "nlerp",
        "slerp",
        "from_mrp",
        "to_mrp",
        "left_divide",
        "right_divide",
        "rotation_distance",
        "rotation_distance_parallel",
        "rotations_equivalent",
        "engine_axis_map",
        "normalize",
        "inverse",
        "align_vectors",
        "power_integer",
        "power_quaternion",
        "power_real",
        "norm",
        "make_sign_continuous",
        "rotation_covariance",
        "rotation_mean",
        "dot",
        "skew_matrix",
        "so3_jacobian",
        "to_axis_angle",
    }
)

_NUMBA_ROTATION_CHECKED = False
_NUMBA_ROTATION_KERNELS: dict[str, Any] = {}


def _numba_rotation_kernel(operation: str) -> Any:
    """Lazily load optional rotation kernels and isolate kernel failures."""
    global _NUMBA_PRANGE, _NUMBA_ROTATION_CHECKED
    if not _NUMBA_ROTATION_CHECKED:
        _NUMBA_ROTATION_CHECKED = True
        try:
            import numba as numba_module
            from numba import guvectorize, njit

            _NUMBA_PRANGE = getattr(numba_module, "prange", range)

            for helper_name in (
                "_quaternion_product_row_numba",
                "_inverse_row_numba",
                "_rotation_distance_row_numba",
                "_log_row_numba",
                "_exp_row_numba",
                "_jacobi_largest_numba",
                "_normalize_quaternion_row_numba",
                "_canonical_quaternion_row_numba",
                "_to_rotvec_row_numba",
                "_from_rotvec_row_numba",
                "_rotation_mean_one_numba",
            ):
                helper = globals()[helper_name]
                if not hasattr(helper, "nopython_signatures"):
                    if helper_name == "_divide_row_numba":
                        globals()[helper_name] = njit(
                            cache=True,
                            nogil=True,
                            inline="always",
                            fastmath={"reassoc"},
                        )(helper)
                    elif helper_name == "_rotation_distance_row_numba":
                        globals()[helper_name] = njit(
                            cache=True, nogil=True, inline="always"
                        )(helper)
                    else:
                        globals()[helper_name] = njit(cache=True, nogil=True)(helper)

            _NUMBA_ROTATION_KERNELS.update(
                {
                    "from_rotation_matrix": njit(cache=True, nogil=True)(
                        _matrix_to_quaternion_numba_impl
                    ),
                    "to_rotation_matrix": njit(cache=True, nogil=True)(
                        _rotation_matrix_numba_impl
                    ),
                    "from_rotvec": njit(cache=True, nogil=True)(
                        _from_rotvec_numba_impl
                    ),
                    "to_rotvec": njit(cache=True, nogil=True)(
                        _to_rotvec_numba_impl
                    ),
                    "integrate_angular_velocity": njit(cache=True, nogil=True)(
                        _integrate_angular_velocity_numba_impl
                    ),
                    "nlerp": njit(cache=True, nogil=True)(_nlerp_numba_impl),
                    "slerp": njit(cache=True, nogil=True)(_slerp_numba_impl),
                    "from_mrp": njit(cache=True, nogil=True)(_from_mrp_numba_impl),
                    "to_mrp": njit(cache=True, nogil=True)(_to_mrp_numba_impl),
                    "normalize": njit(
                        cache=True, nogil=True, fastmath=True
                    )(_normalize_numba_impl),
                    "inverse": njit(
                        cache=True, nogil=True, fastmath=True
                    )(_inverse_numba_impl),
                    "left_divide": njit(cache=True, nogil=True, fastmath=True)(
                        _divide_numba_impl
                    ),
                    "right_divide": njit(cache=True, nogil=True, fastmath=True)(
                        _divide_numba_impl
                    ),
                    "rotation_distance": njit(cache=True, nogil=True)(
                        _rotation_distance_numba_impl
                    ),
                    "rotation_distance_parallel": njit(
                        cache=True, nogil=True, fastmath=True, parallel=True
                    )(_rotation_distance_numba_parallel_impl),
                    "rotations_equivalent": njit(cache=True, nogil=True)(
                        _rotations_equivalent_numba_impl
                    ),
                    "engine_axis_map": njit(cache=True, nogil=True)(
                        _engine_axis_map_numba_impl
                    ),
                    "norm": njit(
                        cache=True, nogil=True, fastmath=True
                    )(_norm_numba_impl),
                    "dot": guvectorize(
                        ["void(float64[:], float64[:], float64[:], boolean[:])"],
                        "(n),(n)->(),()",
                        target="cpu",
                        nopython=True,
                        cache=True,
                    )(_dot_numba_impl),
                    "algebraic": njit(cache=True, nogil=True)(_analytic_numba_impl),
                    "from_axis_angle": njit(cache=True, nogil=True)(
                        _from_axis_angle_numba_impl
                    ),
                    "to_axis_angle": njit(cache=True, nogil=True)(
                        _to_axis_angle_numba_impl
                    ),
                    "so3_jacobian": njit(cache=True, nogil=True)(
                        _so3_jacobian_numba_impl
                    ),
                    "skew_matrix": njit(cache=True, nogil=True)(_skew_numba_impl),
                    "power_integer": njit(cache=True, nogil=True)(
                        _power_integer_numba_impl
                    ),
                    "power_real": njit(cache=True, nogil=True)(_power_real_numba_impl),
                    "power_quaternion": njit(cache=True, nogil=True)(
                        _power_quaternion_numba_impl
                    ),
                    "make_sign_continuous": njit(cache=True, nogil=True)(
                        _make_sign_continuous_numba_impl
                    ),
                    "rotation_mean": njit(cache=True, nogil=True)(
                        _rotation_mean_numba_impl
                    ),
                    "rotation_covariance": njit(cache=True, nogil=True)(
                        _rotation_covariance_numba_impl
                    ),
                    "align_vectors": njit(cache=True, nogil=True)(
                        _align_vectors_numba_impl
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            _NUMBA_ROTATION_KERNELS.clear()
            _warn_numba_failure("rotation conversions", exc)
    return _NUMBA_ROTATION_KERNELS.get(operation)


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


def _map_scalar(
    function: Callable[..., Any],
    *values: Any,
    component_arguments: dict[int, tuple[int, str]],
    result_shape: tuple[int, ...],
    dtype: Any = np.float64,
) -> np.ndarray:
    """Apply a scalar reference operation over broadcast NumPy batch axes."""
    indices = sorted(component_arguments)
    arrays = [
        _components(values[index], *component_arguments[index])
        for index in indices
    ]
    # Component widths differ for mixed operations such as boxplus (4 and 3),
    # so broadcast only the leading batch shapes and keep each final
    # component dimension intact.
    try:
        shape = np.broadcast_shapes(*(array.shape[:-1] for array in arrays))
        broadcast = [
            np.broadcast_to(array, shape + (array.shape[-1],)) for array in arrays
        ]
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    out = np.empty(shape + result_shape, dtype=dtype)
    for batch_index in np.ndindex(shape):
        arguments = list(values)
        for index, array in zip(indices, broadcast):
            arguments[index] = tuple(float(value) for value in array[batch_index])
        out[batch_index] = function(*arguments)
    return _result(out)


def _map_scalar_ordered(
    function: Callable[..., Any],
    *values: Any,
    component_arguments: dict[int, tuple[int, str]],
    result_shape: tuple[int, ...],
    array_scalar_last: bool,
    function_scalar_last: bool,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Map a scalar API while adapting quaternion coefficient order.

    Array wrappers can use a different coefficient order from their scalar
    reference API. Only width-four quaternion arguments and quaternion results
    are reordered; three-vectors and matrices retain their own component order.
    """
    if array_scalar_last == function_scalar_last:
        return _map_scalar(function, *values, component_arguments=component_arguments,
                           result_shape=result_shape, dtype=dtype)

    quaternion_arguments = tuple(
        index for index, (width, _) in component_arguments.items() if width == 4
    )

    def reorder_quaternion(value: Any, source_scalar_last: bool) -> tuple[float, ...]:
        coefficients = tuple(float(component) for component in value)
        if source_scalar_last:
            x, y, z, w = coefficients
            return w, x, y, z
        w, x, y, z = coefficients
        return x, y, z, w

    def adapted(*arguments: Any) -> Any:
        ordered = list(arguments)
        for index in quaternion_arguments:
            ordered[index] = reorder_quaternion(ordered[index], array_scalar_last)
        result = function(*ordered)
        if result_shape == (4,):
            return reorder_quaternion(result, function_scalar_last)
        if result_shape == (2, 4):
            return tuple(reorder_quaternion(item, function_scalar_last) for item in result)
        return result

    return _map_scalar(adapted, *values, component_arguments=component_arguments,
                       result_shape=result_shape, dtype=dtype)


def _from_rotvec_samples(
    function: Callable[..., Any],
    vectors: Any,
    epsilon: int,
    array_scalar_last: bool,
    function_scalar_last: bool,
) -> np.ndarray:
    """Use a compiled batch loop for rotation-vector construction."""
    values = _components(vectors, 3, "rotation vector")
    if values.size // 3 >= 128:
        kernel = _numba_rotation_kernel("from_rotvec")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 3)))
            try:
                result, overflow = kernel(rows, epsilon, array_scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("from_rotvec", None)
                _warn_numba_failure("from_rotvec", exc)
            else:
                if overflow:
                    raise OverflowError(
                        "quaternion arithmetic produced a non-finite result"
                    )
                return result.reshape(values.shape[:-1] + (4,))
    return _map_scalar_ordered(
        function,
        vectors,
        component_arguments={0: (3, "rotation vector")},
        result_shape=(4,),
        array_scalar_last=array_scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _to_rotvec_samples(
    function: Callable[..., Any],
    quaternions: Any,
    epsilon: int,
    scalar_last: bool,
    function_scalar_last: bool,
) -> np.ndarray:
    """Use a compiled stable-angle loop for rotation-vector extraction."""
    values = _components(quaternions, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("to_rotvec")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, valid = kernel(rows, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("to_rotvec", None)
                _warn_numba_failure("to_rotvec", exc)
            else:
                if not valid:
                    raise ValueError("zero quaternion has no rotation")
                return _result(result.reshape(values.shape[:-1] + (3,)))
    return _map_scalar_ordered(
        function,
        quaternions,
        component_arguments={0: (4, "quaternion")},
        result_shape=(3,),
        array_scalar_last=scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _nlerp_samples(
    function: Callable[..., Any], q0: Any, q1: Any, t: Any,
    shortest: bool = True,
    *, array_scalar_last: bool = False, function_scalar_last: bool = False,
) -> np.ndarray:
    """Use a fused batch loop for normalized quaternion interpolation."""
    amount = _factor(t, "t")
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")  # noqa: TRY004
    first = _components(q0, 4, "q0")
    second = _components(q1, 4, "q1")
    try:
        first, second = np.broadcast_arrays(first, second)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    shape = first.shape[:-1]
    if prod(shape) >= 128:
        kernel = _numba_rotation_kernel("nlerp")
        if kernel is not None:
            first_rows = np.ascontiguousarray(first.reshape((-1, 4)))
            second_rows = np.ascontiguousarray(second.reshape((-1, 4)))
            try:
                result, valid = kernel(first_rows, second_rows, amount, shortest)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("nlerp", None)
                _warn_numba_failure("nlerp", exc)
            else:
                if valid:
                    return _result(result.reshape(shape + (4,)))
    return _map_scalar_ordered(
        function, q0, q1, t, shortest,
        component_arguments={0: (4, "q0"), 1: (4, "q1")},
        result_shape=(4,),
        array_scalar_last=array_scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _slerp_samples(
    q0: Any, q1: Any, t: Any, scalar_last: bool, shortest: bool = True
) -> np.ndarray:
    """Use a fused compiled loop for large SLERP batches."""
    amount = _factor(t, "t")
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")  # noqa: TRY004
    first, second = _components(q0, 4, "q0"), _components(q1, 4, "q1")
    try:
        first, second = np.broadcast_arrays(first, second)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    shape = first.shape[:-1]
    if prod(shape) >= 128:
        kernel = _numba_rotation_kernel("slerp")
        if kernel is not None:
            first_rows = np.ascontiguousarray(first.reshape((-1, 4)))
            second_rows = np.ascontiguousarray(second.reshape((-1, 4)))
            try:
                result, valid = kernel(first_rows, second_rows, amount, shortest)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("slerp", None)
                _warn_numba_failure("slerp", exc)
            else:
                if valid:
                    return _result(result.reshape(shape + (4,)))
    return _slerp(q0, q1, amount, scalar_last, shortest)


def _from_mrp_samples(
    function: Callable[..., Any], mrps: Any, epsilon: int,
    array_scalar_last: bool, function_scalar_last: bool,
) -> np.ndarray:
    """Use a compiled loop for convention-specific MRP construction."""
    values = _components(mrps, 3, "sigma")
    if values.size // 3 >= 128:
        kernel = _numba_rotation_kernel("from_mrp")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 3)))
            try:
                result = kernel(rows, epsilon, array_scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("from_mrp", None)
                _warn_numba_failure("from_mrp", exc)
            else:
                return _result(result.reshape(values.shape[:-1] + (4,)))
    return _map_scalar_ordered(
        function, mrps,
        component_arguments={0: (3, "sigma")},
        result_shape=(4,),
        array_scalar_last=array_scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _to_mrp_samples(
    function: Callable[..., Any], quaternions: Any, epsilon: int,
    scalar_last: bool, function_scalar_last: bool,
) -> np.ndarray:
    """Use a compiled loop for canonical quaternion-to-MRP conversion."""
    values = _components(quaternions, 4, "quaternion")
    if values.size // 4 >= 128:
        kernel = _numba_rotation_kernel("to_mrp")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, valid = kernel(rows, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("to_mrp", None)
                _warn_numba_failure("to_mrp", exc)
            else:
                if valid:
                    return _result(result.reshape(values.shape[:-1] + (3,)))
    return _map_scalar_ordered(
        function, quaternions,
        component_arguments={0: (4, "quaternion")},
        result_shape=(3,),
        array_scalar_last=scalar_last,
        function_scalar_last=function_scalar_last,
    )


def _squad_samples(
    q0: Any, q1: Any, a: Any, b: Any, t: Any, scalar_last: bool
) -> np.ndarray:
    """Evaluate SQUAD with broadcasted NumPy SLERP calls."""
    amount = _factor(t, "t")
    values = tuple(
        _components(value, 4, name)
        for value, name in ((q0, "q0"), (q1, "q1"), (a, "a"), (b, "b"))
    )
    try:
        first_q, second_q, first_control, second_control = np.broadcast_arrays(*values)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    first = _slerp(first_q, second_q, amount, scalar_last)
    second = _slerp(first_control, second_control, amount, scalar_last)
    return _slerp(first, second, 2.0 * amount * (1.0 - amount), scalar_last)


def _boxplus_samples(
    from_rotvec: Callable[..., Any],
    q: Any, delta: Any, frame: str, epsilon: int,
    scalar_last: bool, function_scalar_last: bool,
) -> np.ndarray:
    """Compose batched attitudes and rotation-vector increments."""
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    attitude = _components(q, 4, "quaternion")
    vectors = _components(delta, 3, "delta")
    try:
        shape = np.broadcast_shapes(attitude.shape[:-1], vectors.shape[:-1])
        attitude = np.broadcast_to(attitude, shape + (4,))
        vectors = np.broadcast_to(vectors, shape + (3,))
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    scaled = _scaled(attitude)
    base = scaled.unit_scale / np.sqrt(scaled.sum_squares)
    increment = _from_rotvec_samples(
        from_rotvec, vectors, epsilon, scalar_last, function_scalar_last
    )
    if frame == "body":
        product = _multiply(base, increment, scalar_last, epsilon)
    else:
        product = _multiply(increment, base, scalar_last, epsilon)
    normalized = _scaled(product)
    return _result(normalized.unit_scale / np.sqrt(normalized.sum_squares))


def _boxminus_samples(
    to_rotvec: Callable[..., Any], q: Any, reference: Any, frame: str,
    epsilon: int, scalar_last: bool, function_scalar_last: bool,
) -> np.ndarray:
    """Find broadcasted local attitude differences in rotation-vector form."""
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    target, base = _components(q, 4, "q"), _components(reference, 4, "reference")
    try:
        target, base = np.broadcast_arrays(target, base)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc

    def normalize(values: np.ndarray) -> np.ndarray:
        scaled = _scaled(values)
        return scaled.unit_scale / np.sqrt(scaled.sum_squares)

    target, base = normalize(target), normalize(base)
    inverse_base = base.copy()
    if scalar_last:
        inverse_base[..., :3] *= -1.0
    else:
        inverse_base[..., 1:] *= -1.0
    if frame == "body":
        relative = _multiply(inverse_base, target, scalar_last, epsilon)
    else:
        relative = _multiply(target, inverse_base, scalar_last, epsilon)
    return _to_rotvec_samples(
        to_rotvec, relative, epsilon, scalar_last, function_scalar_last
    )


def _quaternion_derivative_samples(
    q: Any, omega: Any, frame: str, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Build batched quaternion derivatives through the compiled product path."""
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    attitude = _components(q, 4, "quaternion")
    rates = _components(omega, 3, "omega")
    try:
        shape = np.broadcast_shapes(attitude.shape[:-1], rates.shape[:-1])
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    attitude = np.broadcast_to(attitude, shape + (4,))
    rates = np.broadcast_to(rates, shape + (3,))
    pure = np.zeros(shape + (4,), dtype=np.float64)
    if scalar_last:
        pure[..., :3] = epsilon * rates
    else:
        pure[..., 1:] = epsilon * rates
    left, right = (attitude, pure) if frame == "body" else (pure, attitude)
    with np.errstate(all="ignore"):
        result = 0.5 * _multiply(left, right, scalar_last, epsilon)
    return _result(result)


def _to_homogeneous_matrix_samples(
    q: Any, translation: Any, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Build homogeneous matrices from batched quaternion and translation data."""
    matrices = _matrix_components(q, epsilon, scalar_last, "rotation")
    offsets = _components(translation, 3, "translation")
    try:
        shape = np.broadcast_shapes(matrices.shape[:-2], offsets.shape[:-1])
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    matrices = np.broadcast_to(matrices, shape + (3, 3))
    offsets = np.broadcast_to(offsets, shape + (3,))
    result = np.zeros(shape + (4, 4), dtype=np.float64)
    result[..., :3, :3] = matrices
    result[..., :3, 3] = offsets
    result[..., 3, 3] = 1.0
    return _result(result)


def _sample_weights(weights: Any, batch_shape: tuple[int, ...], count: int) -> np.ndarray | None:
    if weights is None:
        return None
    try:
        array = np.asarray(weights)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a real array") from exc
    if array.dtype.kind not in "biuf" or array.ndim == 0:
        raise ValueError("weights must be a real array with a sample axis")
    with np.errstate(all="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must be finite")
    if array.shape[-1] != count:
        raise ValueError("weights must match the sample axis")
    try:
        return np.broadcast_to(array, batch_shape + (count,))
    except ValueError as exc:
        raise ValueError("weights are not broadcastable to the sample batch") from exc


def _normalized_weight_rows(
    weights: np.ndarray | None, groups: int, count: int
) -> np.ndarray | None:
    """Normalize valid weight rows without overflowing their finite sum."""
    if weights is None:
        return np.full((groups, count), 1.0 / count, dtype=np.float64)
    rows = np.ascontiguousarray(weights.reshape((groups, count))).copy()
    for group in range(groups):
        scale = np.max(rows[group])
        if scale <= 0.0 or np.any(rows[group] < 0.0):
            return None
        rows[group] /= scale
        total = np.sum(rows[group])
        if not np.isfinite(total) or total <= 0.0:
            return None
        rows[group] /= total
    return rows


def _reorder_coefficient_axis(
    values: Any, source_scalar_last: bool, target_scalar_last: bool
) -> np.ndarray:
    """Reorder width-four coefficients without changing their algebra."""
    array = np.asarray(values, dtype=np.float64)
    if source_scalar_last == target_scalar_last:
        return array
    order = (3, 0, 1, 2) if source_scalar_last else (1, 2, 3, 0)
    return array[..., order]


def _rotation_mean(
    function: Callable[..., Any],
    quaternions: Any,
    weights: Any = None,
    method: str = "markley",
    tolerance: Any = 1e-12,
    max_iterations: int = 100,
    *, array_scalar_last: bool = False, function_scalar_last: bool = False,
) -> np.ndarray:
    samples = _components(quaternions, 4, "quaternions")
    if samples.ndim < 2 or samples.shape[-2] == 0:
        raise ValueError("quaternions must have a nonempty sample axis")
    batch_shape, count = samples.shape[:-2], samples.shape[-2]
    weight_array = _sample_weights(weights, batch_shape, count)
    groups = prod(batch_shape) if batch_shape else 1
    normalized_weights = _normalized_weight_rows(weight_array, groups, count)
    if (
        samples.size // 4 >= 128
        and method in ("markley", "intrinsic")
        and normalized_weights is not None
    ):
        valid_options = method == "markley"
        tolerance_value = 1e-12
        if method == "intrinsic":
            try:
                tolerance_value = _factor(tolerance, "tolerance")
            except ValueError:
                valid_options = False
            else:
                valid_options = (
                    tolerance_value > 0.0
                    and isinstance(max_iterations, int)
                    and max_iterations >= 1
                )
        if valid_options:
            kernel = _numba_rotation_kernel("rotation_mean")
            if kernel is not None:
                sample_rows = np.ascontiguousarray(samples.reshape((groups, count, 4)))
                weight_rows = normalized_weights
                try:
                    result, statuses = kernel(
                        sample_rows,
                        weight_rows,
                        -1 if function_scalar_last else 1,
                        array_scalar_last,
                        method == "intrinsic",
                        tolerance_value,
                        max_iterations if method == "intrinsic" else 1,
                    )
                except Exception as exc:  # noqa: BLE001
                    _NUMBA_ROTATION_KERNELS.pop("rotation_mean", None)
                    _warn_numba_failure("rotation mean", exc)
                else:
                    for group in np.flatnonzero(statuses):
                        ordered_samples = _reorder_coefficient_axis(
                            sample_rows[group], array_scalar_last, function_scalar_last
                        )
                        sample_weights = (
                            None if weight_array is None
                            else weight_array.reshape((groups, count))[group]
                        )
                        result[group] = _reorder_coefficient_axis(
                            function(
                                ordered_samples,
                                sample_weights,
                                method,
                                tolerance,
                                max_iterations,
                            ),
                            function_scalar_last,
                            array_scalar_last,
                        )
                    return _result(result.reshape(batch_shape + (4,)))
    out = np.empty(batch_shape + (4,), dtype=np.float64)
    for index in np.ndindex(batch_shape):
        sample_weights = None if weight_array is None else weight_array[index]
        ordered_samples = _reorder_coefficient_axis(
            samples[index], array_scalar_last, function_scalar_last
        )
        value = function(ordered_samples, sample_weights, method,
                         _factor(tolerance, "tolerance"), max_iterations)
        out[index] = _reorder_coefficient_axis(
            value, function_scalar_last, array_scalar_last
        )
    return _result(out)


def _rotation_covariance(
    function: Callable[..., Any],
    quaternions: Any,
    mean: Any = None,
    weights: Any = None,
    frame: str = "body",
    *, array_scalar_last: bool = False, function_scalar_last: bool = False,
) -> np.ndarray:
    samples = _components(quaternions, 4, "quaternions")
    if samples.ndim < 2 or samples.shape[-2] == 0:
        raise ValueError("quaternions must have a nonempty sample axis")
    batch_shape, count = samples.shape[:-2], samples.shape[-2]
    weight_array = _sample_weights(weights, batch_shape, count)
    if mean is None:
        mean_array = None
    else:
        mean_array = _components(mean, 4, "mean")
        try:
            mean_array = np.broadcast_to(mean_array, batch_shape + (4,))
        except ValueError as exc:
            raise ValueError("mean is not broadcastable to the sample batch") from exc
    groups = prod(batch_shape) if batch_shape else 1
    normalized_weights = _normalized_weight_rows(weight_array, groups, count)
    if (
        samples.size // 4 >= 128
        and frame in ("body", "world")
        and normalized_weights is not None
    ):
        kernel = _numba_rotation_kernel("rotation_covariance")
        if kernel is not None:
            sample_rows = np.ascontiguousarray(samples.reshape((groups, count, 4)))
            weight_rows = normalized_weights
            if mean_array is None:
                means = np.zeros((groups, 4), dtype=np.float64)
            else:
                means = np.ascontiguousarray(mean_array.reshape((groups, 4)))
            try:
                result, statuses = kernel(
                    sample_rows,
                    weight_rows,
                    means,
                    mean_array is not None,
                    frame == "world",
                    -1 if function_scalar_last else 1,
                    array_scalar_last,
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("rotation_covariance", None)
                _warn_numba_failure("rotation covariance", exc)
            else:
                weight_rows_for_fallback = (
                    None if weight_array is None
                    else weight_array.reshape((groups, count))
                )
                mean_rows_for_fallback = (
                    None if mean_array is None else mean_array.reshape((groups, 4))
                )
                for group in np.flatnonzero(statuses):
                    ordered_samples = _reorder_coefficient_axis(
                        sample_rows[group], array_scalar_last, function_scalar_last
                    )
                    center = (
                        None if mean_rows_for_fallback is None
                        else _reorder_coefficient_axis(
                            mean_rows_for_fallback[group],
                            array_scalar_last,
                            function_scalar_last,
                        )
                    )
                    result[group] = function(
                        ordered_samples,
                        center,
                        None if weight_rows_for_fallback is None
                        else weight_rows_for_fallback[group],
                        frame,
                    )
                return _result(result.reshape(batch_shape + (3, 3)))
    out = np.empty(batch_shape + (3, 3), dtype=np.float64)
    for index in np.ndindex(batch_shape):
        sample_weights = None if weight_array is None else weight_array[index]
        ordered_samples = _reorder_coefficient_axis(
            samples[index], array_scalar_last, function_scalar_last
        )
        center = None if mean_array is None else _reorder_coefficient_axis(
            mean_array[index], array_scalar_last, function_scalar_last
        )
        out[index] = function(ordered_samples, center, sample_weights, frame)
    return _result(out)


def _align_vectors(
    function: Callable[..., Any], source: Any, target: Any, weights: Any = None,
    *, array_scalar_last: bool = False, function_scalar_last: bool = False,
) -> np.ndarray:
    a, b = _components(source, 3, "source"), _components(target, 3, "target")
    a, b = np.broadcast_arrays(a, b)
    if a.ndim < 2 or a.shape[-2] == 0:
        raise ValueError("vector arrays must have a nonempty correspondence axis")
    batch_shape, count = a.shape[:-2], a.shape[-2]
    weight_array = _sample_weights(weights, batch_shape, count)
    groups = prod(batch_shape) if batch_shape else 1
    normalized_weights = _normalized_weight_rows(weight_array, groups, count)
    if a.size // 3 >= 128 and normalized_weights is not None:
        kernel = _numba_rotation_kernel("align_vectors")
        if kernel is not None:
            source_rows = np.ascontiguousarray(a.reshape((groups, count, 3)))
            target_rows = np.ascontiguousarray(b.reshape((groups, count, 3)))
            weight_rows = normalized_weights
            epsilon = -1 if function_scalar_last else 1
            try:
                result, statuses = kernel(
                    source_rows, target_rows, weight_rows, epsilon, array_scalar_last
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("align_vectors", None)
                _warn_numba_failure("vector alignment", exc)
            else:
                weights_for_fallback = (
                    None if weight_array is None else weight_array.reshape((groups, count))
                )
                for group in np.flatnonzero(statuses):
                    value = function(
                        source_rows[group],
                        target_rows[group],
                        None if weights_for_fallback is None
                        else weights_for_fallback[group],
                    )
                    result[group] = _reorder_coefficient_axis(
                        value, function_scalar_last, array_scalar_last
                    )
                return _result(result.reshape(batch_shape + (4,)))
    out = np.empty(batch_shape + (4,), dtype=np.float64)
    for index in np.ndindex(batch_shape):
        sample_weights = None if weight_array is None else weight_array[index]
        value = function(a[index], b[index], sample_weights)
        out[index] = _reorder_coefficient_axis(
            value, function_scalar_last, array_scalar_last
        )
    return _result(out)


def _make_sign_continuous(function: Callable[..., Any], quaternions: Any) -> np.ndarray:
    samples = _components(quaternions, 4, "quaternions")
    if samples.ndim < 2:
        raise ValueError("quaternions must have a sequence axis")
    count = samples.shape[-2]
    if samples.size // 4 >= 128:
        kernel = _numba_rotation_kernel("make_sign_continuous")
        if kernel is not None:
            sequences = np.ascontiguousarray(samples.reshape((-1, count, 4)))
            try:
                result, ambiguous = kernel(sequences)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("make_sign_continuous", None)
                _warn_numba_failure("sign-continuous sequence", exc)
            else:
                for index in np.flatnonzero(ambiguous):
                    result[index] = function(sequences[index])
                return _result(result.reshape(samples.shape))
    out = np.empty_like(samples)
    for index in np.ndindex(samples.shape[:-2]):
        out[index] = function(samples[index]) if count else samples[index]
    return _result(out)


def _integrate_angular_velocity_samples(
    function: Callable[..., Any], q: Any, omegas: Any, times: Any,
    frame: str = "body", method: str = "midpoint",
    *, epsilon: int = 1, array_scalar_last: bool = False,
    function_scalar_last: bool = False,
) -> np.ndarray:
    attitude, rates = _components(q, 4, "quaternion"), _components(omegas, 3, "omegas")
    try:
        stamps = np.asarray(times)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be a real numeric array") from exc
    if stamps.dtype.kind not in "biuf" or stamps.ndim == 0:
        raise ValueError("times must have a final sample axis")
    with np.errstate(all="ignore"):
        stamps = stamps.astype(np.float64, copy=False)
    if not np.all(np.isfinite(stamps)) or rates.ndim < 2 or stamps.shape[-1] != rates.shape[-2]:
        raise ValueError("times must be finite and match the angular-rate sample axis")
    if rates.shape[-2] == 0:
        raise ValueError("at least one angular-rate sample is required")
    batch_shape = np.broadcast_shapes(attitude.shape[:-1], rates.shape[:-2], stamps.shape[:-1])
    attitude = np.broadcast_to(attitude, batch_shape + (4,))
    rates = np.broadcast_to(rates, batch_shape + rates.shape[-2:])
    stamps = np.broadcast_to(stamps, batch_shape + (stamps.shape[-1],))
    count = prod(batch_shape)
    work = count * max(rates.shape[-2] - 1, 0)
    if count and work >= 128:
        if frame not in ("body", "world"):
            raise ValueError("frame must be 'body' or 'world'")
        if method not in ("left", "midpoint"):
            raise ValueError("method must be 'left' or 'midpoint'")
        kernel = _numba_rotation_kernel("integrate_angular_velocity")
        if kernel is not None:
            attitude_rows = np.ascontiguousarray(attitude.reshape((-1, 4)))
            rate_rows = np.ascontiguousarray(
                rates.reshape((-1, rates.shape[-2], 3))
            )
            time_rows = np.ascontiguousarray(
                stamps.reshape((-1, stamps.shape[-1]))
            )
            try:
                result, status = kernel(
                    attitude_rows,
                    rate_rows,
                    time_rows,
                    frame == "world",
                    method == "midpoint",
                    epsilon,
                    array_scalar_last,
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("integrate_angular_velocity", None)
                _warn_numba_failure("angular-velocity integration", exc)
            else:
                if status == 1:
                    raise ValueError(
                        "zero quaternion has no normalization or inverse"
                    )
                if status == 2:
                    raise ValueError("times must be strictly increasing")
                if status == 3:
                    raise ValueError("dt must be a finite real number")
                if status == 4:
                    raise ValueError("vector must contain finite real numbers")
                if status == 5:
                    raise OverflowError(
                        "quaternion operation produced a non-finite result"
                    )
                if status == 6:
                    raise OverflowError(
                        "rotation-vector norm is too large for a float"
                    )
                return _result(result.reshape(batch_shape + (4,)))
    out = np.empty(batch_shape + (4,), dtype=np.float64)

    def integrate_one(attitude_value: np.ndarray, rate_values: np.ndarray,
                      time_values: np.ndarray) -> np.ndarray:
        coefficients = tuple(float(value) for value in attitude_value)
        if array_scalar_last != function_scalar_last:
            if array_scalar_last:
                x, y, z, w = coefficients
                coefficients = (w, x, y, z)
            else:
                w, x, y, z = coefficients
                coefficients = (x, y, z, w)
        result = function(coefficients, rate_values, time_values, frame, method)
        if array_scalar_last != function_scalar_last:
            coefficients = tuple(float(value) for value in result)
            if function_scalar_last:
                x, y, z, w = coefficients
                result = (w, x, y, z)
            else:
                w, x, y, z = coefficients
                result = (x, y, z, w)
        return np.asarray(result, dtype=np.float64)

    for index in np.ndindex(batch_shape):
        out[index] = integrate_one(attitude[index], rates[index], stamps[index])
    return _result(out)


def _random_quaternion(size: Any, rng: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    generator = np.random.default_rng() if rng is None else rng
    draw = getattr(generator, "random", None)
    if not callable(draw):
        raise ValueError("rng must provide random(size)")  # noqa: TRY004
    if size is None:
        shape: tuple[int, ...] = ()
    elif isinstance(size, Integral) and not isinstance(size, (bool, np.bool_)):
        if int(size) < 0:
            raise ValueError("size dimensions must be nonnegative")
        shape = (int(size),)
    else:
        try:
            dimensions = tuple(size)
        except TypeError as exc:
            raise ValueError("size must be an integer or tuple of integers") from exc
        if any(not isinstance(item, Integral) or isinstance(item, (bool, np.bool_))
               for item in dimensions):
            raise ValueError("size must be an integer or tuple of integers")
        shape = tuple(int(item) for item in dimensions)
        if any(item < 0 for item in shape):
            raise ValueError("size dimensions must be nonnegative")
    try:
        u = np.asarray(draw(shape + (3,)), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("rng.random must accept the requested output shape") from exc
    if u.shape != shape + (3,) or not np.all(np.isfinite(u)) or np.any((u < 0.0) | (u >= 1.0)):
        raise ValueError("rng.random must return values in [0, 1) with the requested shape")
    u1, u2, u3 = np.moveaxis(u, -1, 0)
    x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    if epsilon < 0:
        x, y, z = -x, -y, -z
    result = np.stack((x, y, z, w), axis=-1) if scalar_last else np.stack((w, x, y, z), axis=-1)
    return _result(result)


def _from_homogeneous_matrix(
    matrix: Any,
    epsilon: int,
    atol: Any = 1e-8,
    *,
    array_scalar_last: bool = False,
) -> np.ndarray:
    values = np.asarray(matrix)
    if values.dtype.kind not in "biuf" or values.ndim < 2 or values.shape[-2:] != (4, 4):
        raise ValueError("matrix must have shape (..., 4, 4) and real numeric dtype")
    with np.errstate(all="ignore"):
        values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain finite float64 values")
    tolerance = _factor(atol, "atol")
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("atol must be in [0, 1)")
    expected_bottom = np.array((0.0, 0.0, 0.0, 1.0))
    if np.any(np.abs(values[..., 3, :] - expected_bottom) > tolerance):
        raise ValueError("matrix must have homogeneous bottom row (0, 0, 0, 1)")
    return _from_rotation_matrix(
        values[..., :3, :3], epsilon, array_scalar_last, tolerance
    )


def _dot_quaternions(lhs: Any, rhs: Any) -> np.ndarray:
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    if a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("dot")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                with np.errstate(all="ignore"):
                    result, needs_exact = kernel(left, right)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("dot", None)
                _warn_numba_failure("dot product", exc)
            else:
                if np.any(needs_exact):
                    from .scalar import dot as scalar_dot

                    for index in np.flatnonzero(needs_exact):
                        result[index] = scalar_dot(left[index], right[index])
                return _result(result.reshape(a.shape[:-1]))
    with np.errstate(all="ignore"):
        terms = a * b
        scale = np.max(np.abs(terms), axis=-1)
        normalized = np.divide(
            terms,
            scale[..., None],
            out=np.zeros_like(terms),
            where=scale[..., None] != 0.0,
        )

        total = np.zeros_like(scale)
        correction = np.zeros_like(scale)
        for component_index in range(4):
            value = normalized[..., component_index]
            updated = total + value
            correction += np.where(
                np.abs(total) >= np.abs(value),
                (total - updated) + value,
                (value - updated) + total,
            )
            total = updated
        result = np.asarray((total + correction) * scale)

        needs_exact = (
            np.any(~np.isfinite(terms), axis=-1)
            | np.any(
                (terms != 0.0)
                & (np.abs(terms) < np.finfo(np.float64).tiny),
                axis=-1,
            )
            | np.any(
                (terms != 0.0)
                & (np.abs(normalized) < np.finfo(np.float64).tiny),
                axis=-1,
            )
            | ~np.isfinite(result)
        )

    if np.any(needs_exact):
        from .scalar import dot as scalar_dot

        flat_a = a.reshape((-1, 4))
        flat_b = b.reshape((-1, 4))
        flat_result = result.reshape(-1)
        for index in np.flatnonzero(np.asarray(needs_exact).reshape(-1)):
            flat_result[index] = scalar_dot(flat_a[index], flat_b[index])
    return _result(result)


def _norm_squared(q: Any) -> np.ndarray:
    return _dot_quaternions(q, q)


def _left_divide(lhs: Any, rhs: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    if a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("left_divide")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                result, status = kernel(left, right, epsilon, scalar_last, True)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("left_divide", None)
                _warn_numba_failure("left division", exc)
            else:
                if status == 1:
                    raise ValueError("zero quaternion has no normalization or inverse")
                if status == 0:
                    return _result(result.reshape(a.shape))
    return _quaternion_product(_inverse_coefficients(a, scalar_last), b, epsilon, scalar_last)


def _right_divide(lhs: Any, rhs: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    if a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("right_divide")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                result, status = kernel(left, right, epsilon, scalar_last, False)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("right_divide", None)
                _warn_numba_failure("right division", exc)
            else:
                if status == 1:
                    raise ValueError("zero quaternion has no normalization or inverse")
                if status == 0:
                    return _result(result.reshape(a.shape))
    return _quaternion_product(a, _inverse_coefficients(b, scalar_last), epsilon, scalar_last)


def _commutator_samples(
    lhs: Any, rhs: Any, epsilon: int, scalar_last: bool
) -> np.ndarray:
    """Return a broadcast quaternion commutator without per-row Python calls."""
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    a_vector = a[..., :3] if scalar_last else a[..., 1:]
    b_vector = b[..., :3] if scalar_last else b[..., 1:]
    with np.errstate(all="ignore"):
        commutator_vector = (2.0 * epsilon) * np.cross(a_vector, b_vector)
        result = np.zeros_like(a)
        if scalar_last:
            result[..., :3] = commutator_vector
        else:
            result[..., 1:] = commutator_vector
    return _result(result)


def _scalar_part(q: Any, scalar_last: bool) -> np.ndarray:
    """Return scalar coefficients in the input batch shape."""
    values = _components(q, 4, "quaternion")
    index = 3 if scalar_last else 0
    return _result(values[..., index].copy())


def _vector_part(q: Any, scalar_last: bool) -> np.ndarray:
    """Return vector coefficients in the input batch shape."""
    values = _components(q, 4, "quaternion")
    vector = values[..., :3] if scalar_last else values[..., 1:]
    return _result(vector.copy())


def _align_sign_samples(q: Any, reference: Any) -> np.ndarray:
    """Broadcast and sign-align coefficient arrays using the robust dot path."""
    value, base = _components(q, 4, "q"), _components(reference, 4, "reference")
    try:
        value, base = np.broadcast_arrays(value, base)
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc
    flip = _dot_quaternions(value, base) < 0.0
    with np.errstate(all="ignore"):
        result = np.where(flip[..., None], -value, value)
    return _result(result)


def _swing_twist_samples(
    q: Any, axis: Any, epsilon: int, scalar_last: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose broadcast quaternion batches with array operations."""
    values = _components(q, 4, "quaternion")
    axes = _components(axis, 3, "axis")
    try:
        shape = np.broadcast_shapes(values.shape[:-1], axes.shape[:-1])
        values = np.broadcast_to(values, shape + (4,))
        axes = np.broadcast_to(axes, shape + (3,))
    except ValueError as exc:
        raise ValueError("component batch dimensions are not broadcastable") from exc

    values = _normalize_samples(values)
    direction, _, _ = _direction(axes)
    if scalar_last:
        x, y, z, scalar = np.moveaxis(values, -1, 0)
    else:
        scalar, x, y, z = np.moveaxis(values, -1, 0)
    axis_x, axis_y, axis_z = np.moveaxis(direction, -1, 0)
    with np.errstate(all="ignore"):
        projection = (x * axis_x + y * axis_y) + z * axis_z
        twist_x, twist_y, twist_z = (
            projection * axis_x,
            projection * axis_y,
            projection * axis_z,
        )
        if scalar_last:
            twist = np.stack((twist_x, twist_y, twist_z, scalar), axis=-1)
        else:
            twist = np.stack((scalar, twist_x, twist_y, twist_z), axis=-1)
        degenerate = np.max(np.abs(twist), axis=-1) == 0.0
        identity = np.zeros_like(twist)
        identity[..., 3 if scalar_last else 0] = 1.0
        twist = np.where(degenerate[..., None], identity, twist)

    twist = _normalize_samples(twist)
    inverse_twist = _inverse_samples(twist, scalar_last)
    swing = _normalize_samples(
        _multiply(values, inverse_twist, scalar_last, epsilon)
    )
    return _result(swing), _result(twist)


def _rotation_distance(q0: Any, q1: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    a, b = _components(q0, 4, "q0"), _components(q1, 4, "q1")
    a, b = np.broadcast_arrays(a, b)
    batch_size = a.size // 4
    if batch_size >= 8192:
        kernel = _numba_rotation_kernel("rotation_distance_parallel")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                result, valid = kernel(left, right, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("rotation_distance_parallel", None)
                _warn_numba_failure("parallel rotation distance", exc)
            else:
                if not np.all(valid):
                    raise ValueError("zero quaternion has no normalization or inverse")
                shaped = result.reshape(a.shape[:-1])
                return _result(shaped[()] if shaped.ndim == 0 else shaped)
    if batch_size >= 128:
        kernel = _numba_rotation_kernel("rotation_distance")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                result, valid = kernel(left, right, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("rotation_distance", None)
                _warn_numba_failure("rotation distance", exc)
            else:
                if not valid:
                    raise ValueError("zero quaternion has no normalization or inverse")
                shaped = result.reshape(a.shape[:-1])
                return _result(shaped[()] if shaped.ndim == 0 else shaped)
    scaled_a, scaled_b = _scaled(a), _scaled(b)
    a = scaled_a.unit_scale / np.sqrt(scaled_a.sum_squares)
    b = scaled_b.unit_scale / np.sqrt(scaled_b.sum_squares)
    relative = _quaternion_product(_inverse_coefficients(a, scalar_last), b,
                                   epsilon, scalar_last)
    if scalar_last:
        vector = relative[..., :3]
        scalar = np.abs(relative[..., 3])
    else:
        vector = relative[..., 1:]
        scalar = np.abs(relative[..., 0])
    vector_norm = np.hypot(np.hypot(vector[..., 0], vector[..., 1]), vector[..., 2])
    return _result(2.0 * np.arctan2(vector_norm, scalar))


def _rotations_equivalent(q0: Any, q1: Any, atol: Any, epsilon: int,
                          scalar_last: bool) -> np.ndarray:
    tolerance = _factor(atol, "atol")
    if tolerance < 0.0:
        raise ValueError("atol must be non-negative")
    a, b = _components(q0, 4, "q0"), _components(q1, 4, "q1")
    a, b = np.broadcast_arrays(a, b)
    batch_size = a.size // 4
    if batch_size >= 8192:
        kernel = _numba_rotation_kernel("rotation_distance_parallel")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                distances, valid = kernel(left, right, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("rotation_distance_parallel", None)
                _warn_numba_failure("parallel rotation equivalence", exc)
            else:
                if not np.all(valid):
                    raise ValueError("zero quaternion has no normalization or inverse")
                return (distances <= tolerance).reshape(a.shape[:-1])
    if batch_size >= 128:
        kernel = _numba_rotation_kernel("rotations_equivalent")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                result, valid = kernel(left, right, tolerance, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("rotations_equivalent", None)
                _warn_numba_failure("rotation equivalence", exc)
            else:
                if not valid:
                    raise ValueError("zero quaternion has no normalization or inverse")
                return result.reshape(a.shape[:-1])
    return _rotation_distance(a, b, epsilon, scalar_last) <= tolerance


def _canonicalize(q: Any, scalar_last: bool, epsilon: int = 1) -> np.ndarray:
    a = _components(q, 4, "quaternion").copy()
    if scalar_last:
        scalar, vector = a[..., 3], a[..., :3]
        tie_positive = epsilon > 0
    else:
        scalar, vector = a[..., 0], a[..., 1:]
        tie_positive = epsilon > 0
    first = np.where(vector[..., 0] != 0.0, vector[..., 0],
                     np.where(vector[..., 1] != 0.0, vector[..., 1], vector[..., 2]))
    flip = (scalar < 0.0) | ((scalar == 0.0) & ((first < 0.0) if tie_positive else (first > 0.0)))
    a = np.where(flip[..., None], -a, a)
    a[a == 0.0] = 0.0
    return _result(a)


def _from_rotation_matrix(matrix: Any, epsilon: int, scalar_last: bool,
                          atol: Any = 1e-8) -> np.ndarray:
    a = np.asarray(matrix)
    if a.dtype.kind not in "biuf" or a.ndim < 2 or a.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape (..., 3, 3) and a real numeric dtype")
    with np.errstate(all="ignore"):
        a = a.astype(np.float64, copy=False)
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must contain finite float64 values")
    tolerance = _factor(atol, "atol")
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("atol must be in [0, 1)")
    from .scalar import from_rotation_matrix as _from_matrix_scalar
    flat = np.ascontiguousarray(a.reshape((-1, 3, 3)))
    if flat.shape[0] >= 128:
        kernel = _numba_rotation_kernel("from_rotation_matrix")
        if kernel is not None:
            with np.errstate(all="ignore"):
                try:
                    quaternions, status = kernel(flat, tolerance)
                except Exception as exc:  # noqa: BLE001
                    _NUMBA_ROTATION_KERNELS.pop("from_rotation_matrix", None)
                    _warn_numba_failure("from_rotation_matrix", exc)
                else:
                    if np.any(status == 1):
                        raise ValueError(
                            "matrix must be a proper orthonormal rotation"
                        )
                    # The compiled check uses ordinary sums for speed. Near
                    # the tolerance boundary, defer to the scalar fsum-based
                    # validator so acceptance remains identical.
                    for index in np.flatnonzero(status == 2):
                        _from_matrix_scalar(flat[index], tolerance)
                    result = quaternions.reshape(a.shape[:-2] + (4,))
                    if epsilon < 0:
                        result[..., 1:] *= -1.0
                    if scalar_last:
                        result = result[..., (1, 2, 3, 0)]
                    return _result(_canonicalize(result, scalar_last, epsilon))
    quaternions = [_from_matrix_scalar(item, tolerance) for item in flat]
    result = np.asarray(quaternions, dtype=np.float64).reshape(a.shape[:-2] + (4,))
    if epsilon < 0:
        result[..., 1:] *= -1.0
    if scalar_last:
        result = result[..., (1, 2, 3, 0)]
    return _result(_canonicalize(result, scalar_last, epsilon))


def _from_two_vectors(source: Any, target: Any, epsilon: int,
                      scalar_last: bool) -> np.ndarray:
    a, b = _components(source, 3, "source"), _components(target, 3, "target")
    a, b = np.broadcast_arrays(a, b)
    ua, _, _ = _direction(a)
    ub, _, _ = _direction(b)
    cosine = np.clip(np.sum(ua * ub, axis=-1), -1.0, 1.0)
    cross = np.cross(ua, ub)
    anti = (cosine < 0.0) & np.all(cross == 0.0, axis=-1)
    least_aligned = np.argmin(np.abs(ua), axis=-1)
    basis = np.eye(3, dtype=np.float64)[least_aligned]
    anti_axis = np.cross(ua, basis)
    anti_axis, _, _ = _direction(anti_axis)
    vector = np.where(anti[..., None], anti_axis, cross)
    scalar = np.where(anti, 0.0, 1.0 + cosine)
    hamilton = np.concatenate((scalar[..., None], vector), axis=-1)
    scaled = _scaled(hamilton)
    hamilton = scaled.unit_scale / np.sqrt(scaled.sum_squares)
    hamilton = _canonicalize(hamilton, False, 1)
    if epsilon < 0:
        hamilton[..., 1:] *= -1.0
    if scalar_last:
        hamilton = hamilton[..., (1, 2, 3, 0)]
    return _result(_canonicalize(hamilton, scalar_last, epsilon))
