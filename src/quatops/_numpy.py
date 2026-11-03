"""Internal validation and numerical helpers for explicit NumPy backends."""

from __future__ import annotations

from math import (
    isfinite,
    prod,
)
from numbers import Integral, Real
from typing import Any, Callable

import numpy as np

from ._euler_sequence import euler_metadata as _euler_sequence_metadata
from ._euler_types import EulerResult
from ._numpy_numba_kernels import (
    _NUMBA_ROTATION_KERNEL_NAMES,  # noqa: F401 - preserved internal test/experiment hook
    _NUMBA_ROTATION_KERNELS,
    _numba_rotation_kernel,
)
from ._numpy_support import _components, _result, _scaled, _warn_numba_failure


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
    if values.size // 3 >= 8192:
        kernel = _numba_rotation_kernel("from_euler")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 3)))
            try:
                result = kernel(
                    rows, i, j, k, ell, epsilon, proper,
                    algebra_epsilon, scalar_last,
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("from_euler", None)
                _warn_numba_failure("from Euler angles", exc)
            else:
                result = result.reshape(values.shape[:-1] + (4,))
                return np.moveaxis(
                    result,
                    -1,
                    _axis_index(component_axis, result.ndim, "component_axis"),
                )
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
    if values.size // 4 >= 8192:
        kernel = _numba_rotation_kernel("to_euler")
        if kernel is not None:
            rows = np.ascontiguousarray(values.reshape((-1, 4)))
            try:
                result, locked, valid = kernel(
                    rows, i, j, k, ell, epsilon, proper, intrinsic,
                    tolerance, algebra_epsilon, scalar_last, degrees,
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("to_euler", None)
                _warn_numba_failure("to Euler angles", exc)
            else:
                if not valid:
                    raise ValueError("quaternion must be nonzero")
                angles = result.reshape(values.shape[:-1] + (3,))
                output_axis = _axis_index(angle_axis, angles.ndim, "angle_axis")
                angles = np.moveaxis(angles, -1, output_axis)
                if not with_status:
                    return _result(angles)
                return EulerResult(angles, locked.reshape(values.shape[:-1]))
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


def _conjugate_samples(q: Any, scalar_last: bool) -> np.ndarray:
    """Conjugate components, using Numba for large scalar-first batches."""
    values = _components(q, 4, "quaternion")
    if not scalar_last and values.size // 4 >= 128 and values.flags.c_contiguous:
        kernel = _numba_rotation_kernel("conjugate")
        if kernel is not None:
            rows = values.reshape((-1, 4))
            try:
                result = kernel(rows)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("conjugate", None)
                _warn_numba_failure("quaternion conjugation", exc)
            else:
                return result.reshape(values.shape)
    result = values.copy()
    vector_start = 0 if scalar_last else 1
    result[..., vector_start : vector_start + 3] *= -1.0
    return result


def _scale(q: Any, factor: Any) -> np.ndarray:
    """Scale quaternion components by one finite real scalar."""
    array = _components(q, 4, "quaternion")
    with np.errstate(all="ignore"):
        result = array * _factor(factor)
    return _result(result)


def _component_transform(
    q: Any,
    source_scalar_last: bool,
    target_scalar_last: bool,
    negate_vector: bool = False,
    name: str = "quaternion",
) -> np.ndarray:
    """Copy, reorder, or convert convention for a quaternion component batch."""
    values = _components(q, 4, name)
    is_copy = source_scalar_last == target_scalar_last and not negate_vector
    if is_copy:
        return values.copy()
    accelerate_transform = (
        not source_scalar_last and target_scalar_last
    ) or (negate_vector and not source_scalar_last)
    if accelerate_transform and values.size // 4 >= 128 and values.flags.c_contiguous:
        kernel = _numba_rotation_kernel("component_transform")
        if kernel is not None:
            rows = values.reshape((-1, 4))
            try:
                result = kernel(
                    rows,
                    source_scalar_last,
                    target_scalar_last,
                    negate_vector,
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("component_transform", None)
                _warn_numba_failure("quaternion component transform", exc)
            else:
                return result.reshape(values.shape)

    if source_scalar_last == target_scalar_last:
        result = values.copy()
    elif source_scalar_last:
        result = values[..., (3, 0, 1, 2)]
    else:
        result = values[..., (1, 2, 3, 0)]
    if negate_vector:
        vector_start = 0 if target_scalar_last else 1
        result[..., vector_start : vector_start + 3] *= -1.0
    return result


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
        if scalar_last:
            result = np.concatenate((out_vector, scalar[..., None]), axis=-1)
        else:
            result = np.concatenate((scalar[..., None], out_vector), axis=-1)
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
        if scalar_last:
            result = np.concatenate((vector, scalar[..., None]), axis=-1)
        else:
            result = np.concatenate((scalar[..., None], vector), axis=-1)
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
    elif kind in ("left", "right") and a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("quaternion_matrix")
        if kernel is not None:
            rows = np.ascontiguousarray(a.reshape((-1, 4)))
            try:
                result = kernel(rows, epsilon, scalar_last, kind == "right")
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("quaternion_matrix", None)
                _warn_numba_failure("quaternion multiplication matrices", exc)
            else:
                return result.reshape(a.shape[:-1] + (4, 4))
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
    first = _slerp_samples(first_q, second_q, amount, scalar_last)
    second = _slerp_samples(first_control, second_control, amount, scalar_last)
    return _slerp_samples(
        first, second, 2.0 * amount * (1.0 - amount), scalar_last
    )


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
            size_values: Any = size
            dimensions = tuple(size_values)
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
    if u.size // 3 >= 128 and u.flags.c_contiguous:
        kernel = _numba_rotation_kernel("random_quaternion")
        if kernel is not None:
            rows = u.reshape((-1, 3))
            try:
                result = kernel(rows, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("random_quaternion", None)
                _warn_numba_failure("random quaternion sampling", exc)
            else:
                return _result(result.reshape(shape + (4,)))
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
    if a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("commutator")
        if kernel is not None:
            left = np.ascontiguousarray(a.reshape((-1, 4)))
            right = np.ascontiguousarray(b.reshape((-1, 4)))
            try:
                result = kernel(left, right, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("commutator", None)
                _warn_numba_failure("quaternion commutator", exc)
            else:
                return _result(result.reshape(a.shape))
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
    if not scalar_last and values.size // 4 >= 128 and values.flags.c_contiguous:
        kernel = _numba_rotation_kernel("vector_part")
        if kernel is not None:
            rows = values.reshape((-1, 4))
            try:
                result = kernel(rows, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("vector_part", None)
                _warn_numba_failure("vector component extraction", exc)
            else:
                return _result(result.reshape(values.shape[:-1] + (3,)))
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
    if a.size // 4 >= 128:
        kernel = _numba_rotation_kernel("canonicalize")
        if kernel is not None:
            rows = np.ascontiguousarray(a.reshape((-1, 4)))
            try:
                result = kernel(rows, epsilon, scalar_last)
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("canonicalize", None)
                _warn_numba_failure("quaternion canonicalization", exc)
            else:
                return _result(result.reshape(a.shape))
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
    if a.size // 3 >= 128:
        kernel = _numba_rotation_kernel("from_two_vectors")
        if kernel is not None:
            source_rows = np.ascontiguousarray(a.reshape((-1, 3)))
            target_rows = np.ascontiguousarray(b.reshape((-1, 3)))
            try:
                result, valid = kernel(
                    source_rows, target_rows, epsilon, scalar_last
                )
            except Exception as exc:  # noqa: BLE001
                _NUMBA_ROTATION_KERNELS.pop("from_two_vectors", None)
                _warn_numba_failure("two-vector rotation", exc)
            else:
                if not valid:
                    raise ValueError("axis must be nonzero")
                return _result(result.reshape(a.shape[:-1] + (4,)))
    ua, _, _ = _direction(a)
    ub, _, _ = _direction(b)
    cross = np.cross(ua, ub)
    summed = ua + ub
    sum_norm = np.hypot(np.hypot(summed[..., 0], summed[..., 1]), summed[..., 2])
    anti = sum_norm == 0.0
    stable_cross = np.cross(ua, summed)
    cross = np.where((sum_norm < 1.0)[..., None], stable_cross, cross)
    least_aligned = np.argmin(np.abs(ua), axis=-1)
    basis = np.eye(3, dtype=np.float64)[least_aligned]
    anti_axis = np.cross(ua, basis)
    anti_axis, _, _ = _direction(anti_axis)
    vector = np.divide(
        cross,
        sum_norm[..., None],
        out=np.zeros_like(cross),
        where=sum_norm[..., None] != 0.0,
    )
    vector = np.where(anti[..., None], anti_axis, vector)
    scalar = np.where(anti, 0.0, 0.5 * sum_norm)
    hamilton = np.concatenate((scalar[..., None], vector), axis=-1)
    scaled = _scaled(hamilton)
    hamilton = scaled.unit_scale / np.sqrt(scaled.sum_squares)
    hamilton = _canonicalize(hamilton, False, 1)
    if epsilon < 0:
        hamilton[..., 1:] *= -1.0
    if scalar_last:
        hamilton = hamilton[..., (1, 2, 3, 0)]
    return _result(_canonicalize(hamilton, scalar_last, epsilon))
