"""Internal validation and numerical helpers for explicit NumPy backends."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, Callable, NamedTuple

import numpy as np

from ._euler_sequence import euler_metadata as _euler_sequence_metadata
from ._euler_types import EulerResult


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
    x, y, z = np.moveaxis(_components(v, 3, "vector"), -1, 0)
    zero = np.zeros_like(x)
    return np.stack((np.stack((zero, -z, y), axis=-1),
                     np.stack((z, zero, -x), axis=-1),
                     np.stack((-y, x, zero), axis=-1)), axis=-2)


def _so3_jacobian(r: Any, side: int, inverse_matrix: bool = False) -> np.ndarray:
    vector = _components(r, 3, "rotation vector")
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


def _matrix_components(q: Any, epsilon: int, scalar_last: bool, kind: str) -> np.ndarray:
    """Build a batched quaternion matrix in the requested algebra and layout."""
    a = _components(q, 4, "quaternion")
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
    out = np.empty_like(samples)
    for index in np.ndindex(samples.shape[:-2]):
        out[index] = function(samples[index]) if count else samples[index]
    return _result(out)


def _integrate_angular_velocity_samples(
    function: Callable[..., Any], q: Any, omegas: Any, times: Any,
    frame: str = "body", method: str = "midpoint",
    *, array_scalar_last: bool = False, function_scalar_last: bool = False,
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
    function: Callable[..., Any],
    atol: Any = 1e-8,
    *,
    array_scalar_last: bool = False,
    function_scalar_last: bool = False,
) -> np.ndarray:
    values = np.asarray(matrix)
    if values.dtype.kind not in "biuf" or values.ndim < 2 or values.shape[-2:] != (4, 4):
        raise ValueError("matrix must have shape (..., 4, 4) and real numeric dtype")
    with np.errstate(all="ignore"):
        values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain finite float64 values")
    flat = values.reshape((-1, 4, 4))
    result = np.asarray([function(item, atol) for item in flat], dtype=np.float64)
    if array_scalar_last != function_scalar_last:
        order = (3, 0, 1, 2) if function_scalar_last else (1, 2, 3, 0)
        result = result[..., order]
    return _result(result.reshape(values.shape[:-2] + (4,)))


def _dot_quaternions(lhs: Any, rhs: Any) -> np.ndarray:
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    with np.errstate(all="ignore"):
        terms = a * b
        scale = np.max(np.abs(terms), axis=-1)
        normalized = np.divide(
            terms,
            scale[..., None],
            out=np.zeros_like(terms),
            where=scale[..., None] != 0.0,
        )

        # Neumaier summation on bounded terms avoids overflow during
        # accumulation while retaining low-order cancellation errors.
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
            | np.any((terms != 0.0) & (np.abs(terms) < np.finfo(np.float64).tiny), axis=-1)
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
        flat_result = np.asarray(result).reshape(-1)
        for flat_index in np.flatnonzero(needs_exact.reshape(-1)):
            flat_result[flat_index] = scalar_dot(flat_a[flat_index], flat_b[flat_index])
    return _result(result)


def _norm_squared(q: Any) -> np.ndarray:
    return _dot_quaternions(q, q)


def _left_divide(lhs: Any, rhs: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    return _quaternion_product(_inverse_coefficients(a, scalar_last), b, epsilon, scalar_last)


def _right_divide(lhs: Any, rhs: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    a, b = _components(lhs, 4, "lhs"), _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    return _quaternion_product(a, _inverse_coefficients(b, scalar_last), epsilon, scalar_last)


def _rotation_distance(q0: Any, q1: Any, epsilon: int, scalar_last: bool) -> np.ndarray:
    a, b = _components(q0, 4, "q0"), _components(q1, 4, "q1")
    a, b = np.broadcast_arrays(a, b)
    a = _scaled(a).unit_scale / np.sqrt(_scaled(a).sum_squares)
    b = _scaled(b).unit_scale / np.sqrt(_scaled(b).sum_squares)
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
    return _rotation_distance(q0, q1, epsilon, scalar_last) <= tolerance


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
    flat = a.reshape((-1, 3, 3))
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
