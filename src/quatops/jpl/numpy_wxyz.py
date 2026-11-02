"""Native NumPy JPL quaternions: scalar-first ``(..., 4)`` in wxyz order.

The product uses a negative vector cross term. Rotations are active and
right-handed; ``multiply(lhs, rhs)`` applies ``rhs`` first. The physically equivalent
Hamilton quaternion is ``(w, -x, -y, -z)``. Arrays use float64 with the same
validation and NumPy floating-point policy as the scalar-first NumPy backend.
"""

from typing import Any

import numpy as np

from .._numpy import (
    _algebraic,
    _binary,
    _canonicalize,
    _components,
    _direction,
    _dot_quaternions,
    _factor,
    _from_rotation_matrix,
    _from_two_vectors,
    _left_divide,
    _matrix_components,
    _norm_squared,
    _power,
    _result,
    _right_divide,
    _rotation_distance,
    _rotations_equivalent,
    _scale,
    _scaled,
    _skew,
    _slerp,
    _so3_jacobian,
    _sum_three,
)

_SCALAR_LAST = False

def exp(q: Any) -> np.ndarray:
    """Return the algebraic quaternion exponential in this storage order."""
    return _algebraic(q, _SCALAR_LAST, "exp")


def log(q: Any) -> np.ndarray:
    """Return the principal algebraic quaternion logarithm in this order."""
    return _algebraic(q, _SCALAR_LAST, "log")


def sin(q: Any) -> np.ndarray:
    """Return the algebraic quaternion sine in this storage order."""
    return _algebraic(q, _SCALAR_LAST, "sin")


def cos(q: Any) -> np.ndarray:
    """Return the algebraic quaternion cosine in this storage order."""
    return _algebraic(q, _SCALAR_LAST, "cos")


def from_xyzw(q: Any) -> np.ndarray:
    """Reorder scalar-last components into JPL scalar-first layout.

    This changes only component layout, not quaternion convention.
    """
    a = _components(q, 4, "quaternion")
    return a[..., [3, 0, 1, 2]]


def to_xyzw(q: Any) -> np.ndarray:
    """Reorder scalar-first JPL components to scalar-last layout."""
    a = _components(q, 4, "quaternion")
    return a[..., [1, 2, 3, 0]]


def from_wxyz(q: Any) -> np.ndarray:
    """Validate and copy scalar-first JPL values without changing convention."""
    return _components(q, 4, "quaternion").copy()


def to_wxyz(q: Any) -> np.ndarray:
    """Validate and copy scalar-first JPL values without changing convention."""
    return _components(q, 4, "quaternion").copy()


def from_hamilton(q: Any) -> np.ndarray:
    """Convert Hamilton wxyz to JPL wxyz without normalization."""
    out = _components(q, 4, "Hamilton quaternion").copy()
    out[..., 1:] *= -1
    return out


def to_hamilton(q: Any) -> np.ndarray:
    """Convert JPL wxyz to Hamilton wxyz without normalization."""
    out = _components(q, 4, "JPL quaternion").copy()
    out[..., 1:] *= -1
    return out


def conjugate(q: Any) -> np.ndarray:
    """Conjugate a wxyz quaternion without changing its magnitude."""
    out = _components(q, 4, "quaternion").copy()
    out[..., 1:] *= -1
    return out


def add(lhs: Any, rhs: Any) -> np.ndarray:
    """Add JPL components with leading-axis broadcasting."""
    return _binary(lhs, rhs, np.add)


def subtract(lhs: Any, rhs: Any) -> np.ndarray:
    """Subtract JPL components with leading-axis broadcasting."""
    return _binary(lhs, rhs, np.subtract)


def negate(q: Any) -> np.ndarray:
    """Negate every JPL quaternion component."""
    return _result(-_components(q, 4, "quaternion"))


def scale(q: Any, factor: Any) -> np.ndarray:
    """Scale every JPL component by one finite real number."""
    return _scale(q, factor)


def multiply(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the JPL product; for active rotations, rhs acts first."""
    a = _components(lhs, 4, "lhs")
    b = _components(rhs, 4, "rhs")
    a, b = np.broadcast_arrays(a, b)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    with np.errstate(all="ignore"):
        result = np.stack((
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw - ay*bz + az*by,
            aw*by + ay*bw - az*bx + ax*bz,
            aw*bz + az*bw - ax*by + ay*bx,
        ), axis=-1)
    return _result(result)


def norm(q: Any) -> Any:
    """Return the norm as a float64 scalar or array of leading shape."""
    a = _components(q, 4, "quaternion")
    with np.errstate(all="ignore"):
        scale = np.max(np.abs(a), axis=-1, keepdims=True)
        scaled = np.divide(a, scale, out=np.zeros_like(a), where=scale != 0)
        result = (scale * np.sqrt(np.sum(scaled * scaled, axis=-1, keepdims=True)))[..., 0]
    return _result(result[()] if result.ndim == 0 else result)


def normalize(q: Any) -> np.ndarray:
    """Return unit quaternions; reject a zero quaternion anywhere in a batch."""
    a = _components(q, 4, "quaternion")
    scaled = _scaled(a)
    with np.errstate(all="ignore"):
        out = scaled.unit_scale / np.sqrt(scaled.sum_squares)
    return _result(out)


def inverse(q: Any) -> np.ndarray:
    """Return multiplicative inverses; reject a zero quaternion in a batch."""
    a = _components(q, 4, "quaternion")
    scaled_state = _scaled(a)
    scaled = scaled_state.unit_scale.copy()
    scaled[..., 1:] *= -1
    with np.errstate(all="ignore"):
        out = (scaled / scaled_state.sum_squares) / scaled_state.scale
    return _result(out)


def divide(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the right quotient ``lhs * inverse(rhs)`` in JPL algebra."""
    return multiply(lhs, inverse(rhs))


def rotate(q: Any, v: Any) -> np.ndarray:
    """Actively rotate vectors with a nonzero JPL quaternion.

    Leading quaternion and vector shapes broadcast. Internally the quaternion
    is rescaled for numerical range; inputs are never normalized or mutated.
    """
    a = _components(q, 4, "quaternion")
    v = _components(v, 3, "vector")
    scaled_state = _scaled(a)
    scaled = scaled_state.unit_scale
    sums = scaled_state.sum_squares
    shape = np.broadcast_shapes(scaled.shape[:-1], v.shape[:-1])
    scaled = np.broadcast_to(scaled, shape + (4,))
    v = np.broadcast_to(v, shape + (3,))
    sums = np.broadcast_to(sums[..., 0], shape)
    w, x, y, z = np.moveaxis(scaled, -1, 0)
    vx, vy, vz = np.moveaxis(v, -1, 0)
    with np.errstate(all="ignore"):
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        m00 = np.clip(((ww + xx) - yy - zz) / sums, -1.0, 1.0)
        m01 = np.clip((2 * (x*y + w*z)) / sums, -1.0, 1.0)
        m02 = np.clip((2 * (x*z - w*y)) / sums, -1.0, 1.0)
        m10 = np.clip((2 * (x*y - w*z)) / sums, -1.0, 1.0)
        m11 = np.clip(((ww - xx) + yy - zz) / sums, -1.0, 1.0)
        m12 = np.clip((2 * (y*z + w*x)) / sums, -1.0, 1.0)
        m20 = np.clip((2 * (x*z + w*y)) / sums, -1.0, 1.0)
        m21 = np.clip((2 * (y*z - w*x)) / sums, -1.0, 1.0)
        m22 = np.clip(((ww - xx) - yy + zz) / sums, -1.0, 1.0)
        out = np.stack((
            _sum_three(m00*vx, m01*vy, m02*vz),
            _sum_three(m10*vx, m11*vy, m12*vz),
            _sum_three(m20*vx, m21*vy, m22*vz),
        ), axis=-1)
    return _result(out)


def _canonical(q: np.ndarray) -> np.ndarray:
    """Choose sign from original wxyz, matching canonical physical rotation."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    first = np.where(x != 0, x, np.where(y != 0, y, z))
    # The Hamilton physical vector is (-x, -y, -z), so the JPL tie is negative.
    flip = (w < 0) | ((w == 0) & (first > 0))
    with np.errstate(all="ignore"):
        out = np.where(flip[..., None], -q, q)
    return np.where(out == 0, 0.0, out)


def _from_unit_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        half = angle * 0.5
        w = np.cos(half)
        w = np.where(np.abs(angle) == np.pi, 0.0, w)
        vector = -axis * np.sin(half)[..., None]
        q = np.concatenate((w[..., None], vector), axis=-1)
    return _canonical(_result(q))


def from_axis_angle(a: Any) -> np.ndarray:
    """Convert (...,4) axis and radians to canonical JPL quaternions.

    Conversion explicitly normalizes the finite nonzero axis, including at
    zero angle. Trigonometric argument reduction at huge angles is platform
    dependent.
    """
    a = _components(a, 4, "axis-angle")
    axis, _, _ = _direction(a[..., :3])
    return _from_unit_axis_angle(axis, a[..., 3])


def from_rotvec(r: Any) -> np.ndarray:
    """Convert (...,3) axis-times-radian vectors to JPL quaternions."""
    r = _components(r, 3, "rotation vector")
    axis, scale, length = _direction(r, allow_zero=True)
    with np.errstate(all="ignore"):
        angle = (scale * length)[..., 0]
    return _from_unit_axis_angle(axis, _result(angle))


def _axis_angle(q: Any) -> tuple:
    a = _components(q, 4, "quaternion")
    if np.any(np.all(a == 0, axis=-1)):
        raise ValueError("zero quaternion has no rotation")
    a = _canonical(a)
    # JPL's vector part is opposite the axis of the equivalent active rotation.
    axis, vector_scale, length = _direction(-a[..., 1:], allow_zero=True)
    vscale = vector_scale[..., 0]
    magnitude = length[..., 0]
    w = a[..., 0]
    with np.errstate(all="ignore"):
        scale = np.maximum(w, vscale)
        angle = 2.0 * np.arctan2((vscale / scale) * magnitude, w / scale)
        tiny = (w > 0) & (vscale <= w * 1e-8)
        v_mantissa, v_exponent = np.frexp(vscale)
        w_mantissa, w_exponent = np.frexp(w)
        small_angle = np.ldexp(
            (2.0 * v_mantissa / w_mantissa) * magnitude,
            v_exponent - w_exponent,
        )
        angle = np.where(tiny, small_angle, angle)
        axis = np.where((angle == 0)[..., None], np.array([1.0, 0.0, 0.0]), axis)
        axis = np.where(axis == 0, 0.0, axis)
        angle = np.where(angle == 0, 0.0, angle)
    return _result(axis), _result(angle)


def to_axis_angle(q: Any) -> np.ndarray:
    """Return (...,4) canonical physical axis and principal angle in radians."""
    axis, angle = _axis_angle(q)
    return np.concatenate((axis, angle[..., None]), axis=-1)


def to_rotvec(q: Any) -> np.ndarray:
    """Return (...,3) canonical physical axis-times-angle vectors."""
    axis, angle = _axis_angle(q)
    original = _canonical(_components(q, 4, "quaternion"))
    vector = -original[..., 1:]
    w = original[..., 0]
    with np.errstate(all="ignore"):
        vector_scale = np.max(np.abs(vector), axis=-1)
        tiny = (w > 0) & (vector_scale <= w * 1e-8)
        small_vector = (2.0 * vector) / w[..., None]
        result = np.where(tiny[..., None], small_vector, axis * angle[..., None])
    return np.where(result == 0, 0.0, _result(result))


def to_rotation_matrix(q: Any) -> np.ndarray:
    """Return active rotation matrices with shape (..., 3, 3)."""
    return _matrix_components(q, -1, False, "rotation")


def to_left_matrix(q: Any) -> np.ndarray:
    """Return left multiplication matrices with shape (..., 4, 4)."""
    return _matrix_components(q, -1, False, "left")


def to_right_matrix(q: Any) -> np.ndarray:
    """Return right multiplication matrices with shape (..., 4, 4)."""
    return _matrix_components(q, -1, False, "right")


def from_rotation_vector(r: Any) -> np.ndarray:
    """Convert a physical rotation vector in radians to a quaternion."""
    return from_rotvec(r)


def to_rotation_vector(q: Any) -> np.ndarray:
    """Extract the canonical physical rotation vector in radians."""
    return to_rotvec(q)


def sinh(q: Any) -> np.ndarray:
    """Return quaternion hyperbolic sine in the selected coefficient order."""
    return _algebraic(q, _SCALAR_LAST, "sinh")


def cosh(q: Any) -> np.ndarray:
    """Return quaternion hyperbolic cosine in the selected coefficient order."""
    return _algebraic(q, _SCALAR_LAST, "cosh")


def tanh(q: Any) -> np.ndarray:
    """Return quaternion hyperbolic tangent as sinh(q) / cosh(q)."""
    return divide(sinh(q), cosh(q))


def power(q: Any, exponent: Any) -> np.ndarray:
    """Raise quaternions to an integer or principal real power."""
    return _power(q, exponent, _SCALAR_LAST, epsilon=-1)


def pow_real(q: Any, exponent: Any) -> np.ndarray:
    """Raise quaternions to a principal real power."""
    return power(q, exponent)


def sqrt(q: Any) -> np.ndarray:
    """Return the principal quaternion square root."""
    return power(q, 0.5)


def slerp(q0: Any, q1: Any, t: Any, shortest: bool = True) -> np.ndarray:
    """Spherically interpolate rotations, preserving batch dimensions."""
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")  # noqa: TRY004
    return _slerp(q0, q1, t, _SCALAR_LAST, shortest)


def skew_matrix(v: Any) -> np.ndarray:
    """Return batched 3x3 cross-product matrices."""
    return _skew(v)


def so3_left_jacobian(r: Any) -> np.ndarray:
    """Return batched SO(3) left Jacobians."""
    return _so3_jacobian(r, 1)


def so3_right_jacobian(r: Any) -> np.ndarray:
    """Return batched SO(3) right Jacobians."""
    return _so3_jacobian(r, -1)


def so3_left_jacobian_inverse(r: Any) -> np.ndarray:
    """Return batched inverse SO(3) left Jacobians."""
    return _so3_jacobian(r, 1, True)


def so3_right_jacobian_inverse(r: Any) -> np.ndarray:
    """Return batched inverse SO(3) right Jacobians."""
    return _so3_jacobian(r, -1, True)


def integrate_angular_velocity(q: Any, omega: Any, dt: Any, frame: str = "body") -> np.ndarray:
    """Integrate constant physical angular velocity in body or world frame."""
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    step = _factor(dt, "dt")
    angular = _components(omega, 3, "angular velocity")
    with np.errstate(all="ignore"):
        delta = _result(angular * step)
    increment = from_rotation_vector(delta)
    attitude = normalize(q)
    product = multiply(attitude, increment) if frame == "body" else multiply(increment, attitude)
    return normalize(product)


def tan(q: Any) -> np.ndarray:
    """Return the algebraic quaternion tangent in JPL coefficient order."""
    return divide(sin(q), cos(q))


def dot(lhs: Any, rhs: Any) -> np.ndarray:
    return _dot_quaternions(lhs, rhs)


def norm_squared(q: Any) -> np.ndarray:
    return _norm_squared(q)


def right_divide(lhs: Any, rhs: Any) -> np.ndarray:
    return _right_divide(lhs, rhs, -1, _SCALAR_LAST)


def left_divide(lhs: Any, rhs: Any) -> np.ndarray:
    return _left_divide(lhs, rhs, -1, _SCALAR_LAST)


def rotation_distance(q0: Any, q1: Any) -> np.ndarray:
    return _rotation_distance(q0, q1, -1, _SCALAR_LAST)


def rotations_equivalent(q0: Any, q1: Any, atol: Any = 1e-12) -> np.ndarray:
    return _rotations_equivalent(q0, q1, atol, -1, _SCALAR_LAST)


def canonicalize(q: Any) -> np.ndarray:
    return _canonicalize(q, _SCALAR_LAST, -1)


def from_rotation_matrix(matrix: Any, atol: Any = 1e-8) -> np.ndarray:
    return _from_rotation_matrix(matrix, -1, _SCALAR_LAST, atol)


def from_two_vectors(source: Any, target: Any) -> np.ndarray:
    return _from_two_vectors(source, target, -1, _SCALAR_LAST)



from .._numpy import (
    _align_vectors,
    _from_homogeneous_matrix,
    _integrate_angular_velocity_samples,
    _make_sign_continuous,
    _map_scalar_ordered,
    _random_quaternion,
    _rotation_covariance,
    _rotation_mean,
)
from . import scalar as _advanced_scalar


def _map_scalar(
    function: Any,
    *values: Any,
    component_arguments: dict[int, tuple[int, str]],
    result_shape: tuple[int, ...],
) -> np.ndarray:
    return _map_scalar_ordered(
        function, *values, component_arguments=component_arguments,
        result_shape=result_shape, array_scalar_last=_SCALAR_LAST,
        function_scalar_last=True,
    )


# Extended quaternion mathematics and attitude APIs.
def asin(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.asin, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def acos(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.acos, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def atan(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.atan, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def asinh(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.asinh, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def acosh(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.acosh, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def atanh(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.atanh, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def expm1(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.expm1, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def log1p(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.log1p, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def log2(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.log2, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def log10(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.log10, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(4,))


def power_quaternion(q: Any, exponent: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.power_quaternion, q, exponent,
                       component_arguments={0: (4, "quaternion"), 1: (4, "exponent")},
                       result_shape=(4,))


def commutator(lhs: Any, rhs: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.commutator, lhs, rhs,
                       component_arguments={0: (4, "lhs"), 1: (4, "rhs")},
                       result_shape=(4,))


def scalar_part(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.scalar_part, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=())


def vector_part(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.vector_part, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(3,))


def quaternion_derivative(q: Any, omega: Any, frame: str = "body") -> np.ndarray:
    return _map_scalar(_advanced_scalar.quaternion_derivative, q, omega, frame,
                       component_arguments={0: (4, "quaternion"), 1: (3, "omega")},
                       result_shape=(4,))


def integrate_angular_velocity_samples(
    q: Any, omegas: Any, times: Any, frame: str = "body", method: str = "midpoint"
) -> np.ndarray:
    return _integrate_angular_velocity_samples(
        _advanced_scalar.integrate_angular_velocity_samples, q, omegas, times, frame, method,
        array_scalar_last=False, function_scalar_last=True,
    )


def boxplus(q: Any, delta: Any, frame: str = "body") -> np.ndarray:
    return _map_scalar(_advanced_scalar.boxplus, q, delta, frame,
                       component_arguments={0: (4, "quaternion"), 1: (3, "delta")},
                       result_shape=(4,))


def boxminus(q: Any, reference: Any, frame: str = "body") -> np.ndarray:
    return _map_scalar(_advanced_scalar.boxminus, q, reference, frame,
                       component_arguments={0: (4, "q"), 1: (4, "reference")},
                       result_shape=(3,))


def nlerp(q0: Any, q1: Any, t: Any, shortest: bool = True) -> np.ndarray:
    return _map_scalar(_advanced_scalar.nlerp, q0, q1, t, shortest,
                       component_arguments={0: (4, "q0"), 1: (4, "q1")},
                       result_shape=(4,))


def squad(q0: Any, q1: Any, a: Any, b: Any, t: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.squad, q0, q1, a, b, t,
                       component_arguments={0: (4, "q0"), 1: (4, "q1"),
                                            2: (4, "a"), 3: (4, "b")},
                       result_shape=(4,))


def align_sign(q: Any, reference: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.align_sign, q, reference,
                       component_arguments={0: (4, "q"), 1: (4, "reference")},
                       result_shape=(4,))


def make_sign_continuous(quaternions: Any) -> np.ndarray:
    return _make_sign_continuous(_advanced_scalar.make_sign_continuous, quaternions)


def from_euler(angles: Any, sequence: str = "xyz") -> np.ndarray:
    return _map_scalar(_advanced_scalar.from_euler, angles, sequence,
                       component_arguments={0: (3, "angles")}, result_shape=(4,))


def to_euler(q: Any, sequence: str = "xyz") -> np.ndarray:
    return _map_scalar(_advanced_scalar.to_euler, q, sequence,
                       component_arguments={0: (4, "quaternion")}, result_shape=(3,))


def from_mrp(sigma: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.from_mrp, sigma,
                       component_arguments={0: (3, "sigma")}, result_shape=(4,))


def to_mrp(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.to_mrp, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(3,))


def swing_twist(q: Any, axis: Any) -> tuple[np.ndarray, np.ndarray]:
    values = _map_scalar(_advanced_scalar.swing_twist, q, axis,
                         component_arguments={0: (4, "quaternion"), 1: (3, "axis")},
                         result_shape=(2, 4))
    return values[..., 0, :], values[..., 1, :]


def rotation_mean(quaternions: Any, weights: Any = None, method: str = "markley",
                  tolerance: Any = 1e-12, max_iterations: int = 100) -> np.ndarray:
    return _rotation_mean(_advanced_scalar.rotation_mean, quaternions, weights,
                          method, tolerance, max_iterations,
                          array_scalar_last=False, function_scalar_last=True)


def rotation_covariance(quaternions: Any, mean: Any = None, weights: Any = None,
                        frame: str = "body") -> np.ndarray:
    return _rotation_covariance(_advanced_scalar.rotation_covariance, quaternions,
                                mean, weights, frame,
                                array_scalar_last=False, function_scalar_last=True)


def align_vectors(source: Any, target: Any, weights: Any = None) -> np.ndarray:
    return _align_vectors(_advanced_scalar.align_vectors, source, target, weights,
                          array_scalar_last=False, function_scalar_last=True)


def random_quaternion(size: Any = None, rng: Any = None) -> np.ndarray:
    return _random_quaternion(size, rng, -1, _SCALAR_LAST)


def random_rotation(size: Any = None, rng: Any = None) -> np.ndarray:
    return random_quaternion(size, rng)


def to_homogeneous_matrix(q: Any, translation: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
    return _map_scalar(_advanced_scalar.to_homogeneous_matrix, q, translation,
                       component_arguments={0: (4, "quaternion"), 1: (3, "translation")},
                       result_shape=(4, 4))


def from_homogeneous_matrix(matrix: Any, atol: Any = 1e-8) -> np.ndarray:
    return _from_homogeneous_matrix(
        matrix, _advanced_scalar.from_homogeneous_matrix, atol,
        array_scalar_last=_SCALAR_LAST, function_scalar_last=True,
    )


def so3_adjoint(q: Any) -> np.ndarray:
    return _map_scalar(_advanced_scalar.so3_adjoint, q,
                       component_arguments={0: (4, "quaternion")}, result_shape=(3, 3))
