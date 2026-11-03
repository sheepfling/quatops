"""Hamilton NumPy API with its quaternion component axis last: ``(..., 4)``.

Each quaternion stores its scalar coefficient last, in ``xyzw`` order. The
product uses a positive vector cross term. Rotations are active and
right-handed; ``multiply(lhs, rhs)`` applies ``rhs`` first. Arrays use float64
with the same validation and NumPy floating-point policy as the ``wxyz`` API.
"""

from typing import Any

import numpy as np

from .._numpy import (
    _algebraic,
    _align_sign_samples,
    _analytic_samples,
    _binary,
    _canonicalize,
    _commutator_samples,
    _component_transform,
    _components,
    _conjugate_samples,
    _direction,
    _dot_quaternions,
    _euler_from_angles,
    _euler_to_angles,
    _factor,
    _from_axis_angle_samples,
    _from_rotation_matrix,
    _from_rotvec_samples,
    _from_two_vectors,
    _inverse_samples,
    _left_divide,
    _matrix_components,
    _multiply,
    _norm_samples,
    _norm_squared,
    _normalize_samples,
    _power,
    _result,
    _right_divide,
    _rotate,
    _rotation_distance,
    _rotations_equivalent,
    _scalar_part,
    _scale,
    _skew,
    _so3_jacobian,
    _to_axis_angle_samples,
    _to_rotvec_samples,
    _vector_part,
)

_SCALAR_LAST = True

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
    """Return a copy of scalar-last ``(..., 4)`` quaternion values."""
    return _component_transform(q, True, True)


def to_xyzw(q: Any) -> np.ndarray:
    """Return a copy of the Hamilton quaternion in scalar-last order."""
    return _component_transform(q, True, True)


def from_wxyz(q: Any) -> np.ndarray:
    """Reorder scalar-first components into Hamilton scalar-last storage.

    This only changes layout; it does not change quaternion convention.
    """
    return _component_transform(q, False, True)


def to_wxyz(q: Any) -> np.ndarray:
    """Reorder Hamilton components to scalar-first layout without changing convention."""
    return _component_transform(q, True, False)


def conjugate(q: Any) -> np.ndarray:
    """Conjugate an xyzw quaternion without changing its magnitude."""
    return _conjugate_samples(q, _SCALAR_LAST)


def add(lhs: Any, rhs: Any) -> np.ndarray:
    """Add quaternion components with leading-axis broadcasting."""
    return _binary(lhs, rhs, np.add)


def subtract(lhs: Any, rhs: Any) -> np.ndarray:
    """Subtract quaternion components with leading-axis broadcasting."""
    return _binary(lhs, rhs, np.subtract)


def negate(q: Any) -> np.ndarray:
    """Negate every quaternion component."""
    return _result(-_components(q, 4, "quaternion"))


def scale(q: Any, factor: Any) -> np.ndarray:
    """Scale every component by one finite real number."""
    return _scale(q, factor)


def multiply(lhs: Any, rhs: Any) -> np.ndarray:
    """Return Hamilton product ``lhs * rhs``; in rotations, rhs acts first."""
    return _multiply(lhs, rhs, _SCALAR_LAST, 1)


def norm(q: Any) -> Any:
    """Return the norm as a float64 scalar or array of leading shape."""
    return _norm_samples(q)


def normalize(q: Any) -> np.ndarray:
    """Return unit quaternions; reject a zero quaternion anywhere in a batch."""
    return _normalize_samples(q)


def inverse(q: Any) -> np.ndarray:
    """Return multiplicative inverses; reject a zero quaternion in a batch."""
    return _inverse_samples(q, _SCALAR_LAST)


def divide(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the right quotient ``lhs * inverse(rhs)``."""
    return multiply(lhs, inverse(rhs))


def rotate(q: Any, v: Any) -> np.ndarray:
    """Actively rotate vectors with a nonzero Hamilton quaternion.

    Leading quaternion and vector shapes broadcast. The scaled cross-product
    calculation rotates without normalizing or mutating the input values.
    """
    return _rotate(q, v, scalar_last=True, epsilon=1)


def _canonical(q: np.ndarray) -> np.ndarray:
    """Choose sign from original xyzw, matching canonical physical rotation."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    first = np.where(x != 0, x, np.where(y != 0, y, z))
    # The first nonzero vector component is positive at an exact half-turn.
    flip = (w < 0) | ((w == 0) & (first < 0))
    with np.errstate(all="ignore"):
        out = np.where(flip[..., None], -q, q)
    return np.where(out == 0, 0.0, out)


def _from_unit_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        half = angle * 0.5
        w = np.cos(half)
        w = np.where(np.abs(angle) == np.pi, 0.0, w)
        vector = axis * np.sin(half)[..., None]
        q = np.concatenate((vector, w[..., None]), axis=-1)
    return _canonical(_result(q))


def from_axis_angle(a: Any) -> np.ndarray:
    """Convert (...,4) axis and radians to canonical Hamilton quaternions.

    Conversion explicitly normalizes the finite nonzero axis, including at
    zero angle. Trigonometric argument reduction at huge angles is platform
    dependent.
    """
    return _from_axis_angle_samples(a, 1, _SCALAR_LAST)


def from_rotvec(r: Any) -> np.ndarray:
    """Convert (...,3) axis-times-radian vectors to Hamilton quaternions."""
    return _from_rotvec_samples(
        _advanced_scalar.from_rotvec, r, 1, _SCALAR_LAST, False
    )


def _axis_angle(q: Any) -> tuple:
    a = _components(q, 4, "quaternion")
    if np.any(np.all(a == 0, axis=-1)):
        raise ValueError("zero quaternion has no rotation")
    a = _canonical(a)
    axis, vector_scale, length = _direction(a[..., :3], allow_zero=True)
    vscale = vector_scale[..., 0]
    magnitude = length[..., 0]
    w = a[..., 3]
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
    return _to_axis_angle_samples(q, 1, _SCALAR_LAST)


def to_rotvec(q: Any) -> np.ndarray:
    """Return (...,3) canonical physical axis-times-angle vectors."""
    return _to_rotvec_samples(
        _advanced_scalar.to_rotvec, q, 1, _SCALAR_LAST, False
    )


def to_rotation_matrix(q: Any) -> np.ndarray:
    """Return active rotation matrices with shape (..., 3, 3)."""
    return _matrix_components(q, 1, True, "rotation")


def to_left_matrix(q: Any) -> np.ndarray:
    """Return left multiplication matrices with shape (..., 4, 4)."""
    return _matrix_components(q, 1, True, "left")


def to_right_matrix(q: Any) -> np.ndarray:
    """Return right multiplication matrices with shape (..., 4, 4)."""
    return _matrix_components(q, 1, True, "right")


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
    return _analytic_samples(_advanced_scalar.tanh, q, "tanh", _SCALAR_LAST, False)


def power(q: Any, exponent: Any) -> np.ndarray:
    """Raise quaternions to an integer or principal real power."""
    return _power(q, exponent, _SCALAR_LAST, epsilon=1)


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
    return _slerp_samples(q0, q1, t, _SCALAR_LAST, shortest)


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
    """Return the algebraic quaternion tangent ``sin(q) / cos(q)``."""
    return _analytic_samples(_advanced_scalar.tan, q, "tan", _SCALAR_LAST, False)


def dot(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the Euclidean coefficient dot product.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _dot_quaternions(lhs, rhs)


def norm_squared(q: Any) -> np.ndarray:
    """Return the squared Euclidean coefficient norm.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _norm_squared(q)


def right_divide(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the right quotient ``lhs * inverse(rhs)``.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _right_divide(lhs, rhs, 1, _SCALAR_LAST)


def left_divide(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the left quotient ``inverse(lhs) * rhs``.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _left_divide(lhs, rhs, 1, _SCALAR_LAST)


def rotation_distance(q0: Any, q1: Any) -> np.ndarray:
    """Return the shortest geodesic angle between two rotations in radians.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _rotation_distance(q0, q1, 1, _SCALAR_LAST)


def rotations_equivalent(q0: Any, q1: Any, atol: Any = 1e-12) -> np.ndarray:
    """Return whether two rotations differ by at most ``atol`` radians.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _rotations_equivalent(q0, q1, atol, 1, _SCALAR_LAST)


def canonicalize(q: Any) -> np.ndarray:
    """Choose a deterministic sign while preserving the quaternion magnitude.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _canonicalize(q, _SCALAR_LAST, 1)


def from_rotation_matrix(matrix: Any, atol: Any = 1e-8) -> np.ndarray:
    """Convert a proper active 3x3 rotation matrix to a canonical unit q.

Orthogonality and determinant +1 are checked within the absolute tolerance.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _from_rotation_matrix(matrix, 1, _SCALAR_LAST, atol)


def from_two_vectors(source: Any, target: Any) -> np.ndarray:
    """Return the shortest active rotation mapping ``source`` onto ``target``.

Inputs must be nonzero. For antiparallel vectors, the half-turn axis is
chosen deterministically by crossing the source with its least-aligned
coordinate axis.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _from_two_vectors(source, target, 1, _SCALAR_LAST)



from .. import scalar as _advanced_scalar
from .._numpy import (
    _align_vectors,
    _boxminus_samples,
    _boxplus_samples,
    _from_homogeneous_matrix,
    _from_mrp_samples,
    _integrate_angular_velocity_samples,
    _make_sign_continuous,
    _nlerp_samples,
    _power_quaternion_samples,
    _quaternion_derivative_samples,
    _random_quaternion,
    _rotation_covariance,
    _rotation_mean,
    _slerp_samples,
    _squad_samples,
    _swing_twist_samples,
    _to_homogeneous_matrix_samples,
    _to_mrp_samples,
)


# Extended quaternion mathematics and attitude APIs.
def asin(q: Any) -> np.ndarray:
    """Return the principal quaternion arcsine.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.asin, q, "asin", _SCALAR_LAST, False)


def acos(q: Any) -> np.ndarray:
    """Return the principal quaternion arccosine.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.acos, q, "acos", _SCALAR_LAST, False)


def atan(q: Any) -> np.ndarray:
    """Return the principal quaternion arctangent.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.atan, q, "atan", _SCALAR_LAST, False)


def asinh(q: Any) -> np.ndarray:
    """Return the principal quaternion inverse hyperbolic sine.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.asinh, q, "asinh", _SCALAR_LAST, False)


def acosh(q: Any) -> np.ndarray:
    """Return the principal quaternion inverse hyperbolic cosine.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.acosh, q, "acosh", _SCALAR_LAST, False)


def atanh(q: Any) -> np.ndarray:
    """Return the principal quaternion inverse hyperbolic tangent.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.atanh, q, "atanh", _SCALAR_LAST, False)


def expm1(q: Any) -> np.ndarray:
    """Return ``exp(q)-1``, using a convergent series near zero.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.expm1, q, "expm1", _SCALAR_LAST, False)


def log1p(q: Any) -> np.ndarray:
    """Return ``log(1+q)``, using a convergent series near zero.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.log1p, q, "log1p", _SCALAR_LAST, False)


def log2(q: Any) -> np.ndarray:
    """Return the principal JPL quaternion logarithm to base two.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.log2, q, "log2", _SCALAR_LAST, False)


def log10(q: Any) -> np.ndarray:
    """Return the principal JPL quaternion logarithm to base ten.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _analytic_samples(_advanced_scalar.log10, q, "log10", _SCALAR_LAST, False)


def power_quaternion(q: Any, exponent: Any) -> np.ndarray:
    """Raise ``q`` to a quaternion exponent as ``exp(exponent ⊙ log(q))``.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _power_quaternion_samples(
        _advanced_scalar.power_quaternion, q, exponent, 1,
        _SCALAR_LAST, False,
    )


def commutator(lhs: Any, rhs: Any) -> np.ndarray:
    """Return the Hamilton algebra commutator ``lhs*rhs - rhs*lhs``.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _commutator_samples(lhs, rhs, 1, _SCALAR_LAST)


def scalar_part(q: Any) -> np.ndarray:
    """Return the real/scalar coefficient of a Hamilton ``wxyz`` value.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _scalar_part(q, _SCALAR_LAST)


def vector_part(q: Any) -> np.ndarray:
    """Return the three imaginary coefficients of a Hamilton ``wxyz`` value.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _vector_part(q, _SCALAR_LAST)


def quaternion_derivative(q: Any, omega: Any, frame: str = "body") -> np.ndarray:
    """Return ``dq/dt`` for a right-handed physical angular rate.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _quaternion_derivative_samples(q, omega, frame, 1, _SCALAR_LAST)


def integrate_angular_velocity_samples(
    q: Any, omegas: Any, times: Any, frame: str = "body", method: str = "midpoint"
) -> np.ndarray:
    """Integrate sampled rates with exact constant-rate increments.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _integrate_angular_velocity_samples(
        _advanced_scalar.integrate_angular_velocity_samples, q, omegas, times, frame, method,
        epsilon=1, array_scalar_last=True, function_scalar_last=False,
    )


def boxplus(q: Any, delta: Any, frame: str = "body") -> np.ndarray:
    """Apply a local physical rotation-vector increment to an attitude.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _boxplus_samples(
        _advanced_scalar.from_rotvec, q, delta, frame, 1, _SCALAR_LAST, False
    )


def boxminus(q: Any, reference: Any, frame: str = "body") -> np.ndarray:
    """Return the physical local rotation vector from ``reference`` to ``q``.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _boxminus_samples(
        _advanced_scalar.to_rotvec, q, reference, frame, 1, _SCALAR_LAST, False
    )


def nlerp(q0: Any, q1: Any, t: Any, shortest: bool = True) -> np.ndarray:
    """Normalized linear interpolation of JPL rotations.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _nlerp_samples(
        _advanced_scalar.nlerp, q0, q1, t, shortest,
        array_scalar_last=_SCALAR_LAST, function_scalar_last=False,
    )


def squad(q0: Any, q1: Any, a: Any, b: Any, t: Any) -> np.ndarray:
    """Evaluate JPL SQUAD from endpoints and their control quaternions.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _squad_samples(q0, q1, a, b, t, _SCALAR_LAST)


def align_sign(q: Any, reference: Any) -> np.ndarray:
    """Choose the sign of ``q`` closest to ``reference`` without normalization.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _align_sign_samples(q, reference)


def make_sign_continuous(quaternions: Any) -> np.ndarray:
    """Flip signs along a sequence so consecutive coefficient dots are nonnegative.

The first input sign and all quaternion magnitudes are preserved.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _make_sign_continuous(_advanced_scalar.make_sign_continuous, quaternions)


def from_euler(angles: Any, sequence: str = "xyz", *, degrees: bool = False,
               angle_axis: int = -1, component_axis: int = -1) -> np.ndarray:
    """Construct a Hamilton rotation from extrinsic or intrinsic Euler angles.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _euler_from_angles(angles, sequence, degrees=degrees,
                              angle_axis=angle_axis, component_axis=component_axis,
                              algebra_epsilon=1, scalar_last=_SCALAR_LAST)


def to_euler(q: Any, sequence: str = "xyz", gimbal_tolerance: float = 1e-12, *,
             degrees: bool = False, component_axis: int = -1,
             angle_axis: int = -1) -> np.ndarray:
    """Extract principal Euler angles from an active Hamilton rotation.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _euler_to_angles(q, sequence, degrees=degrees,
                            gimbal_tolerance=gimbal_tolerance,
                            component_axis=component_axis, angle_axis=angle_axis,
                            algebra_epsilon=1, scalar_last=_SCALAR_LAST,
                            with_status=False)


def to_euler_with_status(q: Any, sequence: str = "xyz", *, degrees: bool = False,
                         gimbal_tolerance: float = 1e-12,
                         component_axis: int = -1, angle_axis: int = -1) -> Any:
    return _euler_to_angles(q, sequence, degrees=degrees,
                            gimbal_tolerance=gimbal_tolerance,
                            component_axis=component_axis, angle_axis=angle_axis,
                            algebra_epsilon=1, scalar_last=_SCALAR_LAST,
                            with_status=True)


def from_mrp(sigma: Any) -> np.ndarray:
    """Convert modified Rodrigues parameters to a unit JPL quaternion.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _from_mrp_samples(
        _advanced_scalar.from_mrp, sigma, 1, _SCALAR_LAST, False
    )


def to_mrp(q: Any) -> np.ndarray:
    """Convert a JPL attitude to principal modified Rodrigues parameters.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _to_mrp_samples(
        _advanced_scalar.to_mrp, q, 1, _SCALAR_LAST, False
    )


def swing_twist(q: Any, axis: Any) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a JPL attitude as ``swing ⊙ twist`` about an axis.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _swing_twist_samples(q, axis, 1, _SCALAR_LAST)


def rotation_mean(quaternions: Any, weights: Any = None, method: str = "markley",
                  tolerance: Any = 1e-12, max_iterations: int = 100) -> np.ndarray:
    """Average JPL rotations by weighted Markley or intrinsic Karcher mean.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _rotation_mean(_advanced_scalar.rotation_mean, quaternions, weights,
                          method, tolerance, max_iterations,
                          array_scalar_last=True, function_scalar_last=False)


def rotation_covariance(quaternions: Any, mean: Any = None, weights: Any = None,
                        frame: str = "body") -> np.ndarray:
    """Return weighted population covariance in a JPL local tangent frame.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _rotation_covariance(_advanced_scalar.rotation_covariance, quaternions,
                                mean, weights, frame,
                                array_scalar_last=True, function_scalar_last=False)


def align_vectors(source: Any, target: Any, weights: Any = None) -> np.ndarray:
    """Solve the weighted Wahba problem with a native JPL quaternion result.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _align_vectors(_advanced_scalar.align_vectors, source, target, weights,
                          array_scalar_last=True, function_scalar_last=False)


def random_quaternion(size: Any = None, rng: Any = None) -> np.ndarray:
    """Draw a uniform unit JPL quaternion on S3 using Shoemake's method.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _random_quaternion(size, rng, 1, _SCALAR_LAST)


def random_rotation(size: Any = None, rng: Any = None) -> np.ndarray:
    """Draw a uniform SO(3) attitude, represented by a unit quaternion.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return random_quaternion(size, rng)


def to_homogeneous_matrix(q: Any, translation: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Return a 4x4 homogeneous matrix for a JPL attitude and translation.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _to_homogeneous_matrix_samples(q, translation, 1, _SCALAR_LAST)


def from_homogeneous_matrix(matrix: Any, atol: Any = 1e-8) -> np.ndarray:
    """Validate a rigid 4x4 transform and return its native JPL rotation.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _from_homogeneous_matrix(
        matrix, 1, atol, array_scalar_last=_SCALAR_LAST,
    )


def so3_adjoint(q: Any) -> np.ndarray:
    """Return the SO(3) adjoint matrix, equal to the active rotation matrix.

NumPy inputs preserve batch axes; compatible leading axes broadcast."""
    return _matrix_components(q, 1, _SCALAR_LAST, "rotation")
