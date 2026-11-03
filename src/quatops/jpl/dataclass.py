"""JPL scalar-last quaternions with named, immutable fields.

Algebra uses a negative cross product and active right-handed rotations.
Constructors retain input without validation; functions validate finite values.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import hypot, isfinite

from .._advanced import EulerResult
from .._typing import FloatLike, FloatT
from . import scalar as _s


@dataclass(frozen=True)
class Quaternion:
    """Immutable slotted Quaternion values; construction is unchecked."""

    __slots__ = ('w', 'x', 'y', 'z')
    x: float
    y: float
    z: float
    w: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z, self.w))


@dataclass(frozen=True)
class AxisAngle:
    """Immutable slotted AxisAngle values; construction is unchecked."""

    __slots__ = ('angle', 'ax', 'ay', 'az')
    ax: float
    ay: float
    az: float
    angle: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.ax, self.ay, self.az, self.angle))


@dataclass(frozen=True)
class RotVec:
    """Immutable slotted RotVec values; construction is unchecked."""

    __slots__ = ('x', 'y', 'z')
    x: float
    y: float
    z: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z))



def _quaternion(q: Iterable[FloatT]) -> Quaternion:
    return Quaternion(*_s._quaternion(q))


def from_xyzw(q: Iterable[FloatT]) -> Quaternion:
    """Validate scalar-last layout without changing convention."""
    return _quaternion(q)


def to_xyzw(q: Iterable[FloatT]) -> tuple[float, float, float, float]:
    """Return scalar-last components in a plain tuple."""
    return _s._quaternion(q)


def from_wxyz(q: Iterable[FloatT]) -> Quaternion:
    """Reorder scalar-first components to JPL scalar-last layout."""
    w, x, y, z = _s._quaternion(q)
    return Quaternion(x, y, z, w)


def to_wxyz(q: Iterable[FloatT]) -> tuple[float, float, float, float]:
    """Reorder to scalar-first without changing convention."""
    x, y, z, w = _s._quaternion(q)
    return w, x, y, z


def from_hamilton(q: Iterable[FloatT]) -> Quaternion:
    """Convert Hamilton wxyz to JPL xyzw without normalization."""
    w, x, y, z = _s._quaternion(q)
    return Quaternion(-x, -y, -z, w)


def to_hamilton(q: Iterable[FloatT]) -> tuple[float, float, float, float]:
    """Convert JPL xyzw to Hamilton wxyz without normalization."""
    x, y, z, w = _s._quaternion(q)
    return w, -x, -y, -z


def conjugate(q: Iterable[FloatT]) -> Quaternion:
    """Return the conjugate, preserving magnitude."""
    x, y, z, w = _s._quaternion(q)
    return Quaternion(-x, -y, -z, w)


def add(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Add JPL components and return a dataclass Quaternion."""
    x, y, z, w = _s._quaternion(lhs)
    a, b, c, d = _s._quaternion(rhs)
    return Quaternion(*_s._checked((x + a, y + b, z + c, w + d)))


def subtract(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Subtract JPL components and return a dataclass Quaternion."""
    x, y, z, w = _s._quaternion(lhs)
    a, b, c, d = _s._quaternion(rhs)
    return Quaternion(*_s._checked((x - a, y - b, z - c, w - d)))


def negate(q: Iterable[FloatT]) -> Quaternion:
    """Negate every component and return a dataclass Quaternion."""
    x, y, z, w = _s._quaternion(q)
    return Quaternion(-x, -y, -z, -w)


def scale(q: Iterable[FloatT], factor: FloatLike) -> Quaternion:
    """Scale every JPL component by a finite real number."""
    value = _s._real_scalar(factor, "factor")
    x, y, z, w = _s._quaternion(q)
    return Quaternion(*_s._checked((value*x, value*y, value*z, value*w)))


def multiply(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the negative-cross JPL product; ``rhs`` acts first."""
    x, y, z, w = _s._quaternion(lhs)
    a, b, c, d = _s._quaternion(rhs)
    values = (
        w*a + x*d - y*c + z*b,
        w*b + y*d - z*a + x*c,
        w*c + z*d - x*b + y*a,
        w*d - x*a - y*b - z*c,
    )
    return Quaternion(*_s._checked(values))


def norm(q: Iterable[FloatT]) -> float:
    """Return Euclidean norm."""
    value = hypot(*_s._quaternion(q))
    if not isfinite(value):
        raise OverflowError("quaternion norm is too large for a float")
    return value


def normalize(q: Iterable[FloatT]) -> Quaternion:
    """Return a unit quaternion, rejecting zero."""
    _, p = _s._scaled(_s._quaternion(q))
    length = hypot(*p)
    return Quaternion(*(value / length for value in p))


def inverse(q: Iterable[FloatT]) -> Quaternion:
    """Return the multiplicative inverse, rejecting zero."""
    scale, p = _s._scaled(_s._quaternion(q))
    den = sum(value * value for value in p)
    x, y, z, w = p
    out = ((-x / den) / scale, (-y / den) / scale,
           (-z / den) / scale, (w / den) / scale)
    return Quaternion(*_s._checked(out))


def divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)`` in JPL algebra."""
    return Quaternion(*_s.divide(lhs, rhs))


rotate = _s.rotate
to_rotation_matrix = _s.to_rotation_matrix
to_left_matrix = _s.to_left_matrix
to_right_matrix = _s.to_right_matrix


def sinh(q: Iterable[FloatT]) -> Quaternion:
    """Return JPL quaternion hyperbolic sine."""
    return Quaternion(*_s.sinh(q))


def cosh(q: Iterable[FloatT]) -> Quaternion:
    """Return JPL quaternion hyperbolic cosine."""
    return Quaternion(*_s.cosh(q))


def tanh(q: Iterable[FloatT]) -> Quaternion:
    """Return JPL quaternion hyperbolic tangent."""
    return Quaternion(*_s.tanh(q))


def power(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Raise a JPL quaternion to an integer or principal real power."""
    return Quaternion(*_s.power(q, exponent))


def pow_real(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Raise a JPL quaternion to a principal real power."""
    return Quaternion(*_s.pow_real(q, exponent))


def sqrt(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal square root of a JPL quaternion."""
    return Quaternion(*_s.sqrt(q))


def slerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Interpolate JPL attitudes by shortest-path spherical interpolation."""
    return Quaternion(*_s.slerp(q0, q1, t, shortest=shortest))


skew_matrix = _s.skew_matrix
so3_left_jacobian = _s.so3_left_jacobian
so3_right_jacobian = _s.so3_right_jacobian
so3_left_jacobian_inverse = _s.so3_left_jacobian_inverse
so3_right_jacobian_inverse = _s.so3_right_jacobian_inverse


def integrate_angular_velocity(q: Iterable[FloatT], omega: Iterable[FloatT], dt: FloatLike,
                               frame: str = "body") -> Quaternion:
    """Integrate JPL attitude using the specified angular-velocity frame."""
    return Quaternion(*_s.integrate_angular_velocity(q, omega, dt, frame=frame))


def from_axis_angle(a: Iterable[FloatT]) -> Quaternion:
    """Normalize a nonzero axis and convert physical radians to JPL."""
    x, y, z, angle = _s._components(a, 4, "axis-angle")
    return Quaternion(*_s._from_unit_axis(_s._axis((x, y, z)), angle))


def to_axis_angle(q: Iterable[FloatT]) -> AxisAngle:
    """Return canonical physical axis and principal radians."""
    axis, angle = _s._axis_angle_from_q(q)
    return AxisAngle(*axis, angle)


def from_rotvec(r: Iterable[FloatT]) -> Quaternion:
    """Convert a physical rotation vector in radians to JPL."""
    v = _s._vector(r)
    if max(abs(value) for value in v) == 0:
        return Quaternion(0.0, 0.0, 0.0, 1.0)
    length = hypot(*v)
    if not isfinite(length):
        raise OverflowError("rotation-vector norm is too large for a float")
    return Quaternion(*_s._from_unit_axis(_s._axis(v), length))


def to_rotvec(q: Iterable[FloatT]) -> RotVec:
    """Extract the canonical physical rotation vector in radians."""
    return RotVec(*_s._rotvec_from_q(q))


def exp(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion exponential as a JPL value."""
    return Quaternion(*_s.exp(q))


def log(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal algebraic quaternion logarithm as a JPL value."""
    return Quaternion(*_s.log(q))


def sin(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion sine as a JPL value."""
    return Quaternion(*_s.sin(q))


def cos(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion cosine as a JPL value."""
    return Quaternion(*_s.cos(q))


def from_rotation_vector(r: Iterable[FloatT]) -> Quaternion:
    """Convert a physical rotation vector in radians to canonical JPL."""
    return from_rotvec(r)


def to_rotation_vector(q: Iterable[FloatT]) -> RotVec:
    """Extract the canonical physical rotation vector from a JPL value."""
    return to_rotvec(q)


def tan(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion tangent ``sin(q) / cos(q)``."""
    return Quaternion(*_s.tan(q))


dot = _s.dot
norm_squared = _s.norm_squared


def right_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)``."""
    return Quaternion(*_s.right_divide(lhs, rhs))


def left_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the left quotient ``inverse(lhs) * rhs``."""
    return Quaternion(*_s.left_divide(lhs, rhs))


rotation_distance = _s.rotation_distance
rotations_equivalent = _s.rotations_equivalent


def canonicalize(q: Iterable[FloatT]) -> Quaternion:
    """Choose a deterministic sign while preserving the quaternion magnitude."""
    return Quaternion(*_s.canonicalize(q))


def from_rotation_matrix(matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8) -> Quaternion:
    """Convert a proper active 3x3 rotation matrix to a canonical unit q.

Orthogonality and determinant +1 are checked within the absolute tolerance."""
    return Quaternion(*_s.from_rotation_matrix(matrix, atol))


def from_two_vectors(source: Iterable[FloatT], target: Iterable[FloatT]) -> Quaternion:
    """Return the shortest active rotation mapping ``source`` onto ``target``.

Inputs must be nonzero. For antiparallel vectors, the half-turn axis is
chosen deterministically by crossing the source with its least-aligned
coordinate axis."""
    return Quaternion(*_s.from_two_vectors(source, target))


# Extended algebra, attitude, statistics, and conversion _ss.
def asin(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion arcsine."""
    return Quaternion(*_s.asin(q))


def acos(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion arccosine."""
    return Quaternion(*_s.acos(q))


def atan(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion arctangent."""
    return Quaternion(*_s.atan(q))


def asinh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal inverse hyperbolic sine."""
    return Quaternion(*_s.asinh(q))


def acosh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal inverse hyperbolic cosine."""
    return Quaternion(*_s.acosh(q))


def atanh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal inverse hyperbolic tangent."""
    return Quaternion(*_s.atanh(q))


def expm1(q: Iterable[FloatT]) -> Quaternion:
    """Return exp(q) - 1 with a stable small-argument path."""
    return Quaternion(*_s.expm1(q))


def log1p(q: Iterable[FloatT]) -> Quaternion:
    """Return log(1 + q) with a stable small-argument path."""
    return Quaternion(*_s.log1p(q))


def log2(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal logarithm to base two."""
    return Quaternion(*_s.log2(q))


def log10(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal logarithm to base ten."""
    return Quaternion(*_s.log10(q))


def power_quaternion(q: Iterable[FloatT], exponent: Iterable[FloatT]) -> Quaternion:
    """Raise q to a quaternion exponent using exp(exponent * log(q))."""
    return Quaternion(*_s.power_quaternion(q, exponent))


def commutator(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return lhs*rhs - rhs*lhs."""
    return Quaternion(*_s.commutator(lhs, rhs))


scalar_part = _s.scalar_part


def vector_part(q: Iterable[FloatT]) -> RotVec:
    """Return the three vector coefficients."""
    return RotVec(*_s.vector_part(q))


def quaternion_derivative(q: Iterable[FloatT], omega: Iterable[FloatT],
                          frame: str = "body") -> Quaternion:
    """Return dq/dt for angular velocity in radians per second."""
    return Quaternion(*_s.quaternion_derivative(q, omega, frame))


def integrate_angular_velocity_samples(
    q: Iterable[FloatT], omegas: Iterable[Iterable[FloatT]], times: Iterable[FloatT],
    frame: str = "body", method: str = "midpoint",
) -> Quaternion:
    """Integrate timestamped angular-rate samples to a final attitude."""
    return Quaternion(*_s.integrate_angular_velocity_samples(q, omegas, times, frame, method))


def boxplus(q: Iterable[FloatT], delta: Iterable[FloatT], frame: str = "body") -> Quaternion:
    """Apply a local radians increment in the body or world frame."""
    return Quaternion(*_s.boxplus(q, delta, frame))


def boxminus(q: Iterable[FloatT], reference: Iterable[FloatT],
             frame: str = "body") -> RotVec:
    """Return the local radians increment from reference to q."""
    return RotVec(*_s.boxminus(q, reference, frame))


def nlerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Interpolate rotations by normalized linear blending."""
    return Quaternion(*_s.nlerp(q0, q1, t, shortest))


def squad(q0: Iterable[FloatT], q1: Iterable[FloatT], a: Iterable[FloatT],
          b: Iterable[FloatT], t: FloatLike) -> Quaternion:
    """Evaluate spherical quadrangle interpolation."""
    return Quaternion(*_s.squad(q0, q1, a, b, t))


def align_sign(q: Iterable[FloatT], reference: Iterable[FloatT]) -> Quaternion:
    """Choose q's sign to be closest to reference."""
    return Quaternion(*_s.align_sign(q, reference))


def make_sign_continuous(quaternions: Iterable[Iterable[FloatT]]) -> tuple[Quaternion, ...]:
    """Choose consecutive signs so adjacent coefficient dots are nonnegative."""
    return tuple(Quaternion(*q) for q in _s.make_sign_continuous(quaternions))


def from_euler(angles: Iterable[FloatT], sequence: str = "xyz", *,
               degrees: bool = False) -> Quaternion:
    """Construct a rotation from angles and an Euler axis sequence."""
    return Quaternion(*_s.from_euler(angles, sequence, degrees=degrees))


def to_euler(q: Iterable[FloatT], sequence: str = "xyz",
             gimbal_tolerance: FloatLike = 1e-12, *,
             degrees: bool = False) -> RotVec:
    """Extract principal angles in the requested Euler sequence."""
    return RotVec(*_s.to_euler(q, sequence, degrees=degrees,
                               gimbal_tolerance=gimbal_tolerance))


def to_euler_with_status(q: Iterable[FloatT], sequence: str = "xyz", *,
                         degrees: bool = False,
                         gimbal_tolerance: FloatLike = 1e-12) -> EulerResult:
    result = _s.to_euler_with_status(q, sequence, degrees=degrees,
                                    gimbal_tolerance=gimbal_tolerance)
    return EulerResult(RotVec(*result.angles), result.gimbal_lock)


def from_mrp(sigma: Iterable[FloatT]) -> Quaternion:
    """Convert modified Rodrigues parameters to a unit rotation."""
    return Quaternion(*_s.from_mrp(sigma))


def to_mrp(q: Iterable[FloatT]) -> RotVec:
    """Convert a rotation to principal modified Rodrigues parameters."""
    return RotVec(*_s.to_mrp(q))


def swing_twist(q: Iterable[FloatT], axis: Iterable[FloatT]) -> tuple[Quaternion, Quaternion]:
    """Return the swing and twist factors with q = swing * twist."""
    swing, twist = _s.swing_twist(q, axis)
    return Quaternion(*swing), Quaternion(*twist)


def rotation_mean(quaternions: Iterable[Iterable[FloatT]],
                  weights: Iterable[FloatT] | None = None, method: str = "markley",
                  tolerance: FloatLike = 1e-12, max_iterations: int = 100) -> Quaternion:
    """Compute a weighted Markley or intrinsic rotation mean."""
    return Quaternion(*_s.rotation_mean(quaternions, weights, method, tolerance, max_iterations))


rotation_covariance = _s.rotation_covariance


def align_vectors(source: Iterable[Iterable[FloatT]], target: Iterable[Iterable[FloatT]],
                  weights: Iterable[FloatT] | None = None) -> Quaternion:
    """Solve weighted attitude alignment from vector correspondences."""
    return Quaternion(*_s.align_vectors(source, target, weights))


def random_quaternion(rng: object | None = None) -> Quaternion:
    """Draw a uniform unit quaternion from S3."""
    return Quaternion(*_s.random_quaternion(rng))


def random_rotation(rng: object | None = None) -> Quaternion:
    """Draw a uniform random SO(3) attitude."""
    return Quaternion(*_s.random_rotation(rng))


to_homogeneous_matrix = _s.to_homogeneous_matrix


def from_homogeneous_matrix(
    matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8
) -> Quaternion:
    """Validate a rigid 4x4 matrix and return its rotation."""
    return Quaternion(*_s.from_homogeneous_matrix(matrix, atol))


so3_adjoint = _s.so3_adjoint
