"""Quaternion operations using immutable, slotted dataclass values.

The ``Quaternion`` type stores scalar-first ``(w, x, y, z)`` values. Its
constructor stores the supplied components as-is; functions validate inputs
and return float-valued results. Numerical behavior matches ``quatops.scalar``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import hypot, isfinite

from . import scalar as _scalar
from ._typing import FloatLike, FloatT


@dataclass(frozen=True)
class Quaternion:
    """A distinct immutable quaternion value with named scalar-first fields.

    Construction does not validate, convert, or normalize its fields. All
    operations in this module validate them when used. Iteration yields the
    fields in ``w, x, y, z`` order for interoperability with scalar functions.
    """

    # Explicit slots support Python 3.9, which lacks dataclass(slots=True).
    __slots__ = ("w", "x", "y", "z")

    w: float
    x: float
    y: float
    z: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.w, self.x, self.y, self.z))


def from_xyzw(q: Iterable[FloatT]) -> Quaternion:
    """Convert scalar-last values to a scalar-first Quaternion."""
    x, y, z, w = _scalar._quaternion(q)
    return Quaternion(w, x, y, z)


def to_xyzw(q: Iterable[FloatT]) -> _scalar.Quaternion:
    """Return scalar-last values as an ordinary tuple, not a Quaternion."""
    w, x, y, z = _scalar._quaternion(q)
    return x, y, z, w


def from_wxyz(q: Iterable[FloatT]) -> Quaternion:
    """Construct a quaternion from scalar-first components."""
    return Quaternion(*_scalar._quaternion(q))


def to_wxyz(q: Iterable[FloatT]) -> _scalar.Quaternion:
    """Return scalar-first components as a plain tuple."""
    return _scalar._quaternion(q)


def conjugate(q: Iterable[FloatT]) -> Quaternion:
    """Return the quaternion conjugate without normalizing."""
    w, x, y, z = _scalar._quaternion(q)
    return Quaternion(w, -x, -y, -z)


def add(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Add quaternion components and return a dataclass Quaternion."""
    w, x, y, z = _scalar._quaternion(lhs)
    a, b, c, d = _scalar._quaternion(rhs)
    return Quaternion(*_scalar._checked((w + a, x + b, y + c, z + d)))


def subtract(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Subtract quaternion components and return a dataclass Quaternion."""
    w, x, y, z = _scalar._quaternion(lhs)
    a, b, c, d = _scalar._quaternion(rhs)
    return Quaternion(*_scalar._checked((w - a, x - b, y - c, z - d)))


def negate(q: Iterable[FloatT]) -> Quaternion:
    """Negate every component and return a dataclass Quaternion."""
    w, x, y, z = _scalar._quaternion(q)
    return Quaternion(-w, -x, -y, -z)


def scale(q: Iterable[FloatT], factor: FloatLike) -> Quaternion:
    """Scale every component by a finite real number."""
    value = _scalar._real_scalar(factor, "factor")
    w, x, y, z = _scalar._quaternion(q)
    return Quaternion(*_scalar._checked((value*w, value*x, value*y, value*z)))


def multiply(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the Hamilton product ``lhs * rhs``; ``rhs`` acts first."""
    w, x, y, z = _scalar._quaternion(lhs)
    a, b, c, d = _scalar._quaternion(rhs)
    values = (
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    )
    return Quaternion(*_scalar._checked(values))


def norm(q: Iterable[FloatT]) -> float:
    """Return the Euclidean quaternion norm without normalizing."""
    result = hypot(*_scalar._quaternion(q))
    if not isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def normalize(q: Iterable[FloatT]) -> Quaternion:
    """Return a unit quaternion; reject zero and non-finite inputs."""
    _, parts = _scalar._scaled(_scalar._quaternion(q))
    length = hypot(*parts)
    return Quaternion(*(value / length for value in parts))


def inverse(q: Iterable[FloatT]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scalar._scaled(_scalar._quaternion(q))
    w, x, y, z = parts
    denominator = sum(value * value for value in parts)
    inv_denominator = 1.0 / denominator
    values = tuple((value * inv_denominator) / scale for value in (w, -x, -y, -z))
    return Quaternion(*_scalar._checked(values))


def divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)``."""
    return Quaternion(*_scalar.divide(lhs, rhs))


def rotate(q: Iterable[FloatT], v: Iterable[FloatT]) -> _scalar.Vector:
    """Actively rotate ``v`` by ``q * (0, v) * inverse(q)``.

    The quaternion need not have unit norm. Internally scaling the components
    avoids unnecessary overflow without changing the input or its rotation.
    """
    _, parts = _scalar._scaled(_scalar._quaternion(q))
    w, x, y, z = parts
    vx, vy, vz = _scalar._vector(v)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    denominator = w2 + x2 + y2 + z2
    def coefficient(value: FloatLike) -> float:
        return min(1.0, max(-1.0, value / denominator))

    m00 = coefficient(w2 + x2 - y2 - z2)
    m01 = coefficient(2*(x*y - w*z))
    m02 = coefficient(2*(x*z + w*y))
    m10 = coefficient(2*(x*y + w*z))
    m11 = coefficient(w2 - x2 + y2 - z2)
    m12 = coefficient(2*(y*z - w*x))
    m20 = coefficient(2*(x*z - w*y))
    m21 = coefficient(2*(y*z + w*x))
    m22 = coefficient(w2 - x2 - y2 + z2)
    values = (_scalar._dot3(m00, m01, m02, vx, vy, vz),
              _scalar._dot3(m10, m11, m12, vx, vy, vz),
              _scalar._dot3(m20, m21, m22, vx, vy, vz))
    _scalar._checked(values)
    return values


def to_rotation_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the active Hamilton rotation as a norm-aware 3x3 matrix."""
    return _scalar.to_rotation_matrix(q)


def to_left_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the Hamilton left-product matrix in scalar-first order."""
    return _scalar.to_left_matrix(q)


def to_right_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the Hamilton right-product matrix in scalar-first order."""
    return _scalar.to_right_matrix(q)

def sinh(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic hyperbolic sine as a dataclass Quaternion."""
    return Quaternion(*_scalar.sinh(q))


def cosh(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic hyperbolic cosine as a dataclass Quaternion."""
    return Quaternion(*_scalar.cosh(q))


def tanh(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic hyperbolic tangent as a dataclass Quaternion."""
    return Quaternion(*_scalar.tanh(q))


def power(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Return an integer or principal real power as a dataclass Quaternion."""
    return Quaternion(*_scalar.power(q, exponent))


def pow_real(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Return the principal real power as a dataclass Quaternion."""
    return Quaternion(*_scalar.pow_real(q, exponent))


def sqrt(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal square root as a dataclass Quaternion."""
    return Quaternion(*_scalar.sqrt(q))


def slerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Spherically interpolate rotations; allow extrapolation beyond [0, 1]."""
    return Quaternion(*_scalar.slerp(q0, q1, t, shortest=shortest))


def skew_matrix(v: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the 3x3 cross-product matrix for a three-vector."""
    return _scalar.skew_matrix(v)


def so3_left_jacobian(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the SO(3) left Jacobian for a physical rotation vector."""
    return _scalar.so3_left_jacobian(r)


def so3_right_jacobian(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the SO(3) right Jacobian for a physical rotation vector."""
    return _scalar.so3_right_jacobian(r)


def so3_left_jacobian_inverse(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the inverse SO(3) left Jacobian."""
    return _scalar.so3_left_jacobian_inverse(r)


def so3_right_jacobian_inverse(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the inverse SO(3) right Jacobian."""
    return _scalar.so3_right_jacobian_inverse(r)


def integrate_angular_velocity(q: Iterable[FloatT], omega: Iterable[FloatT], dt: FloatLike,
                               frame: str = "body") -> Quaternion:
    """Integrate constant body- or world-frame angular velocity exactly."""
    return Quaternion(*_scalar.integrate_angular_velocity(q, omega, dt, frame=frame))



@dataclass(frozen=True)
class AxisAngle:
    """Axis followed by angle in radians; construction is unchecked."""

    __slots__ = ("angle", "ax", "ay", "az")
    ax: float
    ay: float
    az: float
    angle: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.ax, self.ay, self.az, self.angle))


@dataclass(frozen=True)
class RotVec:
    """Rotation vector: axis times angle in radians; construction is unchecked."""

    __slots__ = ("x", "y", "z")
    x: float
    y: float
    z: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z))


def from_axis_angle(a: Iterable[FloatT]) -> Quaternion:
    """Convert a nonzero axis and radians to a canonical unit quaternion."""
    x, y, z, angle = _scalar._components(a, 4, "axis-angle")
    return Quaternion(*_scalar._from_unit_axis(_scalar._axis((x, y, z)), angle))


def to_axis_angle(q: Iterable[FloatT]) -> AxisAngle:
    """Extract canonical axis and radians from a finite nonzero quaternion."""
    axis, angle = _scalar._axis_angle_from_q(q)
    return AxisAngle(*axis, angle)


def from_rotvec(r: Iterable[FloatT]) -> Quaternion:
    """Convert a rotation vector in radians to a canonical unit quaternion."""
    xyz = _scalar._vector(r)
    if max(abs(v) for v in xyz) == 0.0:
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    magnitude = hypot(*xyz)
    if not isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return Quaternion(*_scalar._from_unit_axis(_scalar._axis(xyz), magnitude))


def to_rotvec(q: Iterable[FloatT]) -> RotVec:
    """Extract the canonical rotation vector in radians."""
    return RotVec(*_scalar._rotvec_from_q(q))


def exp(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion exponential as a named value."""
    return Quaternion(*_scalar.exp(q))


def log(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal algebraic quaternion logarithm as a named value."""
    return Quaternion(*_scalar.log(q))


def sin(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion sine as a named value."""
    return Quaternion(*_scalar.sin(q))


def cos(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion cosine as a named value."""
    return Quaternion(*_scalar.cos(q))


def from_rotation_vector(r: Iterable[FloatT]) -> Quaternion:
    """Convert a physical rotation vector in radians to a quaternion."""
    return from_rotvec(r)


def to_rotation_vector(q: Iterable[FloatT]) -> RotVec:
    """Extract a quaternion's canonical physical rotation vector in radians."""
    return to_rotvec(q)


def tan(q: Iterable[FloatT]) -> Quaternion:
    return Quaternion(*_scalar.tan(q))


def dot(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> float:
    return _scalar.dot(lhs, rhs)


def norm_squared(q: Iterable[FloatT]) -> float:
    return _scalar.norm_squared(q)


def right_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    return Quaternion(*_scalar.right_divide(lhs, rhs))


def left_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    return Quaternion(*_scalar.left_divide(lhs, rhs))


def rotation_distance(q0: Iterable[FloatT], q1: Iterable[FloatT]) -> float:
    return _scalar.rotation_distance(q0, q1)


def rotations_equivalent(q0: Iterable[FloatT], q1: Iterable[FloatT], atol: FloatLike = 1e-12) -> bool:
    return _scalar.rotations_equivalent(q0, q1, atol)


def canonicalize(q: Iterable[FloatT]) -> Quaternion:
    return Quaternion(*_scalar.canonicalize(q))


def from_rotation_matrix(matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8) -> Quaternion:
    return Quaternion(*_scalar.from_rotation_matrix(matrix, atol))


def from_two_vectors(source: Iterable[FloatT], target: Iterable[FloatT]) -> Quaternion:
    return Quaternion(*_scalar.from_two_vectors(source, target))


# Extended algebra, attitude, statistics, and conversion _scalars.
def asin(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion arcsine."""
    return Quaternion(*_scalar.asin(q))


def acos(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion arccosine."""
    return Quaternion(*_scalar.acos(q))


def atan(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion arctangent."""
    return Quaternion(*_scalar.atan(q))


def asinh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal inverse hyperbolic sine."""
    return Quaternion(*_scalar.asinh(q))


def acosh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal inverse hyperbolic cosine."""
    return Quaternion(*_scalar.acosh(q))


def atanh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal inverse hyperbolic tangent."""
    return Quaternion(*_scalar.atanh(q))


def expm1(q: Iterable[FloatT]) -> Quaternion:
    """Return exp(q) - 1 with a stable small-argument path."""
    return Quaternion(*_scalar.expm1(q))


def log1p(q: Iterable[FloatT]) -> Quaternion:
    """Return log(1 + q) with a stable small-argument path."""
    return Quaternion(*_scalar.log1p(q))


def log2(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal logarithm to base two."""
    return Quaternion(*_scalar.log2(q))


def log10(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal logarithm to base ten."""
    return Quaternion(*_scalar.log10(q))


def power_quaternion(q: Iterable[FloatT], exponent: Iterable[FloatT]) -> Quaternion:
    """Raise q to a quaternion exponent using exp(exponent * log(q))."""
    return Quaternion(*_scalar.power_quaternion(q, exponent))


def commutator(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return lhs*rhs - rhs*lhs."""
    return Quaternion(*_scalar.commutator(lhs, rhs))


def scalar_part(q: Iterable[FloatT]) -> float:
    """Return the scalar coefficient."""
    return _scalar.scalar_part(q)


def vector_part(q: Iterable[FloatT]) -> RotVec:
    """Return the three vector coefficients."""
    return RotVec(*_scalar.vector_part(q))


def quaternion_derivative(q: Iterable[FloatT], omega: Iterable[FloatT],
                          frame: str = "body") -> Quaternion:
    """Return dq/dt for angular velocity in radians per second."""
    return Quaternion(*_scalar.quaternion_derivative(q, omega, frame))


def integrate_angular_velocity_samples(
    q: Iterable[FloatT], omegas: Iterable[Iterable[FloatT]], times: Iterable[FloatT],
    frame: str = "body", method: str = "midpoint",
) -> Quaternion:
    """Integrate timestamped angular-rate samples to a final attitude."""
    return Quaternion(*_scalar.integrate_angular_velocity_samples(q, omegas, times, frame, method))


def boxplus(q: Iterable[FloatT], delta: Iterable[FloatT], frame: str = "body") -> Quaternion:
    """Apply a local radians increment in the body or world frame."""
    return Quaternion(*_scalar.boxplus(q, delta, frame))


def boxminus(q: Iterable[FloatT], reference: Iterable[FloatT],
             frame: str = "body") -> RotVec:
    """Return the local radians increment from reference to q."""
    return RotVec(*_scalar.boxminus(q, reference, frame))


def nlerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Interpolate rotations by normalized linear blending."""
    return Quaternion(*_scalar.nlerp(q0, q1, t, shortest))


def squad(q0: Iterable[FloatT], q1: Iterable[FloatT], a: Iterable[FloatT],
          b: Iterable[FloatT], t: FloatLike) -> Quaternion:
    """Evaluate spherical quadrangle interpolation."""
    return Quaternion(*_scalar.squad(q0, q1, a, b, t))


def align_sign(q: Iterable[FloatT], reference: Iterable[FloatT]) -> Quaternion:
    """Choose q's sign to be closest to reference."""
    return Quaternion(*_scalar.align_sign(q, reference))


def make_sign_continuous(quaternions: Iterable[Iterable[FloatT]]) -> tuple[Quaternion, ...]:
    """Choose consecutive signs so adjacent coefficient dots are nonnegative."""
    return tuple(Quaternion(*q) for q in _scalar.make_sign_continuous(quaternions))


def from_euler(angles: Iterable[FloatT], sequence: str = "xyz") -> Quaternion:
    """Construct a rotation from three radians and a Tait-Bryan sequence."""
    return Quaternion(*_scalar.from_euler(angles, sequence))


def to_euler(q: Iterable[FloatT], sequence: str = "xyz") -> RotVec:
    """Extract three principal radians in the requested Tait-Bryan sequence."""
    return RotVec(*_scalar.to_euler(q, sequence))


def from_mrp(sigma: Iterable[FloatT]) -> Quaternion:
    """Convert modified Rodrigues parameters to a unit rotation."""
    return Quaternion(*_scalar.from_mrp(sigma))


def to_mrp(q: Iterable[FloatT]) -> RotVec:
    """Convert a rotation to principal modified Rodrigues parameters."""
    return RotVec(*_scalar.to_mrp(q))


def swing_twist(q: Iterable[FloatT], axis: Iterable[FloatT]) -> tuple[Quaternion, Quaternion]:
    """Return the swing and twist factors with q = swing * twist."""
    swing, twist = _scalar.swing_twist(q, axis)
    return Quaternion(*swing), Quaternion(*twist)


def rotation_mean(quaternions: Iterable[Iterable[FloatT]],
                  weights: Iterable[FloatT] | None = None, method: str = "markley",
                  tolerance: FloatLike = 1e-12, max_iterations: int = 100) -> Quaternion:
    """Compute a weighted Markley or intrinsic rotation mean."""
    return Quaternion(*_scalar.rotation_mean(quaternions, weights, method, tolerance, max_iterations))


def rotation_covariance(quaternions: Iterable[Iterable[FloatT]],
                        mean: Iterable[FloatT] | None = None,
                        weights: Iterable[FloatT] | None = None,
                        frame: str = "body") -> tuple[tuple[float, ...], ...]:
    """Return weighted population covariance in a local rotation tangent."""
    return _scalar.rotation_covariance(quaternions, mean, weights, frame)


def align_vectors(source: Iterable[Iterable[FloatT]], target: Iterable[Iterable[FloatT]],
                  weights: Iterable[FloatT] | None = None) -> Quaternion:
    """Solve weighted attitude alignment from vector correspondences."""
    return Quaternion(*_scalar.align_vectors(source, target, weights))


def random_quaternion(rng: object | None = None) -> Quaternion:
    """Draw a uniform unit quaternion from S3."""
    return Quaternion(*_scalar.random_quaternion(rng))


def random_rotation(rng: object | None = None) -> Quaternion:
    """Draw a uniform random SO(3) attitude."""
    return Quaternion(*_scalar.random_rotation(rng))


def to_homogeneous_matrix(
    q: Iterable[FloatT], translation: Iterable[FloatT] = (0.0, 0.0, 0.0)
) -> tuple[tuple[float, ...], ...]:
    """Return a 4x4 homogeneous rotation and translation matrix."""
    return _scalar.to_homogeneous_matrix(q, translation)


def from_homogeneous_matrix(
    matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8
) -> Quaternion:
    """Validate a rigid 4x4 matrix and return its rotation."""
    return Quaternion(*_scalar.from_homogeneous_matrix(matrix, atol))


def so3_adjoint(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the SO(3) adjoint matrix."""
    return _scalar.so3_adjoint(q)


__all__ = [
    "AxisAngle",
    "Quaternion",
    "RotVec",
    "acos",
    "acosh",
    "add",
    "align_sign",
    "align_vectors",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "boxminus",
    "boxplus",
    "canonicalize",
    "commutator",
    "conjugate",
    "cos",
    "cosh",
    "divide",
    "dot",
    "exp",
    "expm1",
    "from_axis_angle",
    "from_euler",
    "from_homogeneous_matrix",
    "from_mrp",
    "from_rotation_matrix",
    "from_rotation_vector",
    "from_rotvec",
    "from_two_vectors",
    "from_wxyz",
    "from_xyzw",
    "integrate_angular_velocity",
    "integrate_angular_velocity_samples",
    "inverse",
    "left_divide",
    "log",
    "log1p",
    "log2",
    "log10",
    "make_sign_continuous",
    "multiply",
    "negate",
    "nlerp",
    "norm",
    "norm_squared",
    "normalize",
    "pow_real",
    "power",
    "power_quaternion",
    "quaternion_derivative",
    "random_quaternion",
    "random_rotation",
    "right_divide",
    "rotate",
    "rotation_covariance",
    "rotation_distance",
    "rotation_mean",
    "rotations_equivalent",
    "scalar_part",
    "scale",
    "sin",
    "sinh",
    "skew_matrix",
    "slerp",
    "so3_adjoint",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse",
    "so3_right_jacobian",
    "so3_right_jacobian_inverse",
    "sqrt",
    "squad",
    "subtract",
    "swing_twist",
    "tan",
    "tanh",
    "to_axis_angle",
    "to_euler",
    "to_homogeneous_matrix",
    "to_left_matrix",
    "to_mrp",
    "to_right_matrix",
    "to_rotation_matrix",
    "to_rotation_vector",
    "to_rotvec",
    "to_wxyz",
    "to_xyzw",
    "vector_part",
]
