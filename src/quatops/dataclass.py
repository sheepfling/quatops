"""Quaternion operations using immutable, slotted dataclass values.

The ``Quaternion`` type stores scalar-first ``(w, x, y, z)`` values. Its
constructor stores the supplied components as-is; functions validate inputs
and return float-valued results. Numerical behavior matches ``quatops.scalar``.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import hypot, isfinite

from . import scalar as _scalar


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


def from_xyzw(q: Iterable[float]) -> Quaternion:
    """Convert scalar-last values to a scalar-first Quaternion."""
    x, y, z, w = _scalar._quaternion(q)
    return Quaternion(w, x, y, z)


def to_xyzw(q: Iterable[float]) -> _scalar.Quaternion:
    """Return scalar-last values as an ordinary tuple, not a Quaternion."""
    w, x, y, z = _scalar._quaternion(q)
    return x, y, z, w


def from_wxyz(q: Iterable[float]) -> Quaternion:
    """Construct a quaternion from scalar-first components."""
    return Quaternion(*_scalar._quaternion(q))


def to_wxyz(q: Iterable[float]) -> _scalar.Quaternion:
    """Return scalar-first components as a plain tuple."""
    return _scalar._quaternion(q)


def conjugate(q: Iterable[float]) -> Quaternion:
    """Return the quaternion conjugate without normalizing."""
    w, x, y, z = _scalar._quaternion(q)
    return Quaternion(w, -x, -y, -z)


def multiply(q: Iterable[float], r: Iterable[float]) -> Quaternion:
    """Return the Hamilton product ``q * r``; ``r`` acts first."""
    w, x, y, z = _scalar._quaternion(q)
    a, b, c, d = _scalar._quaternion(r)
    values = _scalar._checked((
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    ))
    return Quaternion(*values)


def norm(q: Iterable[float]) -> float:
    """Return the Euclidean quaternion norm without normalizing."""
    result = hypot(*_scalar._quaternion(q))
    if not isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def normalize(q: Iterable[float]) -> Quaternion:
    """Return a unit quaternion; reject zero and non-finite inputs."""
    _, parts = _scalar._scaled(_scalar._quaternion(q))
    length = hypot(*parts)
    return Quaternion(*(value / length for value in parts))


def inverse(q: Iterable[float]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scalar._scaled(_scalar._quaternion(q))
    w, x, y, z = parts
    denominator = sum(value * value for value in parts)
    inv_denominator = 1.0 / denominator
    values = tuple((value * inv_denominator) / scale for value in (w, -x, -y, -z))
    return Quaternion(*_scalar._checked(values))


def rotate(q: Iterable[float], v: Iterable[float]) -> _scalar.Vector:
    """Actively rotate ``v`` by ``q * (0, v) * inverse(q)``.

    The quaternion need not have unit norm. Internally scaling the components
    avoids unnecessary overflow without changing the input or its rotation.
    """
    _, parts = _scalar._scaled(_scalar._quaternion(q))
    w, x, y, z = parts
    vx, vy, vz = _scalar._vector(v)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    denominator = w2 + x2 + y2 + z2
    def coefficient(value: float) -> float:
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


def from_axis_angle(a: Iterable[float]) -> Quaternion:
    """Convert a nonzero axis and radians to a canonical unit quaternion."""
    x, y, z, angle = _scalar._components(a, 4, "axis-angle")
    return Quaternion(*_scalar._from_unit_axis(_scalar._axis((x, y, z)), angle))


def to_axis_angle(q: Iterable[float]) -> AxisAngle:
    """Extract canonical axis and radians from a finite nonzero quaternion."""
    axis, angle = _scalar._axis_angle_from_q(q)
    return AxisAngle(*axis, angle)


def from_rotvec(r: Iterable[float]) -> Quaternion:
    """Convert a rotation vector in radians to a canonical unit quaternion."""
    xyz = _scalar._vector(r)
    if max(abs(v) for v in xyz) == 0.0:
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    magnitude = hypot(*xyz)
    if not isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return Quaternion(*_scalar._from_unit_axis(_scalar._axis(xyz), magnitude))


def to_rotvec(q: Iterable[float]) -> RotVec:
    """Extract the canonical rotation vector in radians."""
    return RotVec(*_scalar._rotvec_from_q(q))
