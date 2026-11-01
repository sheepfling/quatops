"""Named-field scalar quaternions with the same operations as ``scalar``.

``Quaternion`` is an immutable, scalar-first tuple with named components.
Its constructor stores its arguments as supplied; operations validate finite
real components and return floats. No operation implicitly normalizes input.
"""

from collections.abc import Iterable
from math import hypot, isfinite
from typing import NamedTuple

from . import scalar


class Quaternion(NamedTuple):
    """A scalar-first quaternion, with fields ``w``, ``x``, ``y``, ``z``.

    Construction does not validate, convert, or normalize the supplied values.
    Operations validate and return float-valued components.
    """

    w: float
    x: float
    y: float
    z: float


def _quaternion(q: Iterable[float]) -> Quaternion:
    return Quaternion(*scalar._quaternion(q))


def _scaled(q: Quaternion) -> tuple[float, Quaternion]:
    scale = max(abs(value) for value in q)
    if scale == 0.0:
        raise ValueError("zero quaternion has no inverse or normalization")
    return scale, Quaternion(q.w / scale, q.x / scale, q.y / scale, q.z / scale)


def from_xyzw(q: Iterable[float]) -> Quaternion:
    """Convert scalar-last ``(x, y, z, w)`` to a named quaternion."""
    x, y, z, w = scalar._quaternion(q)
    return Quaternion(w, x, y, z)


def to_xyzw(q: Iterable[float]) -> tuple[float, float, float, float]:
    """Return scalar-last ``(x, y, z, w)`` as a plain tuple.

    This result is not a ``Quaternion``: its order does not match that type's
    named fields.
    """
    a = _quaternion(q)
    return a.x, a.y, a.z, a.w


def from_wxyz(q: Iterable[float]) -> Quaternion:
    """Construct a quaternion from scalar-first components."""
    return _quaternion(q)


def to_wxyz(q: Iterable[float]) -> tuple[float, float, float, float]:
    """Return scalar-first components as a plain tuple."""
    return scalar._quaternion(q)


def conjugate(q: Iterable[float]) -> Quaternion:
    """Return the conjugate, preserving the quaternion's magnitude."""
    a = _quaternion(q)
    return Quaternion(a.w, -a.x, -a.y, -a.z)


def multiply(q: Iterable[float], r: Iterable[float]) -> Quaternion:
    """Return the Hamilton product ``q * r`` (``r`` acts first)."""
    a, b = _quaternion(q), _quaternion(r)
    result = Quaternion(
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    )
    scalar._checked(result)
    return result


def norm(q: Iterable[float]) -> float:
    """Return the Euclidean norm without normalizing."""
    result = hypot(*_quaternion(q))
    if not isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def normalize(q: Iterable[float]) -> Quaternion:
    """Return a unit quaternion; reject zero, NaN, and infinity."""
    _, parts = _scaled(_quaternion(q))
    length = hypot(*parts)
    return Quaternion(parts.w / length, parts.x / length, parts.y / length, parts.z / length)


def inverse(q: Iterable[float]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scaled(_quaternion(q))
    denominator = sum(value * value for value in parts)
    result = Quaternion(
        (parts.w / denominator) / scale,
        (-parts.x / denominator) / scale,
        (-parts.y / denominator) / scale,
        (-parts.z / denominator) / scale,
    )
    scalar._checked(result)
    return result


def rotate(q: Iterable[float], v: Iterable[float]) -> scalar.Vector:
    """Actively rotate ``v`` by ``q * (0, v) * inverse(q)``.

    Accepts any finite nonzero quaternion; no input is normalized or mutated.
    """
    _, parts = _scaled(_quaternion(q))
    w, x, y, z = parts.w, parts.x, parts.y, parts.z
    vx, vy, vz = scalar._vector(v)
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
    values = (scalar._dot3(m00, m01, m02, vx, vy, vz),
              scalar._dot3(m10, m11, m12, vx, vy, vz),
              scalar._dot3(m20, m21, m22, vx, vy, vz))
    scalar._checked(values)
    return values


class AxisAngle(NamedTuple):
    """Scalar-first axis followed by the angle in radians; construction is unchecked."""

    ax: float
    ay: float
    az: float
    angle: float


class RotVec(NamedTuple):
    """Rotation vector: axis times angle in radians; construction is unchecked."""

    x: float
    y: float
    z: float


def from_axis_angle(a: Iterable[float]) -> Quaternion:
    """Convert a nonzero axis and radians to a canonical unit quaternion."""
    x, y, z, angle = scalar._components(a, 4, "axis-angle")
    return Quaternion(*scalar._from_unit_axis(scalar._axis((x, y, z)), angle))


def to_axis_angle(q: Iterable[float]) -> AxisAngle:
    """Extract canonical axis and radians from a finite nonzero quaternion."""
    axis, angle = scalar._axis_angle_from_q(q)
    return AxisAngle(*axis, angle)


def from_rotvec(r: Iterable[float]) -> Quaternion:
    """Convert a rotation vector in radians to a canonical unit quaternion."""
    xyz = scalar._vector(r)
    if max(abs(v) for v in xyz) == 0.0:
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    magnitude = hypot(*xyz)
    if not isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return Quaternion(*scalar._from_unit_axis(scalar._axis(xyz), magnitude))


def to_rotvec(q: Iterable[float]) -> RotVec:
    """Extract the canonical rotation vector in radians."""
    return RotVec(*scalar._rotvec_from_q(q))
