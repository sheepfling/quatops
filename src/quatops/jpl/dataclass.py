"""JPL scalar-last quaternions with named, immutable fields.

Algebra uses a negative cross product and active right-handed rotations.
Constructors retain input without validation; functions validate finite values.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import hypot, isfinite

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



def _quaternion(q: Iterable[float]) -> Quaternion:
    return Quaternion(*_s._quaternion(q))


def _scaled(q: Quaternion) -> tuple[float, Quaternion]:
    scale = max(abs(value) for value in q)
    if scale == 0.0:
        raise ValueError("zero quaternion has no inverse or normalization")
    return scale, Quaternion(q.x / scale, q.y / scale, q.z / scale, q.w / scale)


def from_xyzw(q: Iterable[float]) -> Quaternion:
    """Validate scalar-last layout without changing convention."""
    return _quaternion(q)


def to_xyzw(q: Iterable[float]) -> tuple[float, float, float, float]:
    """Return scalar-last components in a plain tuple."""
    p = _quaternion(q)
    return p.x, p.y, p.z, p.w


def from_wxyz(q: Iterable[float]) -> Quaternion:
    """Reorder scalar-first components to JPL scalar-last layout."""
    w, x, y, z = _s._quaternion(q)
    return Quaternion(x, y, z, w)


def to_wxyz(q: Iterable[float]) -> tuple[float, float, float, float]:
    """Reorder to scalar-first without changing convention."""
    p = _quaternion(q)
    return p.w, p.x, p.y, p.z


def from_hamilton(q: Iterable[float]) -> Quaternion:
    """Convert Hamilton wxyz to JPL xyzw without normalization."""
    w, x, y, z = _s._quaternion(q)
    return Quaternion(-x, -y, -z, w)


def to_hamilton(q: Iterable[float]) -> tuple[float, float, float, float]:
    """Convert JPL xyzw to Hamilton wxyz without normalization."""
    p = _quaternion(q)
    return p.w, -p.x, -p.y, -p.z


def conjugate(q: Iterable[float]) -> Quaternion:
    """Return the conjugate, preserving magnitude."""
    p = _quaternion(q)
    return Quaternion(-p.x, -p.y, -p.z, p.w)


def multiply(q: Iterable[float], r: Iterable[float]) -> Quaternion:
    """Return the negative-cross JPL product; r acts first."""
    p, t = _quaternion(q), _quaternion(r)
    result = Quaternion(
        p.w*t.x + p.x*t.w - p.y*t.z + p.z*t.y,
        p.w*t.y + p.y*t.w - p.z*t.x + p.x*t.z,
        p.w*t.z + p.z*t.w - p.x*t.y + p.y*t.x,
        p.w*t.w - p.x*t.x - p.y*t.y - p.z*t.z,
    )
    if not all(isfinite(value) for value in result):
        raise OverflowError("quaternion operation produced a non-finite result")
    return result


def norm(q: Iterable[float]) -> float:
    """Return Euclidean norm."""
    value = hypot(*_quaternion(q))
    if not isfinite(value):
        raise OverflowError("quaternion norm is too large for a float")
    return value


def normalize(q: Iterable[float]) -> Quaternion:
    """Return a unit quaternion, rejecting zero."""
    _, p = _scaled(_quaternion(q))
    length = hypot(*p)
    return Quaternion(p.x / length, p.y / length, p.z / length, p.w / length)


def inverse(q: Iterable[float]) -> Quaternion:
    """Return the multiplicative inverse, rejecting zero."""
    scale, p = _scaled(_quaternion(q))
    den = sum(value * value for value in p)
    out = Quaternion((-p.x / den) / scale, (-p.y / den) / scale,
                     (-p.z / den) / scale, (p.w / den) / scale)
    if not all(isfinite(value) for value in out):
        raise OverflowError("quaternion operation produced a non-finite result")
    return out


def rotate(q: Iterable[float], v: Iterable[float]) -> _s.Vector:
    """Actively rotate a vector with the JPL quaternion, without normalization."""
    _, p = _scaled(_quaternion(q))
    vx, vy, vz = _s._vector(v)
    x, y, z, w = p.x, p.y, p.z, p.w
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    den = w2 + x2 + y2 + z2
    def coefficient(value: float) -> float:
        return min(1.0, max(-1.0, value / den))
    m00 = coefficient(w2 + x2 - y2 - z2)
    m01 = coefficient(2*(x*y + w*z))
    m02 = coefficient(2*(x*z - w*y))
    m10 = coefficient(2*(x*y - w*z))
    m11 = coefficient(w2 - x2 + y2 - z2)
    m12 = coefficient(2*(y*z + w*x))
    m20 = coefficient(2*(x*z + w*y))
    m21 = coefficient(2*(y*z - w*x))
    m22 = coefficient(w2 - x2 - y2 + z2)
    out = (_s._dot3(m00, m01, m02, vx, vy, vz),
           _s._dot3(m10, m11, m12, vx, vy, vz),
           _s._dot3(m20, m21, m22, vx, vy, vz))
    _s._checked(out)
    return out


def from_axis_angle(a: Iterable[float]) -> Quaternion:
    """Normalize a nonzero axis and convert physical radians to JPL."""
    x, y, z, angle = _s._components(a, 4, "axis-angle")
    return Quaternion(*_s._from_unit_axis(_s._axis((x, y, z)), angle))


def to_axis_angle(q: Iterable[float]) -> AxisAngle:
    """Return canonical physical axis and principal radians."""
    axis, angle = _s._axis_angle_from_q(q)
    return AxisAngle(*axis, angle)


def from_rotvec(r: Iterable[float]) -> Quaternion:
    """Convert a physical rotation vector in radians to JPL."""
    v = _s._vector(r)
    if max(abs(value) for value in v) == 0:
        return Quaternion(0.0, 0.0, 0.0, 1.0)
    length = hypot(*v)
    if not isfinite(length):
        raise OverflowError("rotation-vector norm is too large for a float")
    return Quaternion(*_s._from_unit_axis(_s._axis(v), length))


def to_rotvec(q: Iterable[float]) -> RotVec:
    """Extract the canonical physical rotation vector in radians."""
    return RotVec(*_s._rotvec_from_q(q))
