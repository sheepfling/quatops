"""JPL scalar-first-vector quaternions as ``(x, y, z, w)`` tuples.

The product uses the negative cross term. Rotations are active and right-handed;
``multiply(q, r)`` composes rotations with ``r`` applied first. The equivalent
Hamilton quaternion for the same rotation is ``(w, -x, -y, -z)``.
"""

import math
from collections.abc import Iterable

from .. import scalar as _h

Quaternion = tuple[float, float, float, float]
Vector = tuple[float, float, float]
AxisAngle = tuple[float, float, float, float]
RotVec = Vector

# Shape checks and scalar floating-point helpers do not select a convention.
_components = _h._components
_vector = _h._vector
_checked = _h._checked
_dot3 = _h._dot3
_axis = _h._axis


def _quaternion(q: Iterable[float]) -> Quaternion:
    return _h._components(q, 4, "quaternion")  # type: ignore[return-value]


def _scaled(q: Quaternion) -> tuple[float, Quaternion]:
    scale = max(abs(value) for value in q)
    if scale == 0.0:
        raise ValueError("zero quaternion has no inverse or normalization")
    return scale, tuple(value / scale for value in q)  # type: ignore[return-value]


def from_xyzw(q: Iterable[float]) -> Quaternion:
    """Validate scalar-last layout; this only reorders, never changes algebra."""
    return _quaternion(q)


def to_xyzw(q: Iterable[float]) -> Quaternion:
    """Return scalar-last components without changing convention."""
    return _quaternion(q)


def from_wxyz(q: Iterable[float]) -> Quaternion:
    """Reorder scalar-first components into JPL scalar-last layout."""
    w, x, y, z = _quaternion(q)
    return x, y, z, w


def to_wxyz(q: Iterable[float]) -> Quaternion:
    """Reorder JPL components into scalar-first layout without changing algebra."""
    x, y, z, w = _quaternion(q)
    return w, x, y, z


def from_hamilton(q: Iterable[float]) -> Quaternion:
    """Convert physical Hamilton wxyz to JPL xyzw, preserving magnitude."""
    w, x, y, z = _quaternion(q)
    return -x, -y, -z, w


def to_hamilton(q: Iterable[float]) -> Quaternion:
    """Convert JPL xyzw to physical Hamilton wxyz, preserving magnitude."""
    x, y, z, w = _quaternion(q)
    return w, -x, -y, -z


def conjugate(q: Iterable[float]) -> Quaternion:
    """Return the conjugate without normalization."""
    x, y, z, w = _quaternion(q)
    return -x, -y, -z, w


def multiply(q: Iterable[float], r: Iterable[float]) -> Quaternion:
    """Return the negative-cross JPL product; ``r`` acts first."""
    x, y, z, w = _quaternion(q)
    a, b, c, d = _quaternion(r)
    return _checked((w*a + x*d - y*c + z*b,
                     w*b + y*d - z*a + x*c,
                     w*c + z*d - x*b + y*a,
                     w*d - x*a - y*b - z*c))  # type: ignore[return-value]


def norm(q: Iterable[float]) -> float:
    """Return the Euclidean quaternion norm."""
    result = math.hypot(*_quaternion(q))
    if not math.isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def normalize(q: Iterable[float]) -> Quaternion:
    """Return a unit quaternion, rejecting zero."""
    _, parts = _scaled(_quaternion(q))
    length = math.hypot(*parts)
    return tuple(value / length for value in parts)  # type: ignore[return-value]


def inverse(q: Iterable[float]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scaled(_quaternion(q))
    x, y, z, w = parts
    denominator = sum(value * value for value in parts)
    return _checked(tuple((v / denominator) / scale for v in (-x, -y, -z, w)))  # type: ignore[return-value]


def rotate(q: Iterable[float], v: Iterable[float]) -> Vector:
    """Return the active right-handed rotation ``q ⊙ (v,0) ⊙ inverse(q)``."""
    _, (x, y, z, w) = _scaled(_quaternion(q))
    vx, vy, vz = _vector(v)
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
    return _checked((_dot3(m00, m01, m02, vx, vy, vz),
                     _dot3(m10, m11, m12, vx, vy, vz),
                     _dot3(m20, m21, m22, vx, vy, vz)))  # type: ignore[return-value]


def _canonical(q: Quaternion) -> Quaternion:
    """Choose positive w; at exactly w=0 choose first nonzero xyz negative."""
    x, y, z, w = q
    first = next((c for c in (x, y, z) if c != 0), -1.0)
    flip = w < 0 or (w == 0 and first > 0)
    return tuple(0.0 if c == 0.0 else (-c if flip else c) for c in (x, y, z, w))  # type: ignore[return-value]


def _from_unit_axis(axis: Vector, angle: float) -> Quaternion:
    """Create a canonical JPL quaternion from a physical axis and radians."""
    half = angle * 0.5
    sine = math.sin(half)
    cosine = 0.0 if abs(angle) == math.pi else math.cos(half)
    x, y, z = axis
    return _canonical((-x*sine, -y*sine, -z*sine, cosine))


def _axis_angle_from_q(q: Iterable[float]) -> tuple[Vector, float]:
    """Extract canonical physical axis and angle without rounding a unit q."""
    x, y, z, w = _canonical(_quaternion(q))
    x, y, z = -x, -y, -z
    scale = max(abs(x), abs(y), abs(z))
    if scale == 0:
        if w == 0:
            raise ValueError("zero quaternion has no rotation")
        return (1.0, 0.0, 0.0), 0.0
    a, b, c = x / scale, y / scale, z / scale
    length = math.hypot(a, b, c)
    axis = tuple(_h._positive_zero(v / length) for v in (a, b, c))
    if w > 0 and scale < w * 1e-8:
        vm, ve = math.frexp(scale)
        wm, we = math.frexp(w)
        angle = math.ldexp((2.0 * (vm / wm)) * length, ve - we)
    else:
        base = max(w, scale)
        angle = 2.0 * math.atan2((scale / base) * length, w / base)
    if angle == 0:
        return (1.0, 0.0, 0.0), 0.0
    return axis, angle  # type: ignore[return-value]


def _rotvec_from_q(q: Iterable[float]) -> RotVec:
    x, y, z, w = _canonical(_quaternion(q))
    x, y, z = -x, -y, -z
    scale = max(abs(x), abs(y), abs(z))
    if w > 0 and scale < w * 1e-8:
        wm, we = math.frexp(w)
        def component(value: float) -> float:
            if value == 0:
                return 0.0
            m, e = math.frexp(value)
            return _h._positive_zero(math.ldexp(2.0 * (m / wm), e - we))
        return component(x), component(y), component(z)
    axis, angle = _axis_angle_from_q((-x, -y, -z, w))
    return tuple(_h._positive_zero(a * angle) for a in axis)  # type: ignore[return-value]


def from_axis_angle(a: Iterable[float]) -> Quaternion:
    """Explicitly normalize a nonzero axis and convert radians to JPL."""
    x, y, z, angle = _h._components(a, 4, "axis-angle")
    return _from_unit_axis(_axis((x, y, z)), angle)


def to_axis_angle(q: Iterable[float]) -> AxisAngle:
    """Return canonical physical axis and principal angle in radians."""
    axis, angle = _axis_angle_from_q(q)
    return (*axis, angle)


def from_rotvec(r: Iterable[float]) -> Quaternion:
    """Convert axis-times-angle radians to a canonical JPL quaternion."""
    v = _vector(r)
    if max(abs(x) for x in v) == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    magnitude = math.hypot(*v)
    if not math.isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return _from_unit_axis(_axis(v), magnitude)


def to_rotvec(q: Iterable[float]) -> RotVec:
    """Extract canonical physical rotation vector in radians."""
    return _rotvec_from_q(q)
