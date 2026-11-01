"""Quaternion operations on scalar-first ``(w, x, y, z)`` tuples.

All operations return new tuples or floats. Inputs must contain finite real
numbers. Underflow follows Python float arithmetic; a non-finite result raises
``OverflowError``. Division-dependent operations reject the zero quaternion.
"""

import math
from collections.abc import Iterable
from math import fsum, hypot, isfinite
from numbers import Real

Quaternion = tuple[float, float, float, float]
Vector = tuple[float, float, float]
AxisAngle = tuple[float, float, float, float]
RotVec = Vector


def _components(values: Iterable[float], length: int, name: str) -> tuple:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of {length} real numbers") from exc
    if len(items) != length:
        raise ValueError(f"{name} must contain exactly {length} components")
    if not all(isinstance(value, Real) for value in items):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        result = tuple(float(value) for value in items)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _quaternion(q: Iterable[float]) -> Quaternion:
    return _components(q, 4, "quaternion")  # type: ignore[return-value]


def _vector(v: Iterable[float]) -> Vector:
    return _components(v, 3, "vector")  # type: ignore[return-value]


def _checked(values: tuple) -> tuple:
    if not all(isfinite(value) for value in values):
        raise OverflowError("quaternion operation produced a non-finite result")
    return values


def from_xyzw(q: Iterable[float]) -> Quaternion:
    """Convert scalar-last ``(x, y, z, w)`` to scalar-first order."""
    x, y, z, w = _quaternion(q)
    return w, x, y, z


def to_xyzw(q: Iterable[float]) -> Quaternion:
    """Convert scalar-first order to scalar-last ``(x, y, z, w)``."""
    w, x, y, z = _quaternion(q)
    return x, y, z, w


def from_wxyz(q: Iterable[float]) -> Quaternion:
    """Validate and copy scalar-first values; no convention conversion."""
    return _quaternion(q)


def to_wxyz(q: Iterable[float]) -> Quaternion:
    """Return scalar-first values as a tuple; no convention conversion."""
    return _quaternion(q)


def conjugate(q: Iterable[float]) -> Quaternion:
    """Return the conjugate, preserving the quaternion's magnitude."""
    w, x, y, z = _quaternion(q)
    return w, -x, -y, -z


def multiply(q: Iterable[float], r: Iterable[float]) -> Quaternion:
    """Return the Hamilton product ``q * r`` (``r`` acts first)."""
    w, x, y, z = _quaternion(q)
    a, b, c, d = _quaternion(r)
    return _checked((
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    ))  # type: ignore[return-value]


def norm(q: Iterable[float]) -> float:
    """Return the Euclidean norm, using a scaled hypotenuse calculation."""
    result = hypot(*_quaternion(q))
    if not isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def _scaled(q: Quaternion) -> tuple:
    scale = max(abs(value) for value in q)
    if scale == 0.0:
        raise ValueError("zero quaternion has no inverse or normalization")
    parts = tuple(value / scale for value in q)
    return scale, parts


def normalize(q: Iterable[float]) -> Quaternion:
    """Return a unit quaternion; reject zero, NaN, and infinity."""
    _, parts = _scaled(_quaternion(q))
    length = hypot(*parts)
    return tuple(value / length for value in parts)  # type: ignore[return-value]


def inverse(q: Iterable[float]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scaled(_quaternion(q))
    w, x, y, z = parts
    denominator = sum(value * value for value in parts)
    result = tuple((value / denominator) / scale for value in (w, -x, -y, -z))
    return _checked(result)  # type: ignore[return-value]


def _dot3(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    """Sum finite component products without avoidable intermediate overflow."""
    terms = [a*x, b*y, c*z]
    terms.sort(key=abs, reverse=True)
    # Add the largest opposite-signed terms first.  fsum otherwise can overflow
    # during its internal accumulation even when the final sum is representable.
    if ((terms[0] > 0.0 and terms[1] > 0.0 and terms[2] < 0.0)
            or (terms[0] < 0.0 and terms[1] < 0.0 and terms[2] > 0.0)):
        terms[1], terms[2] = terms[2], terms[1]
    return fsum(terms)


def rotate(q: Iterable[float], v: Iterable[float]) -> Vector:
    """Actively rotate ``v`` by ``q * (0, v) * inverse(q)``.

    Any nonzero quaternion is valid; its magnitude has no effect. No input is
    normalized. Internally scaling its components avoids unnecessary overflow.
    """
    _, parts = _scaled(_quaternion(q))
    w, x, y, z = parts
    vx, vy, vz = _vector(v)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    denominator = w2 + x2 + y2 + z2
    # Compute each component independently.  A shared vector scale can erase a
    # small component when another one is near the largest representable float.
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
    return _checked((_dot3(m00, m01, m02, vx, vy, vz),
                     _dot3(m10, m11, m12, vx, vy, vz),
                     _dot3(m20, m21, m22, vx, vy, vz)))  # type: ignore[return-value]


def _canonical(q: Quaternion) -> Quaternion:
    """Choose a deterministic quaternion sign without changing its rotation."""
    w, x, y, z = q
    flip = w < 0 or (w == 0 and next((c for c in (x, y, z) if c != 0), 1.0) < 0)
    return tuple(0.0 if c == 0.0 else (-c if flip else c) for c in (w, x, y, z))  # type: ignore[return-value]


def _axis(values: Vector) -> Vector:
    """Return the direction of a finite, nonzero vector using scaled values."""
    x, y, z = values
    s = max(abs(x), abs(y), abs(z))
    if s == 0:
        raise ValueError("axis must be nonzero")
    a, b, c = x / s, y / s, z / s
    length = math.hypot(a, b, c)
    return a / length, b / length, c / length


def _positive_zero(value: float) -> float:
    """Use the canonical positive sign for a floating-point zero."""
    return 0.0 if value == 0.0 else value


def _axis_angle_from_q(q: Iterable[float]) -> tuple[Vector, float]:
    """Extract canonical direction and principal angle without a rounded unit q."""
    w, x, y, z = _canonical(_quaternion(q))
    scale = max(abs(x), abs(y), abs(z))
    if scale == 0.0:
        if w == 0:
            raise ValueError("zero quaternion has no rotation")
        return (1.0, 0.0, 0.0), 0.0
    a, b, c = x / scale, y / scale, z / scale
    scaled_norm = math.hypot(a, b, c)
    # Normalization can underflow a negative component to -0.0. Canonical
    # outputs use positive zero, so fix the sign after the final division.
    axis = (_positive_zero(a / scaled_norm), _positive_zero(b / scaled_norm),
            _positive_zero(c / scaled_norm))
    # Small-angle path multiplies before dividing, preserving e.g. the
    # representable angle of (2, 5e-324, 0, 0), without needing a normalized q.
    if w > 0 and scale < w * 1e-8:
        # Combine mantissas at normal magnitude before the final exponent
        # adjustment to avoid repeatedly rounding subnormal intermediates.
        vector_mantissa, vector_exponent = math.frexp(scale)
        w_mantissa, w_exponent = math.frexp(w)
        angle = math.ldexp(
            (2.0 * (vector_mantissa / w_mantissa)) * scaled_norm,
            vector_exponent - w_exponent,
        )
    else:
        base = max(w, scale)
        angle = 2.0 * math.atan2((scale / base) * scaled_norm, w / base)
    if angle == 0.0:
        return (1.0, 0.0, 0.0), 0.0
    return axis, angle


def _from_unit_axis(axis: Vector, angle: float) -> Quaternion:
    """Create a canonical quaternion from a unit axis and angle in radians."""
    half = angle * 0.5
    sine = math.sin(half)
    cosine = 0.0 if abs(angle) == math.pi else math.cos(half)
    x, y, z = axis
    return _canonical((cosine, x * sine, y * sine, z * sine))


def from_axis_angle(a: Iterable[float]) -> Quaternion:
    """Normalize a nonzero axis and convert its radians to a unit quaternion."""
    x, y, z, angle = _components(a, 4, "axis-angle")
    return _from_unit_axis(_axis((x, y, z)), angle)


def to_axis_angle(q: Iterable[float]) -> AxisAngle:
    """Convert a finite nonzero quaternion to canonical axis and radians."""
    axis, angle = _axis_angle_from_q(q)
    return (*axis, angle)


def from_rotvec(r: Iterable[float]) -> Quaternion:
    """Convert an axis-times-angle vector in radians to a unit quaternion."""
    xyz = _vector(r)
    scale = max(abs(v) for v in xyz)
    if scale == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    magnitude = math.hypot(*xyz)
    if not math.isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return _from_unit_axis(_axis(xyz), magnitude)


def _rotvec_from_q(q: Iterable[float]) -> RotVec:
    """Extract a rotation vector without multiplying by a rounded tiny angle."""
    w, x, y, z = _canonical(_quaternion(q))
    scale = max(abs(x), abs(y), abs(z))
    if w > 0.0 and scale < w * 1e-8:
        w_mantissa, w_exponent = math.frexp(w)
        def component(value: float) -> float:
            if value == 0.0:
                return 0.0
            mantissa, exponent = math.frexp(value)
            result = math.ldexp(2.0 * (mantissa / w_mantissa), exponent - w_exponent)
            return 0.0 if result == 0.0 else result
        return component(x), component(y), component(z)
    axis, angle = _axis_angle_from_q((w, x, y, z))
    components = tuple(a * angle for a in axis)
    # Multiplication can itself underflow a negative value to -0.0.
    return tuple(0.0 if value == 0.0 else value for value in components)  # type: ignore[return-value]


def to_rotvec(q: Iterable[float]) -> RotVec:
    """Convert a nonzero quaternion to a canonical rotation vector in radians."""
    return _rotvec_from_q(q)
