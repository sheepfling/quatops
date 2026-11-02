"""Tuple-based JPL quaternion operations using ``xyzw`` coefficient order.

The product uses the negative cross term. Rotations are active and right-handed;
``multiply(lhs, rhs)`` composes rotations with ``rhs`` applied first. The equivalent
Hamilton quaternion for the same rotation is ``(w, -x, -y, -z)``.
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Callable, Iterable
from numbers import Integral

from .. import _advanced
from .. import scalar as _h
from .._typing import FloatLike, FloatT

Quaternion = tuple[float, float, float, float]
Vector = tuple[float, float, float]
AxisAngle = tuple[float, float, float, float]
RotVec = Vector
EulerResult = _advanced.EulerResult

# Shape checks and scalar floating-point helpers do not select a convention.
_components = _h._components
_vector = _h._vector
_real_scalar = _h._real_scalar
_checked = _h._checked
_dot3 = _h._dot3
_axis = _h._axis
# These kernels operate only on coefficient components or physical 3-vectors,
# so their formulas are convention-independent. Re-export the function objects
# directly instead of adding a JPL forwarding frame.
dot = _h.dot
norm_squared = _h.norm_squared
skew_matrix = _h.skew_matrix
so3_left_jacobian = _h.so3_left_jacobian
so3_right_jacobian = _h.so3_right_jacobian
so3_left_jacobian_inverse = _h.so3_left_jacobian_inverse
so3_right_jacobian_inverse = _h.so3_right_jacobian_inverse
align_sign = _advanced.align_sign
make_sign_continuous = _advanced.make_sign_continuous
_frame = _advanced._frame
_euler_angles = _advanced._euler_angles
_euler_from_matrix = _advanced._euler_from_matrix
_sequence = _advanced._sequence
_weights = _advanced._weights
_jacobi_largest = _advanced._jacobi_largest


def _quaternion(q: Iterable[FloatT]) -> Quaternion:
    return _h._components(q, 4, "quaternion")  # type: ignore[return-value]


def exp(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion exponential in JPL xyzw order."""
    x, y, z, w = _quaternion(q)
    radius = math.hypot(x, y, z)
    try:
        scale = math.exp(w)
        cosine = math.cos(radius)
        sinc = 1.0 if radius == 0.0 else math.sin(radius) / radius
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion exponential is outside the float range") from exc
    return _checked((scale * x * sinc, scale * y * sinc, scale * z * sinc,
                     scale * cosine))  # type: ignore[return-value]


def log(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal algebraic quaternion logarithm in JPL xyzw order."""
    x, y, z, w = _quaternion(q)
    scale = max(abs(w), abs(x), abs(y), abs(z))
    if scale == 0.0:
        raise ValueError("zero quaternion has no logarithm")
    sw, sx, sy, sz = w / scale, x / scale, y / scale, z / scale
    norm_scaled = math.hypot(sw, sx, sy, sz)
    vector_scaled = math.hypot(sx, sy, sz)
    real = math.log(scale) + math.log(norm_scaled)
    if vector_scaled == 0.0:
        if sw < 0.0:
            return math.pi, 0.0, 0.0, real
        return 0.0, 0.0, 0.0, real
    angle = math.atan2(vector_scaled, sw)
    factor = angle / vector_scaled
    return _checked((sx * factor, sy * factor, sz * factor, real))  # type: ignore[return-value]


def sin(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion sine in JPL xyzw order."""
    x, y, z, w = _quaternion(q)
    radius = math.hypot(x, y, z)
    try:
        scalar = math.sin(w) * math.cosh(radius)
        ratio = 1.0 if radius == 0.0 else math.sinh(radius) / radius
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion sine is outside the float range") from exc
    vector_scale = math.cos(w) * ratio
    return _checked((x * vector_scale, y * vector_scale, z * vector_scale,
                     scalar))  # type: ignore[return-value]


def cos(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion cosine in JPL xyzw order."""
    x, y, z, w = _quaternion(q)
    radius = math.hypot(x, y, z)
    try:
        scalar = math.cos(w) * math.cosh(radius)
        ratio = 1.0 if radius == 0.0 else math.sinh(radius) / radius
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion cosine is outside the float range") from exc
    vector_scale = -math.sin(w) * ratio
    return _checked((x * vector_scale, y * vector_scale, z * vector_scale,
                     scalar))  # type: ignore[return-value]


def sinh(q: Iterable[FloatT]) -> Quaternion:
    """Return quaternion hyperbolic sine in JPL xyzw order."""
    x, y, z, w = _quaternion(q)
    radius = math.hypot(x, y, z)
    try:
        scalar = math.sinh(w) * math.cos(radius)
        ratio = 1.0 if radius == 0.0 else math.sin(radius) / radius
        vector_scale = math.cosh(w) * ratio
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion hyperbolic sine is outside the float range") from exc
    return _checked((x * vector_scale, y * vector_scale, z * vector_scale,
                     scalar))  # type: ignore[return-value]


def cosh(q: Iterable[FloatT]) -> Quaternion:
    """Return quaternion hyperbolic cosine in JPL xyzw order."""
    x, y, z, w = _quaternion(q)
    radius = math.hypot(x, y, z)
    try:
        scalar = math.cosh(w) * math.cos(radius)
        ratio = 1.0 if radius == 0.0 else math.sin(radius) / radius
        vector_scale = math.sinh(w) * ratio
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion hyperbolic cosine is outside the float range") from exc
    return _checked((x * vector_scale, y * vector_scale, z * vector_scale,
                     scalar))  # type: ignore[return-value]


def tanh(q: Iterable[FloatT]) -> Quaternion:
    """Return quaternion hyperbolic tangent in JPL xyzw order."""
    return divide(sinh(q), cosh(q))


def tan(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion tangent in JPL xyzw order."""
    return divide(sin(q), cos(q))


def power(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Raise a JPL quaternion to a real power on the principal branch.

    Algebraic powers preserve the coefficient branch used by ``log``; they do
    not convert the quaternion's rotation convention.
    """
    if type(exponent) is float:
        is_integer = False
    elif isinstance(exponent, Integral):
        is_integer = True
    elif isinstance(exponent, FloatLike):
        is_integer = False
    else:
        raise ValueError("exponent must be a finite real number")
    x, y, z, w = _quaternion(q)
    if is_integer:
        n = int(exponent)
        if n == 0:
            return 0.0, 0.0, 0.0, 1.0
        base = (x, y, z, w)
        if n < 0:
            base = inverse(base)
            n = -n
        result: Quaternion = (0.0, 0.0, 0.0, 1.0)
        while n:
            if n & 1:
                result = multiply(result, base)
            n >>= 1
            if n:
                base = multiply(base, base)
        return result
    if type(exponent) is float:
        value = exponent
        if not math.isfinite(value):
            raise ValueError("exponent must be a finite real number")
    else:
        try:
            value = float(exponent)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("exponent must be a finite real number") from exc
        if not math.isfinite(value):
            raise ValueError("exponent must be a finite real number")
    if not any((x, y, z, w)):
        if value == 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        if value > 0.0:
            return (0.0, 0.0, 0.0, 0.0)
        raise ValueError("zero quaternion cannot have a negative power")
    return exp(scale(log((x, y, z, w)), value))


def pow_real(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Alias for the principal real power of a JPL quaternion."""
    return power(q, exponent)


def sqrt(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL quaternion square root."""
    return power(q, 0.5)


def slerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Interpolate JPL attitudes by shortest-path spherical interpolation."""
    amount = t if type(t) is float else _real_scalar(t, "t")
    if not math.isfinite(amount):
        raise ValueError("t must be a finite real number")
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")  # noqa: TRY004
    a, b = normalize(q0), normalize(q1)
    dot = math.fsum(x * y for x, y in zip(a, b))
    if shortest and dot < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        blended = tuple((1.0 - amount) * x + amount * y for x, y in zip(a, b))
        return normalize(blended)
    angle = math.acos(dot)
    sine = math.sin(angle)
    if abs(sine) <= 1e-15:
        raise ValueError("interpolation path is not unique for antipodal endpoints")
    wa = math.sin((1.0 - amount) * angle) / sine
    wb = math.sin(amount * angle) / sine
    return normalize(tuple(wa * x + wb * y for x, y in zip(a, b)))


def integrate_angular_velocity(q: Iterable[FloatT], omega: Iterable[FloatT], dt: FloatLike,
                               frame: str = "body") -> Quaternion:
    """Integrate constant physical angular velocity in the requested frame."""
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    step = dt if type(dt) is float else _real_scalar(dt, "dt")
    if not math.isfinite(step):
        raise ValueError("dt must be a finite real number")
    vector = _vector(omega)
    increment_vector = _checked(tuple(component * step for component in vector))
    increment = from_rotation_vector(increment_vector)
    attitude = normalize(q)
    return normalize(multiply(attitude, increment) if frame == "body"
                     else multiply(increment, attitude))


def _scaled(q: Quaternion) -> tuple[float, Quaternion]:
    scale = max(abs(value) for value in q)
    if scale == 0.0:
        raise ValueError("zero quaternion has no inverse or normalization")
    return scale, tuple(value / scale for value in q)  # type: ignore[return-value]


def from_xyzw(q: Iterable[FloatT]) -> Quaternion:
    """Validate scalar-last layout; this only reorders, never changes algebra."""
    return _quaternion(q)


def to_xyzw(q: Iterable[FloatT]) -> Quaternion:
    """Return scalar-last components without changing convention."""
    return _quaternion(q)


def from_wxyz(q: Iterable[FloatT]) -> Quaternion:
    """Reorder scalar-first components into JPL scalar-last layout."""
    w, x, y, z = _quaternion(q)
    return x, y, z, w


def to_wxyz(q: Iterable[FloatT]) -> Quaternion:
    """Reorder JPL components into scalar-first layout without changing algebra."""
    x, y, z, w = _quaternion(q)
    return w, x, y, z


def from_hamilton(q: Iterable[FloatT]) -> Quaternion:
    """Convert physical Hamilton wxyz to JPL xyzw, preserving magnitude."""
    w, x, y, z = _quaternion(q)
    return -x, -y, -z, w


def to_hamilton(q: Iterable[FloatT]) -> Quaternion:
    """Convert JPL xyzw to physical Hamilton wxyz, preserving magnitude."""
    x, y, z, w = _quaternion(q)
    return w, -x, -y, -z


def conjugate(q: Iterable[FloatT]) -> Quaternion:
    """Return the conjugate without normalization."""
    x, y, z, w = _quaternion(q)
    return -x, -y, -z, w


def add(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Add JPL quaternion components independently."""
    x, y, z, w = _quaternion(lhs)
    a, b, c, d = _quaternion(rhs)
    return _checked((x + a, y + b, z + c, w + d))  # type: ignore[return-value]


def subtract(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Subtract JPL quaternion components independently."""
    x, y, z, w = _quaternion(lhs)
    a, b, c, d = _quaternion(rhs)
    return _checked((x - a, y - b, z - c, w - d))  # type: ignore[return-value]


def negate(q: Iterable[FloatT]) -> Quaternion:
    """Negate every JPL quaternion component."""
    return tuple(-value for value in _quaternion(q))  # type: ignore[return-value]


def scale(q: Iterable[FloatT], factor: FloatLike) -> Quaternion:
    """Scale every JPL quaternion component by a finite real number."""
    if type(factor) is float:
        value = factor
        if not math.isfinite(value):
            raise ValueError("factor must be a finite real number")
    else:
        value = _real_scalar(factor, "factor")
    x, y, z, w = _quaternion(q)
    return _checked((value*x, value*y, value*z, value*w))  # type: ignore[return-value]


def multiply(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the negative-cross JPL product; ``rhs`` acts first."""
    x, y, z, w = _quaternion(lhs)
    a, b, c, d = _quaternion(rhs)
    return _checked((w*a + x*d - y*c + z*b,
                     w*b + y*d - z*a + x*c,
                     w*c + z*d - x*b + y*a,
                     w*d - x*a - y*b - z*c))  # type: ignore[return-value]


def norm(q: Iterable[FloatT]) -> float:
    """Return the Euclidean quaternion norm."""
    result = math.hypot(*_quaternion(q))
    if not math.isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def normalize(q: Iterable[FloatT]) -> Quaternion:
    """Return a unit quaternion, rejecting zero."""
    _, parts = _scaled(_quaternion(q))
    length = math.hypot(*parts)
    return tuple(value / length for value in parts)  # type: ignore[return-value]


def inverse(q: Iterable[FloatT]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scaled(_quaternion(q))
    x, y, z, w = parts
    denominator = sum(value * value for value in parts)
    return _checked(tuple((v / denominator) / scale for v in (-x, -y, -z, w)))  # type: ignore[return-value]


def divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)`` in JPL algebra."""
    return multiply(lhs, inverse(rhs))


def right_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)`` in JPL algebra."""
    return divide(lhs, rhs)


def left_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the left quotient ``inverse(lhs) * rhs`` in JPL algebra."""
    return multiply(inverse(lhs), rhs)


def rotation_distance(q0: Iterable[FloatT], q1: Iterable[FloatT]) -> float:
    """Return the shortest physical attitude distance in radians."""
    a, b = normalize(q0), normalize(q1)
    relative = normalize(multiply(inverse(a), b))
    x, y, z, w = relative
    return 2.0 * math.atan2(math.hypot(x, y, z), abs(w))


def rotations_equivalent(q0: Iterable[FloatT], q1: Iterable[FloatT],
                         atol: FloatLike = 1e-12) -> bool:
    """Return whether two JPL rotations differ by at most ``atol`` radians."""
    tolerance = _real_scalar(atol, "atol")
    if tolerance < 0.0:
        raise ValueError("atol must be non-negative")
    return rotation_distance(q0, q1) <= tolerance


def canonicalize(q: Iterable[FloatT]) -> Quaternion:
    """Choose a deterministic sign while preserving JPL quaternion magnitude."""
    return _canonical(_quaternion(q))


def rotate(q: Iterable[FloatT], v: Iterable[FloatT]) -> Vector:
    """Return the active right-handed rotation ``q ⊙ (v,0) ⊙ inverse(q)``."""
    _, (x, y, z, w) = _scaled(_quaternion(q))
    vx, vy, vz = _vector(v)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    den = w2 + x2 + y2 + z2
    m00 = min(1.0, max(-1.0, (w2 + x2 - y2 - z2) / den))
    m01 = min(1.0, max(-1.0, (2*(x*y + w*z)) / den))
    m02 = min(1.0, max(-1.0, (2*(x*z - w*y)) / den))
    m10 = min(1.0, max(-1.0, (2*(x*y - w*z)) / den))
    m11 = min(1.0, max(-1.0, (w2 - x2 + y2 - z2) / den))
    m12 = min(1.0, max(-1.0, (2*(y*z + w*x)) / den))
    m20 = min(1.0, max(-1.0, (2*(x*z + w*y)) / den))
    m21 = min(1.0, max(-1.0, (2*(y*z - w*x)) / den))
    m22 = min(1.0, max(-1.0, (w2 - x2 - y2 + z2) / den))
    return _checked((_dot3(m00, m01, m02, vx, vy, vz),
                     _dot3(m10, m11, m12, vx, vy, vz),
                     _dot3(m20, m21, m22, vx, vy, vz)))  # type: ignore[return-value]


def to_rotation_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the norm-aware 3x3 active JPL rotation matrix."""
    _, (x, y, z, w) = _scaled(_quaternion(q))
    d = w*w + x*x + y*y + z*z
    rows = (((w*w+x*x-y*y-z*z)/d, 2*(x*y+w*z)/d, 2*(x*z-w*y)/d),
            (2*(x*y-w*z)/d, (w*w-x*x+y*y-z*z)/d, 2*(y*z+w*x)/d),
            (2*(x*z+w*y)/d, 2*(y*z-w*x)/d, (w*w-x*x-y*y+z*z)/d))
    return tuple(tuple(_checked((v,))[0] for v in row) for row in rows)


def from_rotation_matrix(
    matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8
) -> Quaternion:
    """Convert an active right-handed matrix to a canonical unit JPL q."""
    tolerance = _real_scalar(atol, "atol")
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("atol must be in [0, 1)")
    try:
        rows = tuple(_vector(row) for row in matrix)
    except TypeError as exc:
        raise ValueError("matrix must be a 3x3 real array") from exc
    if len(rows) != 3:
        raise ValueError("matrix must have shape (3, 3)")
    r = rows
    if max(abs(value) for row in r for value in row) > 1.0 + tolerance:
        raise ValueError("matrix must be a proper orthonormal rotation")
    errors = [
        abs(math.fsum(r[k][i] * r[k][j] for k in range(3))
            - (1.0 if i == j else 0.0))
        for i in range(3) for j in range(3)
    ]
    determinant = math.fsum((
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]),
        -r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]),
        r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]),
    ))
    if max(errors) > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError("matrix must be a proper orthonormal rotation")

    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x, y, z = (0.25 * s, (r[2][1] - r[1][2]) / s,
                      (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    elif r[0][0] >= r[1][1] and r[0][0] >= r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2])
        w, x, y, z = ((r[2][1] - r[1][2]) / s, 0.25 * s,
                      (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s)
    elif r[1][1] >= r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2])
        w, x, y, z = ((r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s,
                      0.25 * s, (r[1][2] + r[2][1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1])
        w, x, y, z = ((r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s,
                      (r[1][2] + r[2][1]) / s, 0.25 * s)
    return _canonical(normalize((-x, -y, -z, w)))


def to_left_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the 4x4 matrix L(q) satisfying L(q) r = q⊙r (xyzw order)."""
    x, y, z, w = _quaternion(q)
    return ((w,z,-y,x),(-z,w,x,y),(y,-x,w,z),(-x,-y,-z,w))


def to_right_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the 4x4 matrix P(q) satisfying P(q) r = r⊙q (xyzw order)."""
    x, y, z, w = _quaternion(q)
    return ((w,-z,y,x),(z,w,-x,y),(-y,x,w,z),(-x,-y,-z,w))


def _canonical(q: Quaternion) -> Quaternion:
    """Choose positive w; at exactly w=0 choose first nonzero xyz negative."""
    x, y, z, w = q
    flip = w < 0
    if w == 0:
        first = next((c for c in (x, y, z) if c != 0), -1.0)
        flip = first > 0
    return tuple(0.0 if c == 0.0 else (-c if flip else c) for c in (x, y, z, w))  # type: ignore[return-value]


def _from_unit_axis(axis: Vector, angle: FloatLike) -> Quaternion:
    """Create a canonical JPL quaternion from a physical axis and radians."""
    half = angle * 0.5
    sine = math.sin(half)
    cosine = 0.0 if abs(angle) == math.pi else math.cos(half)
    x, y, z = axis
    return _canonical((-x*sine, -y*sine, -z*sine, cosine))


def _axis_angle_from_q(q: Iterable[FloatT]) -> tuple[Vector, float]:
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


def _rotvec_from_q(q: Iterable[FloatT]) -> RotVec:
    x, y, z, w = _canonical(_quaternion(q))
    x, y, z = -x, -y, -z
    scale = max(abs(x), abs(y), abs(z))
    if w > 0 and scale < w * 1e-8:
        wm, we = math.frexp(w)
        def component(value: FloatLike) -> float:
            if value == 0:
                return 0.0
            m, e = math.frexp(value)
            return _h._positive_zero(math.ldexp(2.0 * (m / wm), e - we))
        return component(x), component(y), component(z)
    axis, angle = _axis_angle_from_q((-x, -y, -z, w))
    return tuple(_h._positive_zero(a * angle) for a in axis)  # type: ignore[return-value]


def from_axis_angle(a: Iterable[FloatT]) -> Quaternion:
    """Explicitly normalize a nonzero axis and convert radians to JPL."""
    x, y, z, angle = _h._components(a, 4, "axis-angle")
    return _from_unit_axis(_axis((x, y, z)), angle)


def to_axis_angle(q: Iterable[FloatT]) -> AxisAngle:
    """Return canonical physical axis and principal angle in radians."""
    axis, angle = _axis_angle_from_q(q)
    return (*axis, angle)


def from_rotvec(r: Iterable[FloatT]) -> Quaternion:
    """Convert axis-times-angle radians to a canonical JPL quaternion."""
    v = _vector(r)
    if max(abs(x) for x in v) == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    magnitude = math.hypot(*v)
    if not math.isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return _from_unit_axis(_axis(v), magnitude)


def from_two_vectors(source: Iterable[FloatT], target: Iterable[FloatT]) -> Quaternion:
    """Return the shortest active JPL rotation mapping source onto target."""
    a, b = _axis(_vector(source)), _axis(_vector(target))
    cosine = min(1.0, max(-1.0, math.fsum(x * y for x, y in zip(a, b))))
    cross = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
    if cosine < 0.0 and cross == (0.0, 0.0, 0.0):
        basis_index = min(range(3), key=lambda index: abs(a[index]))
        basis = tuple(1.0 if index == basis_index else 0.0 for index in range(3))
        axis = _axis((a[1] * basis[2] - a[2] * basis[1],
                      a[2] * basis[0] - a[0] * basis[2],
                      a[0] * basis[1] - a[1] * basis[0]))
        return canonicalize((-axis[0], -axis[1], -axis[2], 0.0))
    return canonicalize(normalize((-cross[0], -cross[1], -cross[2], 1.0 + cosine)))


def to_rotvec(q: Iterable[FloatT]) -> RotVec:
    """Extract canonical physical rotation vector in radians."""
    return _rotvec_from_q(q)


def from_rotation_vector(r: Iterable[FloatT]) -> Quaternion:
    """Convert a physical rotation vector in radians to canonical JPL xyzw."""
    return from_rotvec(r)


def to_rotation_vector(q: Iterable[FloatT]) -> RotVec:
    """Extract the canonical physical rotation vector in radians from JPL xyzw."""
    return to_rotvec(q)


def asin(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL coefficient arcsine."""
    return _complex_slice(q, cmath.asin)


def acos(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL coefficient arccosine."""
    return _complex_slice(q, cmath.acos)


def atan(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL coefficient arctangent."""
    return _complex_slice(q, cmath.atan)


def asinh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL coefficient inverse hyperbolic sine."""
    return _complex_slice(q, cmath.asinh)


def acosh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL coefficient inverse hyperbolic cosine."""
    return _complex_slice(q, cmath.acosh)


def atanh(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL coefficient inverse hyperbolic tangent."""
    return _complex_slice(q, cmath.atanh)


def _complex_slice(q: Iterable[FloatT],
                   function: Callable[[complex], complex]) -> Quaternion:
    """Lift a complex function along the JPL coefficient-vector direction."""
    x, y, z, w = _quaternion(q)
    radius = math.hypot(x, y, z)
    value = function(complex(w, radius))
    if radius == 0.0:
        return _checked((-value.imag, 0.0, 0.0, value.real))  # type: ignore[return-value]
    factor = value.imag / radius
    return _checked((factor * x, factor * y, factor * z, value.real))  # type: ignore[return-value]


def expm1(q: Iterable[FloatT]) -> Quaternion:
    """Return ``exp(q)-1``, using a convergent series near zero."""
    value = _quaternion(q)
    if max(map(abs, value)) >= 0.5:
        result = exp(value)
        return result[0], result[1], result[2], result[3] - 1.0
    total = [0.0, 0.0, 0.0, 0.0]
    term: Quaternion = (0.0, 0.0, 0.0, 1.0)
    for index in range(1, 48):
        term = scale(multiply(term, value), 1.0 / index)
        for component in range(4):
            total[component] += term[component]
        if max(map(abs, term)) <= 2e-17 * max(1.0, max(map(abs, total))):
            break
    return _checked(tuple(total))  # type: ignore[return-value]


def log1p(q: Iterable[FloatT]) -> Quaternion:
    """Return ``log(1+q)``, using a convergent series near zero."""
    value = _quaternion(q)
    if max(map(abs, value)) >= 0.25:
        return log((value[0], value[1], value[2], 1.0 + value[3]))
    total = [0.0, 0.0, 0.0, 0.0]
    term: Quaternion = value
    for index in range(1, 96):
        factor = (1.0 if index & 1 else -1.0) / index
        for component in range(4):
            total[component] += factor * term[component]
        if max(map(abs, term)) <= 2e-17 * max(1.0, max(map(abs, total))):
            break
        term = multiply(term, value)
    return _checked(tuple(total))  # type: ignore[return-value]


def log2(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL quaternion logarithm to base two."""
    return scale(log(q), 1.0 / math.log(2.0))


def log10(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal JPL quaternion logarithm to base ten."""
    return scale(log(q), 1.0 / math.log(10.0))


def power_quaternion(q: Iterable[FloatT], exponent: Iterable[FloatT]) -> Quaternion:
    """Raise ``q`` to a quaternion exponent as ``exp(exponent ⊙ log(q))``."""
    base, power_value = _quaternion(q), _quaternion(exponent)
    if not any(power_value):
        return (0.0, 0.0, 0.0, 1.0)
    if not any(base):
        raise ValueError("zero quaternion cannot have a nonzero quaternion power")
    return exp(multiply(power_value, log(base)))


def commutator(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the JPL algebra commutator ``lhs⊙rhs - rhs⊙lhs``."""
    return subtract(multiply(lhs, rhs), multiply(rhs, lhs))


def scalar_part(q: Iterable[FloatT]) -> float:
    """Return the real/scalar coefficient of a Hamilton ``wxyz`` value."""
    return _quaternion(q)[3]


def vector_part(q: Iterable[FloatT]) -> Vector:
    """Return the three imaginary coefficients of a Hamilton ``wxyz`` value."""
    return _quaternion(q)[:3]


def quaternion_derivative(q: Iterable[FloatT], omega: Iterable[FloatT],
                          frame: str = "body") -> Quaternion:
    """Return ``dq/dt`` for a right-handed physical angular rate."""
    side = _frame(frame)
    wx, wy, wz = _vector(omega)
    pure_rate = (-wx, -wy, -wz, 0.0)
    product = (multiply(q, pure_rate) if side == "body"
               else multiply(pure_rate, q))
    return scale(product, 0.5)


def integrate_angular_velocity_samples(
    q: Iterable[FloatT], omegas: Iterable[Iterable[FloatT]], times: Iterable[FloatT],
    frame: str = "body", method: str = "midpoint",
) -> Quaternion:
    """Integrate sampled rates with exact constant-rate increments."""
    side = _frame(frame)
    if method not in ("left", "midpoint"):
        raise ValueError("method must be 'left' or 'midpoint'")
    samples = tuple(_vector(omega) for omega in omegas)
    stamps = tuple(_real_scalar(value, "time") for value in times)
    if not samples or len(samples) != len(stamps):
        raise ValueError("omegas and times must have the same nonzero length")
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("times must be strictly increasing")
    result = normalize(q)
    for index, (left, right) in enumerate(zip(samples, samples[1:])):
        if method == "left":
            rate = left
        else:
            rate = ((left[0] + right[0]) * 0.5,
                    (left[1] + right[1]) * 0.5,
                    (left[2] + right[2]) * 0.5)
        result = integrate_angular_velocity(
            result, rate, stamps[index + 1] - stamps[index], frame=side
        )
    return normalize(result)


def boxplus(q: Iterable[FloatT], delta: Iterable[FloatT], frame: str = "body") -> Quaternion:
    """Apply a local physical rotation-vector increment to an attitude."""
    side = _frame(frame)
    base = normalize(q)
    increment = from_rotvec(delta)
    result = (multiply(base, increment) if side == "body"
              else multiply(increment, base))
    return normalize(result)


def boxminus(q: Iterable[FloatT], reference: Iterable[FloatT],
             frame: str = "body") -> RotVec:
    """Return the physical local rotation vector from ``reference`` to ``q``."""
    side = _frame(frame)
    target, base = normalize(q), normalize(reference)
    relative = (multiply(inverse(base), target) if side == "body"
                else multiply(target, inverse(base)))
    return to_rotvec(relative)


def nlerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Normalized linear interpolation of JPL rotations."""
    amount = _real_scalar(t, "t")
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")  # noqa: TRY004
    a, b = normalize(q0), normalize(q1)
    if shortest and dot(a, b) < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
    return normalize(tuple((1.0 - amount) * x + amount * y
                           for x, y in zip(a, b)))  # type: ignore[return-value]


def squad(q0: Iterable[FloatT], q1: Iterable[FloatT], a: Iterable[FloatT],
          b: Iterable[FloatT], t: FloatLike) -> Quaternion:
    """Evaluate JPL SQUAD from endpoints and their control quaternions."""
    amount = _real_scalar(t, "t")
    first = slerp(q0, q1, amount)
    second = slerp(a, b, amount)
    return slerp(first, second, 2.0 * amount * (1.0 - amount))


def from_euler(
    angles: Iterable[FloatT], sequence: str = "xyz", *, degrees: bool = False
) -> Quaternion:
    """Construct a JPL rotation from extrinsic or intrinsic Euler angles."""
    values = _vector(angles)
    intrinsic, i, j, k, ell, epsilon, proper = _advanced._euler_sequence_metadata(sequence)
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    if degrees:
        values = tuple(math.radians(value) for value in values)  # type: ignore[assignment]
    if not intrinsic:
        values = tuple(reversed(values))  # type: ignore[assignment]
    return _advanced._euler_from_validated_angles(
        *values, i, j, k, ell, epsilon, proper, algebra_epsilon=-1,
        scalar_last=True,
    )


def to_euler(
    q: Iterable[FloatT], sequence: str = "xyz",
    gimbal_tolerance: FloatLike = 1e-12, *, degrees: bool = False,
) -> RotVec:
    """Extract principal Euler angles from an active JPL rotation."""
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    x, y, z, w = _quaternion(q)
    angles = _advanced._euler_from_components(w, -x, -y, -z, sequence,
                                               gimbal_tolerance)
    if degrees:
        return tuple(math.degrees(value) for value in angles)  # type: ignore[arg-type,return-value]
    return angles  # type: ignore[return-value]


def to_euler_with_status(
    q: Iterable[FloatT], sequence: str = "xyz", *, degrees: bool = False,
    gimbal_tolerance: FloatLike = 1e-12,
) -> _advanced.EulerResult:
    """Extract Euler angles and the gimbal-lock flag from an active JPL rotation."""
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    x, y, z, w = _quaternion(q)
    result = _advanced._euler_from_components(w, -x, -y, -z, sequence,
                                               gimbal_tolerance,
                                               with_status=True)
    assert isinstance(result, _advanced.EulerResult)
    if not degrees:
        return result
    return _advanced.EulerResult(
        tuple(math.degrees(value) for value in result.angles), result.gimbal_lock
    )


def from_mrp(sigma: Iterable[FloatT]) -> Quaternion:
    """Convert modified Rodrigues parameters to a unit JPL quaternion."""
    x, y, z = _vector(sigma)
    scale_value = max(abs(x), abs(y), abs(z))
    if scale_value == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    if scale_value <= 1.0:
        norm2 = x*x + y*y + z*z
        denominator = 1.0 + norm2
        return canonicalize(normalize((-2.0*x/denominator, -2.0*y/denominator,
                                       -2.0*z/denominator,
                                       (1.0 - norm2)/denominator)))
    sx, sy, sz = x / scale_value, y / scale_value, z / scale_value
    inverse_scale = 1.0 / scale_value
    norm2 = sx*sx + sy*sy + sz*sz
    denominator = inverse_scale*inverse_scale + norm2
    return canonicalize(normalize((
        (-2.0 * inverse_scale * sx) / denominator,
        (-2.0 * inverse_scale * sy) / denominator,
        (-2.0 * inverse_scale * sz) / denominator,
        (inverse_scale*inverse_scale - norm2) / denominator,
    )))


def to_mrp(q: Iterable[FloatT]) -> RotVec:
    """Convert a JPL attitude to principal modified Rodrigues parameters."""
    x, y, z, w = canonicalize(normalize(q))
    denominator = 1.0 + w
    if denominator == 0.0:
        raise ValueError("MRP is singular for a 360-degree attitude")
    return -x / denominator, -y / denominator, -z / denominator


def swing_twist(q: Iterable[FloatT], axis: Iterable[FloatT]) -> tuple[Quaternion, Quaternion]:
    """Decompose a JPL attitude as ``swing ⊙ twist`` about an axis."""
    value = normalize(q)
    direction = _axis(_vector(axis))
    x, y, z, w = value
    projection = x*direction[0] + y*direction[1] + z*direction[2]
    try:
        twist = normalize((projection*direction[0], projection*direction[1],
                           projection*direction[2], w))
    except ValueError:
        twist = (0.0, 0.0, 0.0, 1.0)
    swing = normalize(multiply(value, inverse(twist)))
    return swing, twist


def rotation_mean(quaternions: Iterable[Iterable[FloatT]],
                  weights: Iterable[FloatT] | None = None, method: str = "markley",
                  tolerance: FloatLike = 1e-12, max_iterations: int = 100) -> Quaternion:
    """Average JPL rotations by weighted Markley or intrinsic Karcher mean."""
    samples = tuple(normalize(q) for q in quaternions)
    if not samples:
        raise ValueError("at least one quaternion is required")
    sample_weights = _weights(weights, len(samples))
    if method not in ("markley", "intrinsic"):
        raise ValueError("method must be 'markley' or 'intrinsic'")
    scatter = [[math.fsum(weight * sample[i] * sample[j]
                          for sample, weight in zip(samples, sample_weights))
                for j in range(4)] for i in range(4)]
    _, eigenvector, _ = _jacobi_largest(scatter)
    mean = canonicalize(normalize(eigenvector))
    if method == "markley":
        return mean
    tol = _real_scalar(tolerance, "tolerance")
    if tol <= 0.0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    for _ in range(max_iterations):
        errors = tuple(to_rotvec(multiply(inverse(mean), sample)) for sample in samples)
        delta = tuple(math.fsum(weight * error[i]
                                for error, weight in zip(errors, sample_weights))
                      for i in range(3))
        if math.hypot(*delta) <= tol:
            return mean
        mean = boxplus(mean, delta, frame="body")
    raise RuntimeError("intrinsic rotation mean did not converge")


def rotation_covariance(quaternions: Iterable[Iterable[FloatT]],
                        mean: Iterable[FloatT] | None = None,
                        weights: Iterable[FloatT] | None = None,
                        frame: str = "body") -> tuple[tuple[float, ...], ...]:
    """Return weighted population covariance in a JPL local tangent frame."""
    side = _frame(frame)
    samples = tuple(normalize(q) for q in quaternions)
    if not samples:
        raise ValueError("at least one quaternion is required")
    sample_weights = _weights(weights, len(samples))
    center = normalize(mean) if mean is not None else rotation_mean(samples, sample_weights)
    errors = tuple(boxminus(sample, center, frame=side) for sample in samples)
    return tuple(tuple(math.fsum(weight * error[i] * error[j]
                                 for error, weight in zip(errors, sample_weights))
                       for j in range(3)) for i in range(3))


def align_vectors(source: Iterable[Iterable[FloatT]], target: Iterable[Iterable[FloatT]],
                  weights: Iterable[FloatT] | None = None) -> Quaternion:
    """Solve the weighted Wahba problem with a native JPL quaternion result."""
    raw_sources = tuple(_vector(value) for value in source)
    raw_targets = tuple(_vector(value) for value in target)
    sources, targets = raw_sources, raw_targets
    if not sources or len(sources) != len(targets):
        raise ValueError("source and target must contain the same nonzero number of vectors")
    if any(not any(vector) for vector in sources + targets):
        raise ValueError("source and target vectors must be nonzero")
    sample_weights = _weights(weights, len(sources))
    source_scale = max(abs(value) for vector in sources for value in vector)
    target_scale = max(abs(value) for vector in targets for value in vector)
    source_vectors = tuple(tuple(value / source_scale for value in vector)
                           for vector in raw_sources)
    target_vectors = tuple(tuple(value / target_scale for value in vector)
                           for vector in raw_targets)
    b = [[math.fsum(weight * src[i] * dst[j]
                    for src, dst, weight in zip(source_vectors, target_vectors, sample_weights))
          for j in range(3)] for i in range(3)]
    sigma = b[0][0] + b[1][1] + b[2][2]
    symmetric = [[b[i][j] + b[j][i] for j in range(3)] for i in range(3)]
    z = (b[1][2] - b[2][1], b[2][0] - b[0][2], b[0][1] - b[1][0])
    k = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            k[i][j] = symmetric[i][j] - (sigma if i == j else 0.0)
        k[i][3] = k[3][i] = z[i]
    k[3][3] = sigma
    matrix_scale = max(abs(value) for row in k for value in row)
    if matrix_scale == 0.0:
        raise ValueError("vector correspondences do not determine a unique rotation")
    scaled_k = [[value / matrix_scale for value in row] for row in k]
    largest, eigenvector, runner_up = _jacobi_largest(scaled_k)
    if largest - runner_up <= 1e-12 * max(1.0, abs(largest)):
        raise ValueError("vector correspondences do not determine a unique rotation")
    return canonicalize(normalize((-eigenvector[0], -eigenvector[1],
                                   -eigenvector[2], eigenvector[3])))


def random_quaternion(rng: object | None = None) -> Quaternion:
    """Draw a uniform unit JPL quaternion on S3 using Shoemake's method."""
    source = random if rng is None else rng
    draw = getattr(source, "random", None)
    if not callable(draw):
        raise ValueError("rng must provide random()")  # noqa: TRY004
    u1, u2, u3 = (float(draw()) for _ in range(3))
    if any(not math.isfinite(u) or not 0.0 <= u < 1.0 for u in (u1, u2, u3)):
        raise ValueError("rng.random() must return values in [0, 1)")
    x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return -x, -y, -z, w


random_rotation = random_quaternion


def to_homogeneous_matrix(
    q: Iterable[FloatT], translation: Iterable[FloatT] = (0.0, 0.0, 0.0)
) -> tuple[tuple[float, ...], ...]:
    """Return a 4x4 homogeneous matrix for a JPL attitude and translation."""
    rotation = to_rotation_matrix(q)
    offset = _vector(translation)
    return tuple((*rotation[row], offset[row]) for row in range(3)) + (
        (0.0, 0.0, 0.0, 1.0),
    )


def from_homogeneous_matrix(
    matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8
) -> Quaternion:
    """Validate a rigid 4x4 transform and return its native JPL rotation."""
    tolerance = _real_scalar(atol, "atol")
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("atol must be in [0, 1)")
    try:
        rows = tuple(tuple(_real_scalar(item, "matrix element") for item in row)
                     for row in matrix)
    except TypeError as exc:
        raise ValueError("matrix must have shape (4, 4)") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("matrix must have shape (4, 4)")
    if any(abs(a - b) > tolerance
           for a, b in zip(rows[3], (0.0, 0.0, 0.0, 1.0))):
        raise ValueError("matrix must have homogeneous bottom row (0, 0, 0, 1)")
    return from_rotation_matrix(tuple(row[:3] for row in rows[:3]), tolerance)


so3_adjoint = to_rotation_matrix


__all__ = [
    'AxisAngle',
    'EulerResult',
    'Quaternion',
    'RotVec',
    'Vector',
    'add',
    'canonicalize',
    'conjugate',
    'cos',
    'cosh',
    'divide',
    'dot',
    'exp',
    'from_axis_angle',
    'from_hamilton',
    'from_rotation_matrix',
    'from_rotation_vector',
    'from_rotvec',
    'from_two_vectors',
    'from_wxyz',
    'from_xyzw',
    'integrate_angular_velocity',
    'inverse',
    'left_divide',
    'log',
    'multiply',
    'negate',
    'norm',
    'norm_squared',
    'normalize',
    'pow_real',
    'power',
    'right_divide',
    'rotate',
    'rotation_distance',
    'rotations_equivalent',
    'scale',
    'sin',
    'sinh',
    'skew_matrix',
    'slerp',
    'so3_left_jacobian',
    'so3_left_jacobian_inverse',
    'so3_right_jacobian',
    'so3_right_jacobian_inverse',
    'sqrt',
    'subtract',
    'tan',
    'tanh',
    'to_axis_angle',
    'to_hamilton',
    'to_left_matrix',
    'to_right_matrix',
    'to_rotation_matrix',
    'to_rotation_vector',
    'to_rotvec',
    'to_wxyz',
    'to_xyzw',
]

__all__ += [
    'acos',
    'acosh',
    'align_sign',
    'align_vectors',
    'asin',
    'asinh',
    'atan',
    'atanh',
    'boxminus',
    'boxplus',
    'commutator',
    'expm1',
    'from_euler',
    'from_homogeneous_matrix',
    'from_mrp',
    'integrate_angular_velocity_samples',
    'log1p',
    'log2',
    'log10',
    'make_sign_continuous',
    'nlerp',
    'power_quaternion',
    'quaternion_derivative',
    'random_quaternion',
    'random_rotation',
    'rotation_covariance',
    'rotation_mean',
    'scalar_part',
    'so3_adjoint',
    'squad',
    'swing_twist',
    'to_euler',
    'to_euler_with_status',
    'to_homogeneous_matrix',
    'to_mrp',
    'vector_part',
]
