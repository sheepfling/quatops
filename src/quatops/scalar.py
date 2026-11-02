"""Hamilton quaternion operations on Python tuples in ``wxyz`` coefficient order.

All operations return new tuples or floats. Inputs must contain finite real
numbers. Underflow follows Python float arithmetic; a non-finite result raises
``OverflowError``. Division-dependent operations reject the zero quaternion.
"""

import math
from collections.abc import Iterable
from math import fsum, hypot, isfinite
from numbers import Integral, Real
from sys import float_info

from ._typing import FloatLike, FloatT

Quaternion = tuple[float, float, float, float]
Vector = tuple[float, float, float]
AxisAngle = tuple[float, float, float, float]
RotVec = Vector


def _components(
    values: Iterable[FloatT], length: int, name: str
) -> tuple[float, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of {length} real numbers") from exc
    if len(items) != length:
        raise ValueError(f"{name} must contain exactly {length} components")
    if type(items[0]) is float and all(type(value) is float for value in items):
        # Built-in float tuples are the common hot path. Keep the static
        # SupportsFloat contract without protocol checks or redundant casts.
        result = items
    else:
        if not all(isinstance(value, Real) or isinstance(value, FloatLike)
                   for value in items):
            raise ValueError(f"{name} must contain finite real numbers")
        try:
            result = tuple(float(value) for value in items)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite real numbers") from exc
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _quaternion(q: Iterable[FloatT]) -> Quaternion:
    return _components(q, 4, "quaternion")  # type: ignore[return-value]


def _vector(v: Iterable[FloatT]) -> Vector:
    return _components(v, 3, "vector")  # type: ignore[return-value]


def _real_scalar(value: FloatLike, name: str) -> float:
    if type(value) is float:
        result = value
    elif isinstance(value, Real) or isinstance(value, FloatLike):
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite real number") from exc
    else:
        raise ValueError(f"{name} must be a finite real number")
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _checked(values: tuple) -> tuple:
    if not all(isfinite(value) for value in values):
        raise OverflowError("quaternion operation produced a non-finite result")
    return values


def exp(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion exponential in scalar-first order."""
    w, x, y, z = _quaternion(q)
    radius = hypot(x, y, z)
    try:
        scale = math.exp(w)
        cosine = math.cos(radius)
        sinc = 1.0 if radius == 0.0 else math.sin(radius) / radius
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion exponential is outside the float range") from exc
    return _checked((scale * cosine, scale * x * sinc, scale * y * sinc,
                     scale * z * sinc))  # type: ignore[return-value]


def log(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal algebraic quaternion logarithm.

    A negative real quaternion has no unique logarithm direction; this API
    chooses +X, so ``log((-a, 0, 0, 0)) == (log(a), pi, 0, 0)`` for ``a > 0``.
    """
    w, x, y, z = _quaternion(q)
    scale = max(abs(w), abs(x), abs(y), abs(z))
    if scale == 0.0:
        raise ValueError("zero quaternion has no logarithm")
    sw, sx, sy, sz = w / scale, x / scale, y / scale, z / scale
    norm_scaled = hypot(sw, sx, sy, sz)
    vector_scaled = hypot(sx, sy, sz)
    real = math.log(scale) + math.log(norm_scaled)
    if vector_scaled == 0.0:
        if sw < 0.0:
            return real, math.pi, 0.0, 0.0
        return real, 0.0, 0.0, 0.0
    angle = math.atan2(vector_scaled, sw)
    factor = angle / vector_scaled
    return _checked((real, sx * factor, sy * factor, sz * factor))  # type: ignore[return-value]


def sin(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion sine in scalar-first order."""
    w, x, y, z = _quaternion(q)
    radius = hypot(x, y, z)
    try:
        scalar = math.sin(w) * math.cosh(radius)
        ratio = 1.0 if radius == 0.0 else math.sinh(radius) / radius
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion sine is outside the float range") from exc
    vector_scale = math.cos(w) * ratio
    return _checked((scalar, x * vector_scale, y * vector_scale,
                     z * vector_scale))  # type: ignore[return-value]


def cos(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion cosine in scalar-first order."""
    w, x, y, z = _quaternion(q)
    radius = hypot(x, y, z)
    try:
        scalar = math.cos(w) * math.cosh(radius)
        ratio = 1.0 if radius == 0.0 else math.sinh(radius) / radius
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion cosine is outside the float range") from exc
    vector_scale = -math.sin(w) * ratio
    return _checked((scalar, x * vector_scale, y * vector_scale,
                     z * vector_scale))  # type: ignore[return-value]


def sinh(q: Iterable[FloatT]) -> Quaternion:
    """Return quaternion hyperbolic sine using the principal component formula."""
    w, x, y, z = _quaternion(q)
    radius = hypot(x, y, z)
    try:
        scalar = math.sinh(w) * math.cos(radius)
        ratio = 1.0 if radius == 0.0 else math.sin(radius) / radius
        vector_scale = math.cosh(w) * ratio
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion hyperbolic sine is outside the float range") from exc
    return _checked((scalar, x * vector_scale, y * vector_scale,
                     z * vector_scale))  # type: ignore[return-value]


def cosh(q: Iterable[FloatT]) -> Quaternion:
    """Return quaternion hyperbolic cosine using the principal component formula."""
    w, x, y, z = _quaternion(q)
    radius = hypot(x, y, z)
    try:
        scalar = math.cosh(w) * math.cos(radius)
        ratio = 1.0 if radius == 0.0 else math.sin(radius) / radius
        vector_scale = math.sinh(w) * ratio
    except (OverflowError, ValueError) as exc:
        raise OverflowError("quaternion hyperbolic cosine is outside the float range") from exc
    return _checked((scalar, x * vector_scale, y * vector_scale,
                     z * vector_scale))  # type: ignore[return-value]


def tanh(q: Iterable[FloatT]) -> Quaternion:
    """Return ``sinh(q) / cosh(q)`` using the right quotient."""
    return divide(sinh(q), cosh(q))


def tan(q: Iterable[FloatT]) -> Quaternion:
    """Return the algebraic quaternion tangent ``sin(q) / cos(q)``."""
    return divide(sin(q), cos(q))


def power(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Raise a quaternion to an integer or real power on the principal branch.

    Integer exponents use exponentiation by squaring. Other real exponents use
    ``exp(exponent * log(q))``; the logarithm chooses +X for negative reals.
    Zero to a positive power is zero, zero to zero is identity, and a negative
    power of zero is undefined.
    """
    if type(exponent) is float:
        is_integer = False
    elif isinstance(exponent, Integral):
        is_integer = True
    elif isinstance(exponent, FloatLike):
        is_integer = False
    else:
        raise ValueError("exponent must be a finite real number")
    base = _quaternion(q)
    if is_integer:
        n = int(exponent)
        if n == 0:
            return (1.0, 0.0, 0.0, 0.0)
        if n < 0:
            base = inverse(base)
            n = -n
        result: Quaternion = (1.0, 0.0, 0.0, 0.0)
        while n:
            if n & 1:
                result = multiply(result, base)
            n >>= 1
            if n:
                base = multiply(base, base)
        return result
    if type(exponent) is float:
        value = exponent
    else:
        try:
            value = float(exponent)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("exponent must be a finite real number") from exc
    if not isfinite(value):
        raise ValueError("exponent must be a finite real number")
    if not any(base):
        if value == 0.0:
            return (1.0, 0.0, 0.0, 0.0)
        if value > 0.0:
            return (0.0, 0.0, 0.0, 0.0)
        raise ValueError("zero quaternion cannot have a negative power")
    return exp(scale(log(base), value))


def pow_real(q: Iterable[FloatT], exponent: FloatLike) -> Quaternion:
    """Alias for the principal real power of a quaternion."""
    return power(q, exponent)


def sqrt(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal square root on the logarithm branch used by ``log``."""
    return pow_real(q, 0.5)


def slerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Spherically interpolate unit rotations; allow extrapolation outside [0, 1].

    Endpoints are normalized. By default, the sign of ``q1`` is selected to
    take the shortest path; ``shortest=False`` preserves both input signs.
    """
    if type(t) is float:
        amount = t
    else:
        amount = _real_scalar(t, "t")
    if not isfinite(amount):
        raise ValueError("t must be a finite real number")
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")
    a, b = normalize(q0), normalize(q1)
    dot = fsum(x * y for x, y in zip(a, b))
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


def skew_matrix(v: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the 3x3 cross-product matrix for a three-vector."""
    x, y, z = _vector(v)
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def _so3_jacobian(r: Iterable[FloatT], side: int, inverse_matrix: bool = False
                  ) -> tuple[tuple[float, ...], ...]:
    x, y, z = _vector(r)
    theta = hypot(x, y, z)
    if not isfinite(theta):
        raise OverflowError("rotation-vector norm is too large for a float")
    axis = (0.0, 0.0, 0.0) if theta == 0.0 else (x / theta, y / theta, z / theta)
    t2 = theta * theta
    if inverse_matrix:
        if theta < 1e-4:
            t4 = t2 * t2
            a = -0.5 * side * theta
            b = t2 * (1.0 / 12.0 + t2 / 720.0 + t4 / 30240.0)
        else:
            half = theta * 0.5
            sine_half = math.sin(half)
            if abs(sine_half) <= 1e-15:
                raise ValueError("SO(3) Jacobian is singular at a nonzero 2*pi multiple")
            a = -0.5 * side * theta
            b = 1.0 - half * math.cos(half) / sine_half
    else:
        if theta < 1e-4:
            t4 = t2 * t2
            a = theta * (0.5 - t2 / 24.0 + t4 / 720.0)
            b = t2 * (1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0)
        else:
            a = (1.0 - math.cos(theta)) / theta
            b = 1.0 - math.sin(theta) / theta
        a *= side
    x, y, z = axis
    k = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
    k2 = tuple(tuple(fsum(k[i][m] * k[m][j] for m in range(3))
                     for j in range(3)) for i in range(3))
    rows = tuple(tuple((1.0 if i == j else 0.0) + a * k[i][j] + b * k2[i][j]
                       for j in range(3)) for i in range(3))
    return tuple(tuple(_checked((value,))[0] for value in row) for row in rows)


def so3_left_jacobian(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the SO(3) left Jacobian for a physical rotation vector."""
    return _so3_jacobian(r, 1)


def so3_right_jacobian(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the SO(3) right Jacobian for a physical rotation vector."""
    return _so3_jacobian(r, -1)


def so3_left_jacobian_inverse(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the inverse SO(3) left Jacobian; reject its 2*pi singularities."""
    return _so3_jacobian(r, 1, True)


def so3_right_jacobian_inverse(r: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the inverse SO(3) right Jacobian; reject its 2*pi singularities."""
    return _so3_jacobian(r, -1, True)


def integrate_angular_velocity(q: Iterable[FloatT], omega: Iterable[FloatT], dt: FloatLike,
                               frame: str = "body") -> Quaternion:
    """Integrate constant angular velocity exactly over ``dt``.

    ``omega`` is a right-handed physical angular-velocity vector in radians
    per second. ``frame='body'`` applies the increment on the right; ``'world'``
    applies it on the left. The result is normalized.
    """
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    if type(dt) is float:
        step = dt
    else:
        step = _real_scalar(dt, "dt")
    if not isfinite(step):
        raise ValueError("dt must be a finite real number")
    vector = _vector(omega)
    increment_vector = _checked(tuple(component * step for component in vector))
    increment = from_rotation_vector(increment_vector)
    attitude = normalize(q)
    return normalize(multiply(attitude, increment) if frame == "body"
                     else multiply(increment, attitude))


def from_xyzw(q: Iterable[FloatT]) -> Quaternion:
    """Convert scalar-last ``(x, y, z, w)`` to scalar-first order."""
    x, y, z, w = _quaternion(q)
    return w, x, y, z


def to_xyzw(q: Iterable[FloatT]) -> Quaternion:
    """Convert scalar-first order to scalar-last ``(x, y, z, w)``."""
    w, x, y, z = _quaternion(q)
    return x, y, z, w


def from_wxyz(q: Iterable[FloatT]) -> Quaternion:
    """Validate and copy scalar-first values; no convention conversion."""
    return _quaternion(q)


def to_wxyz(q: Iterable[FloatT]) -> Quaternion:
    """Return scalar-first values as a tuple; no convention conversion."""
    return _quaternion(q)


def conjugate(q: Iterable[FloatT]) -> Quaternion:
    """Return the conjugate, preserving the quaternion's magnitude."""
    w, x, y, z = _quaternion(q)
    return w, -x, -y, -z


def add(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Add quaternion components independently."""
    w, x, y, z = _quaternion(lhs)
    a, b, c, d = _quaternion(rhs)
    return _checked((w + a, x + b, y + c, z + d))  # type: ignore[return-value]


def subtract(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Subtract quaternion components independently."""
    w, x, y, z = _quaternion(lhs)
    a, b, c, d = _quaternion(rhs)
    return _checked((w - a, x - b, y - c, z - d))  # type: ignore[return-value]


def negate(q: Iterable[FloatT]) -> Quaternion:
    """Negate every quaternion component."""
    return tuple(-value for value in _quaternion(q))  # type: ignore[return-value]


def scale(q: Iterable[FloatT], factor: FloatLike) -> Quaternion:
    """Multiply every quaternion component by one finite real factor."""
    if type(factor) is float:
        value = factor
        if not isfinite(value):
            raise ValueError("factor must be a finite real number")
    else:
        value = _real_scalar(factor, "factor")
    w, x, y, z = _quaternion(q)
    return _checked((value * w, value * x, value * y, value * z))  # type: ignore[return-value]


def multiply(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the Hamilton product ``lhs * rhs`` (``rhs`` acts first)."""
    w, x, y, z = _quaternion(lhs)
    a, b, c, d = _quaternion(rhs)
    return _checked((
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    ))  # type: ignore[return-value]


def norm(q: Iterable[FloatT]) -> float:
    """Return the Euclidean norm, using a scaled hypotenuse calculation."""
    result = hypot(*_quaternion(q))
    if not isfinite(result):
        raise OverflowError("quaternion norm is too large for a float")
    return result


def dot(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> float:
    """Return the Euclidean coefficient dot product."""
    a, b = _quaternion(lhs), _quaternion(rhs)

    def exact_fallback() -> float:
        # Exact rational products retain information when a float product
        # overflows or underflows before cancellation in the final sum.
        from fractions import Fraction

        exact = sum((Fraction.from_float(x) * Fraction.from_float(y)
                     for x, y in zip(a, b)), Fraction())
        try:
            value = float(exact)
        except OverflowError as exc:
            raise OverflowError("quaternion dot product is outside the float range") from exc
        if not isfinite(value):
            raise OverflowError("quaternion dot product is outside the float range")
        return value

    terms = tuple(x * y for x, y in zip(a, b))
    if any(
        not isfinite(term)
        or (x != 0.0 and y != 0.0 and abs(term) < float_info.min)
        for x, y, term in zip(a, b, terms)
    ):
        return exact_fallback()
    try:
        result = fsum(terms)
    except OverflowError:
        return exact_fallback()
    if not isfinite(result):
        return exact_fallback()
    return result


def norm_squared(q: Iterable[FloatT]) -> float:
    """Return the squared Euclidean coefficient norm."""
    return dot(q, q)


def _scaled(q: Quaternion) -> tuple:
    scale = max(abs(value) for value in q)
    if scale == 0.0:
        raise ValueError("zero quaternion has no inverse or normalization")
    parts = tuple(value / scale for value in q)
    return scale, parts


def normalize(q: Iterable[FloatT]) -> Quaternion:
    """Return a unit quaternion; reject zero, NaN, and infinity."""
    _, parts = _scaled(_quaternion(q))
    length = hypot(*parts)
    return tuple(value / length for value in parts)  # type: ignore[return-value]


def inverse(q: Iterable[FloatT]) -> Quaternion:
    """Return the multiplicative inverse of a nonzero quaternion."""
    scale, parts = _scaled(_quaternion(q))
    w, x, y, z = parts
    denominator = sum(value * value for value in parts)
    result = tuple((value / denominator) / scale for value in (w, -x, -y, -z))
    return _checked(result)  # type: ignore[return-value]


def divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)``."""
    return multiply(lhs, inverse(rhs))


def right_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the right quotient ``lhs * inverse(rhs)``."""
    return divide(lhs, rhs)


def left_divide(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the left quotient ``inverse(lhs) * rhs``."""
    return multiply(inverse(lhs), rhs)


def rotation_distance(q0: Iterable[FloatT], q1: Iterable[FloatT]) -> float:
    """Return the shortest geodesic angle between two rotations in radians."""
    a, b = normalize(q0), normalize(q1)
    relative = normalize(multiply(inverse(a), b))
    w, x, y, z = relative
    return 2.0 * math.atan2(hypot(x, y, z), abs(w))


def rotations_equivalent(q0: Iterable[FloatT], q1: Iterable[FloatT],
                         atol: FloatLike = 1e-12) -> bool:
    """Return whether two rotations differ by at most ``atol`` radians."""
    tolerance = _real_scalar(atol, "atol")
    if tolerance < 0.0:
        raise ValueError("atol must be non-negative")
    return rotation_distance(q0, q1) <= tolerance


def _dot3(a: FloatLike, b: FloatLike, c: FloatLike, x: FloatLike, y: FloatLike, z: FloatLike) -> float:
    """Sum finite component products without avoidable intermediate overflow."""
    terms = [a*x, b*y, c*z]
    terms.sort(key=abs, reverse=True)
    # Add the largest opposite-signed terms first.  fsum otherwise can overflow
    # during its internal accumulation even when the final sum is representable.
    if ((terms[0] > 0.0 and terms[1] > 0.0 and terms[2] < 0.0)
            or (terms[0] < 0.0 and terms[1] < 0.0 and terms[2] > 0.0)):
        terms[1], terms[2] = terms[2], terms[1]
    return fsum(terms)


def rotate(q: Iterable[FloatT], v: Iterable[FloatT]) -> Vector:
    """Actively rotate ``v`` by ``q * (0, v) * inverse(q)``.

    Any nonzero quaternion is valid; its magnitude has no effect. No input is
    normalized. Internally scaling its components avoids unnecessary overflow.
    """
    _, parts = _scaled(_quaternion(q))
    w, x, y, z = parts
    vx, vy, vz = _vector(v)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    inv_denominator = 1./ (w2 + x2 + y2 + z2)
    # Compute each component independently.  A shared vector scale can erase a
    # small component when another one is near the largest representable float.
    def coefficient(value: FloatLike) -> float:
        return min(1.0, max(-1.0, value * inv_denominator))

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


def to_rotation_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the norm-aware 3x3 active Hamilton rotation matrix."""
    _, (w, x, y, z) = _scaled(_quaternion(q))
    d = w*w + x*x + y*y + z*z
    rows = (((w*w+x*x-y*y-z*z)/d, 2*(x*y-w*z)/d, 2*(x*z+w*y)/d),
            (2*(x*y+w*z)/d, (w*w-x*x+y*y-z*z)/d, 2*(y*z-w*x)/d),
            (2*(x*z-w*y)/d, 2*(y*z+w*x)/d, (w*w-x*x-y*y+z*z)/d))
    return tuple(tuple(_checked((v,))[0] for v in row) for row in rows)


def from_rotation_matrix(
    matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8
) -> Quaternion:
    """Convert a proper active 3x3 rotation matrix to a canonical unit q.

    Orthogonality and determinant +1 are checked within the absolute tolerance.
    """
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
        abs(fsum(r[k][i] * r[k][j] for k in range(3)) - (1.0 if i == j else 0.0))
        for i in range(3) for j in range(3)
    ]
    determinant = fsum((
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]),
        -r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]),
        r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]),
    ))
    if max(errors) > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError("matrix must be a proper orthonormal rotation")

    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (0.25 * s, (r[2][1] - r[1][2]) / s,
             (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    elif r[0][0] >= r[1][1] and r[0][0] >= r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2])
        q = ((r[2][1] - r[1][2]) / s, 0.25 * s,
             (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s)
    elif r[1][1] >= r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2])
        q = ((r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s,
             0.25 * s, (r[1][2] + r[2][1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1])
        q = ((r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s,
             (r[1][2] + r[2][1]) / s, 0.25 * s)
    return canonicalize(normalize(q))


def to_left_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the 4x4 matrix L(q) satisfying L(q) r = q*r (wxyz order)."""
    w, x, y, z = _quaternion(q)
    return ((w,-x,-y,-z),(x,w,-z,y),(y,z,w,-x),(z,-y,x,w))


def to_right_matrix(q: Iterable[FloatT]) -> tuple[tuple[float, ...], ...]:
    """Return the 4x4 matrix P(q) satisfying P(q) r = r*q (wxyz order)."""
    w, x, y, z = _quaternion(q)
    return ((w,-x,-y,-z),(x,w,z,-y),(y,-z,w,x),(z,y,-x,w))


def _canonical(q: Quaternion) -> Quaternion:
    """Choose a deterministic quaternion sign without changing its rotation."""
    w, x, y, z = q
    flip = w < 0 or (w == 0 and next((c for c in (x, y, z) if c != 0), 1.0) < 0)
    return tuple(0.0 if c == 0.0 else (-c if flip else c) for c in (w, x, y, z))  # type: ignore[return-value]


def canonicalize(q: Iterable[FloatT]) -> Quaternion:
    """Choose a deterministic sign while preserving the quaternion magnitude."""
    return _canonical(_quaternion(q))


def _axis(values: Vector) -> Vector:
    """Return the direction of a finite, nonzero vector using scaled values."""
    x, y, z = values
    s = max(abs(x), abs(y), abs(z))
    if s == 0:
        raise ValueError("axis must be nonzero")
    a, b, c = x / s, y / s, z / s
    length = math.hypot(a, b, c)
    return a / length, b / length, c / length


def _positive_zero(value: FloatLike) -> float:
    """Use the canonical positive sign for a floating-point zero."""
    return 0.0 if value == 0.0 else value


def _axis_angle_from_q(q: Iterable[FloatT]) -> tuple[Vector, float]:
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


def _from_unit_axis(axis: Vector, angle: FloatLike) -> Quaternion:
    """Create a canonical quaternion from a unit axis and angle in radians."""
    half = angle * 0.5
    sine = math.sin(half)
    cosine = 0.0 if abs(angle) == math.pi else math.cos(half)
    x, y, z = axis
    return _canonical((cosine, x * sine, y * sine, z * sine))


def from_axis_angle(a: Iterable[FloatT]) -> Quaternion:
    """Normalize a nonzero axis and convert its radians to a unit quaternion."""
    x, y, z, angle = _components(a, 4, "axis-angle")
    return _from_unit_axis(_axis((x, y, z)), angle)


def to_axis_angle(q: Iterable[FloatT]) -> AxisAngle:
    """Convert a finite nonzero quaternion to canonical axis and radians."""
    axis, angle = _axis_angle_from_q(q)
    return *axis, angle


def from_rotvec(r: Iterable[FloatT]) -> Quaternion:
    """Convert an axis-times-angle vector in radians to a unit quaternion."""
    xyz = _vector(r)
    scale = max(abs(v) for v in xyz)
    if scale == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    magnitude = math.hypot(*xyz)
    if not math.isfinite(magnitude):
        raise OverflowError("rotation-vector norm is too large for a float")
    return _from_unit_axis(_axis(xyz), magnitude)


def from_two_vectors(source: Iterable[FloatT], target: Iterable[FloatT]) -> Quaternion:
    """Return the shortest active rotation mapping ``source`` onto ``target``.

    Inputs must be nonzero. For antiparallel vectors, the half-turn axis is
    chosen deterministically by crossing the source with its least-aligned
    coordinate axis.
    """
    a, b = _axis(_vector(source)), _axis(_vector(target))
    cosine = min(1.0, max(-1.0, fsum(x * y for x, y in zip(a, b))))
    cross = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
    if cosine < 0.0 and cross == (0.0, 0.0, 0.0):
        basis_index = min(range(3), key=lambda index: abs(a[index]))
        basis = tuple(1.0 if index == basis_index else 0.0 for index in range(3))
        axis = _axis((a[1] * basis[2] - a[2] * basis[1],
                      a[2] * basis[0] - a[0] * basis[2],
                      a[0] * basis[1] - a[1] * basis[0]))
        return canonicalize((0.0, *axis))
    return canonicalize(normalize((1.0 + cosine, *cross)))


def _rotvec_from_q(q: Iterable[FloatT]) -> RotVec:
    """Extract a rotation vector without multiplying by a rounded tiny angle."""
    w, x, y, z = _canonical(_quaternion(q))
    scale = max(abs(x), abs(y), abs(z))
    if w > 0.0 and scale < w * 1e-8:
        w_mantissa, w_exponent = math.frexp(w)
        def component(value: FloatLike) -> float:
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


def to_rotvec(q: Iterable[FloatT]) -> RotVec:
    """Convert a nonzero quaternion to a canonical rotation vector in radians."""
    return _rotvec_from_q(q)


def from_rotation_vector(r: Iterable[FloatT]) -> Quaternion:
    """Convert a physical rotation vector in radians to a unit quaternion."""
    return from_rotvec(r)


def to_rotation_vector(q: Iterable[FloatT]) -> RotVec:
    """Convert a nonzero quaternion to its canonical rotation vector in radians."""
    return to_rotvec(q)


from ._advanced import (
    acos,
    acosh,
    align_sign,
    align_vectors,
    asin,
    asinh,
    atan,
    atanh,
    EulerResult,
    boxminus,
    boxplus,
    commutator,
    expm1,
    from_euler,
    from_homogeneous_matrix,
    from_mrp,
    integrate_angular_velocity_samples,
    log1p,
    log10,
    log2,
    make_sign_continuous,
    nlerp,
    power_quaternion,
    quaternion_derivative,
    random_quaternion,
    random_rotation,
    rotation_covariance,
    rotation_mean,
    scalar_part,
    so3_adjoint,
    squad,
    swing_twist,
    to_euler,
    to_euler_with_status,
    to_homogeneous_matrix,
    to_mrp,
    vector_part,
)

__all__ = [
    "AxisAngle",
    "EulerResult",
    "Quaternion",
    "RotVec",
    "Vector",
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
    "to_euler_with_status",
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
