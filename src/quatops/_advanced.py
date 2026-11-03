"""Convention-neutral rotation and extended quaternion operations.

The public convention modules adapt these helpers when their stored algebra is
JPL. Scalar algorithms intentionally depend only on the standard library.
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Iterable, Sequence
from typing import Callable

from . import scalar as _h
from ._euler_sequence import euler_metadata as _euler_sequence_metadata
from ._euler_types import EulerResult
from ._typing import FloatLike, FloatT

Quaternion = _h.Quaternion
Vector = _h.Vector
Matrix3 = tuple[tuple[float, float, float], ...]


def _frame(frame: str) -> str:
    if frame not in ("body", "world"):
        raise ValueError("frame must be 'body' or 'world'")
    return frame


def _sequence(sequence: str) -> tuple[str, bool]:
    if not isinstance(sequence, str) or len(sequence) != 3:
        raise ValueError("sequence must contain three axis letters")
    if sequence.islower():
        intrinsic = False
    elif sequence.isupper():
        intrinsic = True
    else:
        raise ValueError("sequence must use all lowercase or all uppercase letters")
    axes = sequence.lower()
    if any(axis not in "xyz" for axis in axes):
        raise ValueError("sequence must contain only x, y, and z")
    if len(set(axes)) == 2 and axes[0] == axes[2] and axes[0] != axes[1]:
        return axes, intrinsic
    if set(axes) != {"x", "y", "z"}:
        raise ValueError("sequence must be Tait-Bryan or proper Euler")
    return axes, intrinsic


def _euler_angles(angles: Iterable[FloatT], sequence: str) -> tuple[Vector, str]:
    xyz = _h._vector(angles)
    axes, intrinsic = _sequence(sequence)
    if intrinsic:
        return tuple(reversed(xyz)), axes[::-1]  # type: ignore[return-value]
    return xyz, axes


def _static_euler_from_matrix(
    matrix: Sequence[Sequence[float]], axes: str
) -> tuple[Vector, bool]:
    """Extract static-axis Tait-Bryan angles and report gimbal lock."""
    i = "xyz".index(axes[0])
    parity = 0 if axes in ("xyz", "yzx", "zxy") else 1
    next_axis = (1, 2, 0, 1)
    j = next_axis[i + parity]
    k = next_axis[i - parity + 1]
    cy = math.hypot(matrix[i][i], matrix[j][i])
    if cy > 1e-15:
        first = math.atan2(matrix[k][j], matrix[k][k])
        second = math.atan2(-matrix[k][i], cy)
        third = math.atan2(matrix[j][i], matrix[i][i])
        singular = False
    else:
        first = math.atan2(-matrix[j][k], matrix[j][j])
        second = math.atan2(-matrix[k][i], cy)
        third = 0.0
        singular = True
    if parity:
        first, second, third = -first, -second, -third
    return (first, second, third), singular


def _euler_from_matrix(matrix: Sequence[Sequence[float]], sequence: str) -> Vector:
    """Extract Tait-Bryan angles, setting the final angle to zero at lock."""
    axes, intrinsic = _sequence(sequence)
    if not intrinsic:
        return _static_euler_from_matrix(matrix, axes)[0]

    reversed_axes = axes[::-1]
    result, singular = _static_euler_from_matrix(matrix, reversed_axes)
    if not singular:
        return tuple(reversed(result))  # type: ignore[return-value]

    # Intrinsic abc equals extrinsic cba with the angles reversed. The static
    # singular branch sets the last cba angle to zero (the original first
    # angle); choose the equivalent locked solution with the original final
    # angle zero instead, matching the public policy for every sequence.
    parity = 0 if reversed_axes in ("xyz", "yzx", "zxy") else 1
    direction = 1.0 if ((result[1] > 0.0) == (parity == 1)) else -1.0
    return direction * result[0], result[1], 0.0


def _wrap_euler_outer(angle: float) -> float:
    wrapped = math.remainder(angle, math.tau)
    return -math.pi if wrapped == math.pi else wrapped


def _euler_from_components(
    w: float,
    x: float,
    y: float,
    z: float,
    sequence: str,
    gimbal_tolerance: FloatLike = 1e-12,
    with_status: bool = False,
) -> Vector | EulerResult:
    """Extract Euler angles from already validated Hamilton coefficients."""
    tolerance = _h._real_scalar(gimbal_tolerance, "gimbal_tolerance")
    if tolerance < 0.0 or tolerance >= math.pi / 2.0:
        raise ValueError("gimbal_tolerance must be in [0, pi/2)")

    magnitude = max(abs(w), abs(x), abs(y), abs(z))
    if magnitude == 0.0:
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = (value / magnitude for value in (w, x, y, z))

    intrinsic, i, j, k, ell, epsilon, proper = _euler_sequence_metadata(sequence)
    coefficients = (w, x, y, z)
    if proper:
        a = w
        b = coefficients[i]
        c = coefficients[j]
        d = epsilon * coefficients[ell]
        kappa = 1.0
    else:
        a = w + coefficients[j]
        b = coefficients[i] + epsilon * coefficients[k]
        c = w - coefficients[j]
        d = coefficients[i] - epsilon * coefficients[k]
        kappa = epsilon

    r_plus = math.hypot(a, b)
    r_minus = math.hypot(c, d)
    chi = 2.0 * math.atan2(r_minus, r_plus)
    gimbal_lock = chi <= tolerance or math.pi - chi <= tolerance
    if chi <= tolerance:
        chi = 0.0
        u = math.atan2(b, a)
        alpha, gamma = ((2.0 * u, 0.0) if intrinsic
                        else (0.0, 2.0 * kappa * u))
    elif math.pi - chi <= tolerance:
        chi = math.pi
        v = math.atan2(d, c)
        alpha, gamma = ((2.0 * v, 0.0) if intrinsic
                        else (0.0, -2.0 * kappa * v))
    else:
        u = math.atan2(b, a)
        v = math.atan2(d, c)
        alpha, gamma = u + v, kappa * (u - v)

    beta = chi if proper else math.pi / 2.0 - chi
    if not intrinsic:
        alpha, gamma = gamma, alpha
    result = (_wrap_euler_outer(alpha), beta, _wrap_euler_outer(gamma))
    return EulerResult(result, gimbal_lock) if with_status else result


def _euler_from_quaternion(
    q: Iterable[FloatT], sequence: str, gimbal_tolerance: FloatLike = 1e-12,
    *, with_status: bool = False,
) -> Vector | EulerResult:
    return _euler_from_components(*_h._quaternion(q), sequence, gimbal_tolerance,
                                  with_status)


def from_euler(
    angles: Iterable[FloatT], sequence: str = "xyz", *, degrees: bool = False
) -> Quaternion:
    """Construct a rotation from angles and an Euler axis sequence.

    Lowercase sequences are extrinsic (fixed-axis); uppercase sequences are
    intrinsic (body-axis). All six Tait-Bryan and six proper Euler sequences
    are supported. Angles use radians unless ``degrees=True``.
    """
    values = _h._vector(angles)
    intrinsic, i, j, k, ell, epsilon, proper = _euler_sequence_metadata(sequence)
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    if degrees:
        values = tuple(math.radians(value) for value in values)  # type: ignore[assignment]
    if not intrinsic:
        values = tuple(reversed(values))  # type: ignore[assignment]
    a1, a2, a3 = values
    return _euler_from_validated_angles(
        a1, a2, a3, i, j, k, ell, epsilon, proper,
        algebra_epsilon=1, scalar_last=False,
    )


def _euler_from_validated_angles(
    a1: float,
    a2: float,
    a3: float,
    i: int,
    j: int,
    k: int,
    ell: int,
    epsilon: int,
    proper: bool,
    *,
    algebra_epsilon: int,
    scalar_last: bool,
) -> Quaternion:
    """Build canonical quaternion coefficients in the requested storage order."""
    if proper:
        u = 0.5 * a1 + 0.5 * a3
        v = 0.5 * a1 - 0.5 * a3
        w = math.cos(0.5 * a2) * math.cos(u)
        qi = algebra_epsilon * math.cos(0.5 * a2) * math.sin(u)
        qj = algebra_epsilon * math.sin(0.5 * a2) * math.cos(v)
        qell = algebra_epsilon * epsilon * math.sin(0.5 * a2) * math.sin(v)
        components = [0.0, 0.0, 0.0, 0.0]
        components[0], components[i], components[j], components[ell] = w, qi, qj, qell
    else:
        c1, c2, c3 = math.cos(a1 * 0.5), math.cos(a2 * 0.5), math.cos(a3 * 0.5)
        s1, s2, s3 = math.sin(a1 * 0.5), math.sin(a2 * 0.5), math.sin(a3 * 0.5)
        w = c1*c2*c3 - epsilon*s1*s2*s3
        qi = s1*c2*c3 + epsilon*c1*s2*s3
        qj = c1*s2*c3 - epsilon*s1*c2*s3
        qk = c1*c2*s3 + epsilon*s1*s2*c3
        components = [w, 0.0, 0.0, 0.0]
        components[i], components[j], components[k] = (
            algebra_epsilon * qi, algebra_epsilon * qj, algebra_epsilon * qk
        )

    if scalar_last:
        result = [components[1], components[2], components[3], components[0]]
        tie_component = next((value for value in result[:3] if value != 0.0), -1.0)
        flip = result[3] < 0.0 or (
            result[3] == 0.0 and algebra_epsilon * tie_component < 0.0
        )
    else:
        result = components
        tie_component = next((value for value in result[1:] if value != 0.0), 1.0)
        flip = result[0] < 0.0 or (result[0] == 0.0 and tie_component < 0.0)
    if flip:
        result = [-value for value in result]
    result = [0.0 if value == 0.0 else value for value in result]
    return tuple(result)  # type: ignore[return-value]


def to_euler(
    q: Iterable[FloatT], sequence: str = "xyz",
    gimbal_tolerance: FloatLike = 1e-12, *, degrees: bool = False,
) -> Vector:
    """Extract principal angles in a Tait-Bryan or proper Euler sequence.

    Lowercase sequences are extrinsic; uppercase sequences are intrinsic. At
    gimbal lock, the third angle of the requested sequence is set to zero.
    ``gimbal_tolerance`` is the angular distance in radians at which the
    singular representative is used. Angles are in radians unless
    ``degrees=True``.
    """
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    angles: Vector = _euler_from_quaternion(
        q, sequence, gimbal_tolerance
    )  # type: ignore[assignment]
    if degrees:
        return tuple(math.degrees(value) for value in angles)  # type: ignore[arg-type,return-value]
    return angles


def to_euler_with_status(
    q: Iterable[FloatT], sequence: str = "xyz", *, degrees: bool = False,
    gimbal_tolerance: FloatLike = 1e-12,
) -> EulerResult:
    """Extract Euler angles together with the scalar gimbal-lock flag."""
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a bool")  # noqa: TRY004
    result: EulerResult = _euler_from_quaternion(
        q, sequence, gimbal_tolerance, with_status=True
    )  # type: ignore[assignment]
    if not degrees:
        return result
    return EulerResult(
        tuple(math.degrees(value) for value in result.angles),  # type: ignore[arg-type]
        result.gimbal_lock,
    )


def scalar_part(q: Iterable[FloatT]) -> float:
    """Return the real/scalar coefficient of a Hamilton ``wxyz`` value."""
    return _h._quaternion(q)[0]


def vector_part(q: Iterable[FloatT]) -> Vector:
    """Return the three imaginary coefficients of a Hamilton ``wxyz`` value."""
    return _h._quaternion(q)[1:]


def commutator(lhs: Iterable[FloatT], rhs: Iterable[FloatT]) -> Quaternion:
    """Return the algebra commutator ``lhs*rhs - rhs*lhs``."""
    return _h.subtract(_h.multiply(lhs, rhs), _h.multiply(rhs, lhs))


def quaternion_derivative(
    q: Iterable[FloatT], omega: Iterable[FloatT], frame: str = "body"
) -> Quaternion:
    """Return ``dq/dt`` for angular velocity in radians per second.

    For body-frame rates this is ``0.5*q*(0, omega)``; for world-frame rates
    it is ``0.5*(0, omega)*q``. The input quaternion is not normalized.
    """
    side = _frame(frame)
    pure_rate = (0.0, *_h._vector(omega))
    product = (_h.multiply(q, pure_rate) if side == "body"
               else _h.multiply(pure_rate, q))
    return _h.scale(product, 0.5)


def integrate_angular_velocity_samples(
    q: Iterable[FloatT],
    omegas: Iterable[Iterable[FloatT]],
    times: Iterable[FloatT],
    frame: str = "body",
    method: str = "midpoint",
) -> Quaternion:
    """Integrate sampled rates using exact constant-rate increments per step.

    ``method='left'`` holds each interval's first rate. ``method='midpoint'``
    uses the average of adjacent samples, giving a second-order quadrature for
    smooth rates. Timestamps must be finite and strictly increasing.
    """
    side = _frame(frame)
    if method not in ("left", "midpoint"):
        raise ValueError("method must be 'left' or 'midpoint'")
    samples = tuple(_h._vector(omega) for omega in omegas)
    stamps = tuple(_h._real_scalar(value, "time") for value in times)
    if not samples or len(samples) != len(stamps):
        raise ValueError("omegas and times must have the same nonzero length")
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("times must be strictly increasing")
    result = _h.normalize(q)
    for index, (left, right) in enumerate(zip(samples, samples[1:])):
        if method == "left":
            rate: Vector = left
        else:
            rate = tuple((a + b) * 0.5 for a, b in zip(left, right))  # type: ignore[assignment]
        result = _h.integrate_angular_velocity(
            result, rate, stamps[index + 1] - stamps[index], frame=side
        )
    return _h.normalize(result)


def boxplus(q: Iterable[FloatT], delta: Iterable[FloatT], frame: str = "body") -> Quaternion:
    """Apply a local rotation-vector increment to an attitude.

    Body increments compose on the right; world increments compose on the
    left. ``delta`` is a three-vector in radians.
    """
    side = _frame(frame)
    base = _h.normalize(q)
    increment = _h.from_rotvec(delta)
    return _h.normalize(
        _h.multiply(base, increment) if side == "body"
        else _h.multiply(increment, base)
    )


def boxminus(q: Iterable[FloatT], reference: Iterable[FloatT],
             frame: str = "body") -> Vector:
    """Return the local radians vector taking ``reference`` to ``q``."""
    side = _frame(frame)
    target, base = _h.normalize(q), _h.normalize(reference)
    relative = (_h.multiply(_h.inverse(base), target) if side == "body"
                else _h.multiply(target, _h.inverse(base)))
    return _h.to_rotvec(relative)


def nlerp(q0: Iterable[FloatT], q1: Iterable[FloatT], t: FloatLike,
          shortest: bool = True) -> Quaternion:
    """Normalized linear interpolation of unit rotations; extrapolation works."""
    amount = _h._real_scalar(t, "t")
    if not isinstance(shortest, bool):
        raise ValueError("shortest must be a bool")  # noqa: TRY004
    a, b = _h.normalize(q0), _h.normalize(q1)
    if shortest and _h.dot(a, b) < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
    return _h.normalize(tuple((1.0 - amount) * x + amount * y
                              for x, y in zip(a, b)))


def squad(q0: Iterable[FloatT], q1: Iterable[FloatT], a: Iterable[FloatT],
          b: Iterable[FloatT], t: FloatLike) -> Quaternion:
    """Evaluate SQUAD using endpoint quaternions and their control quaternions."""
    amount = _h._real_scalar(t, "t")
    first = _h.slerp(q0, q1, amount)
    second = _h.slerp(a, b, amount)
    return _h.slerp(first, second, 2.0 * amount * (1.0 - amount))


def align_sign(q: Iterable[FloatT], reference: Iterable[FloatT]) -> Quaternion:
    """Choose the sign of ``q`` closest to ``reference`` without normalization."""
    value, base = _h._quaternion(q), _h._quaternion(reference)
    return tuple(-item for item in value) if _h.dot(value, base) < 0.0 else value  # type: ignore[return-value]


def make_sign_continuous(quaternions: Iterable[Iterable[FloatT]]) -> tuple[Quaternion, ...]:
    """Flip signs along a sequence so consecutive coefficient dots are nonnegative.

    The first input sign and all quaternion magnitudes are preserved.
    """
    values = tuple(_h._quaternion(q) for q in quaternions)
    if not values:
        return ()
    out = [values[0]]
    for value in values[1:]:
        out.append(align_sign(value, out[-1]))
    return tuple(out)


def to_mrp(q: Iterable[FloatT]) -> Vector:
    """Convert an attitude to principal modified Rodrigues parameters."""
    w, x, y, z = _h.canonicalize(_h.normalize(q))
    denominator = 1.0 + w
    if denominator == 0.0:
        raise ValueError("MRP is singular for a 360-degree attitude")
    return x / denominator, y / denominator, z / denominator


def from_mrp(sigma: Iterable[FloatT]) -> Quaternion:
    """Convert principal modified Rodrigues parameters to a unit attitude."""
    x, y, z = _h._vector(sigma)
    scale = max(abs(x), abs(y), abs(z))
    if scale == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    if scale <= 1.0:
        norm2 = x*x + y*y + z*z
        denominator = 1.0 + norm2
        return _h.canonicalize(_h.normalize(((1.0 - norm2) / denominator,
                                             2.0*x / denominator,
                                             2.0*y / denominator,
                                             2.0*z / denominator)))
    sx, sy, sz = x / scale, y / scale, z / scale
    inverse_scale = 1.0 / scale
    norm2 = sx*sx + sy*sy + sz*sz
    denominator = inverse_scale*inverse_scale + norm2
    return _h.canonicalize(_h.normalize((
        (inverse_scale*inverse_scale - norm2) / denominator,
        (2.0 * inverse_scale * sx) / denominator,
        (2.0 * inverse_scale * sy) / denominator,
        (2.0 * inverse_scale * sz) / denominator,
    )))


def swing_twist(q: Iterable[FloatT], axis: Iterable[FloatT]) -> tuple[Quaternion, Quaternion]:
    """Decompose the normalized input as ``q = swing * twist`` about an axis.

    If the projected twist is degenerate (a half-turn perpendicular to the
    axis), twist is identity and swing is the normalized input. The signs of
    both factors are retained so their coefficient product matches the
    normalized input, including when its scalar coefficient is negative.
    """
    value = _h.normalize(q)
    direction = _h._axis(_h._vector(axis))
    w, x, y, z = value
    projection = x*direction[0] + y*direction[1] + z*direction[2]
    try:
        twist = _h.normalize((w, projection*direction[0],
                              projection*direction[1], projection*direction[2]))
    except ValueError:
        twist = (1.0, 0.0, 0.0, 0.0)
    swing = _h.normalize(_h.multiply(value, _h.inverse(twist)))
    return swing, twist


def _jacobi_largest(matrix: Sequence[Sequence[float]]) -> tuple[float, tuple[float, ...], float]:
    """Return the largest eigenpair and runner-up for a real symmetric matrix."""
    size = len(matrix)
    a = [list(row) for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(100 * size * size):
        p, q = max(((i, j) for i in range(size) for j in range(i + 1, size)),
                   key=lambda pair: abs(a[pair[0]][pair[1]]))
        off = a[p][q]
        diagonal_scale = max(1.0, *(abs(a[i][i]) for i in range(size)))
        if abs(off) <= 2e-16 * diagonal_scale:
            break
        tau = (a[q][q] - a[p][p]) / (2.0 * off)
        t = math.copysign(1.0, tau) / (abs(tau) + math.hypot(1.0, tau))
        if tau == 0.0:
            t = 1.0
        c = 1.0 / math.hypot(1.0, t)
        s = t * c
        app, aqq = a[p][p], a[q][q]
        a[p][p] = app - t * off
        a[q][q] = aqq + t * off
        a[p][q] = a[q][p] = 0.0
        for k in range(size):
            if k != p and k != q:
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c*akp - s*akq
                a[k][q] = a[q][k] = s*akp + c*akq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = c*vkp - s*vkq
            vectors[k][q] = s*vkp + c*vkq
    order = sorted(range(size), key=lambda i: a[i][i], reverse=True)
    largest_index = order[0]
    eigenvector = tuple(vectors[i][largest_index] for i in range(size))
    return a[largest_index][largest_index], eigenvector, a[order[1]][order[1]]


def _weights(values: Iterable[FloatT] | None, count: int) -> tuple[float, ...]:
    if values is None:
        return (1.0 / count,) * count
    weights = tuple(_h._real_scalar(value, "weight") for value in values)
    if len(weights) != count or any(value < 0.0 for value in weights):
        raise ValueError("weights must be nonnegative and match the sample count")
    if not any(weights):
        raise ValueError("at least one weight must be positive")
    # Normalize after scaling by the largest value. This makes all weighted
    # rotation results invariant to a common factor and prevents an otherwise
    # avoidable overflow when several finite weights are near float64's limit.
    scale = max(weights)
    scaled = tuple(value / scale for value in weights)
    total = math.fsum(scaled)
    return tuple(value / total for value in scaled)


def rotation_mean(
    quaternions: Iterable[Iterable[FloatT]],
    weights: Iterable[FloatT] | None = None,
    method: str = "markley",
    tolerance: FloatLike = 1e-12,
    max_iterations: int = 100,
) -> Quaternion:
    """Average rotations by weighted Markley or intrinsic Karcher mean.

    ``method='markley'`` computes the dominant eigenvector of the weighted
    quaternion scatter matrix. ``method='intrinsic'`` iterates in the body
    tangent space, initialized from the Markley mean.
    """
    samples = tuple(_h.normalize(q) for q in quaternions)
    if not samples:
        raise ValueError("at least one quaternion is required")
    sample_weights = _weights(weights, len(samples))
    if method not in ("markley", "intrinsic"):
        raise ValueError("method must be 'markley' or 'intrinsic'")
    scatter = [[math.fsum(weight * sample[i] * sample[j]
                          for sample, weight in zip(samples, sample_weights))
                for j in range(4)] for i in range(4)]
    _, eigenvector, _ = _jacobi_largest(scatter)
    mean = _h.canonicalize(_h.normalize(eigenvector))
    if method == "markley":
        return mean
    tol = _h._real_scalar(tolerance, "tolerance")
    if tol <= 0.0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    for _ in range(max_iterations):
        errors = tuple(_h.to_rotvec(_h.multiply(_h.inverse(mean), sample))
                       for sample in samples)
        delta = tuple(math.fsum(weight * error[i]
                                for error, weight in zip(errors, sample_weights))
                      for i in range(3))
        if math.hypot(*delta) <= tol:
            return mean
        mean = boxplus(mean, delta, frame="body")
    raise RuntimeError("intrinsic rotation mean did not converge")


def rotation_covariance(
    quaternions: Iterable[Iterable[FloatT]],
    mean: Iterable[FloatT] | None = None,
    weights: Iterable[FloatT] | None = None,
    frame: str = "body",
) -> Matrix3:
    """Return the weighted population covariance in a local rotation tangent."""
    side = _frame(frame)
    samples = tuple(_h.normalize(q) for q in quaternions)
    if not samples:
        raise ValueError("at least one quaternion is required")
    sample_weights = _weights(weights, len(samples))
    center = _h.normalize(mean) if mean is not None else rotation_mean(samples, sample_weights)
    errors = tuple(
        _h.boxminus(sample, center, frame=side) for sample in samples
    )  # type: ignore[assignment]
    covariance = tuple(tuple(math.fsum(weight * error[i] * error[j]
                                        for error, weight in zip(errors, sample_weights))
                              for j in range(3)) for i in range(3))
    return covariance  # type: ignore[return-value]


def align_vectors(
    source: Iterable[Iterable[FloatT]],
    target: Iterable[Iterable[FloatT]],
    weights: Iterable[FloatT] | None = None,
) -> Quaternion:
    """Solve the weighted Wahba problem mapping source vectors onto targets.

    At least two independent vector directions are needed; ambiguous data raise
    ``ValueError`` rather than selecting an arbitrary attitude about a vector.
    Vector lengths are retained, as in SciPy's Wahba solver, so they also act
    as observation strengths. Common rescaling of either vector set or all
    explicit weights does not change the result.
    """
    raw_sources = tuple(_h._vector(value) for value in source)
    raw_targets = tuple(_h._vector(value) for value in target)
    sources, targets = raw_sources, raw_targets
    if not sources or len(sources) != len(targets):
        raise ValueError("source and target must contain the same nonzero number of vectors")
    if any(not any(vector) for vector in sources + targets):
        raise ValueError("source and target vectors must be nonzero")
    sample_weights = _weights(weights, len(sources))
    source_scale = max(abs(value) for vector in sources for value in vector)
    target_scale = max(abs(value) for vector in targets for value in vector)
    source_vectors = tuple(
        tuple(value / source_scale for value in vector) for vector in raw_sources
    )  # type: ignore[assignment]
    target_vectors = tuple(
        tuple(value / target_scale for value in vector) for vector in raw_targets
    )  # type: ignore[assignment]
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
    return _h.canonicalize(_h.normalize((eigenvector[3], *eigenvector[:3])))


def random_quaternion(rng: object | None = None) -> Quaternion:
    """Draw a uniform unit quaternion on S3 using Shoemake's method.

    Supply an object with a ``random()`` method for deterministic draws. The
    returned sign is not canonicalized, so both representatives occur equally.
    """
    source = random if rng is None else rng
    draw = getattr(source, "random", None)
    if not callable(draw):
        raise ValueError("rng must provide random()")  # noqa: TRY004
    draw_float: Callable[[], FloatLike] = draw  # type: ignore[assignment]
    u1, u2, u3 = (float(draw_float()) for _ in range(3))
    if any(not math.isfinite(u) or not 0.0 <= u < 1.0 for u in (u1, u2, u3)):
        raise ValueError("rng.random() must return values in [0, 1)")
    x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return w, x, y, z


def random_rotation(rng: object | None = None) -> Quaternion:
    """Draw a uniform SO(3) attitude, represented by a unit quaternion."""
    return random_quaternion(rng)


def power_quaternion(q: Iterable[FloatT], exponent: Iterable[FloatT]) -> Quaternion:
    """Raise ``q`` to a quaternion exponent as ``exp(exponent * log(q))``.

    The left product order is part of the definition. Zero exponent returns
    identity; a zero base with nonzero exponent is undefined.
    """
    base, power = _h._quaternion(q), _h._quaternion(exponent)
    if not any(power):
        return (1.0, 0.0, 0.0, 0.0)
    if not any(base):
        raise ValueError("zero quaternion cannot have a nonzero quaternion power")
    return _h.exp(_h.multiply(power, _h.log(base)))


def expm1(q: Iterable[FloatT]) -> Quaternion:
    """Return ``exp(q)-1`` with a convergent series near zero."""
    value = _h._quaternion(q)
    if max(map(abs, value)) >= 0.5:
        result = _h.exp(value)
        return result[0] - 1.0, result[1], result[2], result[3]
    total = [0.0, 0.0, 0.0, 0.0]
    term: Quaternion = (1.0, 0.0, 0.0, 0.0)
    for index in range(1, 48):
        term = _h.scale(_h.multiply(term, value), 1.0 / index)
        for component in range(4):
            total[component] += term[component]
        if max(map(abs, term)) <= 2e-17 * max(1.0, max(map(abs, total))):
            break
    return _h._checked(tuple(total))  # type: ignore[return-value]


def log1p(q: Iterable[FloatT]) -> Quaternion:
    """Return ``log(1+q)`` using a convergent series near zero."""
    value = _h._quaternion(q)
    if max(map(abs, value)) >= 0.25:
        return _h.log((1.0 + value[0], value[1], value[2], value[3]))
    total = [0.0, 0.0, 0.0, 0.0]
    term: Quaternion = value
    for index in range(1, 96):
        factor = (1.0 if index & 1 else -1.0) / index
        for component in range(4):
            total[component] += factor * term[component]
        if max(map(abs, term)) <= 2e-17 * max(1.0, max(map(abs, total))):
            break
        term = _h.multiply(term, value)
    return _h._checked(tuple(total))  # type: ignore[return-value]


def log2(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion logarithm to base two."""
    return _h.scale(_h.log(q), 1.0 / math.log(2.0))


def log10(q: Iterable[FloatT]) -> Quaternion:
    """Return the principal quaternion logarithm to base ten."""
    return _h.scale(_h.log(q), 1.0 / math.log(10.0))


def _complex_slice(q: Iterable[FloatT], function: Callable[[complex], complex]) -> Quaternion:
    w, x, y, z = _h._quaternion(q)
    radius = math.hypot(x, y, z)
    value = function(complex(w, radius))
    if radius == 0.0:
        return _h._checked((value.real, value.imag, 0.0, 0.0))  # type: ignore[return-value]
    factor = value.imag / radius
    return _h._checked((value.real, factor * x, factor * y, factor * z))  # type: ignore[return-value]


def asin(q: Iterable[FloatT]) -> Quaternion:
    """Return principal quaternion arcsine using the selected complex slice."""
    return _complex_slice(q, cmath.asin)


def acos(q: Iterable[FloatT]) -> Quaternion:
    """Return principal quaternion arccosine using the selected complex slice."""
    return _complex_slice(q, cmath.acos)


def atan(q: Iterable[FloatT]) -> Quaternion:
    """Return principal quaternion arctangent using the selected complex slice."""
    return _complex_slice(q, cmath.atan)


def asinh(q: Iterable[FloatT]) -> Quaternion:
    """Return principal quaternion inverse hyperbolic sine."""
    return _complex_slice(q, cmath.asinh)


def acosh(q: Iterable[FloatT]) -> Quaternion:
    """Return principal quaternion inverse hyperbolic cosine."""
    return _complex_slice(q, cmath.acosh)


def atanh(q: Iterable[FloatT]) -> Quaternion:
    """Return principal quaternion inverse hyperbolic tangent."""
    return _complex_slice(q, cmath.atanh)


def to_homogeneous_matrix(
    q: Iterable[FloatT], translation: Iterable[FloatT] = (0.0, 0.0, 0.0)
) -> tuple[tuple[float, ...], ...]:
    """Return a 4x4 homogeneous active rotation and optional translation."""
    rotation = _h.to_rotation_matrix(q)
    offset = _h._vector(translation)
    return tuple((*rotation[row], offset[row]) for row in range(3)) + (
        (0.0, 0.0, 0.0, 1.0),
    )


def from_homogeneous_matrix(
    matrix: Iterable[Iterable[FloatT]], atol: FloatLike = 1e-8
) -> Quaternion:
    """Validate a rigid 4x4 matrix and return its rotation quaternion."""
    tolerance = _h._real_scalar(atol, "atol")
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("atol must be in [0, 1)")
    try:
        rows = tuple(tuple(_h._real_scalar(item, "matrix element") for item in row)
                     for row in matrix)
    except TypeError as exc:
        raise ValueError("matrix must have shape (4, 4)") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("matrix must have shape (4, 4)")
    if any(abs(a - b) > tolerance for a, b in zip(rows[3], (0.0, 0.0, 0.0, 1.0))):
        raise ValueError("matrix must have homogeneous bottom row (0, 0, 0, 1)")
    return _h.from_rotation_matrix(tuple(row[:3] for row in rows[:3]), tolerance)


def so3_adjoint(q: Iterable[FloatT]) -> Matrix3:
    """Return the SO(3) adjoint matrix, equal to the active rotation matrix."""
    return _h.to_rotation_matrix(q)  # type: ignore[return-value]


def _public_names() -> tuple[str, ...]:
    return (
        "acosh", "acos", "align_sign", "align_vectors", "asin", "asinh", "atan",
        "atanh", "boxminus", "boxplus", "commutator", "expm1", "from_euler",
        "from_homogeneous_matrix", "from_mrp", "integrate_angular_velocity_samples",
        "log1p", "log10", "log2", "make_sign_continuous", "nlerp", "power_quaternion",
        "quaternion_derivative", "random_quaternion", "random_rotation", "rotation_covariance",
        "rotation_mean", "scalar_part", "so3_adjoint", "squad", "swing_twist",
        "to_euler", "to_euler_with_status", "to_homogeneous_matrix", "to_mrp",
        "vector_part",
    )


__all__ = list(_public_names())
