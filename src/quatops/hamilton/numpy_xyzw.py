"""Native NumPy Hamilton quaternions: scalar-last ``(..., 4)`` in xyzw order.

The product uses a positive vector cross term. Rotations are active and
right-handed; ``multiply(q, r)`` applies ``r`` first. Scalar-last ordering
does not change the Hamilton algebra. Arrays use float64 with the same
validation and NumPy floating-point policy as the scalar-first NumPy backend.
"""

from typing import Any

import numpy as np

from .._numpy import _components, _direction, _result, _scaled, _sum_three


def from_xyzw(q: Any) -> np.ndarray:
    """Return a copy of scalar-last ``(..., 4)`` quaternion values."""
    return _components(q, 4, "quaternion").copy()


def to_xyzw(q: Any) -> np.ndarray:
    """Return a copy of the Hamilton quaternion in scalar-last order."""
    return _components(q, 4, "quaternion").copy()


def from_wxyz(q: Any) -> np.ndarray:
    """Reorder scalar-first components into Hamilton scalar-last storage.

    This only changes layout; it does not change quaternion convention.
    """
    a = _components(q, 4, "quaternion")
    return a[..., [1, 2, 3, 0]]


def to_wxyz(q: Any) -> np.ndarray:
    """Reorder Hamilton components to scalar-first layout without changing convention."""
    a = _components(q, 4, "quaternion")
    return a[..., [3, 0, 1, 2]]


def conjugate(q: Any) -> np.ndarray:
    """Conjugate an xyzw quaternion without changing its magnitude."""
    out = _components(q, 4, "quaternion").copy()
    out[..., :3] *= -1
    return out


def multiply(q: Any, r: Any) -> np.ndarray:
    """Return the Hamilton product; for active rotations, ``r`` acts first."""
    a = _components(q, 4, "q")
    b = _components(r, 4, "r")
    a, b = np.broadcast_arrays(a, b)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    with np.errstate(all="ignore"):
        result = np.stack((
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by + ay*bw + az*bx - ax*bz,
            aw*bz + az*bw + ax*by - ay*bx,
            aw*bw - ax*bx - ay*by - az*bz,
        ), axis=-1)
    return _result(result)


def norm(q: Any) -> Any:
    """Return the norm as a float64 scalar or array of leading shape."""
    a = _components(q, 4, "quaternion")
    with np.errstate(all="ignore"):
        scale = np.max(np.abs(a), axis=-1, keepdims=True)
        scaled = np.divide(a, scale, out=np.zeros_like(a), where=scale != 0)
        result = (scale * np.sqrt(np.sum(scaled * scaled, axis=-1, keepdims=True)))[..., 0]
    return _result(result[()] if result.ndim == 0 else result)


def normalize(q: Any) -> np.ndarray:
    """Return unit quaternions; reject a zero quaternion anywhere in a batch."""
    a = _components(q, 4, "quaternion")
    scaled = _scaled(a)
    with np.errstate(all="ignore"):
        out = scaled.unit_scale / np.sqrt(scaled.sum_squares)
    return _result(out)


def inverse(q: Any) -> np.ndarray:
    """Return multiplicative inverses; reject a zero quaternion in a batch."""
    a = _components(q, 4, "quaternion")
    scaled_state = _scaled(a)
    scaled = scaled_state.unit_scale.copy()
    scaled[..., :3] *= -1
    with np.errstate(all="ignore"):
        out = (scaled / scaled_state.sum_squares) / scaled_state.scale
    return _result(out)


def rotate(q: Any, v: Any) -> np.ndarray:
    """Actively rotate vectors with a nonzero Hamilton quaternion.

    Leading quaternion and vector shapes broadcast. Internally the quaternion
    is rescaled for numerical range; inputs are never normalized or mutated.
    """
    a = _components(q, 4, "quaternion")
    v = _components(v, 3, "vector")
    scaled_state = _scaled(a)
    scaled = scaled_state.unit_scale
    sums = scaled_state.sum_squares
    shape = np.broadcast_shapes(scaled.shape[:-1], v.shape[:-1])
    scaled = np.broadcast_to(scaled, shape + (4,))
    v = np.broadcast_to(v, shape + (3,))
    sums = np.broadcast_to(sums[..., 0], shape)
    x, y, z, w = np.moveaxis(scaled, -1, 0)
    vx, vy, vz = np.moveaxis(v, -1, 0)
    with np.errstate(all="ignore"):
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        m00 = np.clip(((ww + xx) - yy - zz) / sums, -1.0, 1.0)
        m01 = np.clip((2 * (x*y - w*z)) / sums, -1.0, 1.0)
        m02 = np.clip((2 * (x*z + w*y)) / sums, -1.0, 1.0)
        m10 = np.clip((2 * (x*y + w*z)) / sums, -1.0, 1.0)
        m11 = np.clip(((ww - xx) + yy - zz) / sums, -1.0, 1.0)
        m12 = np.clip((2 * (y*z - w*x)) / sums, -1.0, 1.0)
        m20 = np.clip((2 * (x*z - w*y)) / sums, -1.0, 1.0)
        m21 = np.clip((2 * (y*z + w*x)) / sums, -1.0, 1.0)
        m22 = np.clip(((ww - xx) - yy + zz) / sums, -1.0, 1.0)
        out = np.stack((
            _sum_three(m00*vx, m01*vy, m02*vz),
            _sum_three(m10*vx, m11*vy, m12*vz),
            _sum_three(m20*vx, m21*vy, m22*vz),
        ), axis=-1)
    return _result(out)


def _canonical(q: np.ndarray) -> np.ndarray:
    """Choose sign from original xyzw, matching canonical physical rotation."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    first = np.where(x != 0, x, np.where(y != 0, y, z))
    # The first nonzero vector component is positive at an exact half-turn.
    flip = (w < 0) | ((w == 0) & (first < 0))
    with np.errstate(all="ignore"):
        out = np.where(flip[..., None], -q, q)
    return np.where(out == 0, 0.0, out)


def _from_unit_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        half = angle * 0.5
        w = np.cos(half)
        w = np.where(np.abs(angle) == np.pi, 0.0, w)
        vector = axis * np.sin(half)[..., None]
        q = np.concatenate((vector, w[..., None]), axis=-1)
    return _canonical(_result(q))


def from_axis_angle(a: Any) -> np.ndarray:
    """Convert (...,4) axis and radians to canonical Hamilton quaternions.

    Conversion explicitly normalizes the finite nonzero axis, including at
    zero angle. Trigonometric argument reduction at huge angles is platform
    dependent.
    """
    a = _components(a, 4, "axis-angle")
    axis, _, _ = _direction(a[..., :3])
    return _from_unit_axis_angle(axis, a[..., 3])


def from_rotvec(r: Any) -> np.ndarray:
    """Convert (...,3) axis-times-radian vectors to Hamilton quaternions."""
    r = _components(r, 3, "rotation vector")
    axis, scale, length = _direction(r, allow_zero=True)
    with np.errstate(all="ignore"):
        angle = (scale * length)[..., 0]
    return _from_unit_axis_angle(axis, _result(angle))


def _axis_angle(q: Any) -> tuple:
    a = _components(q, 4, "quaternion")
    if np.any(np.all(a == 0, axis=-1)):
        raise ValueError("zero quaternion has no rotation")
    a = _canonical(a)
    axis, vector_scale, length = _direction(a[..., :3], allow_zero=True)
    vscale = vector_scale[..., 0]
    magnitude = length[..., 0]
    w = a[..., 3]
    with np.errstate(all="ignore"):
        scale = np.maximum(w, vscale)
        angle = 2.0 * np.arctan2((vscale / scale) * magnitude, w / scale)
        tiny = (w > 0) & (vscale <= w * 1e-8)
        v_mantissa, v_exponent = np.frexp(vscale)
        w_mantissa, w_exponent = np.frexp(w)
        small_angle = np.ldexp(
            (2.0 * v_mantissa / w_mantissa) * magnitude,
            v_exponent - w_exponent,
        )
        angle = np.where(tiny, small_angle, angle)
        axis = np.where((angle == 0)[..., None], np.array([1.0, 0.0, 0.0]), axis)
        axis = np.where(axis == 0, 0.0, axis)
        angle = np.where(angle == 0, 0.0, angle)
    return _result(axis), _result(angle)


def to_axis_angle(q: Any) -> np.ndarray:
    """Return (...,4) canonical physical axis and principal angle in radians."""
    axis, angle = _axis_angle(q)
    return np.concatenate((axis, angle[..., None]), axis=-1)


def to_rotvec(q: Any) -> np.ndarray:
    """Return (...,3) canonical physical axis-times-angle vectors."""
    axis, angle = _axis_angle(q)
    original = _canonical(_components(q, 4, "quaternion"))
    vector = original[..., :3]
    w = original[..., 3]
    with np.errstate(all="ignore"):
        vector_scale = np.max(np.abs(vector), axis=-1)
        tiny = (w > 0) & (vector_scale <= w * 1e-8)
        small_vector = (2.0 * vector) / w[..., None]
        result = np.where(tiny[..., None], small_vector, axis * angle[..., None])
    return np.where(result == 0, 0.0, _result(result))
