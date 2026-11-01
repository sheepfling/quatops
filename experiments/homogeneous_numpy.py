"""Experimental vectorized homogeneous float64 rotation, not a public backend.

Power-of-two scaling and compensated coefficient products reduce some rounding
errors.  This prototype deliberately exposes known underflow limitations; it is
not a correctly rounded implementation and does not replace quatops.numpy.hamilton.wxyz.
"""

import numpy as np

from quatops._numpy import _components, _result, _sum_three

_SPLIT = 134217729.0  # 2**27 + 1 for Dekker splitting of binary64 values.


def _two_sum(a, b):
    total = a + b
    tail = total - a
    return total, (a - (total - tail)) + (b - tail)


def _two_product(a, b):
    # Quaternion inputs have magnitude at most 0.5, so splitting does not
    # overflow. The residual is not exact for underflowed products.
    product = a * b
    ca = _SPLIT * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = _SPLIT * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    error = ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return product, error


def _sum_expansion(terms):
    # Summing high terms plus product and addition residuals, sufficient for
    # the examples but not a general exact floating expansion.
    total = np.zeros_like(terms[0])
    error = np.zeros_like(total)
    for term in terms:
        total, correction = _two_sum(total, term)
        error = error + correction
    return total + error


def _coefficient(pairs):
    highs = []
    lows = []
    for a, b, sign in pairs:
        high, low = _two_product(a, b)
        highs.append(sign * high)
        lows.append(sign * low)
    return _sum_expansion(highs + lows)


def rotate(q, v):
    """Compute q*(0,v)*q^-1, vectorized, as a precision experiment."""
    q = _components(q, 4, "quaternion")
    v = _components(v, 3, "vector")
    with np.errstate(all="ignore"):
        largest = np.max(np.abs(q), axis=-1, keepdims=True)
        if np.any(largest == 0):
            raise ValueError("zero quaternion has no rotation")
        _, exponent = np.frexp(largest)
        # One extra binary scale factor ensures each component <= 0.5,
        # making the homogeneous matrix coefficients bounded by 1.
        q = np.ldexp(q, -exponent - 1)
        shape = np.broadcast_shapes(q.shape[:-1], v.shape[:-1])
        q = np.broadcast_to(q, shape + (4,))
        v = np.broadcast_to(v, shape + (3,))
        w, x, y, z = np.moveaxis(q, -1, 0)
        vx, vy, vz = np.moveaxis(v, -1, 0)
        norm2 = _coefficient(((w, w, 1), (x, x, 1), (y, y, 1), (z, z, 1)))
        m00 = _coefficient(((w, w, 1), (x, x, 1), (y, y, -1), (z, z, -1)))
        m01 = 2 * _coefficient(((x, y, 1), (w, z, -1)))
        m02 = 2 * _coefficient(((x, z, 1), (w, y, 1)))
        m10 = 2 * _coefficient(((x, y, 1), (w, z, 1)))
        m11 = _coefficient(((w, w, 1), (x, x, -1), (y, y, 1), (z, z, -1)))
        m12 = 2 * _coefficient(((y, z, 1), (w, x, -1)))
        m20 = 2 * _coefficient(((x, z, 1), (w, y, -1)))
        m21 = 2 * _coefficient(((y, z, 1), (w, x, 1)))
        m22 = _coefficient(((w, w, 1), (x, x, -1), (y, y, -1), (z, z, 1)))
        numer = np.stack(
            (
                _sum_three(m00 * vx, m01 * vy, m02 * vz),
                _sum_three(m10 * vx, m11 * vy, m12 * vz),
                _sum_three(m20 * vx, m21 * vy, m22 * vz),
            ),
            axis=-1,
        )
        return _result(numer / norm2[..., None])
