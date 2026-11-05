"""Experimental homogeneous quaternion rotation, not a production backend.

The quaternion is power-of-two scaled before a compensated calculation of the
quadratic rotation numerator. The resulting matrix numerator is applied to the
vector, then divided by the common squared norm. Uses only Python 3.9 APIs.

This prototype is not production safe. In particular, an identity rotation of
(1e308, 1e-308, 5e-324) loses its final component when multiplication by the
homogeneous numerator underflows before division by the norm. Scaling can
also erase very small quaternion components: (1, 0, 0, 5e-324) rotating
(0, 1e308, 0) loses an approximately -9.88e-16 component. Tiny quadratic
products can underflow even when their effect after applying a large vector is
representable. No compensated product algorithm here prevents underflow.
"""

import math
from collections.abc import Iterable

from quatops import scalar

_SPLITTER = 134217729.0  # 2**27 + 1 for binary64 Dekker splitting


def _two_product(a: float, b: float) -> tuple[float, float]:
    """Return product and residual when no intermediate underflow occurs."""
    high = a * b
    a_temp, b_temp = _SPLITTER * a, _SPLITTER * b
    a_hi, b_hi = a_temp - (a_temp - a), b_temp - (b_temp - b)
    a_lo, b_lo = a - a_hi, b - b_hi
    low = ((a_hi * b_hi - high) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return high, low


def _sum_products(*terms: tuple[int, float, float]) -> float:
    parts = []
    for sign, a, b in terms:
        high, low = _two_product(a, b)
        parts.extend((sign * high, sign * low))
    return math.fsum(parts)


def _dot(a: float, b: float, c: float, x: float, y: float, z: float) -> float:
    terms = sorted((a * x, b * y, c * z), key=abs, reverse=True)
    if (terms[0] > 0 and terms[1] > 0 and terms[2] < 0) or (
        terms[0] < 0 and terms[1] < 0 and terms[2] > 0
    ):
        terms[1], terms[2] = terms[2], terms[1]
    return math.fsum(terms)


def rotate(q: Iterable[float], v: Iterable[float]) -> tuple[float, float, float]:
    """Rotate a vector, using compensated quadratic numerator coefficients.

    Experimental: scaling and individual products can underflow, and matrix
    coefficients and vector products are still individually rounded.
    """
    values = scalar._quaternion(q)
    x_v, y_v, z_v = scalar._vector(v)
    largest = max(map(abs, values))
    if largest == 0:
        raise ValueError("zero quaternion")
    # Largest q magnitude becomes [0.25, 0.5). Hence sum(q*q) is < 1,
    # reducing (not eliminating) spurious overflow in the numerator dot.
    _, exponent = math.frexp(largest)
    w, x, y, z = (math.ldexp(item, -exponent - 1) for item in values)
    denom = _sum_products((1, w, w), (1, x, x), (1, y, y), (1, z, z))
    rows = (
        (
            _sum_products((1, w, w), (1, x, x), (-1, y, y), (-1, z, z)),
            _sum_products((2, x, y), (-2, w, z)),
            _sum_products((2, x, z), (2, w, y)),
        ),
        (
            _sum_products((2, x, y), (2, w, z)),
            _sum_products((1, w, w), (-1, x, x), (1, y, y), (-1, z, z)),
            _sum_products((2, y, z), (-2, w, x)),
        ),
        (
            _sum_products((2, x, z), (-2, w, y)),
            _sum_products((2, y, z), (2, w, x)),
            _sum_products((1, w, w), (-1, x, x), (-1, y, y), (1, z, z)),
        ),
    )
    result = tuple(_dot(*row, x_v, y_v, z_v) / denom for row in rows)
    return scalar._checked(result)


def main() -> None:
    """Print a small example when this prototype module is run directly."""
    q = (1.0, 2.0, 3.0, 4.0)
    vector = (1.0, -2.0, 3.0)
    print("Experimental scalar rotation prototype")
    print("q:", q, "v:", vector)
    print("rotated:", rotate(q, vector))


if __name__ == "__main__":
    main()
