import sys
import timeit
from decimal import Decimal, localcontext
from fractions import Fraction as F

import numpy as np
import scipy

from quatops import scalar
from quatops.numpy.hamilton import wxyz as nq


def frac_rot(q, v):
    w, x, y, z = map(F.from_float, map(float, q))
    vx, vy, vz = map(F.from_float, map(float, v))
    den = w * w + x * x + y * y + z * z
    return tuple(
        float((a * vx + b * vy + c * vz) / den)
        for a, b, c in (
            (w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z),
        )
    )


def decimal_rot(q, v):
    with localcontext() as c:
        c.prec = 1200
        w, x, y, z = map(Decimal.from_float, map(float, q))
        vx, vy, vz = map(Decimal.from_float, map(float, v))
        den = w * w + x * x + y * y + z * z
        return tuple(
            float((a * vx + b * vy + c * vz) / den)
            for a, b, c in (
                (
                    w * w + x * x - y * y - z * z,
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ),
                (
                    2 * (x * y + w * z),
                    w * w - x * x + y * y - z * z,
                    2 * (y * z - w * x),
                ),
                (
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    w * w - x * x - y * y + z * z,
                ),
            )
        )


cases = [
    ("regular", (1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0)),
    ("zero diag", (5.0, 12.0, 13.0, 0.0), (1e308, 1.0, 1.0)),
    ("near-zero diag", (1.0, 1.0, 1e-8, 0.0), (0.0, 1e308, 0.0)),
]
print(
    "versions:",
    sys.version.replace("\n", " "),
    "numpy",
    np.__version__,
    "scipy",
    scipy.__version__,
)
for label, q, v in cases:
    print(
        label,
        "current:",
        scalar.rotate(q, v),
        "numpy:",
        nq.rotate(q, v),
        "fraction:",
        frac_rot(q, v),
        "decimal:",
        decimal_rot(q, v),
    )
    for name, fn in [
        ("scalar", scalar.rotate),
        ("fraction", frac_rot),
        ("decimal1200", decimal_rot),
    ]:
        loops = 3000
        timings = timeit.repeat(lambda fn=fn, q=q, v=v: fn(q, v), number=loops, repeat=5)
        print(
            label,
            name,
            "best us/op",
            round(min(timings) * 1e6 / loops, 3),
            "median us/op",
            round(sorted(timings)[2] * 1e6 / loops, 3),
        )
q = np.tile((1.0, 2.0, 3.0, 4.0), (10000, 1))
v = np.tile((1.0, 2.0, 3.0), (10000, 1))
for name, fn, loops in [
    ("numpy_10k", lambda: nq.rotate(q, v), 10),
    ("fraction_10k", lambda: [frac_rot(qi, vi) for qi, vi in zip(q, v)], 1),
]:
    timing = timeit.repeat(fn, number=loops, repeat=5)
    print(
        name,
        "best ms per batch",
        round(min(timing) * 1000 / loops, 3),
        "median ms",
        round(sorted(timing)[2] * 1000 / loops, 3),
    )
