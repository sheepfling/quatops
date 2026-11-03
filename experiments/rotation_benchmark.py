import sys
import timeit
from collections.abc import Iterable
from decimal import Decimal, localcontext
from fractions import Fraction as F

import numpy as np
import scipy
from numpy import ndarray

from quatops import scalar
from quatops.numpy.hamilton import wxyz as nq

def frac_rot(q: Iterable[float] | ndarray, v: Iterable[float] | ndarray) -> tuple[float, ...]:
    w, x, y, z = map(F.from_float, map(float, q))
    vx, vy, vz = map(F.from_float, map(float, v))
    den = w * w + x * x + y * y + z * z
    return (
        float(
            ((w * w + x * x - y * y - z * z) * vx
             + 2 * (x * y - w * z) * vy
             + 2 * (x * z + w * y) * vz)
            / den
        ),
        float(
            (2 * (x * y + w * z) * vx
             + (w * w - x * x + y * y - z * z) * vy
             + 2 * (y * z - w * x) * vz)
            / den
        ),
        float(
            (2 * (x * z - w * y) * vx
             + 2 * (y * z + w * x) * vy
             + (w * w - x * x - y * y + z * z) * vz)
            / den
        ),
    )

def decimal_rot(q: Iterable[float], v: Iterable[float]) -> tuple[float, ...]:
    with localcontext() as c:
        c.prec = 1200
        w, x, y, z = map(Decimal.from_float, map(float, q))
        vx, vy, vz = map(Decimal.from_float, map(float, v))
        den = w * w + x * x + y * y + z * z
        return (
            float(
                ((w * w + x * x - y * y - z * z) * vx
                 + 2 * (x * y - w * z) * vy
                 + 2 * (x * z + w * y) * vz)
                / den
            ),
            float(
                (2 * (x * y + w * z) * vx
                 + (w * w - x * x + y * y - z * z) * vy
                 + 2 * (y * z - w * x) * vz)
                / den
            ),
            float(
                (2 * (x * z - w * y) * vx
                 + 2 * (y * z + w * x) * vy
                 + (w * w - x * x - y * y + z * z) * vz)
                / den
            ),
        )

def main() -> None:
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

    def run1():
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

    def run2() -> None:
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

    run1()
    run2()

if __name__ == "__main__":
    main()
