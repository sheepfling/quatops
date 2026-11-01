"""Local, workload-specific timing; no performance guarantees.

Run: PYTHONPATH=src python experiments/convention_benchmark.py
Inputs are preconstructed. The scalar and NumPy timings include each
backend's public input validation and output construction.
"""

import platform
import sys
import timeit

import numpy as np

from quatops.hamilton import dataclass as hd
from quatops.hamilton import namedtuple as hn
from quatops.hamilton import numpy as ha
from quatops.hamilton import scalar as hs
from quatops.jpl import dataclass as jd
from quatops.jpl import namedtuple as jn
from quatops.jpl import numpy as ja
from quatops.jpl import scalar as js


def report(label, fn, count, scale, unit):
    values = sorted(timeit.repeat(fn, number=count, repeat=5))
    print(
        f"{label}: best={values[0] * scale / count:.3f} median={values[2] * scale / count:.3f} {unit}"
    )


print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform(), "NumPy:", np.__version__)
hq, hr, vector = (1.0, 2.0, 3.0, 4.0), (2.0, -3.0, 1.0, 4.0), (1.0, -2.0, 3.0)
jq, jr = js.from_hamilton(hq), js.from_hamilton(hr)
for name, ham, jpl, make_h, make_j in (
    ("tuple", hs, js, tuple, tuple),
    ("namedtuple", hn, jn, lambda q: hn.Quaternion(*q), lambda q: jn.Quaternion(*q)),
    ("dataclass", hd, jd, lambda q: hd.Quaternion(*q), lambda q: jd.Quaternion(*q)),
):
    qh, rh, qj, rj = make_h(hq), make_h(hr), make_j(jq), make_j(jr)
    for operation, hargs, jargs in (
        ("multiply", (qh, rh), (qj, rj)),
        ("rotate", (qh, vector), (qj, vector)),
    ):
        for convention, module, args in (("H", ham, hargs), ("J", jpl, jargs)):
            report(
                f"{name} {convention} {operation}",
                lambda module=module, operation=operation, args=args: getattr(
                    module, operation
                )(*args),
                3000,
                1e6,
                "us/call",
            )
qh = np.tile(hq, (10000, 1))
rh = np.tile(hr, (10000, 1))
qj = np.tile(jq, (10000, 1))
rj = np.tile(jr, (10000, 1))
v = np.tile(vector, (10000, 1))
for operation, hargs, jargs in (
    ("multiply", (qh, rh), (qj, rj)),
    ("rotate", (qh, v), (qj, v)),
):
    for convention, module, args in (("H", ha, hargs), ("J", ja, jargs)):
        report(
            f"NumPy {convention} {operation}",
            lambda module=module, operation=operation, args=args: getattr(
                module, operation
            )(*args),
            10,
            1e3,
            "ms/10k pairs",
        )
