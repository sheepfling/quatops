"""Local four-layout NumPy timing, not a general performance guarantee.

Run from the project root: PYTHONPATH=src python experiments/numpy_layout_benchmark.py
Inputs are preconstructed; public operations include validation and allocation.
"""

import platform
import sys
import timeit

import numpy as np

from quatops.hamilton import numpy_wxyz as hw
from quatops.hamilton import numpy_xyzw as hx
from quatops.jpl import numpy_wxyz as jw
from quatops.jpl import numpy_xyzw as jx

print("Python", sys.version.replace("\n", " "), "NumPy", np.__version__)
print("Platform", platform.platform())
q_h = np.tile((1.0, 2.0, 3.0, 4.0), (10000, 1))
r_h = np.tile((2.0, -3.0, 1.0, 4.0), (10000, 1))
vectors = np.tile((1.0, -2.0, 3.0), (10000, 1))
backends = (
    ("H wxyz", hw, q_h, r_h),
    (
        "H xyzw",
        hx,
        np.ascontiguousarray(hx.from_wxyz(q_h)),
        np.ascontiguousarray(hx.from_wxyz(r_h)),
    ),
    (
        "J wxyz",
        jw,
        np.ascontiguousarray(jw.from_hamilton(q_h)),
        np.ascontiguousarray(jw.from_hamilton(r_h)),
    ),
    (
        "J xyzw",
        jx,
        np.ascontiguousarray(jx.from_hamilton(q_h)),
        np.ascontiguousarray(jx.from_hamilton(r_h)),
    ),
)
for name, module, q, r in backends:
    for operation, args in (("multiply", (q, r)), ("rotate", (q, vectors))):
        call = lambda module=module, operation=operation, args=args: getattr(
            module, operation
        )(*args)
        results = sorted(timeit.repeat(call, number=10, repeat=5))
        print(
            f"{name} {operation}: best={results[0] * 100:.3f} median={results[2] * 100:.3f} ms/10000 pairs"
        )
