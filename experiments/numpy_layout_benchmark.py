"""Time four NumPy convention/layout paths on a fixed local workload.

Run through ``python3 experiments/run.py numpy-layout-benchmark``. Inputs are
preconstructed; public operations include validation and allocation. Results
are not general performance guarantees.
"""

import platform
import sys
import timeit
from typing import Callable

import numpy as np

from quatops.hamilton import numpy_wxyz as hw
from quatops.hamilton import numpy_xyzw as hx
from quatops.jpl import numpy_wxyz as jw
from quatops.jpl import numpy_xyzw as jx


def main() -> None:
    """Run multiply and rotation timings for all four NumPy modules."""
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
    for name, module_, q, r in backends:
        for operation_, args_ in (("multiply", (q, r)), ("rotate", (q, vectors))):
            function: Callable[[], object] = (
                lambda module=module_, operation=operation_, args=args_: getattr(
                    module, operation
                )(*args)
            )
            timings = sorted(timeit.repeat(function, number=10, repeat=5))
            print(
                f"{name} {operation_}: best={timings[0] * 100:.3f} "
                f"median={timings[2] * 100:.3f} ms/10000 pairs"
            )


if __name__ == "__main__":
    main()
