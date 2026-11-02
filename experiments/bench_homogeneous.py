"""Compare experimental homogeneous kernels with current backends.

Run through ``python3 experiments/run.py homogeneous-benchmark``. This is a
microbenchmark, not a speed or correctness guarantee.
"""

import platform
import statistics
import sys
import timeit
from functools import partial
from typing import Callable

import numpy as np

from experiments import homogeneous_numpy, homogeneous_scalar
from quatops import scalar as current_scalar
from quatops.numpy.hamilton import wxyz as current_numpy

CASES: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "ordinary": ((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0)),
    "zero coefficient": ((5.0, 12.0, 13.0, 0.0), (1e308, 1.0, 1.0)),
    "extreme scales": ((1e-300, 2e-300, 3e-300, 4e-300), (1e308, 1e-308, 0.0)),
}


def report(label: str, function: Callable[[], object], count: int, scale: float,
           unit: str) -> None:
    """Print best and median time per requested work unit."""
    timings = sorted(timeit.repeat(function, number=count, repeat=5))
    print(
        f"{label}: best={timings[0] * scale / count:.3f} "
        f"median={statistics.median(timings) * scale / count:.3f} {unit}"
    )


def main() -> None:
    """Run scalar and batched NumPy comparisons on fixed workloads."""
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform(), platform.machine())
    print("NumPy:", np.__version__)
    for name, (q, v) in CASES.items():
        print("\n", name)
        for label, function in (
            ("current scalar", partial(current_scalar.rotate, q, v)),
            ("homogeneous scalar", partial(homogeneous_scalar.rotate, q, v)),
        ):
            report(label, function, 3000, 1e6, "us/item")

        q_array = np.tile(q, (10000, 1))
        v_array = np.tile(v, (10000, 1))
        for label, function in (
            ("current NumPy", partial(current_numpy.rotate, q_array, v_array)),
            ("homogeneous NumPy", partial(homogeneous_numpy.rotate, q_array, v_array)),
        ):
            report(label, function, 10, 1e3, "ms/10000 items")


if __name__ == "__main__":
    main()
