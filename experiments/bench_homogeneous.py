"""Compare experimental homogeneous kernels with the current backends.

Run with ``PYTHONPATH=src:. python experiments/bench_homogeneous.py``.
This is a microbenchmark, not a speed or correctness guarantee.
"""

import platform
import statistics
import sys
import timeit
from functools import partial

import numpy as np

from experiments import homogeneous_numpy, homogeneous_scalar
from quatops import scalar as current_scalar
from quatops.numpy.hamilton import wxyz as current_numpy

CASES = {
    "ordinary": ((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0)),
    "zero coefficient": ((5.0, 12.0, 13.0, 0.0), (1e308, 1.0, 1.0)),
    "extreme scales": ((1e-300, 2e-300, 3e-300, 4e-300), (1e308, 1e-308, 0.0)),
}

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform(), platform.machine())
print("NumPy:", np.__version__)
for name, (q, v) in CASES.items():
    print("\n", name)
    for label, fn in (
        ("current scalar", current_scalar.rotate),
        ("homogeneous scalar", homogeneous_scalar.rotate),
    ):
        timings = timeit.repeat(partial(fn, q, v), number=3000, repeat=5)
        print(
            label,
            "best/median us per item",
            round(min(timings) * 1e6 / 3000, 3),
            round(statistics.median(timings) * 1e6 / 3000, 3),
        )
    q_array, v_array = np.tile(q, (10000, 1)), np.tile(v, (10000, 1))
    for label, fn in (
        ("current NumPy", current_numpy.rotate),
        ("homogeneous NumPy", homogeneous_numpy.rotate),
    ):
        timings = timeit.repeat(partial(fn, q_array, v_array), number=10, repeat=5)
        print(
            label,
            "best/median ms per 10000 items",
            round(min(timings) * 1e3 / 10, 3),
            round(statistics.median(timings) * 1e3 / 10, 3),
        )
