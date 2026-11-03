"""Compare rotation candidates and save opt-in benchmark results as JSON.

Run from the repository root with ``PYTHONPATH=src`` and the local environment::

    PYTHONPATH=src .venv/bin/python experiments/rotation_accuracy_comparison.py \
        --json /tmp/quatops-rotation-accuracy.json

The compensated prototypes are evaluated for correctness but are not timed
unless they meet the exact-rounding contract. The integer-domain candidate is
an experiment: it uses a certified fast path and falls back to the exact path.
"""

import argparse
import json
import math
import platform
import random
import statistics
import sys
import timeit
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from check_homogeneous import CASES, exact_rotate
from homogeneous_numpy import rotate as homogeneous_numpy
from homogeneous_scalar import rotate as homogeneous_scalar
from quatops import scalar
from quatops.numpy.hamilton import wxyz as array


def certified_integer_fallback(q, v):
    """Round exactly for small integers; use exact arithmetic for all other inputs.

    With |q_i| <= 2**10 and |v_i| <= 2**20, the integer quadratic
    numerators have magnitude below 2**44 and the denominator is at most
    2**22. Both are exactly representable as binary64, so their single
    floating-point division is correctly rounded on IEEE binary64 machines.
    """
    q = scalar._quaternion(q)
    v = scalar._vector(v)
    if (
        all(value.is_integer() for value in q + v)
        and max(map(abs, q)) <= 2**10
        and max(map(abs, v)) <= 2**20
    ):
        w, x, y, z = map(int, q)
        vx, vy, vz = map(int, v)
        denominator = w*w + x*x + y*y + z*z
        if denominator == 0:
            raise ValueError("zero quaternion has no normalization or inverse")
        numerators = (
            (w*w+x*x-y*y-z*z)*vx + 2*(x*y-w*z)*vy + 2*(x*z+w*y)*vz,
            2*(x*y+w*z)*vx + (w*w-x*x+y*y-z*z)*vy + 2*(y*z-w*x)*vz,
            2*(x*z-w*y)*vx + 2*(y*z+w*x)*vy + (w*w-x*x-y*y+z*z)*vz,
        )
        return tuple(float(value) / float(denominator) for value in numerators)
    return scalar.rotate(q, v, accurate=True)


def same_components(lhs, rhs):
    return len(lhs) == len(rhs) and all(
        a == b and (a != 0.0 or math.copysign(1.0, a) == math.copysign(1.0, b))
        for a, b in zip(lhs, rhs)
    )


def assess(function):
    results = []
    for name, q, v in CASES:
        try:
            expected = exact_rotate(q, v)
        except OverflowError:
            try:
                function(q, v)
            except OverflowError:
                passed, detail = True, "raised OverflowError"
            except (ArithmeticError, TypeError, ValueError) as exc:
                passed, detail = False, f"raised {type(exc).__name__}"
            else:
                passed, detail = False, "failed to raise OverflowError"
        else:
            try:
                actual = tuple(float(value) for value in function(q, v))
            except (ArithmeticError, TypeError, ValueError) as exc:
                passed, detail = False, f"raised {type(exc).__name__}"
            else:
                passed = same_components(actual, expected)
                detail = {"actual": [value.hex() for value in actual],
                          "expected": [value.hex() for value in expected]}
        results.append({"case": name, "passed": passed, "detail": detail})
    return results


def check_integer_domain():
    rng = random.Random(683908)
    for _ in range(1000):
        q = tuple(float(rng.randint(-(2**10), 2**10)) for _ in range(4))
        v = tuple(float(rng.randint(-(2**20), 2**20)) for _ in range(3))
        if not any(q):
            q = (1.0, 0.0, 0.0, 0.0)
        if not same_components(certified_integer_fallback(q, v), exact_rotate(q, v)):
            raise AssertionError("certified integer-domain candidate failed its reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path, help="run artifact path")
    args = parser.parse_args()
    candidates = {
        "homogeneous_scalar": homogeneous_scalar,
        "homogeneous_numpy": homogeneous_numpy,
        "certified_integer_fallback": certified_integer_fallback,
        "accurate_scalar": lambda q, v: scalar.rotate(q, v, accurate=True),
        "accurate_numpy": lambda q, v: array.rotate(q, v, accurate=True),
    }
    accuracy = {name: assess(function) for name, function in candidates.items()}
    check_integer_domain()
    for name in ("certified_integer_fallback", "accurate_scalar", "accurate_numpy"):
        if not all(result["passed"] for result in accuracy[name]):
            raise AssertionError(f"{name} did not meet the exact-rounding contract")

    q, v = (1.0, 2.0, 3.0, 4.0), (15.0, 30.0, 45.0)
    q_array, v_array = np.tile(q, (64, 1)), np.tile(v, (64, 1))
    reference = exact_rotate(q, v)
    timed = {
        "default_scalar": (lambda: scalar.rotate(q, v), 1000),
        "certified_integer_fallback": (lambda: certified_integer_fallback(q, v), 1000),
        "accurate_scalar": (lambda: scalar.rotate(q, v, accurate=True), 1000),
        "default_numpy_64": (lambda: array.rotate(q_array, v_array), 100),
        "accurate_numpy_64": (lambda: array.rotate(q_array, v_array, accurate=True), 100),
    }
    for name, (function, _) in timed.items():
        actual = np.asarray(function()).reshape(-1, 3)
        if not all(same_components(tuple(row), reference) for row in actual):
            raise AssertionError(f"timed input for {name} failed the reference")
    timings = {
        name: {"seconds_per_call": [value / count for value in
                                      timeit.repeat(function, number=count, repeat=3)]}
        for name, (function, count) in timed.items()
    }
    for result in timings.values():
        result["median_seconds_per_call"] = statistics.median(result["seconds_per_call"])
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "integer_domain_samples": 1000,
        "accuracy": accuracy,
        "timings": timings,
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, results in accuracy.items():
        print(f"{name}: {sum(result['passed'] for result in results)}/{len(results)} exact")
    print(f"Saved run artifact: {args.json}")


if __name__ == "__main__":
    main()
