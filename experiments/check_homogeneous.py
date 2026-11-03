"""Characterize experimental kernels against an exact rational oracle.

Run with the repository and experiments on the Python path. A failed check or
unavailable backend produces a nonzero exit status.
Fraction is used only in this test oracle, never as a runtime kernel.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Iterable, Sequence
from fractions import Fraction
from types import ModuleType
from typing import Callable

Vec3 = tuple[float, float, float]
PlainQuat = tuple[float, float, float, float]

def exact_rotate(q: Iterable[float], v: Iterable[float]) -> Vec3:
    w, x, y, z = (Fraction.from_float(float(t)) for t in q)
    vx, vy, vz = (Fraction.from_float(float(t)) for t in v)
    denominator = w * w + x * x + y * y + z * z
    if not denominator:
        raise ValueError("zero quaternion")
    rows = (
        (w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z),
    )
    return tuple(
        float(sum(a * b for a, b in zip(row, (vx, vy, vz))) / denominator)
        for row in rows
    )  # type: ignore[return-value]

TINY = math.nextafter(0.0, 1.0)
CASES = (
    ("identity-mixed", (1, 0, 0, 0), (1e308, 1e-308, TINY)),
    ("signed-permutation", (1, 0, 0, 1), (1e308, 1e-308, TINY)),
    ("numpy-near-diagonal", (1, 1, 1e-8, 0), (0, 1e308, 0)),
    ("numpy-near-diagonal-subnormal", (1, 1, 1e-8, 0), (0, 1e-300, 0)),
    ("pythagorean", (5, 12, 13, 0), (1e308, 1, 1)),
    (
        "large-pythagorean",
        (2000001.0, 2000002000000.0, 2000002000001.0, 0.0),
        (1e308, 1, 1),
    ),
    ("tiny-ratio-impact", (1e308, 0, 0, 1e-308), (0, 1e308, 0)),
    ("tiny-square-impact", (1, 1, 1e-200, 0), (0, 1e308, 0)),
    ("subnormal-q-huge-vector", (1, TINY, 0, 0), (0, 1e308, 0)),
    ("tiny-q", (TINY, TINY, 0, 0), (1e308, 1e-308, 0)),
    ("large-q", (1e308, 0, 0, 1e308), (1e308, 1e-308, TINY)),
    ("small-q-and-vector", (1e-300, 2e-300, -1e-300, 3e-300), (TINY, 0, 0)),
    ("generic-cancellation", (-2, -1, 0, 1), (1e308, -1e308, 3e-308)),
    (
        "final-dot-cancellation",
        (4576996.0, 641332161.0, 462136015.0, 0.0),
        (0.0, math.ldexp(4576996.0, 950), -math.ldexp(641332161.0, 950)),
    ),
    ("ordinary", (1, 2, 3, 4), (1, 2, 3)),
    ("unrepresentable-result", (-2, -1, 0, 1), (1.7e308, -1.7e308, -1.7e308)),
)

def compare_one(label: str, function: Callable[..., Iterable[float]], q: Iterable[float],
                v: Iterable[float]) -> bool:
    try:
        expected = exact_rotate(q, v)
    except OverflowError:
        try:
            actual = function(q, v)
        except OverflowError:
            print("  PASS", label, "correctly raised OverflowError")
            return True
        except (ArithmeticError, TypeError, ValueError, IndexError) as exc:
            print(
                "  FAIL",
                label,
                "expected OverflowError; raised",
                type(exc).__name__,
                str(exc),
            )
            return False
        print("  FAIL", label, "expected OverflowError; returned", actual)
        return False
    try:
        actual = tuple(float(x) for x in function(q, v))
    except (ArithmeticError, TypeError, ValueError, IndexError) as exc:
        print(
            "  FAIL",
            label,
            "expected",
            expected,
            "raised",
            type(exc).__name__,
            str(exc),
        )
        return False
    passed = len(actual) == 3 and all(
        math.isclose(a, e, rel_tol=1e-13, abs_tol=0.0) for a, e in zip(actual, expected)
    )
    print(
        " ", "PASS" if passed else "FAIL", label, "expected", expected, "actual", actual
    )
    return passed

def extra_checks(module: ModuleType) -> bool:
    print("  API checks:")
    passed = 0
    for q, v in (
            ((0, 0, 0, 0), (1, 0, 0)),
            ((1, 0, 0), (1, 0, 0)),
            ((1, 0, 0, 0), (1, 0)),
            ((float("nan"), 0, 0, 0), (1, 0, 0)),
    ):
        try:
            module.rotate(q, v)
        except ValueError:
            print("   PASS rejects", q, v)
            passed += 1
        except (ArithmeticError, TypeError, IndexError) as exc:
            print("   FAIL rejects", q, v, type(exc).__name__, str(exc))
        else:
            print("   FAIL accepts", q, v)
    return passed == 4

def array_checks(module: ModuleType) -> bool:
    import numpy as np

    q = np.array([[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 1.0]]])
    v = np.array([[[1e308, 1e-308, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    q_copy, v_copy = q.copy(), v.copy()
    q.flags.writeable = False
    v.flags.writeable = False
    old = np.seterr(all="raise")
    batch_ok = False
    empty_ok = False
    try:
        try:
            out = module.rotate(q, v)
            expected = np.array(
                [[exact_rotate(q[i, 0], v[0, j]) for j in range(3)] for i in range(2)]
            )
            shape_ok = out.shape == (2, 3, 3)
            values_ok = np.array_equal(out, expected)
            immutable = np.array_equal(q, q_copy) and np.array_equal(v, v_copy)
            errstate_ok = all(value == "raise" for value in np.geterr().values())
            batch_ok = all((shape_ok, values_ok, immutable, errstate_ok))
            print(
                "  broadcast/read-only/seterr:",
                "PASS"
                if batch_ok
                else "FAIL",
                "shape",
                shape_ok,
                "values",
                values_ok,
                "unchanged",
                immutable,
                "seterr",
                errstate_ok,
            )
        except (ArithmeticError, TypeError, ValueError, IndexError) as exc:
            print("  broadcast/read-only/seterr: FAIL", type(exc).__name__, str(exc))
        try:
            out = module.rotate(np.empty((0, 1, 4)), np.empty((1, 3, 3)))
            empty_ok = out.shape == (0, 3, 3)
            print(
                "  empty broadcast:",
                "PASS" if empty_ok else "FAIL",
                out.shape,
            )
        except (ArithmeticError, TypeError, ValueError, IndexError) as exc:
            print("  empty broadcast: FAIL", type(exc).__name__, str(exc))
    finally:
        np.seterr(**old)
    return batch_ok and empty_ok

def main(names: Sequence[str] | None = None) -> int:
    names = list(names if names is not None else sys.argv[1:])
    names = names or ["homogeneous_scalar", "homogeneous_numpy"]
    all_passed = True
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            print(name, "not available:", str(exc))
            all_passed = False
            continue
        print("\n" + name)
        passed = sum(compare_one(label, module.rotate, q, v) for label, q, v in CASES)
        print("  numeric cases:", passed, "/", len(CASES), "passed")
        api_passed = extra_checks(module)
        all_passed = all_passed and passed == len(CASES) and api_passed
        if "numpy" in name:
            array_passed = array_checks(module)
            all_passed = all_passed and array_passed
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
