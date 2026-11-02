"""Compile public batch paths and report Numba's actual signatures.

Run from the repository root with ``.venv/bin/python
experiments/numba_coverage.py``. The report distinguishes registered kernels
from dispatchers that compiled a nopython signature in this process.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

import numpy as np

from quatops import _numpy
from quatops.engines import SignedAxisMap
from quatops.engines import numpy as engine_numpy
from quatops.numpy.hamilton import wxyz as hamilton_wxyz
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw

_BACKENDS = (
    ("Hamilton", "wxyz", hamilton_wxyz),
    ("Hamilton", "xyzw", hamilton_xyzw),
    ("JPL", "wxyz", jpl_wxyz),
    ("JPL", "xyzw", jpl_xyzw),
)


def _describe_kernel(name: str, kernel: Any) -> dict[str, Any]:
    """Report compiled signatures separately from lazy registration."""
    if kernel is None:
        return {"name": name, "state": "missing", "kind": "unknown", "signatures": []}

    nopython_signatures = getattr(kernel, "nopython_signatures", None)
    if nopython_signatures is not None:
        signatures = [str(signature) for signature in nopython_signatures]
        return {
            "name": name,
            "state": "compiled" if signatures else "registered",
            "kind": "njit",
            "signatures": signatures,
        }

    gufunc_signatures = getattr(kernel, "types", None)
    if gufunc_signatures is not None:
        signatures = [str(signature) for signature in gufunc_signatures]
        return {
            "name": name,
            "state": "compiled" if signatures else "registered",
            "kind": "gufunc",
            "signatures": signatures,
        }

    return {"name": name, "state": "unknown", "kind": type(kernel).__name__, "signatures": []}


def _exercise_public_batch_paths() -> None:
    """Cross each dispatcher threshold through the public NumPy APIs."""
    count = 128
    rng = np.random.default_rng(2026)
    rotation_vectors = rng.normal(scale=0.25, size=(count, 3))
    axis_angles = np.column_stack((rotation_vectors, np.full(count, 0.3)))
    euler_angles = rng.normal(scale=0.2, size=(count, 3))
    mrps = rng.normal(scale=0.1, size=(count, 3))
    vectors = rng.normal(size=(count, 3))
    rates = np.zeros((count, 2, 3))
    times = np.broadcast_to((0.0, 1.0), (count, 2))
    translations = rng.normal(size=(count, 3))
    mean_samples = rng.normal(scale=0.1, size=(count, 6, 3))
    mean_weights = rng.random((count, 6))
    source_vectors = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    sequence = np.empty((count, 4, 4), dtype=np.float64)

    for _convention, _layout, backend in _BACKENDS:
        q0 = backend.from_rotvec(rotation_vectors)
        q1 = backend.from_euler(euler_angles)
        backend.from_axis_angle(axis_angles)
        backend.to_axis_angle(q0)
        backend.multiply(q0, q1)
        backend.rotate(q0, vectors)
        rotation_matrices = backend.to_rotation_matrix(q0)
        backend.from_rotation_matrix(rotation_matrices)
        backend.to_rotvec(q0)
        backend.to_euler(q0)
        backend.integrate_angular_velocity_samples(q0, rates, times)
        backend.nlerp(q0, q1, 0.4)
        backend.slerp(q0, q1, 0.4)
        backend.squad(q0, q1, q0, q1, 0.4)
        backend.from_mrp(mrps)
        backend.to_mrp(q0)
        backend.norm(q0)
        backend.normalize(q0)
        backend.inverse(q0)
        backend.dot(q0, q1)
        backend.left_divide(q0, q1)
        backend.right_divide(q0, q1)
        backend.rotation_distance(q0, q1)
        backend.rotations_equivalent(q0, q1)
        for operation in (
            "exp", "log", "sin", "cos", "sinh", "cosh", "tan", "tanh",
            "asin", "acos", "atan", "asinh", "acosh", "atanh", "expm1",
            "log1p", "log2", "log10",
        ):
            getattr(backend, operation)(q0)
        backend.power(q0, 3)
        backend.power(q0, 1.25)
        backend.power_quaternion(q0, q1)
        backend.skew_matrix(vectors)
        backend.so3_left_jacobian(rotation_vectors)
        backend.so3_right_jacobian(rotation_vectors)
        backend.so3_left_jacobian_inverse(rotation_vectors)
        backend.so3_right_jacobian_inverse(rotation_vectors)
        backend.quaternion_derivative(q0, vectors)
        backend.boxplus(q0, rotation_vectors)
        backend.boxminus(q0, q1)
        homogeneous = backend.to_homogeneous_matrix(q0, translations)
        backend.from_homogeneous_matrix(homogeneous)
        backend.so3_adjoint(q0)
        backend.make_sign_continuous(
            np.broadcast_to(q0[:, None, :], sequence.shape).copy()
        )
        samples = backend.from_rotvec(mean_samples)
        backend.rotation_mean(samples, mean_weights)
        backend.rotation_mean(samples, mean_weights, method="intrinsic")
        backend.rotation_covariance(samples, weights=mean_weights)
        target_vectors = backend.rotate(q0[:, None, :], source_vectors)
        backend.align_vectors(source_vectors, target_vectors)

    large_rotation_vectors = rng.normal(scale=0.25, size=(8192, 3))
    large_other_vectors = rng.normal(scale=0.2, size=(8192, 3))
    for _convention, _layout, backend in _BACKENDS:
        large_q0 = backend.from_rotvec(large_rotation_vectors)
        large_q1 = backend.from_rotvec(large_other_vectors)
        backend.rotation_distance(large_q0, large_q1)
        backend.rotations_equivalent(large_q0, large_q1)

    axis_map = SignedAxisMap(("y", "-z", "x"))
    engine_numpy.map_hamilton_xyzw(
        hamilton_xyzw.from_rotvec(large_rotation_vectors), axis_map
    )
    engine_numpy.map_hamilton_wxyz(
        hamilton_wxyz.from_rotvec(large_rotation_vectors), axis_map
    )
    engine_numpy.map_jpl_xyzw(
        jpl_xyzw.from_rotvec(large_rotation_vectors), axis_map
    )
    engine_numpy.map_jpl_wxyz(
        jpl_wxyz.from_rotvec(large_rotation_vectors), axis_map
    )


def collect_numba_coverage() -> dict[str, Any]:
    """Warm public kernels and return their compiled-signature inventory."""
    if find_spec("numba") is None:
        return {
            "numba_available": False,
            "python": platform.python_version(),
            "numba_version": None,
            "complete": False,
            "summary": {
                "compiled": 0,
                "registered": 0,
                "missing": 0,
                "unknown": 0,
                "total": 0,
            },
            "kernels": [],
        }

    try:
        numba_version = version("numba")
    except PackageNotFoundError:
        numba_version = "installed, version metadata unavailable"

    _exercise_public_batch_paths()

    kernels = [
        _describe_kernel(name, _numpy._NUMBA_ROTATION_KERNELS.get(name))
        for name in sorted(_numpy._NUMBA_ROTATION_KERNEL_NAMES)
    ]
    for scalar_last, epsilon in sorted(_numpy._NUMBA_MULTIPLY_IMPLEMENTATIONS):
        convention = "Hamilton" if epsilon == 1 else "JPL"
        layout = "xyzw" if scalar_last else "wxyz"
        kernels.append(
            _describe_kernel(
                f"multiply[{convention},{layout}]",
                _numpy._NUMBA_MULTIPLY.get((scalar_last, epsilon)),
            )
        )
    kernels.append(_describe_kernel("rotate", _numpy._NUMBA_ROTATE))
    summary = {
        state: sum(item["state"] == state for item in kernels)
        for state in ("compiled", "registered", "missing", "unknown")
    }
    return {
        "numba_available": True,
        "python": platform.python_version(),
        "numba_version": numba_version,
        "complete": summary["compiled"] == len(kernels),
        "summary": {**summary, "total": len(kernels)},
        "kernels": kernels,
    }


def main() -> int:
    report = collect_numba_coverage()
    if not report["numba_available"]:
        print(
            f"CPython {report['python']}: Numba is unavailable; "
            "optional NumPy fallbacks are active."
        )
        return 0

    print(
        f"Numba coverage: CPython {report['python']}, "
        f"Numba {report['numba_version']}; "
        f"{report['summary']['compiled']}/{report['summary']['total']} "
        "dispatchers compiled"
    )
    for item in report["kernels"]:
        signatures = ", ".join(item["signatures"]) or "no compiled signatures"
        print(
            f"[{item['state']}] {item['name']} "
            f"({item['kind']}): {signatures}"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
