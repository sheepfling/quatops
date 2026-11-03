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
from quatops.numpy.hamilton.components_first import wxyz as hamilton_wxyz_first
from quatops.numpy.hamilton.components_first import xyzw as hamilton_xyzw_first
from quatops.numpy.jpl import wxyz as jpl_wxyz
from quatops.numpy.jpl import xyzw as jpl_xyzw
from quatops.numpy.jpl.components_first import wxyz as jpl_wxyz_first
from quatops.numpy.jpl.components_first import xyzw as jpl_xyzw_first

_BACKENDS = (
    ("Hamilton", "wxyz", hamilton_wxyz),
    ("Hamilton", "xyzw", hamilton_xyzw),
    ("JPL", "wxyz", jpl_wxyz),
    ("JPL", "xyzw", jpl_xyzw),
)

_COMPONENT_FIRST_BACKENDS = (
    hamilton_wxyz_first,
    hamilton_xyzw_first,
    jpl_wxyz_first,
    jpl_xyzw_first,
)

_NUMBA_API_PATHS = {
    "algebraic": (
        "exp",
        "log",
        "sin",
        "cos",
        "sinh",
        "cosh",
        "tan",
        "tanh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "expm1",
        "log1p",
        "log2",
        "log10",
    ),
    "multiply": ("multiply",),
    "rotate": ("rotate",),
    "norm": ("norm",),
    "normalize": ("normalize",),
    "inverse": ("inverse",),
    "dot": ("dot", "norm_squared"),
    "from_axis_angle": ("from_axis_angle",),
    "to_axis_angle": ("to_axis_angle",),
    "from_rotation_matrix": ("from_rotation_matrix",),
    "to_rotation_matrix": ("to_rotation_matrix",),
    "quaternion_matrix": ("to_left_matrix", "to_right_matrix"),
    "canonicalize": ("canonicalize",),
    "commutator": ("commutator",),
    "from_two_vectors": ("from_two_vectors",),
    "conjugate": ("conjugate",),
    "component_transform": (
        "from_wxyz",
        "to_xyzw",
        "from_hamilton",
        "to_hamilton",
    ),
    "vector_part": ("vector_part",),
    "random_quaternion": ("random_quaternion", "random_rotation"),
    "from_euler": ("from_euler",),
    "to_euler": ("to_euler", "to_euler_with_status"),
    "from_rotvec": ("from_rotvec", "from_rotation_vector"),
    "to_rotvec": ("to_rotvec", "to_rotation_vector"),
    "integrate_angular_velocity": ("integrate_angular_velocity_samples",),
    "nlerp": ("nlerp",),
    "slerp": ("slerp",),
    "from_mrp": ("from_mrp",),
    "to_mrp": ("to_mrp",),
    "left_divide": ("left_divide",),
    "right_divide": ("right_divide",),
    "rotation_distance": ("rotation_distance",),
    "rotations_equivalent": ("rotations_equivalent",),
    "power_integer/power_real": ("power", "pow_real", "sqrt"),
    "power_quaternion": ("power_quaternion",),
    "make_sign_continuous": ("make_sign_continuous",),
    "rotation_mean": ("rotation_mean",),
    "rotation_covariance": ("rotation_covariance",),
    "skew_matrix": ("skew_matrix",),
    "so3_jacobian": (
        "so3_left_jacobian",
        "so3_right_jacobian",
        "so3_left_jacobian_inverse",
        "so3_right_jacobian_inverse",
    ),
    "align_vectors": ("align_vectors",),
}

_COMPOSED_NUMBA_API_PATHS = {
    "shared Numba kernels": (
        "align_sign",
        "divide",
        "from_homogeneous_matrix",
        "integrate_angular_velocity",
        "quaternion_derivative",
        "squad",
        "swing_twist",
        "to_homogeneous_matrix",
        "so3_adjoint",
        "boxplus",
        "boxminus",
    ),
}

_NUMPY_API_PATHS = {
    "component and arithmetic arrays": (
        "add",
        "from_xyzw",
        "negate",
        "scale",
        "subtract",
        "to_wxyz",
    ),
    "scalar extraction": ("scalar_part",),
}

_ENGINE_NUMBA_API_PATHS = {
    "engine_axis_map": tuple(f"engines.numpy.{name}" for name in engine_numpy.__all__),
}


def _api_catalog() -> dict[str, dict[str, list[str]]]:
    """Return the full exported NumPy API route inventory."""
    return {
        "numba": {route: list(names) for route, names in _NUMBA_API_PATHS.items()},
        "composed": {
            route: list(names) for route, names in _COMPOSED_NUMBA_API_PATHS.items()
        },
        "numpy": {route: list(names) for route, names in _NUMPY_API_PATHS.items()},
        "engine_numba": {
            route: list(names) for route, names in _ENGINE_NUMBA_API_PATHS.items()
        },
    }


def _api_catalog_names() -> tuple[set[str], set[str]]:
    """Return categorized and exported names for the route completeness check."""
    categorized = {
        name
        for routes in (
            _NUMBA_API_PATHS,
            _COMPOSED_NUMBA_API_PATHS,
            _NUMPY_API_PATHS,
            _ENGINE_NUMBA_API_PATHS,
        )
        for names in routes.values()
        for name in names
    }
    expected: set[str] = set()
    expected.update(hamilton_xyzw.__all__)
    expected.update(jpl_xyzw.__all__)
    for backend in _COMPONENT_FIRST_BACKENDS:
        expected.update(backend.__all__)
    expected.update(f"engines.numpy.{name}" for name in engine_numpy.__all__)
    return categorized, expected


def _api_catalog_complete() -> bool:
    """Check that every public backend and engine export has one route."""
    categorized, expected = _api_catalog_names()
    catalog_size = sum(
        len(names)
        for routes in (
            _NUMBA_API_PATHS,
            _COMPOSED_NUMBA_API_PATHS,
            _NUMPY_API_PATHS,
            _ENGINE_NUMBA_API_PATHS,
        )
        for names in routes.values()
    )
    return catalog_size == len(categorized) and categorized == expected


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

    return {
        "name": name,
        "state": "unknown",
        "kind": type(kernel).__name__,
        "signatures": [],
    }


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
        backend.add(q0, q1)
        backend.conjugate(q0)
        backend.from_xyzw(backend.to_xyzw(q0))
        backend.from_wxyz(backend.to_wxyz(q0))
        backend.subtract(q0, q1)
        backend.negate(q0)
        backend.scale(q0, 0.75)
        backend.scalar_part(q0)
        backend.vector_part(q0)
        backend.random_quaternion(count, rng=np.random.default_rng(73))
        backend.random_rotation(count, rng=np.random.default_rng(74))
        backend.multiply(q0, q1)
        backend.divide(q0, q1)
        backend.rotate(q0, vectors)
        rotation_matrices = backend.to_rotation_matrix(q0)
        backend.from_rotation_matrix(rotation_matrices)
        backend.to_rotvec(q0)
        backend.to_rotation_vector(q0)
        backend.from_rotation_vector(rotation_vectors)
        backend.to_euler(q0)
        backend.to_euler_with_status(q0)
        backend.integrate_angular_velocity_samples(q0, rates, times)
        backend.integrate_angular_velocity(q0, vectors, 0.01)
        backend.nlerp(q0, q1, 0.4)
        backend.slerp(q0, q1, 0.4)
        backend.squad(q0, q1, q0, q1, 0.4)
        backend.from_mrp(mrps)
        backend.to_mrp(q0)
        backend.norm(q0)
        backend.norm_squared(q0)
        backend.normalize(q0)
        backend.inverse(q0)
        backend.dot(q0, q1)
        backend.align_sign(q0, q1)
        backend.left_divide(q0, q1)
        backend.right_divide(q0, q1)
        backend.rotation_distance(q0, q1)
        backend.rotations_equivalent(q0, q1)
        backend.canonicalize(q0)
        backend.commutator(q0, q1)
        backend.from_two_vectors(vectors, vectors)
        backend.scalar_part(q0)
        backend.vector_part(q0)
        for operation in (
            "exp",
            "log",
            "sin",
            "cos",
            "sinh",
            "cosh",
            "tan",
            "tanh",
            "asin",
            "acos",
            "atan",
            "asinh",
            "acosh",
            "atanh",
            "expm1",
            "log1p",
            "log2",
            "log10",
        ):
            getattr(backend, operation)(q0)
        backend.power(q0, 3)
        backend.power(q0, 1.25)
        backend.pow_real(q0, 1.25)
        backend.sqrt(q0)
        backend.power_quaternion(q0, q1)
        backend.skew_matrix(vectors)
        backend.to_left_matrix(q0)
        backend.to_right_matrix(q0)
        backend.so3_left_jacobian(rotation_vectors)
        backend.so3_right_jacobian(rotation_vectors)
        backend.so3_left_jacobian_inverse(rotation_vectors)
        backend.so3_right_jacobian_inverse(rotation_vectors)
        backend.quaternion_derivative(q0, vectors)
        backend.boxplus(q0, rotation_vectors)
        backend.boxminus(q0, q1)
        backend.swing_twist(q0, vectors)
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
        backend.random_quaternion(count, rng=np.random.default_rng(73))
        backend.random_rotation(count, rng=np.random.default_rng(74))
        if hasattr(backend, "from_hamilton"):
            backend.from_hamilton(backend.to_hamilton(q0))

    large_rotation_vectors = rng.normal(scale=0.25, size=(8192, 3))
    large_other_vectors = rng.normal(scale=0.2, size=(8192, 3))
    large_euler_angles = rng.normal(scale=0.2, size=(8192, 3))
    for _convention, _layout, backend in _BACKENDS:
        large_euler = backend.from_euler(large_euler_angles)
        backend.to_euler(large_euler)
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
    engine_numpy.map_jpl_xyzw(jpl_xyzw.from_rotvec(large_rotation_vectors), axis_map)
    engine_numpy.map_jpl_wxyz(jpl_wxyz.from_rotvec(large_rotation_vectors), axis_map)

    component_first_angles = np.moveaxis(large_euler_angles, -1, 0)
    component_first_vectors = np.moveaxis(large_rotation_vectors, -1, 0)
    for backend in _COMPONENT_FIRST_BACKENDS:
        q0 = backend.from_rotvec(component_first_vectors)
        q1 = backend.from_euler(component_first_angles)
        backend.to_rotvec(q0)
        backend.to_euler(q1)
        backend.multiply(q0, q1)
        backend.rotate(q0, component_first_vectors)
        backend.to_rotation_matrix(q0)
        backend.add(q0, q1)
        backend.subtract(q0, q1)
        backend.conjugate(q0)
        backend.negate(q0)
        backend.scale(q0, 0.75)
        backend.scalar_part(q0)
        backend.vector_part(q0)
        backend.from_xyzw(backend.to_xyzw(q0))
        backend.from_wxyz(backend.to_wxyz(q0))
        backend.random_quaternion(count, rng=np.random.default_rng(73))
        backend.random_rotation(count, rng=np.random.default_rng(74))
        if hasattr(backend, "from_hamilton"):
            hamilton_values = hamilton_wxyz_first.from_rotvec(component_first_vectors)
            backend.from_hamilton(hamilton_values)


def collect_numba_coverage() -> dict[str, Any]:
    """Warm public kernels and return their compiled-signature inventory."""
    api_catalog = _api_catalog()
    catalog_complete = _api_catalog_complete()
    if find_spec("numba") is None:
        return {
            "numba_available": False,
            "python": platform.python_version(),
            "numba_version": None,
            "complete": False,
            "api_catalog_complete": catalog_complete,
            "api_catalog": api_catalog,
            "summary": {
                "compiled": 0,
                "registered": 0,
                "missing": 0,
                "unknown": 0,
                "total": 0,
                "api_routes": sum(
                    len(names)
                    for routes in api_catalog.values()
                    for names in routes.values()
                ),
                "api_expected": len(_api_catalog_names()[1]),
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
    summary["api_routes"] = sum(
        len(names) for routes in api_catalog.values() for names in routes.values()
    )
    summary["api_expected"] = len(_api_catalog_names()[1])
    return {
        "numba_available": True,
        "python": platform.python_version(),
        "numba_version": numba_version,
        "complete": summary["compiled"] == len(kernels) and catalog_complete,
        "api_catalog_complete": catalog_complete,
        "api_catalog": api_catalog,
        "summary": {**summary, "total": len(kernels)},
        "kernels": kernels,
    }


def main() -> int:
    report = collect_numba_coverage()
    if not report["numba_available"]:
        print(
            f"Public API catalog: {report['summary']['api_routes']}/"
            f"{report['summary']['api_expected']} routes classified; "
            f"CPython {report['python']} has no Numba installation."
        )
        _print_api_catalog(report["api_catalog"])
        print(
            f"CPython {report['python']}: Numba is unavailable; "
            "optional NumPy fallbacks are active."
        )
        return 0

    print(
        f"Numba coverage: CPython {report['python']}, "
        f"Numba {report['numba_version']}; "
        f"{report['summary']['compiled']}/{report['summary']['total']} "
        f"dispatchers compiled; "
        f"{report['summary']['api_routes']}/{report['summary']['api_expected']} "
        "API routes classified"
    )
    _print_api_catalog(report["api_catalog"])
    for item in report["kernels"]:
        signatures = ", ".join(item["signatures"]) or "no compiled signatures"
        print(f"[{item['state']}] {item['name']} ({item['kind']}): {signatures}")
    return 0 if report["complete"] else 1


def _print_api_catalog(catalog: dict[str, dict[str, list[str]]]) -> None:
    """Print every exported operation grouped by its implementation path."""
    for category, routes in catalog.items():
        title = (
            "NumPy-only API paths"
            if category == "numpy"
            else f"{category.replace('_', ' ').title()} API paths"
        )
        if not routes:
            print(f"\n{title}: none")
            continue
        print(f"\n{title}:")
        for route, names in routes.items():
            print(f"  {route}: {', '.join(names)}")


if __name__ == "__main__":
    sys.exit(main())
