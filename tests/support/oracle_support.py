"""Backend and layout helpers shared by independent-reference tests."""

from importlib import import_module

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation


def random_rotations(count: int, rng: np.random.Generator) -> Rotation:
    """Use the SciPy random-generator keyword supported by the installed version."""
    try:
        return Rotation.random(count, rng=rng)
    except TypeError:
        # SciPy before 1.15 calls this parameter ``random_state``.
        return Rotation.random(count, random_state=rng)

SCALAR_MODULES = (
    "quatops.scalar",
    "quatops.python.hamilton",
    "quatops.hamilton.scalar",
    "quatops.hamilton.namedtuple",
    "quatops.hamilton.dataclass",
    "quatops.types.hamilton_namedtuple",
    "quatops.types.hamilton_dataclass",
    "quatops.jpl.scalar",
    "quatops.python.jpl",
    "quatops.jpl.namedtuple",
    "quatops.jpl.dataclass",
    "quatops.types.jpl_namedtuple",
    "quatops.types.jpl_dataclass",
    "quatops.namedtuple",
    "quatops.dataclass",
)
NUMPY_MODULES = tuple(
    f"quatops.numpy.{convention}."
    + (f"components_first.{order}" if component_first else order)
    for convention in ("hamilton", "jpl")
    for component_first in (False, True)
    for order in ("wxyz", "xyzw")
)
BACKENDS = tuple(
    (
        name,
        import_module(name),
        name.startswith("quatops.jpl") or ".jpl." in name
        or ".jpl_" in name or name.endswith(".jpl"),
        name.startswith("quatops.numpy."),
        ".components_first." in name,
    )
    for name in SCALAR_MODULES + NUMPY_MODULES
)
COEFFICIENT_SIGN = np.array((1.0, -1.0, -1.0, -1.0))


def as_hamilton_coefficients(values: np.ndarray, is_jpl: bool) -> np.ndarray:
    """Map public JPL wxyz coefficients into equivalent Hamilton coefficients."""
    return values * COEFFICIENT_SIGN if is_jpl else values


def reference_quaternions(values: np.ndarray, is_jpl: bool) -> np.ndarray:
    return quaternion.as_quat_array(as_hamilton_coefficients(values, is_jpl))


def reference_coefficients(values: np.ndarray, is_jpl: bool) -> np.ndarray:
    coefficients = np.asarray(quaternion.as_float_array(values), dtype=float)
    return as_hamilton_coefficients(coefficients, is_jpl)


def from_wxyz(module, values: np.ndarray, is_numpy: bool, component_first: bool):
    coefficients = np.asarray(values, dtype=float)
    if not is_numpy:
        return module.from_wxyz(tuple(float(value) for value in coefficients))
    if component_first:
        coefficients = np.moveaxis(coefficients, -1, 0)
    return module.from_wxyz(coefficients)


def to_wxyz(module, value, is_numpy: bool, component_first: bool) -> np.ndarray:
    coefficients = np.asarray(module.to_wxyz(value), dtype=float)
    if is_numpy and component_first:
        coefficients = np.moveaxis(coefficients, 0, -1)
    return coefficients


def orient_vectors(values: np.ndarray, component_first: bool) -> np.ndarray:
    return np.moveaxis(values, -1, 0) if component_first else values


def orient_matrices(values: np.ndarray, component_first: bool) -> np.ndarray:
    return np.moveaxis(values, (-2, -1), (0, 1)) if component_first else values


def vectors_to_trailing(values: np.ndarray, component_first: bool) -> np.ndarray:
    return np.moveaxis(values, 0, -1) if component_first else np.asarray(values)


def matrices_to_trailing(values: np.ndarray, component_first: bool) -> np.ndarray:
    return np.moveaxis(values, (0, 1), (-2, -1)) if component_first else np.asarray(values)
