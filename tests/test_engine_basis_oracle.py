"""SciPy-oracle tests for low-overhead game-engine basis conversions."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from quatops.engines import SignedAxisMap
from quatops.engines import numpy as engine_numpy
from quatops.numpy.hamilton import xyzw as hamilton_xyzw
from quatops.types.hamilton_dataclass import Quaternion as HamiltonDataclass
from quatops.types.hamilton_namedtuple import Quaternion as HamiltonNamedTuple
from quatops.types.jpl_dataclass import Quaternion as JPLDataclass
from quatops.types.jpl_namedtuple import Quaternion as JPLNamedTuple


def _all_signed_axes():
    for permutation in itertools.permutations("xyz"):
        for signs in itertools.product((1, -1), repeat=3):
            yield tuple(
                ("-" if sign < 0 else "") + axis
                for axis, sign in zip(permutation, signs)
            )


def _matrix_from_axes(axes):
    """Build B independently from the map object under test."""
    matrix = np.zeros((3, 3), dtype=float)
    for row, axis in enumerate(axes):
        sign = -1.0 if axis.startswith("-") else 1.0
        name = axis[1:] if axis.startswith("-") else axis
        matrix[row, "xyz".index(name)] = sign
    return matrix


@pytest.mark.parametrize("axes", tuple(_all_signed_axes()))
def test_scalar_maps_match_scipy_rotation_matrix_for_all_signed_axis_maps(axes):
    rng = np.random.default_rng(0xE113 + sum(map(ord, "".join(axes))))
    rotations = Rotation.random(24, random_state=rng)
    source_xyzw = rotations.as_quat()
    source_wxyz = source_xyzw[:, (3, 0, 1, 2)]
    source_jpl_xyzw = source_xyzw * np.array((-1.0, -1.0, -1.0, 1.0))
    source_jpl_wxyz = source_jpl_xyzw[:, (3, 0, 1, 2)]
    axis_map = SignedAxisMap(axes)
    basis = _matrix_from_axes(axes)
    expected = basis @ rotations.as_matrix() @ basis.T
    source_vector = rng.normal(size=3)
    np.testing.assert_array_equal(
        axis_map.vector(source_vector), basis @ source_vector
    )

    mapped_values = (
        np.stack([axis_map.map_hamilton_xyzw(q) for q in source_xyzw]),
        np.stack([axis_map.map_hamilton_wxyz(q) for q in source_wxyz]),
        np.stack([axis_map.map_jpl_xyzw(q) for q in source_jpl_xyzw]),
        np.stack([axis_map.map_jpl_wxyz(q) for q in source_jpl_wxyz]),
    )
    for mapped in mapped_values:
        actual = Rotation.from_quat(mapped).as_matrix()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)


def test_numpy_maps_preserve_layout_dtype_and_match_scipy():
    rng = np.random.default_rng(0xE11E)
    source_xyzw = Rotation.random(32, random_state=rng).as_quat().astype(np.float32)
    axes = ("y", "z", "-x")
    axis_map = SignedAxisMap(axes)
    basis = _matrix_from_axes(axes)
    expected = basis @ Rotation.from_quat(source_xyzw).as_matrix() @ basis.T

    trailing = engine_numpy.map_hamilton_xyzw(source_xyzw, axis_map)
    assert trailing.dtype == source_xyzw.dtype
    np.testing.assert_allclose(
        Rotation.from_quat(trailing).as_matrix(), expected, rtol=0.0, atol=2e-6
    )

    leading = engine_numpy.map_hamilton_xyzw(source_xyzw.T, axis_map,
                                              component_axis=0)
    assert leading.shape == source_xyzw.T.shape
    np.testing.assert_array_equal(leading, trailing.T)

    source_wxyz = source_xyzw[:, (3, 0, 1, 2)]
    mapped_wxyz = engine_numpy.map_hamilton_wxyz(source_wxyz, axis_map)
    np.testing.assert_array_equal(mapped_wxyz, trailing)

    source_jpl = source_xyzw * np.array((-1.0, -1.0, -1.0, 1.0), dtype=np.float32)
    mapped_jpl = engine_numpy.map_jpl_xyzw(source_jpl, axis_map)
    np.testing.assert_array_equal(mapped_jpl, trailing)

    source_jpl_wxyz = source_wxyz * np.array((1.0, -1.0, -1.0, -1.0),
                                              dtype=np.float32)
    mapped_jpl_wxyz = engine_numpy.map_jpl_wxyz(source_jpl_wxyz, axis_map)
    np.testing.assert_array_equal(mapped_jpl_wxyz, trailing)

    component_middle = np.moveaxis(source_xyzw.reshape((4, 8, 4)), -1, 1)
    mapped_middle = engine_numpy.map_hamilton_xyzw(
        component_middle, axis_map, component_axis=1
    )
    np.testing.assert_array_equal(
        np.moveaxis(mapped_middle, 1, -1),
        engine_numpy.map_hamilton_xyzw(np.moveaxis(component_middle, 1, -1), axis_map),
    )


@pytest.mark.parametrize("axes", tuple(_all_signed_axes()))
def test_basis_map_commutes_with_shortest_slerp_against_scipy(axes):
    rng = np.random.default_rng(0x51E2)
    source = Rotation.random(2, random_state=rng)
    times = np.linspace(0.0, 1.0, 21)
    source_path = Slerp((0.0, 1.0), source)(times)
    axis_map = SignedAxisMap(axes)
    basis = _matrix_from_axes(axes)

    mapped_keys = np.stack(
        [axis_map.map_hamilton_xyzw(q) for q in source.as_quat()]
    )
    target_samples = np.stack([
        hamilton_xyzw.slerp(mapped_keys[0], mapped_keys[1], float(weight))
        for weight in times
    ])
    target_matrices = Rotation.from_quat(target_samples).as_matrix()
    mapped_source_matrices = basis @ source_path.as_matrix() @ basis.T
    np.testing.assert_allclose(
        target_matrices, mapped_source_matrices, rtol=0.0, atol=3e-14
    )


@pytest.mark.parametrize(
    "axes",
    [(), ("x", "y"), ("x", "y", "y"), ("x", "y", "z", "w"),
     ("x", "y", "q"), ("++x", "y", "z")],
)
def test_signed_axis_map_rejects_invalid_axes(axes):
    with pytest.raises((TypeError, ValueError)):
        SignedAxisMap(axes)


def test_map_rejects_quaternion_with_wrong_component_count():
    axis_map = SignedAxisMap(("x", "y", "z"))
    with pytest.raises(ValueError, match="exactly four"):
        axis_map.map_hamilton_xyzw((1.0, 0.0, 0.0))


def test_signed_axis_map_is_immutable_after_compilation():
    axis_map = SignedAxisMap(("x", "y", "z"))
    with pytest.raises(AttributeError, match="immutable"):
        axis_map.destination_axes = ("y", "x", "z")


def test_scalar_map_accepts_quatops_namedtuple_and_dataclass_values():
    axis_map = SignedAxisMap(("x", "y", "-z"))
    hamilton_wxyz = (0.5, 0.1, -0.2, 0.3)
    jpl_xyzw = (-0.1, 0.2, -0.3, 0.5)

    expected_hamilton = axis_map.map_hamilton_wxyz(hamilton_wxyz)
    expected_jpl = axis_map.map_jpl_xyzw(jpl_xyzw)
    assert (
        axis_map.map_hamilton_wxyz(HamiltonNamedTuple(*hamilton_wxyz))
        == expected_hamilton
    )
    assert (
        axis_map.map_hamilton_wxyz(HamiltonDataclass(*hamilton_wxyz))
        == expected_hamilton
    )
    assert axis_map.map_jpl_xyzw(JPLNamedTuple(*jpl_xyzw)) == expected_jpl
    assert axis_map.map_jpl_xyzw(JPLDataclass(*jpl_xyzw)) == expected_jpl


def test_numpy_map_rejects_invalid_component_axis_and_shape():
    axis_map = SignedAxisMap(("x", "y", "z"))
    with pytest.raises(ValueError, match="out of range"):
        engine_numpy.map_hamilton_xyzw(np.ones((2, 4)), axis_map, component_axis=3)
    with pytest.raises(ValueError, match="length four"):
        engine_numpy.map_hamilton_xyzw(np.ones((2, 3)), axis_map)
