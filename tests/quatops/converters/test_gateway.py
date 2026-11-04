"""Tests for the public physical-orientation target gateway."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.converters import (
    convert_orientation_batch_from_target,
    convert_orientation_batch_to_target,
    convert_orientation_from_target,
    convert_orientation_to_target,
)
from quatops.engines import OrthogonalBasisMap
from quatops.python import hamilton
from quatops.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    CoordinateFrame,
    PhysicalOrientation,
    PhysicalOrientationBatch,
)
from quatops.types import HamiltonQuaternion


def _source_and_target_frames():
    source_world = CANONICAL_WORLD_RFU.bind(
        "application.world.rfu", "application.world.axes"
    )
    source_body = CANONICAL_BODY_RFU.bind("asset.body.rfu", "asset.body.axes")
    target_basis = OrthogonalBasisMap(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    )
    target_world = CoordinateFrame(
        "application.unity-world",
        source_world.reference,
        ("right", "up", "forward"),
        target_basis,
        "left",
        role="world",
        reference_convention=source_world.reference_convention,
    )
    target_body = CoordinateFrame(
        "asset.unity-local",
        source_body.reference,
        ("right", "up", "forward"),
        target_basis,
        "left",
        role="body",
        reference_convention=source_body.reference_convention,
    )
    return source_world, source_body, target_world, target_body, target_basis


def test_scalar_gateway_matches_explicit_basis_action_and_roundtrips():
    source_world, source_body, target_world, target_body, basis = (
        _source_and_target_frames()
    )
    source_wxyz = (0.8, 0.2, -0.4, 0.4)
    physical = PhysicalOrientation(
        HamiltonQuaternion(*source_wxyz), source_world, source_body
    )

    target = convert_orientation_to_target(
        physical, "unity.quaternion", target_world, target_body
    )
    expected_xyzw = (-source_wxyz[1], -source_wxyz[3], -source_wxyz[2], source_wxyz[0])
    np.testing.assert_allclose(target.values, expected_xyzw, rtol=0.0, atol=1e-15)
    assert (target.from_frame, target.to_frame) == (
        target_body.name,
        target_world.name,
    )

    source_rotation = Rotation.from_quat(hamilton.to_xyzw(source_wxyz)).as_matrix()
    basis_matrix = np.asarray(basis.matrix)
    expected_target_matrix = basis_matrix @ source_rotation @ basis_matrix.T
    target_rotation = Rotation.from_quat(target.values).as_matrix()
    np.testing.assert_allclose(
        target_rotation, expected_target_matrix, rtol=0.0, atol=2e-15
    )
    vector = np.asarray((0.31, -0.72, 0.48))
    np.testing.assert_allclose(
        basis_matrix @ (source_rotation @ vector),
        target_rotation @ (basis_matrix @ vector),
        rtol=0.0,
        atol=2e-15,
    )

    recovered = convert_orientation_from_target(
        target,
        "unity.quaternion",
        source_world,
        source_body,
        target_world,
        target_body,
    )
    assert recovered.parent_frame == source_world
    assert recovered.child_frame == source_body
    assert hamilton.rotations_equivalent(recovered.quaternion_wxyz, source_wxyz)


def test_batch_gateway_preserves_component_axis_and_roundtrips():
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    source_values = np.asarray(
        [
            (0.8, 0.2, -0.4, 0.4),
            (0.7, -0.3, 0.2, 0.6164414002968976),
        ],
        dtype=np.float64,
    ).T
    physical = PhysicalOrientationBatch(source_values, source_world, source_body)

    native = convert_orientation_batch_to_target(
        physical,
        "unity.quaternion",
        target_world,
        target_body,
        component_axis=0,
    )
    assert native.values.shape == source_values.shape
    assert native.from_frame == target_body.name
    assert native.to_frame == target_world.name

    recovered = convert_orientation_batch_from_target(
        native,
        "unity.quaternion",
        source_world,
        source_body,
        target_world,
        target_body,
        component_axis=0,
    )
    assert recovered.values.shape == source_values.shape
    for actual, expected in zip(recovered.values.T, source_values.T):
        assert hamilton.rotations_equivalent(actual, expected, atol=2e-12)


def test_batch_gateway_rejects_non_unit_physical_values():
    source_world, source_body, target_world, target_body, _ = (
        _source_and_target_frames()
    )
    physical = PhysicalOrientationBatch(
        np.asarray(((2.0, 0.0, 0.0, 0.0),)), source_world, source_body
    )

    with pytest.raises(ValueError, match="unit quaternions"):
        convert_orientation_batch_to_target(
            physical, "unity.quaternion", target_world, target_body
        )
