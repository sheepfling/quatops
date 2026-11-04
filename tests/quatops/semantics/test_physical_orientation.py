"""Tests for the physical frame-semantics value layer."""

import numpy as np
import pytest

from quatops.engines import OrthogonalBasisMap, SignedAxisMap
from quatops.python import hamilton
from quatops.semantics import (
    AEROSPACE_BODY_FRD,
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    OPENXR_SPACE_RUB,
    ROS_BODY_FLU,
    ROS_CAMERA_OPTICAL,
    WORLD_ENU,
    WORLD_NED,
    CoordinateFrame,
    PhysicalOrientation,
    PhysicalOrientationBatch,
)
from quatops.types import HamiltonQuaternion


def test_physical_orientation_requires_unit_hamilton_quaternion():
    world = CANONICAL_WORLD_RFU.bind("test.world", "test.world.axes")
    body = CANONICAL_BODY_RFU.bind("test.body", "test.body.axes")

    orientation = PhysicalOrientation.from_wxyz(
        (0.8, 0.2, -0.4, 0.4), parent_frame=world, child_frame=body
    )
    assert orientation.quaternion_wxyz == (0.8, 0.2, -0.4, 0.4)

    with pytest.raises(ValueError, match="unit quaternion"):
        PhysicalOrientation.from_wxyz(
            (2.0, 0.0, 0.0, 0.0), parent_frame=world, child_frame=body
        )


def test_physical_orientation_compose_and_inverse_check_frame_definitions():
    world = CANONICAL_WORLD_RFU.bind("test.world", "test.world.axes")
    base = CANONICAL_BODY_RFU.bind("robot.base", "robot.base.axes")
    sensor = ROS_CAMERA_OPTICAL.bind("robot.sensor", "robot.base.axes")
    q_world_from_base = hamilton.from_rotvec((0.42, -0.31, 0.17))
    q_base_from_sensor = hamilton.from_rotvec((-0.22, 0.63, 0.38))
    world_from_base = PhysicalOrientation(
        HamiltonQuaternion(*q_world_from_base), world, base
    )
    base_from_sensor = PhysicalOrientation(
        HamiltonQuaternion(*q_base_from_sensor), base, sensor
    )

    world_from_sensor = world_from_base.compose(base_from_sensor)
    expected = hamilton.multiply(q_world_from_base, q_base_from_sensor)
    assert world_from_sensor.parent_frame == world
    assert world_from_sensor.child_frame == sensor
    assert hamilton.rotations_equivalent(world_from_sensor.quaternion_wxyz, expected)

    identity = world_from_sensor.inverse().compose(world_from_sensor)
    assert identity.parent_frame == sensor
    assert identity.child_frame == sensor
    assert hamilton.rotations_equivalent(identity.quaternion_wxyz, (1.0, 0.0, 0.0, 0.0))

    different_base = CoordinateFrame(
        name=base.name,
        reference=base.reference,
        axes=base.axes,
        basis_to_reference=SignedAxisMap(("y", "-x", "z")),
        handedness="right",
        role="body",
        reference_convention=base.reference_convention,
    )
    with pytest.raises(ValueError, match="intermediate frame definition"):
        world_from_base.compose(
            PhysicalOrientation(
                HamiltonQuaternion(*q_base_from_sensor), different_base, sensor
            )
        )


def test_physical_orientation_rejects_mixed_handed_frame_pair():
    world = CANONICAL_WORLD_RFU.bind("test.world", "test.world.axes")
    left_handed_body = CoordinateFrame(
        "test.left-body",
        "test.body.axes",
        ("right", "forward", "down"),
        SignedAxisMap(("x", "y", "-z")),
        "left",
        role="body",
    )

    with pytest.raises(ValueError, match="matching handedness"):
        PhysicalOrientation(
            HamiltonQuaternion(1.0, 0.0, 0.0, 0.0), world, left_handed_body
        )
    with pytest.raises(ValueError, match="matching handedness"):
        PhysicalOrientationBatch(
            np.asarray([[1.0, 0.0, 0.0, 0.0]]), world, left_handed_body
        )


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        (
            ROS_BODY_FLU,
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        (
            ROS_CAMERA_OPTICAL,
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ),
        (
            AEROSPACE_BODY_FRD,
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ),
        (
            WORLD_NED,
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ),
        (
            OPENXR_SPACE_RUB,
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        ),
    ],
)
def test_common_convention_presets_have_declared_basis(convention, expected):
    actual = np.asarray(convention.basis_to_reference.matrix)
    np.testing.assert_array_equal(actual, np.asarray(expected))
    assert convention.handedness == "right"


def test_frame_convention_binds_dynamic_provider_basis_explicitly():
    enu_in_ecef = OrthogonalBasisMap(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )

    local_enu = WORLD_ENU.bind(
        "local.enu@site-17",
        reference="earth.ecef",
        basis_to_reference=enu_in_ecef,
        reference_convention="earth.ecef.axes",
    )

    assert local_enu.name == "local.enu@site-17"
    assert local_enu.reference == "earth.ecef"
    assert local_enu.reference_convention == "earth.ecef.axes"
    assert local_enu.basis_to_reference == enu_in_ecef

    with pytest.raises(ValueError, match="provided together"):
        WORLD_ENU.bind(
            "local.enu@site-17",
            reference="earth.ecef",
            basis_to_reference=enu_in_ecef,
        )
