"""Tests for frame-labelled orientation contracts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.experimental.converters import get_converter
from quatops.experimental.engines import OrthogonalBasisMap, SignedAxisMap
from quatops.experimental.semantics import (
    CANONICAL_BODY_FRAME,
    ECEF_FRAME,
    CoordinateFrame,
    ECEFBodyOrientation,
    FrameCorrespondence,
    FramedOrientation,
    SemanticQuaternionMap,
)
from quatops.python import hamilton


def _frame_map():
    canonical_world = CoordinateFrame(
        "canonical.world",
        "world-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    canonical_body = CoordinateFrame(
        "canonical.body",
        "body-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    world_map = SignedAxisMap(("x", "z", "y"))
    body_map = SignedAxisMap(("z", "y", "x"))
    return SemanticQuaternionMap(
        world=FrameCorrespondence(
            canonical_world,
            CoordinateFrame(
                "target.world",
                "world-reference",
                ("right", "up", "back"),
                world_map.inverse(),
                "left",
            ),
        ),
        body=FrameCorrespondence(
            canonical_body,
            CoordinateFrame(
                "target.body",
                "body-reference",
                ("forward", "up", "left"),
                body_map.inverse(),
                "left",
            ),
        ),
    )


def test_semantic_map_preserves_parent_child_labels_and_roundtrips():
    frame_map = _frame_map()
    source_q = hamilton.from_rotvec((0.73, -0.42, 1.17))
    source = FramedOrientation(source_q, "canonical.body", "canonical.world")

    target = frame_map.to_destination(source)
    assert (target.from_frame, target.to_frame) == ("target.body", "target.world")
    recovered = frame_map.from_destination(target)
    assert (recovered.from_frame, recovered.to_frame) == (
        "canonical.body",
        "canonical.world",
    )
    assert hamilton.rotations_equivalent(recovered.quaternion_wxyz, source_q)


def test_semantic_map_rejects_wrong_source_or_destination_frame_labels():
    frame_map = _frame_map()
    wrong_source = FramedOrientation(
        (1.0, 0.0, 0.0, 0.0), "canonical.camera", "canonical.world"
    )
    with pytest.raises(ValueError, match="child frame"):
        frame_map.to_destination(wrong_source)

    wrong_destination = FramedOrientation(
        (1.0, 0.0, 0.0, 0.0), "target.body", "target.stage"
    )
    with pytest.raises(ValueError, match="parent frame"):
        frame_map.from_destination(wrong_destination)


def test_frame_labelled_composition_checks_the_intermediate_frame():
    q_world_from_base = hamilton.from_rotvec((0.42, -0.31, 0.17))
    q_base_from_sensor = hamilton.from_rotvec((-0.22, 0.63, 0.38))
    world_from_base = FramedOrientation(q_world_from_base, "robot.base", "world")
    base_from_sensor = FramedOrientation(
        q_base_from_sensor, "sensor.optical", "robot.base"
    )

    world_from_sensor = world_from_base.compose(base_from_sensor)
    assert (world_from_sensor.from_frame, world_from_sensor.to_frame) == (
        "sensor.optical",
        "world",
    )
    expected = Rotation.from_quat(
        hamilton.to_xyzw(q_world_from_base)
    ) * Rotation.from_quat(hamilton.to_xyzw(q_base_from_sensor))
    actual = Rotation.from_quat(hamilton.to_xyzw(world_from_sensor.quaternion_wxyz))
    np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=2e-14)
    recovered = world_from_sensor.inverse().compose(world_from_sensor)
    assert (recovered.from_frame, recovered.to_frame) == (
        "sensor.optical",
        "sensor.optical",
    )
    assert hamilton.rotations_equivalent(
        recovered.quaternion_wxyz, (1.0, 0.0, 0.0, 0.0)
    )

    mismatched = FramedOrientation(q_base_from_sensor, "sensor.optical", "camera")
    with pytest.raises(ValueError, match="cannot compose"):
        world_from_base.compose(mismatched)


def test_frame_correspondence_checks_handedness_against_map_parity():
    with pytest.raises(ValueError, match="handedness"):
        CoordinateFrame(
            "target",
            "world",
            ("right", "up", "back"),
            SignedAxisMap(("x", "y", "z")),
            "left",
        )


def test_frame_reference_convention_must_match_even_when_one_is_unspecified():
    source = CoordinateFrame(
        "source.world",
        "shared-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
        role="world",
    )
    destination = CoordinateFrame(
        "target.world",
        "shared-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
        role="world",
        reference_convention="explicit-rfu-v1",
    )
    assert not source.has_same_definition(destination)
    with pytest.raises(ValueError, match="reference convention"):
        FrameCorrespondence(source, destination)


def test_general_orthogonal_basis_maps_support_rotated_semantic_frames():
    source_world = CoordinateFrame(
        "source.world",
        "world-reference",
        ("east", "north", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    source_body = CoordinateFrame(
        "source.body",
        "body-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    world = Rotation.from_rotvec((0.31, -0.28, 0.19)).as_matrix()
    body = Rotation.from_rotvec((-0.11, 0.27, 0.36)).as_matrix()
    target_world = CoordinateFrame(
        "target.world",
        "world-reference",
        ("x", "y", "z"),
        OrthogonalBasisMap(world.T),
        "right",
    )
    target_body = CoordinateFrame(
        "target.body",
        "body-reference",
        ("x", "y", "z"),
        OrthogonalBasisMap(body.T),
        "right",
    )
    frame_map = SemanticQuaternionMap(
        FrameCorrespondence(source_world, target_world),
        FrameCorrespondence(source_body, target_body),
    )
    q = hamilton.from_rotvec((0.42, -0.81, 0.23))
    mapped = frame_map.to_destination(
        FramedOrientation(q, "source.body", "source.world")
    )
    expected = world @ Rotation.from_quat(hamilton.to_xyzw(q)).as_matrix() @ body.T
    actual = Rotation.from_quat(hamilton.to_xyzw(mapped.quaternion_wxyz)).as_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=4e-14)


def test_ecef_body_semantics_flow_through_hamilton_to_target_and_back():
    target_world = CoordinateFrame(
        "target.ecef-world",
        ECEF_FRAME.reference,
        ("ECEF +X", "ECEF +Y", "ECEF +Z"),
        SignedAxisMap(("x", "y", "z")),
        reference_convention=ECEF_FRAME.reference_convention,
    )
    target_body = CoordinateFrame(
        "target.body-rfu",
        CANONICAL_BODY_FRAME.reference,
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        reference_convention=CANONICAL_BODY_FRAME.reference_convention,
    )
    mapping = SemanticQuaternionMap(
        FrameCorrespondence(ECEF_FRAME, target_world),
        FrameCorrespondence(CANONICAL_BODY_FRAME, target_body),
    )
    source_q = hamilton.from_rotvec((0.53, -0.27, 0.91))
    semantic = ECEFBodyOrientation.from_wxyz(source_q)
    canonical = semantic.to_framed_hamilton()
    assert (canonical.to_frame, canonical.from_frame) == (
        ECEF_FRAME.name,
        CANONICAL_BODY_FRAME.name,
    )

    profile = get_converter("unity.quaternion")
    target = profile.to_target(canonical, mapping)
    recovered = ECEFBodyOrientation.from_framed_hamilton(
        profile.from_target(target, mapping)
    )
    assert hamilton.rotations_equivalent(recovered.quaternion_wxyz, source_q)


def test_ecef_body_semantics_reject_other_frame_pairs():
    with pytest.raises(ValueError, match="parent frame"):
        ECEFBodyOrientation.from_framed_hamilton(
            FramedOrientation(
                (1.0, 0.0, 0.0, 0.0),
                CANONICAL_BODY_FRAME.name,
                "local-world",
            )
        )


def test_orthogonal_basis_map_rejects_nonorthogonal_and_improper_magnitudes():
    with pytest.raises(ValueError, match="orthogonal"):
        OrthogonalBasisMap(((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="finite"):
        OrthogonalBasisMap(((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
