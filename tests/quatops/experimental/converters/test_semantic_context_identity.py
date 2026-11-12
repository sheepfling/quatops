"""Ordinary Python copies and keys must retain complete semantic context."""

from dataclasses import replace

import numpy as np
import pytest

from quatops.experimental.converters import OrientationConverter
from quatops.experimental.engines import SignedAxisMap
from quatops.experimental.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    FrameAlignment,
    OpenXRSpaceContext,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    bind_semantic_role,
)


def _bound_converter():
    world = CANONICAL_WORLD_RFU.bind("source.world", "world.reference")
    body = CANONICAL_BODY_RFU.bind("source.body", "body.reference")
    parent = bind_semantic_role(
        "ros.rep103.body-flu",
        name="robot.base",
        alignment=FrameAlignment(
            world,
            SignedAxisMap(("-y", "x", "z")),
            "body_alignment",
            "application",
            "application_declaration",
            "app://robot/base-v1",
        ),
    )
    child = bind_semantic_role(
        "ros.rep103.optical",
        name="robot.camera",
        alignment=FrameAlignment(
            body,
            SignedAxisMap(("x", "z", "-y")),
            "sensor_mount",
            "application",
            "calibration",
            "calibration://robot/camera-v1",
        ),
    )
    return OrientationConverter(
        "ros2.geometry-msgs-quaternion", world, body, parent, child
    )


def test_no_change_dataclass_copy_preserves_both_bindings_and_reverse_operations():
    original = _bound_converter()
    copied = replace(original)
    assert copied == original
    assert hash(copied) == hash(original)
    assert copied.target_parent is original.target_parent
    assert copied.target_child is original.target_child
    assert copied.target_semantic_bindings == original.target_semantic_bindings
    assert copied.target_parent_frame == original.target_parent.coordinate_frame
    assert copied.target_child_frame == original.target_child.coordinate_frame
    orientation = PhysicalOrientation.from_wxyz(
        (0.8, 0.2, -0.4, 0.4), original.source_parent, original.source_child
    )
    native = original.to_target(orientation)
    assert copied.to_target(orientation) == native
    np.testing.assert_allclose(
        copied.from_target(native).quaternion_wxyz,
        orientation.quaternion_wxyz,
        atol=1e-15,
    )
    batch = PhysicalOrientationBatch(
        np.asarray((orientation.quaternion_wxyz,)),
        original.source_parent,
        original.source_child,
    )
    target_batch = original.to_target_batch(batch)
    copied_batch = copied.to_target_batch(batch)
    assert copied_batch.parent_semantic_binding == target_batch.parent_semantic_binding
    assert copied_batch.child_semantic_binding == target_batch.child_semantic_binding
    np.testing.assert_allclose(
        copied.from_target_batch(target_batch).values, batch.values, atol=1e-15
    )


@pytest.mark.parametrize("field", ("target_parent", "target_child"))
def test_calibration_revision_distinguishes_converter_keys_even_with_the_same_basis(
    field,
):
    original = _bound_converter()
    binding = getattr(original, field)
    revised_binding = replace(
        binding,
        alignment=replace(
            binding.alignment,
            alignment_source="calibration",
            alignment_evidence_reference="calibration://revision-2",
        ),
    )
    revised = replace(original, **{field: revised_binding})
    assert revised.target_parent_frame == original.target_parent_frame
    assert revised.target_child_frame == original.target_child_frame
    assert original != revised
    assert len({original: "old", revised: "new"}) == 2
    source = PhysicalOrientation.from_wxyz(
        (1.0, 0.0, 0.0, 0.0), original.source_parent, original.source_child
    )
    with pytest.raises(ValueError, match="semantic binding"):
        revised.from_target(original.to_target(source))


def test_explicitly_replacing_a_binding_with_its_frame_removes_only_that_binding():
    original = _bound_converter()
    stripped = replace(original, target_parent=original.target_parent_frame)
    assert stripped.target_semantic_bindings == (None, original.target_child)
    assert stripped != original


def test_generic_conversion_preserves_independently_supplied_openxr_snapshots():
    world = CANONICAL_WORLD_RFU.bind("base.a", "world.reference")
    body = CANONICAL_BODY_RFU.bind("base.b", "body.reference")
    bindings = []
    for reference, role, space, session, time in (
        (world, "openxr.reference-space", "local", "session-a", 10),
        (body, "openxr.view-space", "view", "session-b", 20),
    ):
        bindings.append(
            bind_semantic_role(
                role,
                name=f"{session}/{space}",
                alignment=FrameAlignment(
                    reference,
                    SignedAxisMap(("x", "z", "-y")),
                    "space_alignment",
                    "runtime",
                    "runtime_relation",
                    f"runtime://{session}/{time}",
                    OpenXRSpaceContext(
                        f"{session}/{space}",
                        reference.name,
                        session,
                        time,
                        "test-runtime",
                        space,
                        1,
                    ),
                ),
            )
        )
    converter = OrientationConverter("openxr.xrquaternionf", world, body, *bindings)
    source = PhysicalOrientation.from_wxyz((0.8, 0.2, -0.4, 0.4), world, body)
    target = converter.to_target(source)
    assert (
        target.parent_semantic_binding.alignment.space_context.session_id == "session-a"
    )
    assert target.child_semantic_binding.alignment.space_context.time_ns == 20
    copied = replace(converter)
    np.testing.assert_allclose(
        copied.from_target(target).quaternion_wxyz, source.quaternion_wxyz, atol=1e-15
    )
    child = bindings[1]
    updated = replace(
        child,
        alignment=replace(
            child.alignment,
            space_context=replace(child.alignment.space_context, time_ns=21),
        ),
    )
    revised = replace(converter, target_child=updated)
    assert len({converter: 1, revised: 2}) == 2
    with pytest.raises(ValueError, match="child semantic binding"):
        revised.from_target(target)
