"""Executable role bindings with explicit physical alignment context."""

from dataclasses import replace
from math import sqrt
from urllib.parse import urlsplit

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.experimental.converters import OrientationConverter
from quatops.experimental.converters.targets import get_converter
from quatops.experimental.engines import OrthogonalBasisMap, SignedAxisMap
from quatops.experimental.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    ECEF_FRAME,
    Ellipsoid,
    FrameAlignment,
    FrameCorrespondence,
    GeodeticOrigin,
    OpenXRSpaceContext,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    SemanticFrameBinding,
    bind_cesium_enu_frame,
    bind_semantic_role,
)
from quatops.experimental.semantics.frames import CoordinateFrame


def _canonical_frame_pair():
    world = CANONICAL_WORLD_RFU.bind("source.world", "scene-reference")
    body = CANONICAL_BODY_RFU.bind("source.body", "body-reference")
    return world, body


def test_unity_world_role_binding_and_explicit_model_binding_drive_gateway():
    source_world, source_body = _canonical_frame_pair()
    unity_from_source = OrthogonalBasisMap(((1, 0, 0), (0, 0, 1), (0, 1, 0)))
    unity_world = bind_semantic_role(
        "unity.world-axes",
        name="app.unity-world",
        alignment=FrameAlignment(
            reference_frame=source_world,
            basis_to_reference=unity_from_source,
            purpose="scene_alignment",
            provider="application",
            alignment_source="application_declaration",
            alignment_evidence_reference="app://scene-coordinates/v1",
        ),
    )
    # The model frame uses a separately declared alignment. Unity's world
    # convention does not choose an imported model's local axes.
    unity_body = CoordinateFrame(
        name="vehicle.unity-local",
        reference=source_body.reference,
        axes=("right", "up", "forward"),
        basis_to_reference=unity_from_source,
        handedness="left",
        role="body",
        reference_convention=source_body.reference_convention,
    )
    converter = OrientationConverter(
        "unity.quaternion",
        source_world,
        source_body,
        unity_world,
        unity_body,
    )
    assert converter.target_semantic_bindings == (unity_world, None)
    assert converter.target_semantic_binding_ids == ("unity.world-axes",)
    physical = PhysicalOrientation.from_wxyz(
        (sqrt(0.5), 0.0, 0.0, sqrt(0.5)), source_world, source_body
    )
    native = converter.to_target(physical)
    assert native.parent_semantic_binding == unity_world
    assert native.child_semantic_binding is None
    native_rotation = Rotation.from_quat(native.values).as_matrix()
    # Canonical +X right turns to canonical +Y forward. In Unity local
    # coordinates, that direction is +Z forward.
    np.testing.assert_allclose(
        native_rotation @ np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        rtol=0.0,
        atol=2e-15,
    )
    recovered = converter.from_target(native)
    assert recovered.parent_frame.has_identical_definition(source_world)
    assert recovered.child_frame.has_identical_definition(source_body)

    assert unity_world.alignment is not None
    changed_provenance = replace(
        unity_world.alignment,
        alignment_source="calibration",
        alignment_evidence_reference="calibration://scene-survey/42",
    )
    assert changed_provenance is not None
    rebound = SemanticFrameBinding(
        contract_id=unity_world.contract_id,
        authority=unity_world.authority,
        coordinate_frame=unity_world.coordinate_frame,
        evidence_reference=unity_world.evidence_reference,
        target_family=unity_world.target_family,
        alignment=changed_provenance,
    )
    with pytest.raises(ValueError, match="parent semantic binding"):
        converter.from_target(replace(native, parent_semantic_binding=rebound))
    with pytest.raises(ValueError, match="parent semantic binding does not match"):
        converter.from_target(
            replace(native, from_frame=native.to_frame, to_frame=native.from_frame)
        )

    batch = PhysicalOrientationBatch(
        values=np.asarray(((sqrt(0.5), 0.0, 0.0, sqrt(0.5)),)),
        parent_frame=source_world,
        child_frame=source_body,
    )
    target_batch = converter.to_target_batch(batch)
    assert target_batch.parent_semantic_binding == unity_world
    assert target_batch.child_semantic_binding is None
    assert converter.from_target_batch(target_batch).values.shape == batch.values.shape
    assert unity_world.authority == "target_defined"
    unity_source = urlsplit(unity_world.evidence_reference)
    assert unity_source.scheme == "https"
    assert unity_source.netloc == "docs.unity.com"
    assert unity_world.reference_frame is not None
    assert unity_world.reference_frame.has_identical_definition(source_world)
    with pytest.raises(ValueError, match="target family"):
        SemanticFrameBinding(
            contract_id=unity_world.contract_id,
            authority=unity_world.authority,
            coordinate_frame=unity_world.coordinate_frame,
            evidence_reference=unity_world.evidence_reference,
            target_family="Unreal",
            alignment=unity_world.alignment,
        )

    unreal_world = bind_semantic_role(
        "unreal.world-axes",
        name="app.unreal-world",
        alignment=FrameAlignment(
            reference_frame=source_world,
            basis_to_reference=OrthogonalBasisMap(((0, 1, 0), (1, 0, 0), (0, 0, 1))),
            purpose="scene_alignment",
            provider="application",
            alignment_source="application_declaration",
            alignment_evidence_reference="app://unreal-coordinates/v1",
        ),
    )
    with pytest.raises(ValueError, match="belongs to Unreal, not Unity"):
        OrientationConverter(
            "unity.quaternion",
            source_world,
            source_body,
            unreal_world,
            unity_body,
        )

    unregistered_unity_profile = replace(
        get_converter("unity.quaternion"), converter_id="unity.other-quaternion-api"
    )
    with pytest.raises(ValueError, match="not registered for converter"):
        OrientationConverter(
            unregistered_unity_profile,
            source_world,
            source_body,
            unity_world,
            unity_body,
        )


def test_ros_public_conversion_preserves_orientation_with_distinct_parent_body_maps():
    source_parent, source_child = _canonical_frame_pair()
    parent_alignment = OrthogonalBasisMap(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    child_alignment = OrthogonalBasisMap(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
    ros_parent = bind_semantic_role(
        "ros.rep103.body-flu",
        name="robot.map-flu",
        alignment=FrameAlignment(
            reference_frame=source_parent,
            basis_to_reference=parent_alignment,
            purpose="body_alignment",
            provider="application",
            alignment_source="survey",
            alignment_evidence_reference="survey://robot/map/v1",
        ),
    )
    ros_child = bind_semantic_role(
        "ros.rep103.optical",
        name="robot.camera-optical",
        alignment=FrameAlignment(
            reference_frame=source_child,
            basis_to_reference=child_alignment,
            purpose="sensor_mount",
            provider="application",
            alignment_source="calibration",
            alignment_evidence_reference="calibration://camera-mount/v2",
        ),
    )
    converter = OrientationConverter(
        "ros2.geometry-msgs-quaternion",
        source_parent,
        source_child,
        ros_parent,
        ros_child,
    )
    physical = PhysicalOrientation.from_wxyz(
        (0.5, 0.5, 0.5, 0.5), source_parent, source_child
    )
    target = converter.to_target(physical)
    parent_map = np.asarray(
        FrameCorrespondence(
            source_parent, ros_parent.coordinate_frame
        ).coordinates.matrix,
        dtype=np.float64,
    )
    child_map = np.asarray(
        FrameCorrespondence(
            source_child, ros_child.coordinate_frame
        ).coordinates.matrix,
        dtype=np.float64,
    )
    canonical_rotation = Rotation.from_quat(
        (
            physical.quaternion_wxyz[1],
            physical.quaternion_wxyz[2],
            physical.quaternion_wxyz[3],
            physical.quaternion_wxyz[0],
        )
    ).as_matrix()
    target_rotation = Rotation.from_quat(target.values).as_matrix()
    probe = np.asarray((0.31, -0.47, 0.83))
    np.testing.assert_allclose(
        target_rotation @ child_map @ probe,
        parent_map @ canonical_rotation @ probe,
        rtol=0.0,
        atol=2e-15,
    )
    recovered = converter.from_target(target)
    assert recovered.parent_frame.has_identical_definition(source_parent)
    assert recovered.child_frame.has_identical_definition(source_child)
    with pytest.raises(ValueError, match="child semantic binding"):
        converter.from_target(replace(target, child_semantic_binding=ros_parent))


def test_role_binding_rejects_missing_or_wrong_alignment_context():
    source_world, _ = _canonical_frame_pair()
    with pytest.raises(ValueError, match="requires alignment purpose"):
        bind_semantic_role(
            "unity.world-axes",
            name="unity.world",
            alignment=FrameAlignment(
                reference_frame=source_world,
                basis_to_reference=OrthogonalBasisMap(
                    ((1, 0, 0), (0, 0, 1), (0, 1, 0))
                ),
                purpose="model_alignment",
                provider="application",
                alignment_source="application_declaration",
                alignment_evidence_reference="app://model/v1",
            ),
        )
    with pytest.raises(ValueError, match="requires runtime-supplied alignment"):
        bind_semantic_role(
            "openxr.reference-space",
            name="openxr.local-space",
            alignment=FrameAlignment(
                reference_frame=source_world,
                basis_to_reference=SignedAxisMap(("x", "z", "-y")),
                purpose="space_alignment",
                provider="application",
                alignment_source="application_declaration",
                alignment_evidence_reference="app://guessed-space/v1",
            ),
        )
    with pytest.raises(ValueError, match="typed space context"):
        bind_semantic_role(
            "openxr.reference-space",
            name="openxr.local-space",
            alignment=FrameAlignment(
                reference_frame=source_world,
                basis_to_reference=SignedAxisMap(("x", "z", "-y")),
                purpose="space_alignment",
                provider="runtime",
                alignment_source="runtime_relation",
                alignment_evidence_reference="capture://openxr/local-space",
            ),
        )
    xr_binding = bind_semantic_role(
        "openxr.reference-space",
        name="openxr.local-space",
        alignment=FrameAlignment(
            reference_frame=source_world,
            basis_to_reference=SignedAxisMap(("x", "z", "-y")),
            purpose="space_alignment",
            provider="runtime",
            alignment_source="runtime_relation",
            alignment_evidence_reference="capture://openxr/local-space",
            space_context=OpenXRSpaceContext(
                space_id="session-a/space-17",
                base_space_id=source_world.name,
                session_id="session-a",
                time_ns=1_000_000_000,
                runtime_id="Khronos test runtime",
                reference_space_type="local",
                location_flags=15,
            ),
        ),
    )
    assert xr_binding.coordinate_frame.axes == ("right", "up", "back")
    assert xr_binding.alignment is not None
    assert xr_binding.alignment.space_context is not None
    assert xr_binding.alignment.space_context.reference_space_type == "local"
    assert xr_binding.alignment.space_context.orientation_valid
    assert xr_binding.alignment.space_context.orientation_tracked
    assert xr_binding.reference_frame is not None
    assert xr_binding.reference_frame.has_identical_definition(source_world)
    assert (
        OpenXRSpaceContext(
            space_id="session-a/view-2",
            base_space_id=source_world.name,
            session_id="session-a",
            time_ns=-1,
            runtime_id="Khronos test runtime",
            reference_space_type="view",
            location_flags=15,
        ).time_ns
        == -1
    )
    with pytest.raises(TypeError, match="signed 64-bit integer"):
        OpenXRSpaceContext(
            space_id="session-a/space-18",
            base_space_id=source_world.name,
            session_id="session-a",
            time_ns=1.0,
            runtime_id="Khronos test runtime",
            reference_space_type="local",
            location_flags=15,
        )
    with pytest.raises(ValueError, match="signed 64-bit range"):
        OpenXRSpaceContext(
            space_id="session-a/space-19",
            base_space_id=source_world.name,
            session_id="session-a",
            time_ns=1 << 63,
            runtime_id="Khronos test runtime",
            reference_space_type="local",
            location_flags=15,
        )
    with pytest.raises(ValueError, match="ORIENTATION_VALID_BIT"):
        OpenXRSpaceContext(
            space_id="session-a/space-20",
            base_space_id=source_world.name,
            session_id="session-a",
            time_ns=1,
            runtime_id="Khronos test runtime",
            reference_space_type="local",
            location_flags=0x00000004,
        )
    view_binding = bind_semantic_role(
        "openxr.view-space",
        name="openxr.view",
        alignment=FrameAlignment(
            reference_frame=source_world,
            basis_to_reference=SignedAxisMap(("x", "z", "-y")),
            purpose="space_alignment",
            provider="runtime",
            alignment_source="runtime_relation",
            alignment_evidence_reference="capture://openxr/view",
            space_context=OpenXRSpaceContext(
                space_id="session-a/view-2",
                base_space_id=source_world.name,
                session_id="session-a",
                time_ns=1,
                runtime_id="Khronos test runtime",
                reference_space_type="view",
                location_flags=15,
            ),
        ),
    )
    assert view_binding.coordinate_frame.role == "sensor"
    with pytest.raises(ValueError, match="type does not match"):
        bind_semantic_role(
            "openxr.reference-space",
            name="openxr.stage",
            alignment=FrameAlignment(
                reference_frame=source_world,
                basis_to_reference=SignedAxisMap(("x", "z", "-y")),
                purpose="space_alignment",
                provider="runtime",
                alignment_source="runtime_relation",
                alignment_evidence_reference="capture://openxr/stage",
                space_context=OpenXRSpaceContext(
                    space_id="session-a/stage-1",
                    base_space_id=source_world.name,
                    session_id="session-a",
                    time_ns=1,
                    runtime_id="Khronos test runtime",
                    reference_space_type="stage",
                    location_flags=15,
                ),
            ),
        )
    with pytest.raises(ValueError, match="handedness"):
        bind_semantic_role(
            "unity.world-axes",
            name="unity.world.invalid",
            alignment=FrameAlignment(
                reference_frame=source_world,
                basis_to_reference=SignedAxisMap(("x", "y", "z")),
                purpose="scene_alignment",
                provider="application",
                alignment_source="application_declaration",
                alignment_evidence_reference="app://scene/v1",
            ),
        )
    with pytest.raises(ValueError, match="role-specific typed context"):
        bind_semantic_role(
            "cesium.enu-to-fixed",
            name="enu-without-origin",
            alignment=FrameAlignment(
                reference_frame=source_world,
                basis_to_reference=SignedAxisMap(("x", "y", "z")),
                purpose="scene_alignment",
                provider="application",
                alignment_source="application_declaration",
                alignment_evidence_reference="app://axes-only/v1",
            ),
        )


def test_cesium_enu_binding_requires_typed_origin_and_ellipsoid():
    binding = bind_cesium_enu_frame(
        name="enu@greenwich-equator",
        origin=GeodeticOrigin(0.0, 0.0, 0.0),
        ellipsoid=Ellipsoid("WGS84", 6_378_137.0, 6_356_752.314245179),
    )
    assert binding.authority == "target_defined"
    assert binding.origin == GeodeticOrigin(0.0, 0.0, 0.0)
    assert binding.ellipsoid is not None and binding.ellipsoid.name == "WGS84"
    assert binding.reference_frame is not None
    assert binding.reference_frame.has_identical_definition(ECEF_FRAME)
    basis = binding.coordinate_frame.basis_to_reference
    assert basis.vector((1.0, 0.0, 0.0)) == (0.0, 1.0, 0.0)  # east -> ECEF +Y
    assert basis.vector((0.0, 1.0, 0.0)) == (0.0, 0.0, 1.0)  # north -> ECEF +Z
    assert basis.vector((0.0, 0.0, 1.0)) == (1.0, 0.0, 0.0)  # up -> ECEF +X


def test_openxr_roundtrip_is_bound_to_session_space_and_sample_time():
    source_world, source_body = _canonical_frame_pair()
    xr_binding = bind_semantic_role(
        "openxr.reference-space",
        name="app.openxr-local",
        alignment=FrameAlignment(
            reference_frame=source_world,
            basis_to_reference=SignedAxisMap(("x", "z", "-y")),
            purpose="space_alignment",
            provider="runtime",
            alignment_source="runtime_relation",
            alignment_evidence_reference="runtime-capture://session-a/space-17/1s",
            space_context=OpenXRSpaceContext(
                space_id="session-a/space-17",
                base_space_id=source_world.name,
                session_id="session-a",
                time_ns=1_000_000_000,
                runtime_id="Khronos test runtime",
                reference_space_type="local",
                location_flags=0x00000005,
            ),
        ),
    )
    converter = OrientationConverter(
        "openxr.xrquaternionf",
        source_world,
        source_body,
        xr_binding,
        source_body,
    )
    physical = PhysicalOrientation.from_wxyz(
        (sqrt(0.5), 0.0, 0.0, sqrt(0.5)), source_world, source_body
    )
    native = converter.to_target(physical)
    assert converter.from_target(native).parent_frame.has_identical_definition(
        source_world
    )

    assert xr_binding.alignment is not None
    assert xr_binding.alignment.space_context is not None
    original_context = xr_binding.alignment.space_context
    for changed_context in (
        replace(original_context, session_id="session-b"),
        replace(original_context, time_ns=original_context.time_ns + 1),
    ):
        changed_alignment = replace(xr_binding.alignment, space_context=changed_context)
        changed_binding = replace(xr_binding, alignment=changed_alignment)
        with pytest.raises(ValueError, match="parent semantic binding"):
            converter.from_target(
                replace(native, parent_semantic_binding=changed_binding)
            )


def test_cesium_binding_rejects_invalid_geodetic_context():
    with pytest.raises(ValueError, match="latitude"):
        GeodeticOrigin(0.0, 91.0, 0.0)
    with pytest.raises(ValueError, match="ellipsoid axes"):
        Ellipsoid("invalid", 1.0, 2.0)


def test_cesium_binding_rejects_ellipsoid_center_without_division_error():
    with pytest.raises(ValueError, match="undefined at the ellipsoid center"):
        bind_cesium_enu_frame(
            name="enu@ellipsoid-center",
            origin=GeodeticOrigin(0.0, 0.0, -6_378_137.0),
            ellipsoid=Ellipsoid("WGS84", 6_378_137.0, 6_356_752.314245179),
        )


def test_cesium_enu_basis_uses_height_and_ellipsoid_context():
    ellipsoid = Ellipsoid("WGS84", 6_378_137.0, 6_356_752.314245179)
    at_surface = bind_cesium_enu_frame(
        name="enu@surface",
        origin=GeodeticOrigin(-75.0, 40.0, 0.0),
        ellipsoid=ellipsoid,
    )
    above_surface = bind_cesium_enu_frame(
        name="enu@height",
        origin=GeodeticOrigin(-75.0, 40.0, 1_000.0),
        ellipsoid=ellipsoid,
    )
    sphere = bind_cesium_enu_frame(
        name="enu@sphere",
        origin=GeodeticOrigin(-75.0, 40.0, 1_000.0),
        ellipsoid=Ellipsoid("sphere", 6_378_137.0, 6_378_137.0),
    )
    assert (
        np.max(
            np.abs(
                np.asarray(at_surface.coordinate_frame.basis_to_reference.matrix)
                - np.asarray(above_surface.coordinate_frame.basis_to_reference.matrix)
            )
        )
        > 1e-8
    )
    assert (
        np.max(
            np.abs(
                np.asarray(sphere.coordinate_frame.basis_to_reference.matrix)
                - np.asarray(above_surface.coordinate_frame.basis_to_reference.matrix)
            )
        )
        > 1e-8
    )


def test_semantic_binding_rejects_a_frame_broken_away_from_its_context():
    origin = GeodeticOrigin(12.0, 47.0, 450.0)
    ellipsoid = Ellipsoid("WGS84", 6_378_137.0, 6_356_752.314245179)
    valid = bind_cesium_enu_frame(
        name="enu@site",
        origin=origin,
        ellipsoid=ellipsoid,
    )
    forged_frame = CoordinateFrame(
        name="enu@site",
        reference=valid.coordinate_frame.reference,
        axes=("east", "north", "up"),
        basis_to_reference=SignedAxisMap(("x", "y", "z")),
        handedness="right",
        role="world",
        reference_convention=valid.coordinate_frame.reference_convention,
    )
    with pytest.raises(ValueError, match="basis does not match its typed origin"):
        SemanticFrameBinding(
            contract_id=valid.contract_id,
            authority=valid.authority,
            coordinate_frame=forged_frame,
            evidence_reference=valid.evidence_reference,
            target_family=valid.target_family,
            reference_frame=valid.reference_frame,
            origin=origin,
            ellipsoid=ellipsoid,
        )
