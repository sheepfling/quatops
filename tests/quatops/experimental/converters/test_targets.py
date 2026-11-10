"""Tests for scalar target converter declarations."""

import pytest

from quatops.experimental.converters import (
    CONVERTERS,
    SEMANTIC_CONTRACTS,
    TARGET_DECLARATIONS,
    ConverterProfile,
    CoordinateFrame,
    FrameCorrespondence,
    FramedOrientation,
    OrientationConverter,
    SemanticContract,
    SemanticQuaternionMap,
    TargetOrientation,
    accepted_converters,
    get_converter,
    get_target_declaration,
    reorder_quaternion_components,
)
from quatops.experimental.engines import SignedAxisMap
from quatops.experimental.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    PhysicalOrientation,
)
from quatops.python import hamilton


def _identity_map():
    source_world = CoordinateFrame(
        "canonical.world",
        "world-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    source_body = CoordinateFrame(
        "canonical.body",
        "body-reference",
        ("right", "forward", "up"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    target_world = CoordinateFrame(
        "target.world",
        "world-reference",
        ("x", "y", "z"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    target_body = CoordinateFrame(
        "target.body",
        "body-reference",
        ("x", "y", "z"),
        SignedAxisMap(("x", "y", "z")),
        "right",
    )
    return SemanticQuaternionMap(
        FrameCorrespondence(source_world, target_world),
        FrameCorrespondence(source_body, target_body),
    )


@pytest.mark.parametrize("converter_id", tuple(CONVERTERS))
def test_declared_profiles_convert_and_recover_framed_orientations(converter_id):
    profile = get_converter(converter_id)
    frame_map = _identity_map()
    source_q = hamilton.from_rotvec((0.47, -0.66, 0.89))
    source = FramedOrientation(source_q, "canonical.body", "canonical.world")

    native = profile.to_target(source, frame_map)
    assert native.converter_id == converter_id
    assert (native.from_frame.name, native.to_frame.name) == (
        "target.body",
        "target.world",
    )
    recovered = profile.from_target(native, frame_map)
    assert (recovered.from_frame, recovered.to_frame) == (
        "canonical.body",
        "canonical.world",
    )
    assert hamilton.rotations_equivalent(recovered.quaternion_wxyz, source_q)


@pytest.mark.parametrize("converter_id", tuple(CONVERTERS))
def test_profile_antipode_policy_is_explicit_in_its_representation(converter_id):
    profile = get_converter(converter_id)
    frame_map = _identity_map()
    quaternion = (0.8, 0.2, -0.4, 0.4)
    antipode = tuple(-value for value in quaternion)
    source = FramedOrientation(quaternion, "canonical.body", "canonical.world")
    opposite = FramedOrientation(antipode, "canonical.body", "canonical.world")

    target = profile.to_target(source, frame_map)
    antipodal_target = profile.to_target(opposite, frame_map)
    restored = profile.from_target(target, frame_map)
    antipodal_restored = profile.from_target(antipodal_target, frame_map)

    if profile.representation == "quaternion":
        assert antipodal_target.values == tuple(-value for value in target.values)
        assert restored.quaternion_wxyz == quaternion
        assert antipodal_restored.quaternion_wxyz == antipode
    else:
        assert antipodal_target.values == target.values
        assert antipodal_restored.quaternion_wxyz == restored.quaternion_wxyz
        assert hamilton.rotations_equivalent(
            antipodal_restored.quaternion_wxyz, antipode
        )


def test_target_profiles_preserve_declared_scope_and_acceptance_state():
    assert get_converter("blender.mathutils.quaternion").evidence_status == (
        "accepted_converter"
    )
    openxr = get_converter("openxr.xrquaternionf")
    assert openxr.evidence_status == "declared_contract"
    assert "no vendor-runtime" in openxr.scope
    assert (
        get_converter("sophus.so3-params").evidence_status == "supporting_oracle_only"
    )
    accepted = accepted_converters()
    assert len(accepted) == 14
    assert all(
        profile.evidence_status == "accepted_converter" for profile in accepted.values()
    )


def test_orientation_converter_exposes_nonaccepted_profile_states():
    source_world = CANONICAL_WORLD_RFU.bind("source.world", "world-reference")
    source_body = CANONICAL_BODY_RFU.bind("source.body", "body-reference")
    declared_openxr = OrientationConverter(
        "openxr.xrquaternionf", source_world, source_body, source_world, source_body
    )
    supporting_sophus = OrientationConverter(
        "sophus.so3-params", source_world, source_body, source_world, source_body
    )

    assert declared_openxr.evidence_status == "declared_contract"
    assert supporting_sophus.evidence_status == "supporting_oracle_only"
    physical = PhysicalOrientation.from_wxyz(
        (0.8, 0.2, -0.4, 0.4), source_world, source_body
    )
    assert declared_openxr.to_target(physical).converter_id == "openxr.xrquaternionf"
    assert supporting_sophus.to_target(physical).converter_id == "sophus.so3-params"
    assert declared_openxr.evidence_status == "declared_contract"
    assert supporting_sophus.evidence_status == "supporting_oracle_only"


def test_target_index_distinguishes_converters_from_supporting_evidence():
    assert TARGET_DECLARATIONS["openxr.xrquaternionf"].role == "orientation_converter"
    assert TARGET_DECLARATIONS["proj.9.5.1"].role == "frame_provider"
    assert TARGET_DECLARATIONS["khronos.gltf-validator"].role == "format_validator"
    assert get_target_declaration("blender.gltf-import").role == "loader_observation"


def test_representation_evidence_and_semantic_authority_are_independent():
    eigen = get_converter("eigen.quaternion-coeffs")
    assert eigen.representation_evidence == "native_verified"
    assert eigen.semantic_authority == "representation_only"
    openxr = get_converter("openxr.xrquaternionf")
    assert openxr.representation_evidence == "spec_and_test_runtime"
    assert openxr.semantic_authority == "representation_only"
    assert SEMANTIC_CONTRACTS["openxr.reference-space"].binding_kind == "fixed_axes"
    assert SEMANTIC_CONTRACTS["openxr.view-space"].frame_role == "sensor"
    assert SEMANTIC_CONTRACTS["cesium.enu-to-fixed"].binding_kind == "geographic_enu"
    assert SEMANTIC_CONTRACTS["gltf.asset-front"].authority == "standard_defined"
    assert get_converter("gltf.rotation").semantic_authority == "representation_only"
    assert all(
        profile.semantic_authority == "representation_only"
        for profile in CONVERTERS.values()
    )


def test_semantic_contract_authority_requires_documented_source_kind():
    with pytest.raises(ValueError, match="formal standard"):
        SemanticContract(
            contract_id="bad.standard-claim",
            authority="standard_defined",
            role="fixture axis assignment",
            source_kind="target_documentation",
            binding_kind="fixed_axes",
            source_url="https://example.invalid/spec",
            source_version="1",
            source_section="axes",
            positive_axes=("x", "y", "z"),
            handedness="right",
            alignment_purpose="scene_alignment",
        )


def test_storage_reorder_does_not_change_quaternion_algebra_or_frame():
    q_wxyz = (0.8, 0.2, -0.4, 0.4)
    q_xyzw = reorder_quaternion_components(
        q_wxyz, source_order="wxyz", target_order="xyzw"
    )
    assert q_xyzw == (0.2, -0.4, 0.4, 0.8)
    assert (
        reorder_quaternion_components(q_xyzw, source_order="xyzw", target_order="wxyz")
        == q_wxyz
    )
    with pytest.raises(ValueError, match="four coefficients"):
        reorder_quaternion_components(
            (1.0, 0.0, 0.0), source_order="wxyz", target_order="xyzw"
        )
    with pytest.raises(ValueError, match="target documentation"):
        SemanticContract(
            contract_id="bad.fixture-claim",
            authority="target_defined",
            role="assigned numeric probe frame",
            source_kind="formal_standard",
            binding_kind="fixed_axes",
            source_url="https://example.invalid/spec",
            source_version="1",
            source_section="axes",
            positive_axes=("x", "y", "z"),
            handedness="right",
            alignment_purpose="scene_alignment",
        )


def test_converter_rejects_wrong_profile_or_frame_pair():
    frame_map = _identity_map()
    profile = get_converter("unity.quaternion")
    wrong_profile_value = TargetOrientation(
        converter_id="godot.quaternion",
        values=(0.0, 0.0, 0.0, 1.0),
        representation="quaternion",
        component_order="xyzw",
        from_frame=frame_map.body.destination,
        to_frame=frame_map.world.destination,
    )
    with pytest.raises(ValueError, match="different converter"):
        profile.from_target(wrong_profile_value, frame_map)

    wrong_order = ConverterProfile(
        converter_id="invalid",
        target="Test",
        api="Test",
        version="1",
        representation="quaternion",
        component_order="wxyz",
        evidence_status="accepted_converter",
        representation_evidence="native_verified",
        semantic_authority="representation_only",
        scope="test-only",
    )
    with pytest.raises(ValueError, match="layout"):
        wrong_order.from_target(
            TargetOrientation(
                converter_id="invalid",
                values=(1.0, 0.0, 0.0, 0.0),
                representation="quaternion",
                component_order="xyzw",
                from_frame=frame_map.body.destination,
                to_frame=frame_map.world.destination,
            ),
            frame_map,
        )


def test_manual_native_target_requires_declared_frame_definitions():
    with pytest.raises(TypeError, match="CoordinateFrame definition"):
        TargetOrientation(
            converter_id="unity.quaternion",
            values=(0.0, 0.0, 0.0, 1.0),
            representation="quaternion",
            component_order="xyzw",
            from_frame="native.body",
            to_frame="native.world",
        )
