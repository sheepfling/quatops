"""Tests for scalar target converter declarations."""

import pytest

from quatops.converters import (
    CONVERTERS,
    TARGET_DECLARATIONS,
    ConverterProfile,
    CoordinateFrame,
    FrameCorrespondence,
    FramedOrientation,
    SemanticQuaternionMap,
    TargetOrientation,
    get_converter,
    get_target_declaration,
)
from quatops.engines import SignedAxisMap
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
    assert (native.from_frame, native.to_frame) == ("target.body", "target.world")
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


def test_target_index_distinguishes_converters_from_supporting_evidence():
    assert TARGET_DECLARATIONS["openxr.xrquaternionf"].role == "orientation_converter"
    assert TARGET_DECLARATIONS["proj.9.5.1"].role == "frame_provider"
    assert TARGET_DECLARATIONS["khronos.gltf-validator"].role == "format_validator"
    assert get_target_declaration("blender.gltf-import").role == "loader_observation"


def test_converter_rejects_wrong_profile_or_frame_pair():
    frame_map = _identity_map()
    profile = get_converter("unity.quaternion")
    wrong_profile_value = TargetOrientation(
        converter_id="godot.quaternion",
        values=(0.0, 0.0, 0.0, 1.0),
        representation="quaternion",
        component_order="xyzw",
        from_frame="target.body",
        to_frame="target.world",
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
        scope="test-only",
    )
    with pytest.raises(ValueError, match="layout"):
        wrong_order.from_target(
            TargetOrientation(
                converter_id="invalid",
                values=(1.0, 0.0, 0.0, 0.0),
                representation="quaternion",
                component_order="xyzw",
                from_frame="other.body",
                to_frame="other.world",
            ),
            frame_map,
        )
