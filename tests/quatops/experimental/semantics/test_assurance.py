"""Audit registered semantic claims and their linked evidence artifacts."""

from dataclasses import replace

import pytest

from quatops.experimental.converters import CONVERTERS
from quatops.experimental.semantics import (
    SEMANTIC_CONTRACTS,
    semantic_assurance_rows,
    validate_semantic_assurance,
)
from quatops.experimental.semantics.assurance import SEMANTIC_EVIDENCE
from tests.support.paths import REPOSITORY_ROOT


def test_semantic_claims_have_scoped_sources_and_explicit_exclusions():
    for contract in SEMANTIC_CONTRACTS.values():
        if contract.authority in ("target_defined", "standard_defined"):
            assert contract.source_url is not None
            assert contract.source_version
            assert contract.source_section
            assert contract.source_claim
            assert contract.assurance_scope
            assert contract.exclusions
        for converter_id in contract.converter_ids:
            profile = CONVERTERS[converter_id]
            assert profile.target == contract.target_family
            assert profile.semantic_authority == "representation_only"


def test_registered_axis_claims_match_the_reviewed_primary_sources():
    expected = {
        "unreal.world-axes": (
            "https://dev.epicgames.com/documentation/en-us/unreal-engine/coordinate-system-and-spaces-in-unreal-engine?application_version=5.7",
            ("forward", "right", "up"),
            "left",
            "world",
        ),
        "unity.world-axes": (
            "https://docs.unity.com/en-us/engine/6000.5/manual/scripting/programming-math/unity-engine-math/class-quaternion",
            ("right", "up", "forward"),
            "left",
            "world",
        ),
        "godot.camera-axes": (
            "https://docs.godotengine.org/en/4.7/tutorials/assets_pipeline/importing_3d_scenes/model_export_considerations.html",
            ("right", "up", "back"),
            "right",
            "sensor",
        ),
        "ros.rep103.body-flu": (
            "https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md",
            ("forward", "left", "up"),
            "right",
            "body",
        ),
        "ros.rep103.optical": (
            "https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md",
            ("right", "down", "forward"),
            "right",
            "sensor",
        ),
        "gltf.asset-front": (
            "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html",
            ("left", "up", "forward"),
            "right",
            "body",
        ),
        "openxr.reference-space": (
            "https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html",
            ("right", "up", "back"),
            "right",
            "world",
        ),
        "openxr.view-space": (
            "https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html",
            ("right", "up", "back"),
            "right",
            "sensor",
        ),
    }
    for contract_id, (source_url, axes, handedness, frame_role) in expected.items():
        contract = SEMANTIC_CONTRACTS[contract_id]
        assert contract.source_url == source_url
        assert contract.positive_axes == axes
        assert contract.handedness == handedness
        assert contract.frame_role == frame_role
        assert contract.source_claim
        assert contract.assurance_scope
        if contract_id.startswith("ros.rep103."):
            assert contract.source_version == "REP-103"

    cesium = SEMANTIC_CONTRACTS["cesium.enu-to-fixed"]
    assert cesium.binding_kind == "geographic_enu"
    assert cesium.source_url == (
        "https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/"
        "Transforms.html#eastNorthUpToFixedFrame"
    )


def test_axis_labels_cannot_claim_a_conflicting_handedness():
    contract = SEMANTIC_CONTRACTS["unreal.world-axes"]
    with pytest.raises(ValueError, match="handedness conflicts"):
        replace(contract, handedness="right")


def test_camera_forward_and_oriented_asset_front_remain_distinct_roles():
    camera = SEMANTIC_CONTRACTS["godot.camera-axes"]
    asset = SEMANTIC_CONTRACTS["gltf.asset-front"]
    assert camera.positive_axes == ("right", "up", "back")
    assert camera.frame_role == "sensor"
    assert asset.positive_axes == ("left", "up", "forward")
    assert asset.frame_role == "body"
    assert any("arbitrary" in item for item in camera.exclusions)
    assert "arbitrary mesh geometry" in asset.exclusions[0]


def test_gltf_node_hierarchy_contract_cites_transform_sections():
    node_local = SEMANTIC_CONTRACTS["gltf.node-local"]
    assert node_local.source_section == (
        "3.5.2 Nodes and Hierarchy; 3.5.3 Transformations; 5.25.6 node.rotation"
    )


def test_assurance_matrix_does_not_promote_openxr_test_runtime_to_adapter_acceptance():
    rows = {row["contract"].contract_id: row for row in semantic_assurance_rows()}
    for role_id in ("openxr.reference-space", "openxr.view-space"):
        assert rows[role_id]["representation"] == "R specification/test-runtime only"
        assert "runtime_space_relation" in rows[role_id]["native"]
        assert "adapter conformance not demonstrated" in rows[role_id]["native"]
    openxr = SEMANTIC_EVIDENCE["openxr.test-runtime.1.1.65"]
    assert openxr.proof_scope == "runtime_space_relation"
    assert openxr.adapter_conformance_checked is False


def test_native_proof_scope_does_not_overstate_role_specific_coverage():
    evidence = {item.evidence_id: item for item in SEMANTIC_EVIDENCE.values()}
    assert evidence["gltf.threejs.loader.0.187.0"].proof_scope == "numeric_action"
    assert evidence["godot.native.4.7"].proof_scope == "numeric_action"
    assert evidence["ros2.tf2.0.36.23"].proof_scope == "runtime_space_relation"
    assert evidence["cesium.native.1.124.0"].proof_scope == "role_specific_action"


def test_evidence_ids_resolve_to_existing_artifacts_and_replay_tests():
    validate_semantic_assurance(str(REPOSITORY_ROOT))
