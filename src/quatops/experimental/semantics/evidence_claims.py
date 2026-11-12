"""Exact subjects, operations, scopes, and observations credited as evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SemanticEvidenceOperation = Literal[
    "quaternion_action",
    "quaternion_field",
    "hierarchy_composition",
    "named_frame_relation",
    "geographic_frame_action",
    "runtime_space_relation",
    "camera_node_action",
    "asset_front_observation",
]
SemanticEvidenceScope = Literal[
    "numeric_probe",
    "static_node",
    "named_frame_chain",
    "enu_at_origin",
    "LOCAL",
    "LOCAL_FLOOR",
    "VIEW",
    "camera_node",
    "oriented_mesh",
]


@dataclass(frozen=True)
class SemanticEvidenceClaim:
    """A limited claim whose witness IDs must resolve in a validated capture.

    Association with a semantic role does not establish that entire role or
    physical alignment. The operation and scope are the complete native claim.
    """

    contract_id: str
    converter_id: str
    evidence_id: str
    version: str
    operation: SemanticEvidenceOperation
    scope: SemanticEvidenceScope
    witness_ids: tuple[str, ...]


SEMANTIC_EVIDENCE_CLAIMS = (
    SemanticEvidenceClaim(
        "unreal.world-axes",
        "unreal.fquat",
        "unreal.native.5.7.4",
        "5.7.4",
        "quaternion_action",
        "numeric_probe",
        ("identity", "mixed_a", "mixed_b"),
    ),
    SemanticEvidenceClaim(
        "unity.world-axes",
        "unity.quaternion",
        "unity.native.6000.5.2f1",
        "6000.5.2f1",
        "quaternion_action",
        "numeric_probe",
        ("identity", "mixed_a", "mixed_b"),
    ),
    SemanticEvidenceClaim(
        "godot.camera-axes",
        "godot.quaternion",
        "godot.native.4.7",
        "4.7-stable",
        "quaternion_action",
        "numeric_probe",
        ("identity", "mixed_a", "mixed_b"),
    ),
    SemanticEvidenceClaim(
        "godot.camera-axes",
        "godot.quaternion",
        "godot.native.4.8-dev1",
        "4.8-dev1",
        "quaternion_action",
        "numeric_probe",
        ("identity", "mixed_a", "mixed_b"),
    ),
    SemanticEvidenceClaim(
        "godot.camera-axes",
        "godot.quaternion",
        "godot.camera3d.4.7-stable",
        "4.7-stable",
        "camera_node_action",
        "camera_node",
        ("0_0", "1_2", "2_1"),
    ),
    SemanticEvidenceClaim(
        "godot.camera-axes",
        "godot.quaternion",
        "godot.camera3d.4.8-dev1",
        "4.8-dev1",
        "camera_node_action",
        "camera_node",
        ("0_0", "1_2", "2_1"),
    ),
    SemanticEvidenceClaim(
        "ros.rep103.body-flu",
        "ros2.geometry-msgs-quaternion",
        "ros2.tf2.0.36.23",
        "0.36.23",
        "named_frame_relation",
        "named_frame_chain",
        ("world-base-sensor",),
    ),
    SemanticEvidenceClaim(
        "ros.rep103.optical",
        "ros2.geometry-msgs-quaternion",
        "ros2.tf2.0.36.23",
        "0.36.23",
        "named_frame_relation",
        "named_frame_chain",
        ("world-base-sensor",),
    ),
    SemanticEvidenceClaim(
        "cesium.enu-to-fixed",
        "cesium.quaternion",
        "cesium.native.1.124.0",
        "1.124.0",
        "geographic_frame_action",
        "enu_at_origin",
        ("identity@geo-00", "mixed_a@geo-03"),
    ),
    SemanticEvidenceClaim(
        "gltf.asset-front",
        "gltf.rotation",
        "gltf.threejs.oriented-mesh.0.187.0",
        "0.187.0",
        "asset_front_observation",
        "oriented_mesh",
        ("OrientedMesh",),
    ),
    SemanticEvidenceClaim(
        "gltf.node-local",
        "gltf.rotation",
        "gltf.threejs.loader.0.187.0",
        "0.187.0",
        "quaternion_field",
        "static_node",
        ("QuatOpsParent", "QuatOpsChild"),
    ),
    SemanticEvidenceClaim(
        "gltf.node-local",
        "gltf.rotation",
        "gltf.threejs.loader.0.187.0",
        "0.187.0",
        "hierarchy_composition",
        "static_node",
        ("QuatOpsParent", "QuatOpsChild"),
    ),
    SemanticEvidenceClaim(
        "openxr.reference-space",
        "openxr.xrquaternionf",
        "openxr.test-runtime.1.1.65",
        "1.1.65",
        "runtime_space_relation",
        "LOCAL",
        ("A_from_B", "B_from_C", "A_from_C"),
    ),
    SemanticEvidenceClaim(
        "openxr.view-space",
        "openxr.xrquaternionf",
        "openxr.test-runtime.view.1.1.65",
        "1.1.65",
        "runtime_space_relation",
        "VIEW",
        ("A_from_B", "B_from_C", "A_from_C"),
    ),
    SemanticEvidenceClaim(
        "openxr.reference-space",
        "openxr.xrquaternionf",
        "openxr.test-runtime.local_floor.1.1.65",
        "1.1.65",
        "runtime_space_relation",
        "LOCAL_FLOOR",
        ("A_from_B", "B_from_C", "A_from_C"),
    ),
)
