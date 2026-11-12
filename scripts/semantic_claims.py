"""Check semantic evidence credit against validated native capture contents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from quatops.experimental.converters.targets import CONVERTERS
from quatops.experimental.semantics.assurance import SemanticEvidenceArtifact
from quatops.experimental.semantics.evidence_claims import (
    SemanticEvidenceClaim,
    SemanticEvidenceOperation,
    SemanticEvidenceScope,
)
from quatops.experimental.semantics.roles import SemanticContract
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.crystallized import (
    EngineQuaternionSerialization,
    FrameChainSerialization,
    GeographicSerialization,
    SceneTransportSerialization,
)
from scripts.evidence_schema.godot_camera import load_camera_capture
from scripts.evidence_schema.openxr import (
    OpenXRSemanticConfig,
    OpenXRTestRuntimeCapture,
)
from scripts.evidence_schema.openxr_spaces import validate_space_capture
from scripts.evidence_schema.oriented_mesh import validate_oriented_mesh
from scripts.evidence_schema.utilities import sha256_file
from scripts.evidence_schema.validation import (
    ValidatedCapture,
    validate_evidence_packages,
)

_NATIVE_TARGET_NAMES = {
    "Unreal": "Unreal Engine",
    "Unity": "Unity Editor",
    "Godot": "Godot Engine",
    "ROS 2": "ROS 2 tf2",
    "CesiumJS": "CesiumJS",
    "glTF": "Three.js GLTFLoader",
}
CaptureCoverage = dict[
    tuple[SemanticEvidenceOperation, SemanticEvidenceScope], set[str]
]


def _capture_coverage(capture: ValidatedCapture) -> CaptureCoverage:
    serialization = capture.serialization
    if isinstance(serialization, EngineQuaternionSerialization):
        return {
            ("quaternion_action", "numeric_probe"): {
                row.case_id
                for row in serialization.witnesses
                if row.kind == "orientation"
            }
        }
    if isinstance(serialization, FrameChainSerialization):
        return {
            ("named_frame_relation", "named_frame_chain"): {
                row.case_id
                for row in serialization.witnesses
                if row.lookups and row.point_actions and row.messages
            }
        }
    if isinstance(serialization, GeographicSerialization):
        return {
            ("geographic_frame_action", "enu_at_origin"): {
                row.case_id
                for row in serialization.witnesses
                if row.world_directions_xyz and row.ecef_origin_xyz_m
            }
        }
    if isinstance(serialization, SceneTransportSerialization):
        nodes = {row.node_id: row for row in serialization.witnesses}
        hierarchy = set()
        for row in nodes.values():
            if row.parent_id in nodes and (
                row.world_matrix_rows is not None
                or row.world_basis_images_xyz is not None
            ):
                parent = nodes[row.parent_id]
                hierarchy.update((row.node_id, parent.node_id))
        return {
            ("quaternion_field", "static_node"): set(nodes),
            ("hierarchy_composition", "static_node"): hierarchy,
        }
    return {}


def _openxr_coverage(
    root: Path, artifact: SemanticEvidenceArtifact
) -> tuple[str, CaptureCoverage]:
    path = root / artifact.artifact_path
    if path.name in {"test_runtime_view.json", "test_runtime_local_floor.json"}:
        version, space, witnesses = validate_space_capture(root, path)
        if artifact.adapter_conformance_checked:
            raise ValueError(
                "OpenXR static-offset capture did not execute QuatOps adapter"
            )
        return version, {("runtime_space_relation", space): witnesses}
    capture = OpenXRTestRuntimeCapture.model_validate(
        read_evidence_document(path), strict=True
    )
    manifest = read_evidence_document(path.parent.parent / "manifest.json")
    config_path = path.parent.parent / "semantic_contract.json"
    config = OpenXRSemanticConfig.model_validate(
        read_evidence_document(config_path), strict=True
    )
    semantic_contract = manifest.get("semantic_contract")
    if not isinstance(semantic_contract, dict) or semantic_contract.get(
        "sha256"
    ) != sha256_file(config_path):
        raise ValueError(
            f"{artifact.evidence_id}: OpenXR semantic config digest differs from provenance"
        )
    if (
        capture.api_version != config.api_version
        or capture.reference_space_type != config.reference_space.reference_space_type
    ):
        raise ValueError(
            f"{artifact.evidence_id}: OpenXR capture differs from its semantic config"
        )
    cases = {case.observation: case for case in config.relation_cases}
    if set(capture.observations) != set(cases) or any(
        row.target != cases[key].target or row.base != cases[key].base
        for key, row in capture.observations.items()
    ):
        raise ValueError(
            f"{artifact.evidence_id}: OpenXR observation direction differs from its semantic config"
        )
    artifacts = manifest.get("artifacts")
    record = (
        artifacts.get("test_runtime_frame_chain")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(record, dict) or record.get("witness_sha256") != sha256_file(
        path
    ):
        raise ValueError(
            f"{artifact.evidence_id}: OpenXR witness digest differs from provenance"
        )
    if artifact.adapter_conformance_checked and not capture.quatops_executed:
        raise ValueError(
            f"{artifact.evidence_id}: capture did not execute the QuatOps adapter"
        )
    return ".".join(str(value) for value in capture.api_version), {
        ("runtime_space_relation", capture.reference_space_type): set(
            capture.observations
        )
    }


def validate_evidence_claims(
    root: Path,
    claims: Sequence[SemanticEvidenceClaim],
    evidence: Mapping[str, SemanticEvidenceArtifact],
    contracts: Mapping[str, SemanticContract],
) -> None:
    """Reject unsupported operations, scopes, versions, subjects, and witnesses.

    This development check deliberately uses the same target-specific parsing,
    semantic serializers, normalization, and provenance checks as evidence replay.
    New native capabilities need a matching typed capture interpretation here.
    """
    captures = validate_evidence_packages(root)
    linked: dict[str, set[str]] = {}
    seen: set[SemanticEvidenceClaim] = set()
    for claim in claims:
        if claim in seen:
            raise ValueError("semantic evidence repeats an identical claim")
        seen.add(claim)
        contract = contracts.get(claim.contract_id)
        artifact = evidence.get(claim.evidence_id)
        profile = CONVERTERS.get(claim.converter_id)
        if contract is None or artifact is None or profile is None:
            raise ValueError(
                "semantic evidence claim references an unknown subject or artifact"
            )
        if not artifact.native_runtime_capture:
            raise ValueError(
                f"{claim.evidence_id}: native claim requires a native runtime capture"
            )
        if (
            claim.converter_id not in contract.converter_ids
            or profile.target != artifact.target
        ):
            raise ValueError(
                f"{claim.contract_id}: evidence claim profile does not match its subject"
            )
        if claim.evidence_id not in contract.execution_evidence_ids:
            raise ValueError(
                f"{claim.contract_id}: evidence claim has no declared role association"
            )
        if claim.operation == "runtime_space_relation":
            allowed_spaces = {
                "openxr.reference-space": {"LOCAL", "LOCAL_FLOOR"},
                "openxr.view-space": {"VIEW"},
            }
            if claim.scope not in allowed_spaces.get(claim.contract_id, set()):
                raise ValueError(
                    f"{claim.contract_id}: reference-space scope conflicts with the role"
                )
        linked.setdefault(claim.contract_id, set()).add(claim.evidence_id)
        path = (root / artifact.artifact_path).resolve(strict=True)
        matches = [
            capture
            for capture in captures
            if capture.witness_path.resolve() == path
            and capture.version == claim.version
        ]
        if matches:
            if matches[0].target != _NATIVE_TARGET_NAMES.get(artifact.target):
                raise ValueError(
                    f"{claim.evidence_id}: validated capture target differs from the claimed target"
                )
            coverage = _capture_coverage(matches[0])
        elif artifact.target == "Godot" and claim.evidence_id.startswith(
            "godot.camera3d."
        ):
            camera = load_camera_capture(root, path)
            if camera.version.split(" ")[0] != claim.version:
                raise ValueError(
                    "Camera3D claim version differs from validated capture"
                )
            coverage = {
                ("camera_node_action", "camera_node"): {row.id for row in camera.cases}
            }
        elif (
            artifact.target == "glTF"
            and claim.evidence_id == "gltf.threejs.oriented-mesh.0.187.0"
        ):
            version, witnesses = validate_oriented_mesh(root, path)
            if version != claim.version:
                raise ValueError(
                    "oriented mesh claim version differs from validated capture"
                )
            if artifact.adapter_conformance_checked:
                raise ValueError(
                    "oriented mesh native capture did not execute QuatOps adapter"
                )
            coverage = {("asset_front_observation", "oriented_mesh"): witnesses}
        elif artifact.target == "OpenXR":
            version, coverage = _openxr_coverage(root, artifact)
            if claim.version != version:
                raise ValueError(
                    f"{claim.evidence_id}: claim version differs from the validated capture"
                )
        else:
            raise ValueError(
                f"{claim.evidence_id}: claim version or artifact differs from the validated capture"
            )
        available = coverage.get((claim.operation, claim.scope), set())
        if not available:
            raise ValueError(
                f"{claim.evidence_id}: capture does not support {claim.operation} in scope {claim.scope}"
            )
        if (
            not claim.witness_ids
            or len(set(claim.witness_ids)) != len(claim.witness_ids)
            or not set(claim.witness_ids) <= available
        ):
            raise ValueError(
                f"{claim.evidence_id}: claim witness IDs do not resolve for its operation and scope"
            )
    for contract in contracts.values():
        if linked.get(contract.contract_id, set()) != set(
            contract.execution_evidence_ids
        ):
            raise ValueError(
                f"{contract.contract_id}: evidence association lacks a scoped claim"
            )
