"""Compact, generated assurance claims for registered semantic roles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from .roles import SEMANTIC_CONTRACTS

if TYPE_CHECKING:
    from ..converters.targets import ConverterProfile


@dataclass(frozen=True)
class SemanticEvidenceArtifact:
    """Repository artifacts backing a native proof at an explicit scope."""

    evidence_id: str
    artifact_path: str
    replay_test_path: str
    target: str
    version: str
    native_runtime_capture: bool
    proof_scope: Literal[
        "numeric_action", "role_specific_action", "runtime_space_relation"
    ]
    adapter_conformance_checked: bool
    claim: str


_EVIDENCE_ITEMS = (
    SemanticEvidenceArtifact(
        "unreal.native.5.7.4",
        "tests/oracles/unreal/5.7.4/witnesses.json",
        "tests/quatops/experimental/engines/test_canonical_interoperability.py",
        "Unreal",
        "5.7.4",
        True,
        "numeric_action",
        True,
        "Frozen Unreal quaternion vector actions replayed against the declared axis "
        "map; no application scene alignment claim.",
    ),
    SemanticEvidenceArtifact(
        "unity.native.6000.5.2f1",
        "tests/oracles/unity/6000.5.2f1/witnesses.json",
        "tests/quatops/experimental/engines/test_canonical_interoperability.py",
        "Unity",
        "6000.5.2f1",
        True,
        "numeric_action",
        True,
        "Frozen Unity quaternion vector actions replayed against the declared axis "
        "map; no application scene alignment claim.",
    ),
    SemanticEvidenceArtifact(
        "godot.native.4.7",
        "tests/oracles/godot/shared/witnesses.json",
        "tests/quatops/experimental/engines/test_canonical_interoperability.py",
        "Godot",
        "4.7-stable",
        True,
        "numeric_action",
        True,
        "Frozen Godot quaternion vector actions replayed for the captured probe frame; no Camera3D action claim.",
    ),
    SemanticEvidenceArtifact(
        "godot.native.4.8-dev1",
        "tests/oracles/godot/shared/witnesses.json",
        "tests/quatops/experimental/engines/test_canonical_interoperability.py",
        "Godot",
        "4.8-dev1",
        True,
        "numeric_action",
        True,
        "Frozen Godot quaternion vector actions replayed for the captured probe frame; no Camera3D action claim.",
    ),
    SemanticEvidenceArtifact(
        "ros2.tf2.0.36.23",
        "tests/golden/oracles/ros2/0.36.23/witnesses.json",
        "tests/experiments/converters/ros2/test_ros_tf2_converter.py",
        "ROS 2",
        "0.36.23",
        True,
        "runtime_space_relation",
        True,
        "Native tf2 named-frame chain and point action; does not prove camera-to-body extrinsics or sensor mounting.",
    ),
    SemanticEvidenceArtifact(
        "cesium.native.1.124.0",
        "tests/golden/transports/cesium/1.124.0/witnesses.json",
        "tests/experiments/converters/cesium/test_cesium_converter.py",
        "CesiumJS",
        "1.124.0",
        True,
        "role_specific_action",
        True,
        "Native ENU-to-fixed frame actions replayed at explicit geodetic origins and "
        "ellipsoid; body alignment remains supplied.",
    ),
    SemanticEvidenceArtifact(
        "gltf.threejs.loader.0.187.0",
        "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json",
        "tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py",
        "glTF",
        "2.0 / Three.js 0.187.0",
        True,
        "numeric_action",
        True,
        "A QuatOps-produced static hierarchy was loaded by Three.js GLTFLoader; no "
        "oriented mesh geometry verifies asset-front.",
    ),
    SemanticEvidenceArtifact(
        "openxr.test-runtime.1.1.65",
        "experiments/oracles/openxr/1.1.65/witnesses/test_runtime_frame_chain.json",
        "tests/experiments/oracles/openxr/test_openxr_test_runtime.py",
        "OpenXR",
        "1.1.65 / Khronos test runtime",
        True,
        "runtime_space_relation",
        False,
        "Native static-space xrLocateSpace relation from the Khronos test runtime; "
        "no vendor hardware or QuatOps adapter acceptance claimed.",
    ),
)

SEMANTIC_EVIDENCE: Mapping[str, SemanticEvidenceArtifact] = MappingProxyType(
    {item.evidence_id: item for item in _EVIDENCE_ITEMS}
)


def semantic_assurance_rows() -> tuple[dict[str, object], ...]:
    """Return derived R/S/B/N claims; B is always tied to supplied alignment."""
    from ..converters.targets import CONVERTERS

    rows: list[dict[str, object]] = []
    for contract in SEMANTIC_CONTRACTS.values():
        profiles = tuple(
            CONVERTERS[converter_id]
            for converter_id in contract.converter_ids
            if converter_id in CONVERTERS
        )
        evidence = tuple(
            SEMANTIC_EVIDENCE[evidence_id]
            for evidence_id in contract.execution_evidence_ids
            if evidence_id in SEMANTIC_EVIDENCE
        )
        if contract.converter_ids and len(profiles) != len(contract.converter_ids):
            raise ValueError(f"{contract.contract_id} references an unknown converter")
        if len(evidence) != len(contract.execution_evidence_ids):
            raise ValueError(f"{contract.contract_id} references unknown evidence")
        representation = _representation_claim(profiles)
        source = (
            "pinned source and scoped claim recorded"
            if contract.source_url and contract.source_claim
            else "application contract"
            if contract.application_source
            else "not established"
        )
        if contract.binding_kind == "fixed_axes":
            binding = (
                "executable binder; explicit alignment required; structure checked, "
                "physical truth supplied by caller/runtime"
            )
        elif contract.binding_kind == "geographic_enu":
            binding = (
                "executable contextual resolver; typed origin and ellipsoid required; "
                "body alignment remains caller supplied"
            )
        else:
            binding = "declared contract only; no executable semantic binder"
        native = (
            "; ".join(
                f"{item.proof_scope}: {item.evidence_id}"
                + (
                    " (adapter conformance replayed)"
                    if item.adapter_conformance_checked
                    else " (native runtime only; adapter conformance not demonstrated)"
                )
                for item in evidence
            )
            or "not demonstrated"
        )
        rows.append(
            {
                "contract": contract,
                "representation": representation,
                "source": source,
                "binding": binding,
                "native": native,
                "profiles": profiles,
                "evidence": evidence,
            }
        )
    return tuple(rows)


def _representation_claim(profiles: tuple[ConverterProfile, ...]) -> str:
    if not profiles:
        return "not applicable"
    claims = {profile.representation_evidence for profile in profiles}
    states = {profile.evidence_status for profile in profiles}
    if claims == {"native_verified"} and states == {"accepted_converter"}:
        return "R verified (native representation)"
    if claims == {"spec_and_test_runtime"}:
        return "R specification/test-runtime only"
    if claims == {"native_supporting"}:
        return "R supporting oracle only"
    return "R partial/mixed; see profile state"


def validate_semantic_assurance(repository_root: str) -> None:
    """Fail if role claims reference absent or mismatched declarations/evidence."""
    from pathlib import Path

    from ..converters.targets import CONVERTERS

    root = Path(repository_root).resolve(strict=True)
    for contract in SEMANTIC_CONTRACTS.values():
        if contract.authority in ("target_defined", "standard_defined") and not all(
            (
                contract.source_url,
                contract.source_version,
                contract.source_section,
                contract.source_claim,
                contract.assurance_scope,
            )
        ):
            raise ValueError(f"{contract.contract_id} has an incomplete source record")
        if len(set(contract.converter_ids)) != len(contract.converter_ids):
            raise ValueError(f"{contract.contract_id} repeats a converter ID")
        if len(set(contract.execution_evidence_ids)) != len(
            contract.execution_evidence_ids
        ):
            raise ValueError(f"{contract.contract_id} repeats an evidence ID")
        unknown_converters = set(contract.converter_ids).difference(CONVERTERS)
        if unknown_converters:
            raise ValueError(
                f"{contract.contract_id} references unknown converters: "
                f"{sorted(unknown_converters)}"
            )
        unknown_evidence = set(contract.execution_evidence_ids).difference(
            SEMANTIC_EVIDENCE
        )
        if unknown_evidence:
            raise ValueError(
                f"{contract.contract_id} references unknown evidence: "
                f"{sorted(unknown_evidence)}"
            )
        profiles = [CONVERTERS[item] for item in contract.converter_ids]
        for profile in profiles:
            if contract.target_family != profile.target:
                raise ValueError(
                    f"{contract.contract_id} targets {contract.target_family!r}, "
                    f"but {profile.converter_id} targets {profile.target!r}"
                )
        for evidence_id in contract.execution_evidence_ids:
            artifact = SEMANTIC_EVIDENCE[evidence_id]
            if artifact.evidence_id != evidence_id:
                raise ValueError(f"{evidence_id} key does not match its evidence ID")
            if (
                artifact.adapter_conformance_checked
                and not artifact.native_runtime_capture
            ):
                raise ValueError(
                    f"{evidence_id} claims adapter conformance without a native capture"
                )
            if not (root / artifact.artifact_path).is_file():
                raise ValueError(f"{evidence_id} artifact is missing")
            if not (root / artifact.replay_test_path).is_file():
                raise ValueError(f"{evidence_id} replay test is missing")
            if artifact.target != contract.target_family:
                raise ValueError(
                    f"{evidence_id} target does not match {contract.contract_id}"
                )
