"""Strict serialization models for native Unreal, Unity, Godot, and Blender captures."""

# Pydantic annotations must remain importable on the supported Python 3.9 floor.
# ruff: noqa: UP007, UP045

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from scripts.evidence_schema.primitives import (
    NATIVE_RETAINED_CASE_IDS,
    SHA256,
    NativeCaptureModel,
    Quaternion4,
    Vector3,
)

ENGINE_RETAINED_CASE_IDS = NATIVE_RETAINED_CASE_IDS


class EngineIdentity(NativeCaptureModel):
    engine: str
    api: str
    version: Optional[str] = None
    build: Optional[Union[int, str]] = None
    platform: str
    native_scalar_profiles: list[str]
    native_quaternion_order: Literal["wxyz", "xyzw"]
    native_algebra: Literal["Hamilton", "JPL"]
    native_vector_action: str


class EngineFrame(NativeCaptureModel):
    declaration: Optional[str] = None
    declared_probe_semantics: Optional[str] = None
    handedness: Optional[str] = None
    positive_x: Optional[str] = None
    positive_y: Optional[str] = None
    positive_z: Optional[str] = None
    forward_direction: Optional[str] = None
    positive_x_image: Optional[Vector3] = None
    positive_y_image: Optional[Vector3] = None
    positive_z_image: Optional[Vector3] = None
    semantic_imported_object_forward_axis: Optional[str] = None


class EngineCanonicalFrame(NativeCaptureModel):
    declaration: Optional[str] = None
    handedness: str
    positive_x: str
    positive_y: str
    positive_z: str


class EngineConversionContract(NativeCaptureModel):
    contract_id: str
    canonical_algebra: Literal["Hamilton", "JPL"]
    canonical_order: Literal["wxyz", "xyzw"]
    canonical_action: Literal["active", "passive"]
    canonical_frame: EngineCanonicalFrame
    native_frame: EngineFrame
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    conversion_rule: str
    inverse_rule: str
    world_and_body_map: Optional[str] = None
    input_domain: str
    native_precision_tolerance: dict[str, float]
    scope: str
    excludes: list[str]


class EngineNativeBuildRecord(NativeCaptureModel):
    build_branch: Optional[str] = None
    build: Optional[str] = None
    build_hash: Optional[str] = None
    hash: Optional[str] = None
    major: Optional[int] = None
    minor: Optional[int] = None
    patch: Optional[int] = None
    changelist: Optional[int] = None
    branch: Optional[str] = None
    name: Optional[str] = None
    platform: Optional[str] = None
    platform_architecture: Optional[str] = None
    build_target: Optional[str] = None
    precision_types: Optional[str] = None
    precision: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None
    python_version: Optional[str] = None
    capture_method: Optional[str] = None


class EngineDirectCapture(NativeCaptureModel):
    capture_source: str
    capture_source_sha256: SHA256
    capture_output_sha256: SHA256
    selector_sha256: SHA256
    executable_sha256: SHA256
    utc: str
    comparison: str


class EngineProvenance(NativeCaptureModel):
    semantic_contract_file: str
    semantic_contract_sha256: SHA256
    original_capture_sha256: Optional[SHA256] = None
    capture_source: Optional[str] = None
    capture_source_sha256: Optional[SHA256] = None
    canonical_input_file: Optional[str] = None
    canonical_input_sha256: Optional[SHA256] = None
    native_build_record: Optional[EngineNativeBuildRecord] = None
    capture_method: Optional[str] = None
    retention_method: Optional[str] = None
    witness_file: Optional[str] = None
    witness_sha256: Optional[SHA256] = None
    direct_capture: Optional[EngineDirectCapture] = None


class EngineContractDocument(NativeCaptureModel):
    engine: EngineIdentity
    conversion: EngineConversionContract
    provenance: EngineProvenance


class EngineSemanticIdentity(NativeCaptureModel):
    """Stable target identity and representation facts owned by an experiment."""

    engine: str
    api: str
    version: Union[str, list[str]]
    native_scalar_profiles: list[str]
    native_quaternion_order: Literal["wxyz", "xyzw"]
    native_algebra: Literal["Hamilton", "JPL"]
    native_vector_action: str

    def supports_version(self, version: str) -> bool:
        """Allow only versions explicitly named by the experiment contract."""
        return (
            version in self.version
            if isinstance(self.version, list)
            else version == self.version
        )


def engine_semantic_contract_path(root: Path, engine: str, version: str) -> Path:
    """Select a version-specific contract or an explicitly shared engine contract."""
    directory = root / "experiments" / engine
    versioned = directory / version / "semantic_contract.json"
    return versioned if versioned.is_file() else directory / "semantic_contract.json"


class EngineSemanticConfig(NativeCaptureModel):
    """Experiment-owned semantic contract, separate from run provenance."""

    schema_version: Literal[1]
    engine: EngineSemanticIdentity
    conversion: EngineConversionContract


class EngineOrientation(NativeCaptureModel):
    case_id: str
    native_quaternion: Quaternion4
    native_rotated_positive_basis_vectors_xyz: Annotated[
        list[Vector3], Field(min_length=3, max_length=3)
    ]
    native_rotated_probe_vector_xyz: Optional[Vector3] = None


class EngineComposition(NativeCaptureModel):
    left_case_id: str
    right_case_id: str
    product_order: str
    native_product_quaternion: Quaternion4
    native_rotated_positive_basis_vectors_xyz: Annotated[
        list[Vector3], Field(min_length=3, max_length=3)
    ]


class EnginePrecisionCapture(NativeCaptureModel):
    orientation_cases: list[EngineOrientation]
    composition: EngineComposition
    reverse_composition: EngineComposition


class EngineCanonicalInterop(NativeCaptureModel):
    contract_id: str
    native_quaternion_order: Literal["wxyz", "xyzw"]
    frame_map_destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    orientation_cases: Optional[list[EngineOrientation]] = None
    composition: Optional[EngineComposition] = None
    reverse_composition: Optional[EngineComposition] = None
    precision_captures: Optional[dict[str, EnginePrecisionCapture]] = None


class EngineCaptureDocument(NativeCaptureModel):
    schema_version: Literal[1]
    target: EngineIdentity
    contract: EngineConversionContract
    provenance: Optional[EngineProvenance] = None
    canonical_interop: EngineCanonicalInterop
