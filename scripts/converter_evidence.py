"""Pydantic normalization for versioned native converter evidence packages."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import (
    CesiumCapture,
    CesiumContract,
    CesiumInputs,
    EigenCapture,
    EigenContract,
    EigenInputs,
    MuJoCoCapture,
    MuJoCoContract,
    MuJoCoInputs,
    OpenCVCapture,
    OpenCVContract,
    OpenCVInputs,
    OpenUsdCapture,
    OpenUsdContract,
    OpenUsdInputs,
    Ros2Capture,
    Ros2Contract,
    SpiceyPyCapture,
    SpiceyPyContract,
    SpiceyPyInputs,
    TargetCapture,
    TargetContract,
    TargetInputs,
    ThreeGltfCapture,
    ThreeGltfContract,
    ThreeGltfInputs,
)
from scripts.evidence_schema.primitives import SemanticContractReference
from scripts.evidence_schema.utilities import sha256_file

_VERSION_FIELDS = (
    "version",
    "package_version",
    "wrapper_version",
    "distribution_version",
    "tf2_version",
    "engine_version",
    "api_version",
)


class ConverterEvidenceTarget(BaseModel):
    """Common target identity with all native version details retained."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    version: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ConverterEvidencePackage(BaseModel):
    """Package envelope with a validated, target-specific native capture."""

    model_config = ConfigDict(extra="forbid", strict=True)

    fixture_directory: str
    # Pydantic evaluates annotations at runtime on the Python 3.9 CI floor.
    schema_version: Union[str, int]  # noqa: UP007
    evidence_kind: str
    contract_id: str
    target: ConverterEvidenceTarget
    contract: TargetContract
    inputs: Optional[TargetInputs] = None  # noqa: UP045
    target_capture: TargetCapture
    manifest: dict[str, JsonValue]
    observation_group_count: int
    contract_source_path: Path = Field(exclude=True)
    inputs_source_path: Optional[Path] = Field(default=None, exclude=True)  # noqa: UP045


_CAPTURE_MODELS = {
    "Eigen": EigenCapture,
    "MuJoCo": MuJoCoCapture,
    "OpenCV": OpenCVCapture,
    "ROS 2 tf2": Ros2Capture,
    "SpiceyPy / CSPICE": SpiceyPyCapture,
    "CesiumJS": CesiumCapture,
    "Three.js GLTFLoader": ThreeGltfCapture,
    "OpenUSD": OpenUsdCapture,
}

_CONTRACT_MODELS = {
    "Eigen": EigenContract,
    "MuJoCo": MuJoCoContract,
    "OpenCV": OpenCVContract,
    "ROS 2 tf2": Ros2Contract,
    "SpiceyPy / CSPICE": SpiceyPyContract,
    "CesiumJS": CesiumContract,
    "Three.js GLTFLoader": ThreeGltfContract,
    "OpenUSD": OpenUsdContract,
}

_INPUT_MODELS = {
    "Eigen": EigenInputs,
    "MuJoCo": MuJoCoInputs,
    "OpenCV": OpenCVInputs,
    "SpiceyPy / CSPICE": SpiceyPyInputs,
    "CesiumJS": CesiumInputs,
    "Three.js GLTFLoader": ThreeGltfInputs,
    "OpenUSD": OpenUsdInputs,
}

# Converter input documents are authored beside their experiment adapters. When
# a retained native package uses the same bytes, load that source directly
# instead of checking in a duplicate under tests/golden.
_INPUT_DOCUMENT_PATHS = {
    ("Eigen", "5.0.1"): "experiments/converters/eigen/5.0.1/converter_inputs.json",
    ("MuJoCo", "3.17.0"): "experiments/converters/mujoco/3.17.0/converter_inputs.json",
    ("OpenCV", "4.13.0"): "experiments/converters/opencv/4.13.0/converter_inputs.json",
    (
        "SpiceyPy / CSPICE",
        "8.3.0",
    ): "experiments/converters/spiceypy/8.3.0/converter_inputs.json",
    (
        "CesiumJS",
        "1.124.0",
    ): "experiments/converters/cesium/1.124.0/converter_inputs.json",
    (
        "Three.js GLTFLoader",
        "0.187.0",
    ): "experiments/converters/gltf/2.0.0/converter_inputs.json",
    ("OpenUSD", "26.11"): "experiments/converters/openusd/26.11/converter_inputs.json",
}


def _target_capture(
    name: str, document: dict[str, JsonValue], *, directory: Path
) -> TargetCapture:
    model = _CAPTURE_MODELS.get(name)
    if model is None:
        raise ValueError(f"{directory}: unsupported typed capture target {name!r}")
    return model.model_validate(document, strict=True)


def _target_contract(name: str, document: dict[str, JsonValue], *, directory: Path):
    model = _CONTRACT_MODELS.get(name)
    if model is None:
        raise ValueError(f"{directory}: unsupported typed contract target {name!r}")
    return model.model_validate(document, strict=True)


def _target_inputs(
    name: str, document: dict[str, JsonValue], *, directory: Path
) -> TargetInputs:
    model = _INPUT_MODELS.get(name)
    if model is None:
        raise ValueError(f"{directory}: unsupported typed input target {name!r}")
    return model.model_validate(document, strict=True)


def _observation_group_count(capture: TargetCapture) -> int:
    if isinstance(capture, EigenCapture):
        return 1
    if isinstance(capture, (MuJoCoCapture, OpenCVCapture, SpiceyPyCapture)):
        return 3
    return len(capture.observations)


def converter_fixture_directories(root: Path) -> list[Path]:
    """Find complete converter packages without relying on directory names."""
    golden_root = root / "tests" / "golden"
    return sorted(
        manifest_path.parent
        for manifest_path in golden_root.rglob("manifest.json")
        if (manifest_path.parent / "witnesses.json").is_file()
    )


def _target_identity(manifest: dict[str, JsonValue]) -> ConverterEvidenceTarget:
    raw_target = manifest.get("target")
    if not isinstance(raw_target, dict):
        raw_target = manifest.get("product")
    if not isinstance(raw_target, dict):
        raise TypeError("manifest must declare a target object or product object")

    name = raw_target.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("manifest target must have a non-empty name")

    version_field = next(
        (
            field
            for field in _VERSION_FIELDS
            if isinstance(raw_target.get(field), str) and raw_target[field]
        ),
        None,
    )
    if version_field is None:
        raise ValueError(f"manifest target {name!r} must declare a version")
    version = raw_target[version_field]
    if not isinstance(version, str) or not version:
        raise ValueError(f"manifest target {name!r} must declare a string version")

    details = {
        key: value
        for key, value in raw_target.items()
        if key not in {"name", version_field}
    }
    return ConverterEvidenceTarget(
        name=name,
        version=version,
        details=details,
    )


def load_converter_evidence(
    directory: Path, *, repository_root: Path
) -> ConverterEvidencePackage:
    """Load one proof package into the shared envelope without rewriting it."""
    directory = directory.resolve()
    manifest = read_evidence_document(directory / "manifest.json")
    witnesses = read_evidence_document(directory / "witnesses.json")
    target = _target_identity(manifest)
    inputs_path = directory / "inputs.json"
    if not inputs_path.is_file():
        shared_inputs = _INPUT_DOCUMENT_PATHS.get((target.name, target.version))
        if shared_inputs is not None:
            inputs_path = repository_root / shared_inputs
    inputs_document = (
        read_evidence_document(inputs_path) if inputs_path.is_file() else None
    )
    inputs_source_path = inputs_path.resolve() if inputs_document is not None else None
    contract_reference = SemanticContractReference.model_validate(
        manifest.get("semantic_contract"), strict=True
    )
    relative_contract_path = PurePosixPath(contract_reference.file)
    if (
        relative_contract_path.is_absolute()
        or ".." in relative_contract_path.parts
        or "\\" in contract_reference.file
    ):
        raise ValueError(
            f"{directory}: semantic contract path must be repository-relative"
        )
    normalized_root = repository_root.resolve()
    semantic_contract_path = (
        normalized_root / Path(*relative_contract_path.parts)
    ).resolve(strict=True)
    try:
        semantic_contract_path.relative_to(normalized_root)
    except ValueError as error:
        raise ValueError(
            f"{directory}: semantic contract path escapes the repository"
        ) from error
    if sha256_file(semantic_contract_path) != contract_reference.sha256:
        raise ValueError(f"{directory}: semantic contract digest differs from manifest")
    semantic_contract_document = read_evidence_document(semantic_contract_path)
    contract = _target_contract(
        target.name, semantic_contract_document, directory=semantic_contract_path
    )
    contract_id = getattr(contract, "contract_id", None)
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError(f"{directory}: contract must declare contract_id")
    inputs = (
        _target_inputs(target.name, inputs_document, directory=directory)
        if inputs_document is not None
        else None
    )
    if inputs is not None and getattr(inputs, "contract_id", None) not in (
        None,
        contract_id,
    ):
        raise ValueError(
            f"{directory}: input contract_id does not match semantic contract"
        )

    evidence_kind = witnesses.get("evidence_kind")
    if not isinstance(evidence_kind, str) or not evidence_kind:
        raise ValueError(f"{directory}: witnesses must declare evidence_kind")
    schema_version = witnesses.get("schema_version")
    if not isinstance(schema_version, (str, int)):
        raise TypeError(f"{directory}: witnesses must declare schema_version")

    capture = _target_capture(target.name, witnesses, directory=directory)
    witness_target = getattr(capture, "target", None)
    if witness_target is not None and witness_target.name != target.name:
        raise ValueError(f"{directory}: witness and manifest target names differ")
    contract_target = getattr(contract, "target", None)
    if contract_target is not None and getattr(contract_target, "name", None) not in (
        None,
        target.name,
    ):
        raise ValueError(f"{directory}: contract and manifest target names differ")

    return ConverterEvidencePackage(
        fixture_directory=directory.relative_to(normalized_root).as_posix(),
        schema_version=schema_version,
        evidence_kind=evidence_kind,
        contract_id=contract_id,
        target=target,
        contract=contract,
        inputs=inputs,
        target_capture=capture,
        manifest=manifest,
        observation_group_count=_observation_group_count(capture),
        contract_source_path=semantic_contract_path.resolve(),
        inputs_source_path=inputs_source_path,
    )
