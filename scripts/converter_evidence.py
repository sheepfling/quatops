"""Pydantic normalization for versioned native converter evidence packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from scripts.evidence_json import read_evidence_document

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


class ConverterEvidenceObservation(BaseModel):
    """One normalized observation group with its native payload unchanged."""

    model_config = ConfigDict(extra="forbid", strict=True)

    group_id: str
    payload: JsonValue


class ConverterEvidencePackage(BaseModel):
    """Normalized package envelope; native observation payloads stay target-specific."""

    model_config = ConfigDict(extra="forbid", strict=True)

    fixture_directory: str
    # Pydantic evaluates annotations at runtime on the Python 3.9 CI floor.
    schema_version: Union[str, int]  # noqa: UP007
    evidence_kind: str
    contract_id: str
    target: ConverterEvidenceTarget
    contract: dict[str, JsonValue]
    inputs: Optional[dict[str, JsonValue]] = None  # noqa: UP045
    observation_layout: Literal["keyed-groups", "records"]
    observations: list[ConverterEvidenceObservation]
    witness_metadata: dict[str, JsonValue]
    manifest: dict[str, JsonValue]


def converter_fixture_directories(root: Path) -> list[Path]:
    """Find complete converter packages without relying on directory names."""
    golden_root = root / "tests" / "golden"
    return sorted(
        contract_path.parent
        for contract_path in golden_root.rglob("contract.json")
        if (contract_path.parent / "manifest.json").is_file()
        and (contract_path.parent / "witnesses.json").is_file()
    )


def _target_identity(manifest: dict[str, Any]) -> ConverterEvidenceTarget:
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

    details = {
        key: value
        for key, value in raw_target.items()
        if key not in {"name", version_field}
    }
    return ConverterEvidenceTarget(
        name=name,
        version=raw_target[version_field],
        details=details,
    )


def load_converter_evidence(
    directory: Path, *, repository_root: Path
) -> ConverterEvidencePackage:
    """Load one proof package into the shared envelope without rewriting it."""
    directory = directory.resolve()
    contract = read_evidence_document(directory / "contract.json")
    manifest = read_evidence_document(directory / "manifest.json")
    witnesses = read_evidence_document(directory / "witnesses.json")
    inputs_path = directory / "inputs.json"
    inputs = read_evidence_document(inputs_path) if inputs_path.is_file() else None

    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError(f"{directory}: contract must declare contract_id")
    if inputs is not None and inputs.get("contract_id") not in (None, contract_id):
        raise ValueError(f"{directory}: input contract_id does not match contract.json")

    raw_observations = witnesses.get("observations")
    if isinstance(raw_observations, dict):
        observation_layout: Literal["keyed-groups", "records"] = "keyed-groups"
        observations = [
            ConverterEvidenceObservation(group_id=group_id, payload=payload)
            for group_id, payload in raw_observations.items()
        ]
    elif isinstance(raw_observations, list):
        observation_layout = "records"
        observations = [
            ConverterEvidenceObservation(
                group_id=f"record-{index:04d}", payload=payload
            )
            for index, payload in enumerate(raw_observations)
        ]
    else:
        raise TypeError(
            f"{directory}: witnesses must contain observations as an object or array"
        )
    evidence_kind = witnesses.get("evidence_kind")
    if not isinstance(evidence_kind, str) or not evidence_kind:
        raise ValueError(f"{directory}: witnesses must declare evidence_kind")
    schema_version = witnesses.get("schema_version")
    if not isinstance(schema_version, (str, int)):
        raise TypeError(f"{directory}: witnesses must declare schema_version")

    target = _target_identity(manifest)
    witness_target = witnesses.get("target")
    if isinstance(witness_target, dict) and witness_target.get("name") != target.name:
        raise ValueError(f"{directory}: witness and manifest target names differ")
    contract_target = contract.get("target")
    if isinstance(contract_target, dict) and contract_target.get("name") != target.name:
        raise ValueError(f"{directory}: contract and manifest target names differ")

    normalized_root = repository_root.resolve()
    return ConverterEvidencePackage(
        fixture_directory=directory.relative_to(normalized_root).as_posix(),
        schema_version=schema_version,
        evidence_kind=evidence_kind,
        contract_id=contract_id,
        target=target,
        contract=contract,
        inputs=inputs,
        observation_layout=observation_layout,
        observations=observations,
        witness_metadata={
            key: value
            for key, value in witnesses.items()
            if key not in {"schema_version", "evidence_kind", "target", "observations"}
        },
        manifest=manifest,
    )
