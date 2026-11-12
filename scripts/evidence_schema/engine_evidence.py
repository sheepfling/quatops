"""Load native game-engine quaternion witness packages."""

from __future__ import annotations

from pathlib import Path

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.engine_captures import (
    EngineCaptureDocument,
    EngineContractDocument,
    EngineProvenance,
    EngineSemanticConfig,
    engine_semantic_contract_path,
)
from scripts.evidence_schema.utilities import sha256_file

ENGINE_VERSIONS = (
    ("unreal", "5.7.4"),
    ("unity", "6000.5.2f1"),
    ("godot", "4.7-stable"),
    ("godot", "4.8-dev1"),
    ("blender", "5.1.2"),
)


class LoadedEnginePackage:
    """Typed pair of a versioned convention contract and its native capture."""

    def __init__(
        self,
        contract_path: Path,
        witness_path: Path,
        semantic_config_path: Path,
        contract: EngineContractDocument,
        capture: EngineCaptureDocument,
    ) -> None:
        self.contract_path = contract_path
        self.witness_path = witness_path
        self.semantic_config_path = semantic_config_path
        self.contract = contract
        self.capture = capture


def load_engine_package(
    repository_root: Path, engine: str, version: str
) -> LoadedEnginePackage:
    """Validate one engine contract and the native witness file it names."""
    directory = repository_root / "tests" / "oracles" / engine / version
    contract_path = directory / "contract.json"
    raw_contract = read_evidence_document(contract_path)
    if set(raw_contract) != {"engine", "provenance"}:
        raise ValueError(f"{directory}: expected engine identity and provenance")
    provenance = EngineProvenance.model_validate(
        raw_contract["provenance"], strict=True
    )
    semantic_config_path = engine_semantic_contract_path(
        repository_root, engine, version
    )
    semantic_config = EngineSemanticConfig.model_validate(
        read_evidence_document(semantic_config_path), strict=True
    )
    declared_config_path = provenance.semantic_contract_file
    if (
        declared_config_path
        != semantic_config_path.relative_to(repository_root).as_posix()
    ):
        raise ValueError(f"{directory}: semantic config path differs from provenance")
    declared_config_hash = provenance.semantic_contract_sha256
    if declared_config_hash != sha256_file(semantic_config_path):
        raise ValueError(f"{directory}: semantic config digest differs from provenance")
    contract = EngineContractDocument.model_validate(
        {
            "engine": raw_contract["engine"],
            "conversion": semantic_config.conversion,
            "provenance": provenance,
        },
        strict=True,
    )
    expected_identity = semantic_config.engine
    actual_identity = contract.engine
    for field in (
        "engine",
        "api",
        "native_scalar_profiles",
        "native_quaternion_order",
        "native_algebra",
        "native_vector_action",
    ):
        if getattr(actual_identity, field) != getattr(expected_identity, field):
            raise ValueError(
                f"{directory}: retained engine {field} differs from "
                "experiment semantic config"
            )
    if actual_identity.version != version or not expected_identity.supports_version(
        version
    ):
        raise ValueError(
            f"{directory}: engine version differs from experiment semantic config"
        )
    witness_name = contract.provenance.witness_file or "witnesses.json"
    witness_path = (directory / witness_name).resolve()
    witness_document = read_evidence_document(witness_path)
    capture_document = {
        **witness_document,
        "target": witness_document.get("target", raw_contract["engine"]),
        "contract": witness_document.get("contract", semantic_config.conversion),
        "provenance": witness_document.get("provenance", raw_contract["provenance"]),
    }
    capture = EngineCaptureDocument.model_validate(capture_document, strict=True)

    if contract.engine.engine != capture.target.engine:
        raise ValueError(f"{directory}: contract and capture engine names differ")
    if contract.engine.api != capture.target.api:
        raise ValueError(f"{directory}: contract and capture API identities differ")
    if contract.engine.version and capture.target.version not in (
        None,
        contract.engine.version,
    ):
        raise ValueError(f"{directory}: contract and capture versions differ")
    if contract.conversion.contract_id != capture.contract.contract_id:
        raise ValueError(f"{directory}: contract and capture contract IDs differ")
    if capture.canonical_interop.contract_id != contract.conversion.contract_id:
        raise ValueError(f"{directory}: canonical interop contract ID differs")
    if (
        capture.canonical_interop.native_quaternion_order
        != contract.engine.native_quaternion_order
    ):
        raise ValueError(f"{directory}: native quaternion order differs")
    if (
        capture.canonical_interop.frame_map_destination_axes_from_source
        != contract.conversion.destination_axes_from_source
    ):
        raise ValueError(f"{directory}: native frame map differs")
    if contract.provenance.witness_sha256:
        actual_witness_hash = sha256_file(witness_path)
        if actual_witness_hash != contract.provenance.witness_sha256:
            raise ValueError(
                f"{directory}: native witness digest differs from contract"
            )
    return LoadedEnginePackage(
        contract_path,
        witness_path,
        semantic_config_path,
        contract,
        capture,
    )
