"""Load and crystallize native game-engine quaternion witness packages."""

from __future__ import annotations

from pathlib import Path

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.crystallized import (
    CanonicalInputs,
    CaptureProvenance,
    CrystallizationHeader,
    CrystallizedCapture,
    DependencyPin,
    EvidenceTarget,
)
from scripts.evidence_schema.engine_captures import (
    EngineCaptureDocument,
    EngineContractDocument,
    EngineSemanticConfig,
)
from scripts.evidence_schema.normalizer import normalize_capture
from scripts.evidence_schema.semantic_serializers.engines import (
    serialize_engine_semantics,
)
from scripts.evidence_schema.utilities import sha256_file, slug_identifier

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
    retained_contract = EngineContractDocument.model_validate(raw_contract, strict=True)
    semantic_config_path = (
        repository_root / "experiments" / engine / version / "semantic_contract.json"
    )
    semantic_config = EngineSemanticConfig.model_validate(
        read_evidence_document(semantic_config_path), strict=True
    )
    declared_config_path = retained_contract.provenance.semantic_contract_file
    if (
        declared_config_path
        != semantic_config_path.relative_to(repository_root).as_posix()
    ):
        raise ValueError(f"{directory}: semantic config path differs from provenance")
    declared_config_hash = retained_contract.provenance.semantic_contract_sha256
    if declared_config_hash != sha256_file(semantic_config_path):
        raise ValueError(f"{directory}: semantic config digest differs from provenance")
    expected_identity = semantic_config.engine
    actual_identity = retained_contract.engine
    for field in (
        "engine",
        "api",
        "version",
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
    if retained_contract.conversion != semantic_config.conversion:
        raise ValueError(
            f"{directory}: retained conversion contract differs from "
            "experiment semantic config"
        )
    contract = EngineContractDocument(
        engine=retained_contract.engine,
        conversion=semantic_config.conversion,
        provenance=retained_contract.provenance,
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


def crystallize_engine_package(
    package: LoadedEnginePackage,
    *,
    repository_root: Path,
    canonical_inputs: CanonicalInputs,
) -> CrystallizedCapture:
    """Normalize one engine's semantic serialization with its provenance."""
    contract = package.contract
    engine = contract.engine
    conversion = contract.conversion
    serialization = serialize_engine_semantics(package, canonical_inputs)
    provenance = contract.provenance
    if provenance.capture_source is None or provenance.capture_source_sha256 is None:
        raise ValueError("engine capture contract lacks adapter provenance")
    adapter_path = repository_root / Path(provenance.capture_source)
    checked_in_hash = sha256_file(adapter_path) if adapter_path.is_file() else None
    witness_hash = sha256_file(package.witness_path)
    canonical_hash = provenance.canonical_input_sha256
    if canonical_hash is None:
        raise ValueError("engine contract lacks canonical input digest")
    runtime_version = (
        provenance.native_build_record.python_version
        if provenance.native_build_record is not None
        else None
    )
    native_runtime = engine.version or "version specified by per-version contract"
    if runtime_version:
        native_runtime += f"; embedded Python {runtime_version}"
    if provenance.capture_method:
        native_runtime += f"; {provenance.capture_method}"
    capture_provenance = CaptureProvenance(
        capture_adapter=provenance.capture_source,
        capture_adapter_sha256=provenance.capture_source_sha256,
        input_sha256=canonical_hash,
        contract_sha256=sha256_file(package.semantic_config_path),
        semantic_contract_file=package.semantic_config_path.relative_to(
            repository_root
        ).as_posix(),
        witnesses_sha256=witness_hash,
        native_runtime=native_runtime,
        checked_in_adapter_sha256=(
            checked_in_hash
            if checked_in_hash != provenance.capture_source_sha256
            else None
        ),
        dependencies=(
            [DependencyPin(name="embedded Python", version=runtime_version)]
            if runtime_version
            else []
        ),
    )
    target = EvidenceTarget(
        product=engine.engine,
        version=engine.version or contract.engine.version or "shared",
        api=engine.api,
        native_precision=engine.native_scalar_profiles,
        build_id=str(engine.build) if engine.build is not None else None,
    )
    witness_set_id = (
        "godot-shared-native-witness-v1"
        if package.contract_path.parent.name.startswith("4.")
        and package.contract_path.parent.parent.name == "godot"
        else f"{slug_identifier(package.contract_path.parent.parent.name)}-native-witness-v1"
    )
    header = CrystallizationHeader(
        evidence_id=f"{engine.engine}:{target.version}",
        target=target,
        contract_id=conversion.contract_id,
        witness_set_id=witness_set_id,
        provenance=capture_provenance,
    )
    return normalize_capture(
        header,
        serialization,
        canonical_inputs,
    )
