"""Validate native captures through the shared frame and witness contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scripts.converter_evidence import (
    ConverterEvidencePackage,
    converter_fixture_directories,
    load_converter_evidence,
)
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import ContractTargetIdentity
from scripts.evidence_schema.crystallized import SemanticSerialization
from scripts.evidence_schema.engine_evidence import ENGINE_VERSIONS, load_engine_package
from scripts.evidence_schema.normalizer import normalize_capture
from scripts.evidence_schema.semantic_serializers import serialize_target_semantics
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)
from scripts.evidence_schema.semantic_serializers.engines import (
    serialize_engine_semantics,
)
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    crystallize_shared_inputs,
)
from scripts.evidence_schema.utilities import sha256_file


@dataclass(frozen=True)
class ValidatedCapture:
    """A normalized view of one versioned native capture."""

    target: str
    version: str
    witness_path: Path
    serialization: SemanticSerialization


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _converter_precision(package: ConverterEvidencePackage) -> list[str]:
    target = package.contract.target
    declared: list[str] = []
    if isinstance(target, ContractTargetIdentity):
        if target.scalar_profiles:
            declared = list(target.scalar_profiles)
        elif target.native_scalar_dtype:
            declared = [target.native_scalar_dtype]
    for key in ("scalar_profiles", "native_scalar_profiles", "native_scalar_dtype"):
        recorded = package.target.details.get(key)
        if isinstance(recorded, str):
            recorded = [recorded]
        if isinstance(recorded, list):
            if any(not isinstance(item, str) for item in recorded):
                raise TypeError(
                    f"{package.target.name}: manifest precision must contain strings"
                )
            if declared and set(recorded) != set(declared):
                raise ValueError(
                    f"{package.target.name}: manifest precision differs from contract"
                )
    return declared


def verify_capture_provenance(
    package: ConverterEvidencePackage, directory: Path
) -> None:
    """Check the retained witness digest and the native capture identity."""
    manifest = package.manifest
    provenance = manifest.get("provenance") or manifest.get("capture_provenance")
    if not isinstance(provenance, dict):
        raise TypeError(f"{directory}: capture provenance is missing")
    producer_keys = (
        "capture_runner_sha256",
        "capture_script_sha256",
        "native_capture_script_sha256",
        "native_loader_script_sha256",
    )
    if not any(_is_sha256(provenance.get(key)) for key in producer_keys):
        raise ValueError(f"{directory}: native capture producer digest is missing")
    if package.inputs_source_path is None and not any(
        _is_sha256(provenance.get(key))
        for key in (
            "selected_inputs_sha256",
            "converter_input_sha256",
            "canonical_inputs_sha256",
            "converter_input_spec_sha256",
            "input_sha256",
            "corpus_sha256",
        )
    ):
        raise ValueError(f"{directory}: native capture input digest is missing")
    witness_keys = (
        "witnesses_sha256",
        "native_observations_sha256",
        "selected_witnesses_sha256",
    )
    digests = [provenance.get(key) for key in witness_keys if key in provenance]
    if not digests or any(
        digest != sha256_file(directory / "witnesses.json") for digest in digests
    ):
        raise ValueError(f"{directory}: native witness digest differs from provenance")
    filename = (
        "capture_threejs.mjs"
        if package.target.name == "Three.js GLTFLoader"
        else "capture_native.mjs"
        if package.target.name == "CesiumJS"
        else "capture_native.py"
    )
    if not (package.contract_source_path.parent / filename).is_file():
        raise FileNotFoundError(f"{directory}: native capture adapter is missing")


def validate_evidence_packages(repository_root: Path) -> tuple[ValidatedCapture, ...]:
    """Validate each target's Pydantic capture and its normalized frame semantics."""
    root = repository_root.resolve(strict=True)
    canonical_path = root / "experiments/interoperability/v1/canonical_inputs.json"
    canonical_hash = sha256_file(canonical_path)
    canonical = crystallize_shared_inputs(
        SharedCanonicalInputCapture.model_validate(
            read_evidence_document(canonical_path), strict=True
        )
    )
    context = SemanticSerializationContext(
        canonical_inputs=canonical,
        canonical_input_path=canonical_path,
        canonical_input_sha256=canonical_hash,
        repository_root=root,
    )
    captures: list[ValidatedCapture] = []
    for engine, version in ENGINE_VERSIONS:
        package = load_engine_package(root, engine, version)
        identity = package.contract.engine
        if (
            not identity.engine.strip()
            or not identity.api.strip()
            or identity.version != version
        ):
            raise ValueError(f"{engine} {version}: native engine identity differs")
        provenance = package.contract.provenance
        if provenance.canonical_input_sha256 != canonical_hash:
            raise ValueError(f"{engine} {version} canonical input digest differs")
        if not provenance.capture_source or not provenance.capture_source_sha256:
            raise ValueError(f"{engine} {version} capture provenance is missing")
        serialization = serialize_engine_semantics(package, canonical)
        normalize_capture(
            serialization,
            canonical,
            native_precision=package.contract.engine.native_scalar_profiles,
        )
        captures.append(
            ValidatedCapture(
                package.capture.target.engine,
                version,
                package.witness_path,
                serialization,
            )
        )

    directories = converter_fixture_directories(root)
    if not directories:
        raise ValueError("no converter evidence packages were found")
    for directory in directories:
        package = load_converter_evidence(directory, repository_root=root)
        verify_capture_provenance(package, directory)
        serialization = serialize_target_semantics(package, context)
        normalize_capture(
            serialization, canonical, native_precision=_converter_precision(package)
        )
        captures.append(
            ValidatedCapture(
                package.target.name,
                package.target.version,
                directory / "witnesses.json",
                serialization,
            )
        )

    shared: dict[Path, SemanticSerialization] = {}
    for capture in captures:
        previous = shared.get(capture.witness_path)
        if (
            previous is not None
            and previous.witnesses != capture.serialization.witnesses
        ):
            raise ValueError(
                f"shared witness file {capture.witness_path} normalizes inconsistently"
            )
        shared[capture.witness_path] = capture.serialization
    return tuple(captures)
