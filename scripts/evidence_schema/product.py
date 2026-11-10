"""Assemble the compact product from validated native evidence packages."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import JsonValue

from scripts.converter_evidence import (
    ConverterEvidencePackage,
    converter_fixture_directories,
    load_converter_evidence,
)
from scripts.evidence_json import dump_evidence_json, read_evidence_document
from scripts.evidence_schema.crystallized import (
    CaptureProvenance,
    CrystallizationHeader,
    CrystallizedCapture,
    DependencyPin,
    EvidenceProduct,
    EvidenceTarget,
)
from scripts.evidence_schema.engine_evidence import (
    ENGINE_VERSIONS,
    crystallize_engine_package,
    load_engine_package,
)
from scripts.evidence_schema.normalizer import normalize_capture
from scripts.evidence_schema.semantic_serializers import serialize_target_semantics
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    crystallize_shared_inputs,
)
from scripts.evidence_schema.utilities import sha256_file, slug_identifier

_ADAPTER_PATHS = {
    "Eigen": "experiments/converters/eigen/5.0.1/capture_native.py",
    "MuJoCo": "experiments/converters/mujoco/3.17.0/capture_native.py",
    "OpenCV": "experiments/converters/opencv/4.13.0/capture_native.py",
    "ROS 2 tf2": "experiments/converters/ros2/0.36.23/capture_native.py",
    "SpiceyPy / CSPICE": "experiments/converters/spiceypy/8.3.0/capture_native.py",
    "CesiumJS": "experiments/converters/cesium/1.124.0/capture_native.mjs",
    "Three.js GLTFLoader": "experiments/converters/gltf/2.0.0/capture_threejs.mjs",
    "OpenUSD": "experiments/converters/openusd/26.11/capture_native.py",
}

_RUN_ADAPTER_HASH_KEYS = (
    "capture_runner_sha256",
    "capture_script_sha256",
    "native_capture_script_sha256",
    "native_loader_script_sha256",
    "runner_sha256",
)
_INPUT_HASH_KEYS = (
    "selected_inputs_sha256",
    "converter_input_sha256",
    "canonical_inputs_sha256",
    "converter_input_spec_sha256",
    "input_sha256",
    "corpus_sha256",
)


def _find_manifest_value(value: JsonValue, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
                return candidate
        for child in value.values():
            found = _find_manifest_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_manifest_value(child, keys)
            if found is not None:
                return found
    return None


def _run_details(manifest: dict[str, JsonValue]) -> tuple[str, list[DependencyPin]]:
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        return "runtime details not declared in manifest", []
    runtime_parts: list[str] = []
    dependencies: list[DependencyPin] = []
    for runtime in ("python", "node", "v8"):
        value = execution.get(runtime)
        if isinstance(value, str) and value:
            version = re.search(r"v?\d+(?:\.\d+)+", value)
            short_version = (
                version.group(0).lstrip("v") if version else value.split()[0]
            )
            runtime_parts.append(f"{runtime} {short_version}")
            if runtime != "v8":
                dependencies.append(DependencyPin(name=runtime, version=short_version))
    runtime_value = execution.get("runtime")
    if isinstance(runtime_value, str) and runtime_value:
        runtime_version = re.search(r"\d+(?:\.\d+)+", runtime_value)
        if runtime_version:
            runtime_parts.append(f"native runtime {runtime_version.group(0)}")
            dependencies.append(
                DependencyPin(name="native runtime", version=runtime_version.group(0))
            )
    for system in ("platform", "architecture"):
        value = execution.get(system)
        if isinstance(value, str) and value:
            runtime_parts.append(value)
    return "; ".join(runtime_parts) or "runtime details not declared", dependencies


def _reproduction_dependencies(
    manifest: dict[str, JsonValue],
) -> list[DependencyPin]:
    product = manifest.get("product")
    if not isinstance(product, dict):
        return []
    dependencies = product.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    pins: list[DependencyPin] = []
    for item in dependencies:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        version = item.get("version")
        if (
            isinstance(name, str)
            and name in {"container image", "ROS_DISTRO"}
            and isinstance(version, str)
        ):
            pins.append(DependencyPin(name=name, version=version))
    return pins


def _primary_native_binary(manifest: dict[str, JsonValue], product: str) -> str | None:
    if product in {"Eigen", "ROS 2 tf2"}:
        keys = (
            ("native_executable_sha256", "binary_sha256")
            if product == "Eigen"
            else ("binary_sha256",)
        )
        return _find_manifest_value(manifest, keys)
    if product == "SpiceyPy / CSPICE":
        return _find_manifest_value(manifest, ("loaded_cspice_binary_sha256",))
    if product == "CesiumJS":
        return _find_manifest_value(manifest, ("package_entry_sha256",))
    preferred_name = {
        "MuJoCo": "mujoco.dll",
        "OpenCV": "cv2.pyd",
        "OpenUSD": "usd_ms.dll",
    }.get(product)
    if preferred_name:
        for key in ("native_binary_files", "native_binaries"):
            rows = manifest.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    path = row.get("path")
                    digest = row.get("sha256")
                    if (
                        isinstance(path, str)
                        and path.lower().endswith(preferred_name.lower())
                        and isinstance(digest, str)
                    ):
                        return digest
        binaries = manifest.get("native_binaries_sha256")
        if isinstance(binaries, dict):
            for path, digest in binaries.items():
                if str(path).lower().endswith(preferred_name.lower()) and isinstance(
                    digest, str
                ):
                    return digest
    return None


def _precision_profiles(package: ConverterEvidencePackage) -> list[str]:
    profiles: set[str] = set()
    details = package.target.details
    for key in ("scalar_profiles", "native_scalar_profiles", "precision_profiles"):
        values = details.get(key)
        if isinstance(values, list):
            profiles.update(value for value in values if isinstance(value, str))
    for key in ("native_scalar_dtype", "dtype"):
        value = details.get(key)
        if isinstance(value, str):
            profiles.add(value)
    target_contract = getattr(package.contract, "target", None)
    for key in (
        "scalar_profiles",
        "native_scalar_profiles",
        "native_scalar_dtype",
    ):
        value = getattr(target_contract, key, None)
        if isinstance(value, list):
            profiles.update(item for item in value if isinstance(item, str))
        elif isinstance(value, str):
            profiles.add(value)
    precision = package.manifest.get("precision")
    if isinstance(precision, dict):
        for value in precision.values():
            if isinstance(value, str) and value in {"float32", "float64", "binary64"}:
                profiles.add(value)
    capture_dump = package.target_capture.model_dump(mode="python")

    def collect_dtype(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"dtype", "native_scalar_dtype"} and isinstance(child, str):
                    profiles.add(child)
                else:
                    collect_dtype(child)
        elif isinstance(value, list):
            for child in value:
                collect_dtype(child)

    collect_dtype(capture_dump)
    return sorted(profiles)


def _target_api(package: ConverterEvidencePackage) -> str:
    contract = package.contract
    target = getattr(contract, "target", None)
    if target is not None:
        for name in ("native_api", "entry_points"):
            value = getattr(contract, name, None) or getattr(target, name, None)
            if isinstance(value, list):
                return ", ".join(value)
        for name in ("header", "interface"):
            value = getattr(target, name, None)
            if isinstance(value, str) and value:
                return value
        if isinstance(target, str):
            return target
    return package.target.name


def _provenance(
    package: ConverterEvidencePackage,
    *,
    repository_root: Path,
    directory: Path,
) -> CaptureProvenance:
    manifest = package.manifest
    adapter_relative = _ADAPTER_PATHS[package.target.name]
    adapter_path = repository_root / Path(adapter_relative)
    declared_adapter_hash = _find_manifest_value(manifest, _RUN_ADAPTER_HASH_KEYS)
    if declared_adapter_hash is None:
        raise ValueError(f"{directory}: manifest has no native capture adapter hash")

    if not adapter_path.is_file():
        raise FileNotFoundError(f"capture adapter not found: {adapter_path}")
    checked_in_adapter_hash = sha256_file(adapter_path)
    if package.inputs_source_path is not None:
        input_hash = sha256_file(package.inputs_source_path)
    else:
        input_hash = _find_manifest_value(manifest, _INPUT_HASH_KEYS)
        if input_hash is None:
            raise ValueError(f"{directory}: no input digest in package or manifest")
    runtime, dependencies = _run_details(manifest)
    dependencies.extend(_reproduction_dependencies(manifest))
    contract_path = package.contract_source_path
    witnesses_path = directory / "witnesses.json"
    manifest_path = directory / "manifest.json"
    return CaptureProvenance(
        capture_adapter=adapter_relative,
        capture_adapter_sha256=declared_adapter_hash,
        input_sha256=input_hash,
        contract_sha256=sha256_file(contract_path),
        semantic_contract_file=contract_path.relative_to(repository_root).as_posix(),
        witnesses_sha256=sha256_file(witnesses_path),
        manifest_sha256=sha256_file(manifest_path),
        native_runtime=runtime,
        native_binary_sha256=_primary_native_binary(manifest, package.target.name),
        checked_in_adapter_sha256=(
            checked_in_adapter_hash
            if checked_in_adapter_hash != declared_adapter_hash
            else None
        ),
        transport_asset_sha256=_find_manifest_value(
            manifest, ("asset_sha256", "stage_sha256")
        ),
        dependency_lock_sha256=_find_manifest_value(
            manifest,
            ("native_package_lock_sha256", "pinned_lockfile_sha256", "lockfile_sha256"),
        ),
        dependencies=dependencies,
    )


def _header(
    package: ConverterEvidencePackage,
    *,
    repository_root: Path,
    directory: Path,
) -> CrystallizationHeader:
    build_id = _find_manifest_value(
        package.manifest,
        ("native_executable_sha256", "package_entry_sha256", "binary_sha256"),
    )
    return CrystallizationHeader(
        evidence_id=f"{package.target.name}:{package.target.version}",
        target=EvidenceTarget(
            product=package.target.name,
            version=package.target.version,
            api=_target_api(package),
            native_precision=_precision_profiles(package),
            build_id=build_id,
        ),
        contract_id=package.contract_id,
        witness_set_id=f"{slug_identifier(package.target.name)}-converter-v1",
        provenance=_provenance(
            package,
            repository_root=repository_root,
            directory=directory,
        ),
    )


def build_evidence_product(repository_root: Path) -> EvidenceProduct:
    """Load, serialize, normalize, and collect every converter proof package."""
    repository_root = repository_root.resolve()
    canonical_path = (
        repository_root / "experiments/interoperability/v1/canonical_inputs.json"
    )
    canonical_document = read_evidence_document(canonical_path)
    canonical_hash = sha256_file(canonical_path)
    canonical_capture = SharedCanonicalInputCapture.model_validate(
        canonical_document, strict=True
    )
    canonical_inputs = crystallize_shared_inputs(canonical_capture)
    serialization_context = SemanticSerializationContext(
        canonical_inputs=canonical_inputs,
        canonical_input_path=canonical_path.resolve(),
        canonical_input_sha256=canonical_hash,
        repository_root=repository_root,
    )
    captures: list[CrystallizedCapture] = []
    for engine, version in ENGINE_VERSIONS:
        package = load_engine_package(repository_root, engine, version)
        if package.contract.provenance.canonical_input_sha256 != canonical_hash:
            raise ValueError(f"{engine} {version} canonical input digest differs")
        captures.append(
            crystallize_engine_package(
                package,
                repository_root=repository_root,
                canonical_inputs=canonical_inputs,
            )
        )
    for directory in converter_fixture_directories(repository_root):
        package = load_converter_evidence(directory, repository_root=repository_root)
        serialization = serialize_target_semantics(package, serialization_context)
        captures.append(
            normalize_capture(
                _header(
                    package,
                    repository_root=repository_root,
                    directory=directory,
                ),
                serialization,
                canonical_inputs,
            )
        )
    witness_sets = {}
    for capture in captures:
        witness_set = capture.witness_set
        existing = witness_sets.get(witness_set.witness_set_id)
        if (
            existing is not None
            and existing.model_dump_json() != witness_set.model_dump_json()
        ):
            raise ValueError(
                f"witness set {witness_set.witness_set_id!r} changed across records"
            )
        witness_sets.setdefault(witness_set.witness_set_id, witness_set)
    return EvidenceProduct(
        schema_version=2,
        canonical_inputs=canonical_inputs,
        witness_sets=list(witness_sets.values()),
        records=[capture.record for capture in captures],
    )


def render_evidence_product(product: EvidenceProduct) -> str:
    """Render the canonical product using the repository's evidence JSON style."""
    return dump_evidence_json(
        product.model_dump(mode="json", exclude_none=True), max_inline_length=200
    )
