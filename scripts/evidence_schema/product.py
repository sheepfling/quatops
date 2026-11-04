"""Assemble the compact product from validated native evidence packages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from quatops.scalar import multiply as multiply_wxyz
from scripts.converter_evidence import (
    ConverterEvidencePackage,
    converter_fixture_directories,
    load_converter_evidence,
)
from scripts.evidence_json import dump_evidence_json, read_evidence_document
from scripts.evidence_schema.contracts import crystallize_contract
from scripts.evidence_schema.crystallized import (
    CanonicalInputs,
    CaptureProvenance,
    CrystallizationHeader,
    CrystallizedCapture,
    DependencyPin,
    EvidenceLayer,
    EvidenceProduct,
    EvidenceTarget,
)
from scripts.evidence_schema.engine_product import (
    ENGINE_VERSIONS,
    crystallize_engine_package,
    load_engine_package,
)
from scripts.evidence_schema.reduce import crystallize_capture
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    crystallize_shared_inputs,
)
from scripts.evidence_schema.targets import (
    CesiumInputs,
    EigenInputs,
    MuJoCoInputs,
    OpenCVInputs,
    OpenUsdInputs,
    SpiceyPyInputs,
    ThreeGltfInputs,
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


def _find_manifest_value(value: Any, keys: tuple[str, ...]) -> str | None:
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


def _run_details(manifest: dict[str, Any]) -> tuple[str, list[DependencyPin]]:
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


def _reproduction_dependencies(manifest: dict[str, Any]) -> list[DependencyPin]:
    result: list[DependencyPin] = []
    product = manifest.get("product")
    if isinstance(product, dict) and isinstance(product.get("dependencies"), list):
        for item in product["dependencies"]:
            if (
                isinstance(item, dict)
                and item.get("name") in {"container image", "ROS_DISTRO"}
                and isinstance(item.get("version"), str)
            ):
                result.append(DependencyPin(name=item["name"], version=item["version"]))
    return result


def _primary_native_binary(manifest: dict[str, Any], product: str) -> str | None:
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
                    if (
                        isinstance(row, dict)
                        and row.get("path", "").lower().endswith(preferred_name.lower())
                        and isinstance(row.get("sha256"), str)
                    ):
                        return row["sha256"]
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

    def collect_dtype(value: Any) -> None:
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


def _layer(package: ConverterEvidencePackage) -> EvidenceLayer:
    kind = package.evidence_kind
    if "ros2_tf2" in kind:
        return "frame-transport"
    if "cesium" in kind:
        return "geographic-transport"
    if "threejs" in kind or "openusd" in kind:
        return "scene-transport"
    return "native-quaternion-api"


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
    contract_path = directory / "contract.json"
    witnesses_path = directory / "witnesses.json"
    manifest_path = directory / "manifest.json"
    return CaptureProvenance(
        capture_adapter=adapter_relative,
        capture_adapter_sha256=declared_adapter_hash,
        input_sha256=input_hash,
        contract_sha256=sha256_file(contract_path),
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
    layer = _layer(package)
    evidence_contract = crystallize_contract(package)
    build_id = _find_manifest_value(
        package.manifest,
        ("native_executable_sha256", "package_entry_sha256", "binary_sha256"),
    )
    return CrystallizationHeader(
        evidence_id=f"{package.target.name}:{package.target.version}",
        layer=layer,
        target=EvidenceTarget(
            product=package.target.name,
            version=package.target.version,
            api=_target_api(package),
            native_precision=_precision_profiles(package),
            build_id=build_id,
        ),
        contract_id=package.contract_id,
        contract=evidence_contract,
        witness_set_id=f"{slug_identifier(package.target.name)}-converter-v1",
        provenance=_provenance(
            package,
            repository_root=repository_root,
            directory=directory,
        ),
    )


def _validate_canonical_quaternion(
    target: str,
    case_id: str,
    coefficients: list[float],
    canonical_by_case: dict[str, list[float]],
) -> None:
    expected = canonical_by_case.get(case_id)
    if expected is None or coefficients != expected:
        raise ValueError(
            f"{target} input {case_id!r} differs from shared canonical inputs"
        )


def _validate_converter_input_links(
    package: ConverterEvidencePackage,
    *,
    canonical_inputs: CanonicalInputs,
    canonical_path: Path,
    repository_root: Path,
) -> None:
    inputs = package.inputs
    expected_hash = sha256_file(canonical_path)
    canonical_by_case = {
        row.case_id: row.coefficients_wxyz for row in canonical_inputs.orientations
    }
    canonical_case_ids = list(canonical_by_case)
    target = package.target.name

    # ROS 2 keeps its explicit frame-chain inputs in the native capture itself.
    if inputs is None:
        if target == "ROS 2 tf2":
            return
        raise ValueError(f"{target} package lacks its typed input document")

    if isinstance(inputs, (EigenInputs, MuJoCoInputs, OpenCVInputs, SpiceyPyInputs)):
        referenced_path = (repository_root / inputs.shared_canonical_inputs).resolve()
        if referenced_path != canonical_path.resolve():
            raise ValueError(f"{target} references a different canonical input file")
        if inputs.retained_case_ids != canonical_case_ids:
            raise ValueError(f"{target} retained case IDs differ from shared inputs")
        return

    if isinstance(inputs, CesiumInputs):
        if inputs.canonical_input_sha256 != expected_hash:
            raise ValueError("Cesium canonical input digest differs")
        for row in inputs.cases:
            case_id = row.canonical_orientation_case
            if case_id in canonical_by_case:
                _validate_canonical_quaternion(
                    target, case_id, row.canonical_wxyz, canonical_by_case
                )
            elif case_id == (
                f"{canonical_inputs.composition_left_case_id} * "
                f"{canonical_inputs.composition_right_case_id}"
            ):
                expected = multiply_wxyz(
                    canonical_by_case[canonical_inputs.composition_left_case_id],
                    canonical_by_case[canonical_inputs.composition_right_case_id],
                )
                if any(
                    abs(actual - wanted) > 1e-15
                    for actual, wanted in zip(row.canonical_wxyz, expected)
                ):
                    raise ValueError(
                        "Cesium composed input differs from shared product"
                    )
            else:
                raise ValueError(f"Cesium input references unknown case {case_id!r}")
        return

    if isinstance(inputs, ThreeGltfInputs):
        if inputs.canonical_input_sha256 != expected_hash:
            raise ValueError("glTF canonical input digest differs")
        known_values = {tuple(values) for values in canonical_by_case.values()}
        if any(tuple(row.canonical_wxyz) not in known_values for row in inputs.nodes):
            raise ValueError(
                "glTF node input differs from shared canonical orientations"
            )
        return

    if isinstance(inputs, OpenUsdInputs):
        if inputs.canonical_input_sha256 != expected_hash:
            raise ValueError("OpenUSD canonical input digest differs")
        for row in inputs.nodes:
            _validate_canonical_quaternion(
                target, row.canonical_case_id, row.canonical_wxyz, canonical_by_case
            )
        return

    raise TypeError(f"no canonical input link validation for {type(inputs).__name__}")


def build_evidence_product(repository_root: Path) -> EvidenceProduct:
    """Load, validate, and reduce every versioned converter proof package."""
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
        _validate_converter_input_links(
            package,
            canonical_inputs=canonical_inputs,
            canonical_path=canonical_path,
            repository_root=repository_root,
        )
        captures.append(
            crystallize_capture(
                package.target_capture,
                _header(
                    package,
                    repository_root=repository_root,
                    directory=directory,
                ),
                target_inputs=package.inputs,
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
        schema_version=1,
        canonical_inputs=canonical_inputs,
        witness_sets=list(witness_sets.values()),
        records=[capture.record for capture in captures],
    )


def render_evidence_product(product: EvidenceProduct) -> str:
    """Render the canonical product using the repository's evidence JSON style."""
    return dump_evidence_json(
        product.model_dump(mode="json", exclude_none=True), max_inline_length=200
    )
