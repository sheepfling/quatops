"""Retain selected engine witnesses with native capture provenance."""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue, TypeAdapter

from scripts.evidence_json import (
    dump_evidence_json,
    parse_evidence_document,
    read_evidence_document,
)
from scripts.evidence_schema.engine_captures import (
    ENGINE_RETAINED_CASE_IDS,
    EngineCanonicalFrame,
    EngineCanonicalInterop,
    EngineFrame,
    EngineSemanticConfig,
    engine_semantic_contract_path,
)
from scripts.evidence_schema.primitives import Quaternion4, validate_signed_axis_map
from scripts.evidence_schema.shared_inputs import SharedCanonicalInputCapture

JsonRecord = dict[str, JsonValue]
QUATERNION = TypeAdapter(Quaternion4)


@dataclass(frozen=True)
class EngineRetentionConfig:
    version: str | None
    source: str
    map_key: str
    frame_key: str


ENGINE_CONFIG = {
    "unreal": EngineRetentionConfig(
        version="5.7.4",
        source="experiments/unreal/5.7.4/Source/QuatOpsUnrealCapture/Private/QuatOpsUnrealCaptureTests.cpp",
        map_key="source_to_unreal_axis_map",
        frame_key="unreal_world_frame",
    ),
    "unity": EngineRetentionConfig(
        version="6000.5.2f1",
        source="experiments/unity/6000.5.2f1/Assets/Editor/QuatOpsEngineCapture.cs",
        map_key="source_to_unity_axis_map",
        frame_key="unity_world_frame",
    ),
    "godot": EngineRetentionConfig(
        version=None,
        source="experiments/godot/capture.gd",
        map_key="source_to_godot_axis_map",
        frame_key="godot_world_frame",
    ),
    "blender": EngineRetentionConfig(
        version="5.1.2",
        source="experiments/blender/5.1.2/capture.py",
        map_key="source_to_blender_axis_map",
        frame_key="blender_world_frame_probe",
    ),
}


def _record(value: object, *, context: str) -> JsonRecord:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be a JSON object with string keys")
    return cast(JsonRecord, value)


def _rows(value: object, *, context: str) -> list[JsonRecord]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
    return [
        _record(row, context=f"{context}[{index}]") for index, row in enumerate(value)
    ]


def _native_vector(value: JsonValue) -> list[JsonValue]:
    """Keep the three captured components in a compact, ordered vector."""
    if isinstance(value, dict):
        if set(value) != {"x", "y", "z"}:
            raise ValueError("native vector object must contain only x, y and z")
        value = [value["x"], value["y"], value["z"]]
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)
    ):
        raise ValueError("native vector must have three numeric components")
    return value


def compact_native_vectors(row: JsonRecord) -> JsonRecord:
    """Store native basis images as rows while retaining their exact values."""
    result = dict(row)
    basis_key = "native_rotated_positive_basis_vectors_xyz"
    if basis_key in result:
        basis = result[basis_key]
        if not isinstance(basis, list) or len(basis) != 3:
            raise ValueError("native basis action must contain three vectors")
        result[basis_key] = [_native_vector(value) for value in basis]
    probe_key = "native_rotated_probe_vector_xyz"
    if probe_key in result:
        result[probe_key] = _native_vector(result[probe_key])
    return result


def retain_canonical_interop(
    canonical_interop: JsonRecord, expected_inputs: dict[str, list[float]]
) -> JsonRecord:
    """Retain native outputs and reference canonical inputs by case ID."""
    canonical_interop = deepcopy(canonical_interop)
    if "precision_captures" in canonical_interop:
        profiles = _record(
            canonical_interop["precision_captures"], context="precision_captures"
        )
        retained_profiles = {
            name: _retain_profile(
                _record(profile, context=f"precision_captures.{name}"),
                expected_inputs,
                precision=name,
            )
            for name, profile in profiles.items()
        }
        return cast(
            JsonRecord,
            {**canonical_interop, "precision_captures": retained_profiles},
        )
    return _retain_profile(canonical_interop, expected_inputs, precision="float32")


def _retain_profile(
    profile: JsonRecord,
    expected_inputs: dict[str, list[float]],
    *,
    precision: str,
) -> JsonRecord:
    cases = _rows(profile.get("orientation_cases"), context="orientation_cases")
    seen: set[str] = set()
    retained_cases: list[JsonRecord] = []
    mixed_probe = profile.get("native_mixed_probe_xyz")
    input_tolerance = 2e-14 if precision == "float64" else 2e-6
    for row in cases:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise TypeError("orientation case must have a string case_id")
        if case_id not in expected_inputs:
            raise ValueError(f"unexpected native orientation case: {case_id}")
        if case_id in seen:
            raise ValueError(f"duplicate native orientation case: {case_id}")
        seen.add(case_id)
        native_input = QUATERNION.validate_python(
            row.get("canonical_wxyz"), strict=True
        )
        expected_input = expected_inputs[case_id]
        if any(
            abs(float(actual) - float(expected)) > input_tolerance
            for actual, expected in zip(native_input, expected_input)
        ):
            raise ValueError(f"native input differs from shared case {case_id!r}")
        retained = {key: value for key, value in row.items() if key != "canonical_wxyz"}
        if case_id == "mixed_a" and mixed_probe is not None:
            if "native_rotated_probe_vector_xyz" in retained:
                raise ValueError("native mixed probe must appear only once")
            retained["native_rotated_probe_vector_xyz"] = mixed_probe
        if ("native_rotated_probe_vector_xyz" in retained) != (case_id == "mixed_a"):
            raise ValueError("native mixed probe must appear only on mixed_a")
        retained_cases.append(compact_native_vectors(cast(JsonRecord, retained)))
    missing = set(expected_inputs) - seen
    if missing:
        raise ValueError(
            f"native capture is missing orientation cases: {sorted(missing)}"
        )
    retained_profile = {
        key: value
        for key, value in profile.items()
        if key
        not in {
            "native_mixed_probe_xyz",
            "orientation_cases",
        }
    }
    for product_key in ("composition", "reverse_composition"):
        product = retained_profile.get(product_key)
        if isinstance(product, dict):
            retained_product = dict(product)
            retained_profile[product_key] = compact_native_vectors(retained_product)
    return cast(
        JsonRecord,
        {**retained_profile, "orientation_cases": retained_cases},
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(
    engine: str,
    version: str,
    source_path: Path,
    output_dir: Path,
    repository_root: Path,
) -> None:
    config = ENGINE_CONFIG[engine]
    if config.version and version != config.version:
        raise ValueError(f"{engine} version must be {config.version}")
    semantic_config_path = engine_semantic_contract_path(
        repository_root, engine, version
    )
    semantic_config = EngineSemanticConfig.model_validate(
        read_evidence_document(semantic_config_path), strict=True
    )
    if not semantic_config.engine.supports_version(version):
        raise ValueError(f"{engine} semantic config version must be {version}")
    capture_source_name = config.source.format(version=version)
    original_bytes = source_path.read_bytes()
    captured = _record(
        parse_evidence_document(original_bytes), context="native capture"
    )
    engine_info = _record(captured.get("engine"), context="engine")
    if engine_info.get("name") != semantic_config.engine.engine:
        raise ValueError("native capture engine identity differs from semantic config")
    native_version = engine_info.get("version")
    if not isinstance(native_version, str) or not (
        native_version == version
        or native_version.startswith((version + "-", version + " "))
    ):
        raise ValueError(f"native {engine} version differs from requested version")
    canonical_interop = _record(
        captured.get("canonical_interop"), context="canonical_interop"
    )
    canonical_inputs = (
        repository_root / "experiments/interoperability/v1/canonical_inputs.json"
    )
    inputs = SharedCanonicalInputCapture.model_validate(
        read_evidence_document(canonical_inputs), strict=True
    )
    expected_inputs = {
        row.case_id: [float(value) for value in row.canonical_wxyz]
        for row in inputs.orientations
        if row.case_id in ENGINE_RETAINED_CASE_IDS
    }
    if set(expected_inputs) != set(ENGINE_RETAINED_CASE_IDS):
        raise ValueError("canonical input corpus lacks retained orientation cases")

    canonical_interop = retain_canonical_interop(canonical_interop, expected_inputs)
    checked = EngineCanonicalInterop.model_validate(canonical_interop, strict=True)
    validate_signed_axis_map(
        checked.frame_map_destination_axes_from_source, context="native frame map"
    )
    if (
        checked.contract_id != semantic_config.conversion.contract_id
        or checked.native_quaternion_order
        != semantic_config.engine.native_quaternion_order
        or checked.frame_map_destination_axes_from_source
        != semantic_config.conversion.destination_axes_from_source
    ):
        raise ValueError("native capture quaternion or axis-map contract differs")
    if checked.precision_captures is not None and set(
        checked.precision_captures
    ) != set(semantic_config.engine.native_scalar_profiles):
        raise ValueError("native capture scalar profiles differ from semantic config")
    if checked.precision_captures is None:
        if len(semantic_config.engine.native_scalar_profiles) != 1 or (
            checked.orientation_cases is None
            or checked.composition is None
            or checked.reverse_composition is None
        ):
            raise ValueError("flat native capture lacks its single complete profile")
        products = [(checked.composition, checked.reverse_composition)]
    else:
        products = [
            (profile.composition, profile.reverse_composition)
            for profile in checked.precision_captures.values()
        ]
    left, right = inputs.composition.left_case_id, inputs.composition.right_case_id
    for forward, reverse in products:
        if (forward.left_case_id, forward.right_case_id) != (left, right) or (
            reverse.left_case_id,
            reverse.right_case_id,
        ) != (right, left):
            raise ValueError("native capture product operands differ from the recipe")
        if any(
            product.product_order != "lhs * rhs; rhs acts first"
            for product in (forward, reverse)
        ):
            raise ValueError("native capture product action differs from the recipe")

    if engine == "unreal":
        native_semantics = _record(
            captured.get("native_semantics"), context="native_semantics"
        )
        source_frame = native_semantics["source_frame"]
        map_record = _record(native_semantics[config.map_key], context=config.map_key)
        native_frame = _record(
            native_semantics[config.frame_key], context=config.frame_key
        )
    else:
        source_frame = captured["source_frame"]
        map_record = _record(captured.get(config.map_key), context=config.map_key)
        native_frame = _record(
            captured.get(
                config.frame_key,
                {
                    "declared_probe_semantics": "identity probe; no universal asset axis inferred"
                },
            ),
            context=config.frame_key,
        )
    axes_from_source = map_record.get("destination_axes_from_source")
    observed_source_frame = EngineCanonicalFrame.model_validate(
        source_frame, strict=True
    )
    observed_native_frame = EngineFrame.model_validate(native_frame, strict=True)
    if observed_source_frame != semantic_config.conversion.canonical_frame:
        raise ValueError("native capture source frame differs from semantic config")
    if observed_native_frame != semantic_config.conversion.native_frame:
        raise ValueError("native capture target frame differs from semantic config")
    if axes_from_source != semantic_config.conversion.destination_axes_from_source:
        raise ValueError("native capture axis map differs from semantic config")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite capture: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "witnesses.json"
    output_path.write_text(
        dump_evidence_json(
            {"schema_version": 1, "canonical_interop": canonical_interop}
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "engine": engine_info,
        "requested_version": version,
        "capture_method": captured.get("capture_method"),
        "native_capture_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "capture_source": capture_source_name,
        "capture_source_sha256": sha256(repository_root / capture_source_name),
        "selector_sha256": sha256(Path(__file__)),
        "canonical_input_sha256": sha256(canonical_inputs),
        "semantic_contract": semantic_config_path.relative_to(
            repository_root
        ).as_posix(),
        "semantic_contract_sha256": sha256(semantic_config_path),
        "witness_sha256": sha256(output_path),
    }
    (output_dir / "manifest.json").write_text(
        dump_evidence_json(manifest), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=tuple(ENGINE_CONFIG))
    parser.add_argument("version")
    parser.add_argument(
        "capture_json", type=Path, help="temporary native capture output"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="new or empty directory for reviewed witnesses and run manifest",
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    retain(
        args.engine,
        args.version,
        args.capture_json,
        args.output_dir,
        args.repository_root.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
