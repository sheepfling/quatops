"""Retain selected engine witnesses with native capture provenance."""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue

if __package__:
    from scripts.evidence_json import (
        dump_evidence_json,
        parse_evidence_document,
        read_evidence_document,
    )
else:
    from evidence_json import (
        dump_evidence_json,
        parse_evidence_document,
        read_evidence_document,
    )

ROOT = Path(__file__).resolve().parents[1]
JsonRecord = dict[str, JsonValue]


@dataclass(frozen=True)
class EngineRetentionConfig:
    version: str | None
    source: str
    map_key: str
    frame_key: str
    api: str
    rule: str
    inverse: str


ENGINE_CONFIG = {
    "unreal": EngineRetentionConfig(
        version="5.7.4",
        source="experiments/unreal/5.7.4/Source/QuatOpsUnrealCapture/Private/QuatOpsUnrealCaptureTests.cpp",
        map_key="source_to_unreal_axis_map",
        frame_key="unreal_world_frame",
        api="UE::Math::TQuat<T>(X,Y,Z,W); TQuat::RotateVector",
        rule="native_xyzw=(-y,-x,-z,w)",
        inverse="canonical_wxyz=(W,-Y,-X,-Z)",
    ),
    "unity": EngineRetentionConfig(
        version="6000.5.2f1",
        source="experiments/unity/6000.5.2f1/Assets/Editor/QuatOpsEngineCapture.cs",
        map_key="source_to_unity_axis_map",
        frame_key="unity_world_frame",
        api="UnityEngine.Quaternion(x,y,z,w); Quaternion * Vector3",
        rule="native_xyzw=(-x,-z,-y,w)",
        inverse="canonical_wxyz=(W,-X,-Z,-Y)",
    ),
    "godot": EngineRetentionConfig(
        version=None,
        source="experiments/godot/capture.gd",
        map_key="source_to_godot_axis_map",
        frame_key="godot_world_frame",
        api="Quaternion(x,y,z,w); Quaternion * Vector3",
        rule="native_xyzw=(x,z,-y,w)",
        inverse="canonical_wxyz=(W,X,-Z,Y)",
    ),
    "blender": EngineRetentionConfig(
        version="5.1.2",
        source="experiments/blender/5.1.2/capture.py",
        map_key="source_to_blender_axis_map",
        frame_key="blender_world_frame_probe",
        api="mathutils.Quaternion((w,x,y,z)); Quaternion @ Vector",
        rule="native_wxyz=(w,x,y,z)",
        inverse="canonical_wxyz=native_wxyz",
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


def _quaternion(row: JsonRecord, field: str) -> list[float]:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(component, bool) or not isinstance(component, (int, float))
            for component in value
        )
    ):
        raise TypeError(f"{field} must contain four numeric coefficients")
    numeric_values = cast(list[int | float], value)
    return [float(component) for component in numeric_values]


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
        native_input = _quaternion(row, "canonical_wxyz")
        expected_input = expected_inputs[case_id]
        if any(
            abs(float(actual) - float(expected)) > input_tolerance
            for actual, expected in zip(native_input, expected_input)
        ):
            raise ValueError(f"native input differs from shared case {case_id!r}")
        retained = {
            key: value
            for key, value in row.items()
            if key not in {"canonical_wxyz", "native_input_wxyz"}
        }
        if case_id != "mixed_a":
            retained.pop("native_rotated_probe_vector_xyz", None)
        retained_cases.append(cast(JsonRecord, retained))
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
            "probe_vector_canonical_xyz",
            "probe_vector_native_xyz",
            "orientation_cases",
        }
    }
    for product_key in ("composition", "reverse_composition"):
        product = retained_profile.get(product_key)
        if isinstance(product, dict):
            retained_product = dict(product)
            retained_product.pop("native_rotated_probe_vector_xyz", None)
            retained_profile[product_key] = retained_product
    return cast(
        JsonRecord,
        {**retained_profile, "orientation_cases": retained_cases},
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(engine: str, version: str, source_path: Path, output_dir: Path) -> None:
    config = ENGINE_CONFIG[engine]
    if config.version and version != config.version:
        raise ValueError(f"{engine} version must be {config.version}")
    capture_source_name = config.source.format(version=version)
    original_bytes = source_path.read_bytes()
    captured = _record(
        parse_evidence_document(original_bytes), context="native capture"
    )
    canonical_interop = _record(
        captured.get("canonical_interop"), context="canonical_interop"
    )
    canonical_inputs = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    input_document = read_evidence_document(canonical_inputs)
    input_rows = _rows(input_document.get("orientations"), context="orientations")
    expected_inputs: dict[str, list[float]] = {}
    for row in input_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise TypeError("canonical input must have a string case_id")
        if case_id in expected_inputs:
            raise ValueError(f"duplicate canonical input case: {case_id}")
        expected_inputs[case_id] = _quaternion(row, "canonical_wxyz")

    profiles_to_check = (
        {
            name: _record(profile, context=f"precision_captures.{name}")
            for name, profile in _record(
                canonical_interop["precision_captures"],
                context="precision_captures",
            ).items()
        }
        if "precision_captures" in canonical_interop
        else {"default": canonical_interop}
    )
    for name, profile in profiles_to_check.items():
        if "reverse_composition" not in profile:
            raise ValueError(
                f"native capture profile {name!r} lacks reverse composition"
            )
    canonical_interop = retain_canonical_interop(canonical_interop, expected_inputs)

    engine_info = _record(captured.get("engine"), context="engine")
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
    axes_from_source = _record(
        map_record.get("destination_axes_from_source"),
        context="destination_axes_from_source",
    )
    capture_source = ROOT / capture_source_name
    witness = {
        "schema_version": 1,
        "target": {
            "engine": engine_info["name"],
            "api": config.api,
            "version": version,
            "build": engine_info.get("build_hash")
            or engine_info.get("changelist")
            or engine_info.get("build"),
            "platform": engine_info.get("platform"),
            "native_scalar_profiles": ["float32", "float64"]
            if engine == "unreal"
            else ["float32"],
            "native_quaternion_order": "wxyz" if engine == "blender" else "xyzw",
            "native_algebra": "Hamilton",
            "native_vector_action": "active quaternion sandwich / engine vector rotation API",
        },
        "contract": {
            "contract_id": "quatops-canonical-hamilton-wxyz-active-rh-v1",
            "canonical_algebra": "Hamilton",
            "canonical_order": "wxyz",
            "canonical_action": "active",
            "canonical_frame": source_frame,
            "native_frame": native_frame,
            "destination_axes_from_source": axes_from_source,
            "conversion_rule": config.rule,
            "inverse_rule": config.inverse,
            "world_and_body_map": "same declared signed-axis map",
            "input_domain": "unit Hamilton quaternion; adapters preserve magnitude and do not normalize",
            "native_precision_tolerance": {"float32": 2e-6, "float64": 2e-14}
            if engine == "unreal"
            else {"float32": 2e-6},
            "scope": "unit quaternion coefficient and physical-action conversion",
            "excludes": [
                "asset model-forward defaults",
                "scene import",
                "animation transport",
                "Euler adapters",
            ],
        },
        "provenance": {
            "original_capture_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "capture_source": capture_source_name.replace("\\", "/"),
            "capture_source_sha256": sha256(capture_source),
            "canonical_input_file": "experiments/interoperability/v1/canonical_inputs.json",
            "canonical_input_sha256": sha256(canonical_inputs),
            "native_build_record": engine_info,
            "capture_method": captured.get("capture_method"),
            "retention_method": (
                "selected native outputs retained losslessly; shared inputs "
                "referenced by case ID; redundant input copies and probe samples removed"
            ),
        },
        "canonical_interop": canonical_interop,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if engine == "godot":
        shared_dir = output_dir.parent / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        output_path = shared_dir / "witnesses.json"
        shared_witness = {
            "schema_version": 1,
            "canonical_interop": canonical_interop,
        }
        if output_path.exists():
            existing = read_evidence_document(output_path)
            if existing != shared_witness:
                raise ValueError(
                    "Godot version produced different selected values; retain a "
                    "separate versioned witness instead of sharing this file"
                )
        else:
            output_path.write_text(
                dump_evidence_json(shared_witness),
                encoding="utf-8",
                newline="\n",
            )
        (output_dir / "witnesses.json").unlink(missing_ok=True)
        contract_provenance = {
            "witness_file": "../shared/witnesses.json",
            "witness_sha256": sha256(output_path),
            "capture_source": witness["provenance"]["capture_source"],
            "capture_source_sha256": witness["provenance"]["capture_source_sha256"],
            "original_capture_sha256": witness["provenance"]["original_capture_sha256"],
            "canonical_input_file": witness["provenance"]["canonical_input_file"],
            "canonical_input_sha256": witness["provenance"]["canonical_input_sha256"],
            "native_build_record": engine_info,
            "capture_method": witness["provenance"]["capture_method"],
            "retention_method": witness["provenance"]["retention_method"],
        }
    else:
        output_path = output_dir / "witnesses.json"
        retained_witness = {
            "schema_version": 1,
            "canonical_interop": canonical_interop,
        }
        output_path.write_text(
            dump_evidence_json(retained_witness), encoding="utf-8", newline="\n"
        )
        contract_provenance = {
            "witness_file": "witnesses.json",
            "witness_sha256": sha256(output_path),
            "capture_source": witness["provenance"]["capture_source"],
            "capture_source_sha256": witness["provenance"]["capture_source_sha256"],
            "original_capture_sha256": witness["provenance"]["original_capture_sha256"],
            "canonical_input_file": witness["provenance"]["canonical_input_file"],
            "canonical_input_sha256": witness["provenance"]["canonical_input_sha256"],
        }

    contract = {
        "engine": witness["target"],
        "conversion": witness["contract"],
        "provenance": contract_provenance,
    }
    (output_dir / "contract.json").write_text(
        dump_evidence_json(contract), encoding="utf-8", newline="\n"
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
        help="versioned directory for retained contract and witnesses",
    )
    args = parser.parse_args()
    retain(args.engine, args.version, args.capture_json, args.output_dir)


if __name__ == "__main__":
    main()
