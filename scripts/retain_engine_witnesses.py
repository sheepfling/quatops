"""Select compact, provenance-linked engine witnesses from a discovery capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidence_json import parse_evidence_document, read_evidence_document

ROOT = Path(__file__).resolve().parents[1]
ENGINE_CONFIG = {
    "unreal": {
        "version": "5.7.4",
        "source": "experiments/unreal/5.7.4/Source/QuatOpsUnrealCapture/Private/QuatOpsUnrealCaptureTests.cpp",
        "map_key": "source_to_unreal_axis_map",
        "frame_key": "unreal_world_frame",
        "api": "UE::Math::TQuat<T>(X,Y,Z,W); TQuat::RotateVector",
        "rule": "native_xyzw=(-y,-x,-z,w)",
        "inverse": "canonical_wxyz=(W,-Y,-X,-Z)",
    },
    "unity": {
        "version": "6000.5.2f1",
        "source": "experiments/unity/6000.5.2f1/Assets/Editor/QuatOpsEngineCapture.cs",
        "map_key": "source_to_unity_axis_map",
        "frame_key": "unity_world_frame",
        "api": "UnityEngine.Quaternion(x,y,z,w); Quaternion * Vector3",
        "rule": "native_xyzw=(-x,-z,-y,w)",
        "inverse": "canonical_wxyz=(W,-X,-Z,-Y)",
    },
    "godot": {
        "version": None,
        "source": "experiments/godot/{version}/capture.gd",
        "map_key": "source_to_godot_axis_map",
        "frame_key": "godot_world_frame",
        "api": "Quaternion(x,y,z,w); Quaternion * Vector3",
        "rule": "native_xyzw=(x,z,-y,w)",
        "inverse": "canonical_wxyz=(W,X,-Z,Y)",
    },
    "blender": {
        "version": "5.1.2",
        "source": "experiments/blender/5.1.2/capture.py",
        "map_key": "source_to_blender_axis_map",
        "frame_key": "blender_world_frame_probe",
        "api": "mathutils.Quaternion((w,x,y,z)); Quaternion @ Vector",
        "rule": "native_wxyz=(w,x,y,z)",
        "inverse": "canonical_wxyz=native_wxyz",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(engine: str, version: str, source_path: Path, output_dir: Path) -> None:
    config = ENGINE_CONFIG[engine]
    if config["version"] and version != config["version"]:
        raise ValueError(f"{engine} version must be {config['version']}")
    config = dict(config)
    config["source"] = config["source"].format(version=version)
    original_bytes = source_path.read_bytes()
    captured = parse_evidence_document(original_bytes)
    canonical_interop = captured["canonical_interop"]
    canonical_inputs = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    input_rows = read_evidence_document(canonical_inputs)["orientations"]
    expected_inputs = {row["case_id"]: row["canonical_wxyz"] for row in input_rows}
    expected_cases = set(expected_inputs)
    precision_capture = (
        canonical_interop["precision_captures"]["float32"]
        if "precision_captures" in canonical_interop
        else canonical_interop
    )
    actual_cases = {row["case_id"] for row in precision_capture["orientation_cases"]}
    if expected_cases != actual_cases:
        raise ValueError(
            f"native capture witness IDs differ: expected {sorted(expected_cases)}, got {sorted(actual_cases)}"
        )
    if "reverse_composition" not in precision_capture:
        raise ValueError("native capture is missing the reverse composition witness")
    profiles = (
        canonical_interop["precision_captures"].values()
        if "precision_captures" in canonical_interop
        else (canonical_interop,)
    )
    for profile in profiles:
        for row in profile["orientation_cases"]:
            row["native_input_wxyz"] = row["canonical_wxyz"]
            row["canonical_wxyz"] = expected_inputs[row["case_id"]]

    engine_info = captured["engine"]
    if engine == "unreal":
        source_frame = captured["native_semantics"]["source_frame"]
    else:
        source_frame = captured["source_frame"]
    if engine == "unreal":
        map_record = captured["native_semantics"][config["map_key"]]
        native_frame = captured["native_semantics"][config["frame_key"]]
    else:
        map_record = captured[config["map_key"]]
        native_frame = captured.get(
            config["frame_key"],
            {
                "declared_probe_semantics": "identity probe; no universal asset axis inferred"
            },
        )
    capture_source = ROOT / config["source"]
    witness = {
        "schema_version": 1,
        "target": {
            "engine": engine_info["name"],
            "api": config["api"],
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
            "destination_axes_from_source": map_record["destination_axes_from_source"],
            "conversion_rule": config["rule"],
            "inverse_rule": config["inverse"],
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
            "capture_source": config["source"].replace("\\", "/"),
            "capture_source_sha256": sha256(capture_source),
            "canonical_input_file": "experiments/interoperability/v1/canonical_inputs.json",
            "canonical_input_sha256": sha256(canonical_inputs),
            "native_build_record": engine_info,
            "capture_method": captured.get("capture_method"),
            "retention_method": "selected values copied losslessly from native capture; no values regenerated",
        },
        "canonical_interop": canonical_interop,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "witnesses.json"
    output_path.write_text(
        json.dumps(witness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    contract = {
        "engine": witness["target"],
        "conversion": witness["contract"],
        "provenance": {
            "witness_file": "witnesses.json",
            "witness_sha256": sha256(output_path),
            "capture_source": witness["provenance"]["capture_source"],
            "capture_source_sha256": witness["provenance"]["capture_source_sha256"],
            "original_capture_sha256": witness["provenance"]["original_capture_sha256"],
            "canonical_input_sha256": witness["provenance"]["canonical_input_sha256"],
        },
    }
    (output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=tuple(ENGINE_CONFIG))
    parser.add_argument("version")
    parser.add_argument(
        "capture_json", type=Path, help="temporary full native discovery output"
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
