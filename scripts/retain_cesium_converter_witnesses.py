"""Retain selected Cesium converter outputs and pinned run provenance."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from evidence_json import (
    compact_numeric_evidence,
    dump_evidence_json,
    read_evidence_document,
    read_evidence_rows,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/converters/cesium/1.124.0"
DESTINATION = ROOT / "tests/golden/transports/cesium/1.124.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vectors(value: dict, *, start: int = 0, stop: int | None = None) -> dict:
    return {
        "values": value["values"][start:stop],
        "dtype": value["dtype"],
    }


def _retained_outputs(case_id: str, outputs: dict) -> dict:
    retained = {
        "model_to_ecef_quaternion_xyzw": outputs["model_to_ecef_quaternion_xyzw"],
        "local_quaternion_recovered_from_ecef_xyzw": outputs[
            "local_quaternion_recovered_from_ecef_xyzw"
        ],
        "world_directions_xyz": _vectors(
            outputs["world_directions_xyz"],
            stop=4 if case_id == "mixed_a@geo-03" else 3,
        ),
        "world_points_xyz_m": _vectors(outputs["world_points_xyz_m"], start=-1),
        "point_roundtrip_local_xyz_m": _vectors(
            outputs["point_roundtrip_local_xyz_m"], start=-1
        ),
        "ecef_origin_xyz_m": outputs["ecef_origin_xyz_m"],
    }
    if case_id == "mixed_a@geo-03":
        retained["local_basis_and_probe_images"] = outputs[
            "local_basis_and_probe_images"
        ]
    return compact_numeric_evidence(retained)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    capture = args.capture_dir.resolve()
    run = read_evidence_document(capture / "manifest.json")
    observations = read_evidence_rows(capture / "observations.json")
    inputs = read_evidence_document(HERE / "converter_inputs.json")
    case_ids = [case["case_id"] for case in inputs["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Cesium converter inputs contain duplicate case IDs")
    matches_by_id = {
        case_id: [row for row in observations if row.get("case_id") == case_id]
        for case_id in case_ids
    }
    duplicate_ids = [
        case_id for case_id, matches in matches_by_id.items() if len(matches) > 1
    ]
    if duplicate_ids:
        raise ValueError(
            f"duplicate Cesium capture case IDs: {', '.join(duplicate_ids)}"
        )
    missing_ids = [case_id for case_id, matches in matches_by_id.items() if not matches]
    if missing_ids:
        raise ValueError(
            f"Cesium capture is missing selected cases: {', '.join(missing_ids)}"
        )
    selected_observations = [matches_by_id[case_id][0] for case_id in case_ids]

    DESTINATION.mkdir(parents=True, exist_ok=True)
    witnesses_path = DESTINATION / "witnesses.json"
    retained = [
        {
            "case_id": row["case_id"],
            "outputs": _retained_outputs(row["case_id"], row["outputs"]),
        }
        for row in selected_observations
    ]
    witnesses_path.write_text(
        dump_evidence_json(
            {
                "schema_version": "1.0",
                "evidence_kind": "native_cesium_quatops_converter_capture",
                "observations": retained,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    contract = {
        "contract_id": "quatops-cesium-local-enu-ecef-v1",
        "target": "CesiumJS 1.124.0",
        "algebra": "Hamilton active quaternion",
        "storage": "QuatOps WXYZ; Cesium XYZW",
        "local_frame": {
            "handedness": "right",
            "positive_x": "east/right",
            "positive_y": "north/forward",
            "positive_z": "up/ellipsoid normal",
        },
        "source_map": ["x", "y", "z"],
        "native_api": [
            "Cartesian3.fromDegrees",
            "Transforms.eastNorthUpToFixedFrame",
            "Matrix3.fromQuaternion",
            "Matrix4.fromRotationTranslation",
            "Matrix4.multiply",
            "Matrix4.inverseTransformation",
        ],
        "forward": (
            "R_ECEF_from_model = R_ECEF_from_ENU * R_ENU_from_model; "
            "t_ECEF = t_ECEF_from_ENU + R_ECEF_from_ENU * t_ENU_from_model"
        ),
        "inverse": (
            "R_ENU_from_model = transpose(R_ECEF_from_ENU) * R_ECEF_from_model; "
            "t_ENU = transpose(R_ECEF_from_ENU) * (t_ECEF - t_ECEF_from_ENU)"
        ),
        "geodesy": "WGS84; lon/lat input in degrees, height and translations in metres",
        "normalization": "unit input quaternion; no hidden sign canonicalization in QuatOps input mapping",
        "scope": [
            "QuatOps-produced local quaternion to Cesium ECEF transform",
            "native basis/probe action",
            "multiple geodetic origins",
            "inverse point and orientation recovery",
            "noncommuting composition and antipode",
        ],
        "excludes": [
            "Cesium Model/glTF importer",
            "asset model-forward defaults",
            "rendering",
            "time-dependent EOP/ICRF",
        ],
    }
    contract_path = DESTINATION / "contract.json"
    contract_path.write_text(
        dump_evidence_json(contract), encoding="utf-8", newline="\n"
    )
    source_inputs = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    geo_inputs = ROOT / "experiments/oracles/shared/geographic_inputs_v1.json"
    native_numeric_helper_sha256 = run.get("numeric_helper_sha256")
    manifest = {
        "schema_version": "1.0",
        "review_state": (
            "native capture retained; QuatOps replay covered by "
            "tests/experiments/converters/cesium/test_cesium_converter.py"
        ),
        "target": run["target"],
        "execution": run["execution"],
        "provenance": {
            "run_manifest_sha256": sha256(capture / "manifest.json"),
            "full_observation_sha256": sha256(capture / "observations.json"),
            "converter_input_sha256": sha256(HERE / "converter_inputs.json"),
            "canonical_inputs_sha256": sha256(source_inputs),
            "geographic_inputs_sha256": sha256(geo_inputs),
            "native_capture_script_sha256": run["exporter"]["sha256"],
            "native_numeric_helper_sha256": native_numeric_helper_sha256,
            "native_package_lock_sha256": run["package_lock_sha256"],
            "witnesses_sha256": sha256(witnesses_path),
            "contract_sha256": sha256(contract_path),
        },
        "reproduction": {
            "input_generator_sha256": sha256(HERE / "create_inputs.py"),
            "capture_script_sha256": sha256(HERE / "capture_native.mjs"),
            "numeric_helper_sha256": sha256(
                ROOT / "experiments/converters/shared/numeric_evidence.mjs"
            ),
            "retention_script_sha256": sha256(Path(__file__)),
            "retention_helper_sha256": sha256(ROOT / "scripts/evidence_json.py"),
            "retention_formatter_sha256": sha256(ROOT / "scripts/evidence_format.py"),
            "package_lock_sha256": sha256(HERE / "package-lock.json"),
            "geographic_inputs_sha256": sha256(geo_inputs),
        },
        "selection": {
            "retained_observations": len(selected_observations),
            "native_observations": run["observations"]["count"],
            "policy": (
                "retain local basis action once; retain ECEF orientation, "
                "world basis action, one probe point round trip, and origin "
                "for identity, mixed, antipodal, and composed cases"
            ),
            "numeric_values": (
                "round-trippable decimals and dtype; exact bits only for "
                "signed-zero or nonfinite values"
            ),
        },
    }
    (DESTINATION / "manifest.json").write_text(
        dump_evidence_json(manifest), encoding="utf-8", newline="\n"
    )
    print(
        f"retained {len(selected_observations)} Cesium converter witnesses at {DESTINATION}"
    )


if __name__ == "__main__":
    main()
