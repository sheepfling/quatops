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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    capture = args.capture_dir.resolve()
    run = read_evidence_document(capture / "manifest.json")
    observations = read_evidence_rows(capture / "observations.json")
    inputs = read_evidence_document(HERE / "converter_inputs.json")
    if len(observations) != len(inputs["cases"]) or len(observations) != 5:
        raise ValueError("unexpected Cesium converter capture case count")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    inputs_path = DESTINATION / "inputs.json"
    inputs_path.write_bytes((HERE / "converter_inputs.json").read_bytes())
    witnesses_path = DESTINATION / "witnesses.json"
    retained = [
        {
            "case_id": row["case_id"],
            "outputs": compact_numeric_evidence(row["outputs"]),
        }
        for row in observations
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
    geo_inputs = ROOT / "experiments/converters/cesium/1.124.0/geographic_inputs.json"
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
            "converter_input_sha256": sha256(inputs_path),
            "canonical_inputs_sha256": sha256(source_inputs),
            "geographic_inputs_sha256": sha256(geo_inputs),
            "native_capture_script_sha256": run["exporter"]["sha256"],
            "native_numeric_helper_sha256": run["numeric_helper_sha256"],
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
            "package_lock_sha256": sha256(HERE / "package-lock.json"),
            "geographic_inputs_sha256": sha256(geo_inputs),
        },
        "selection": {
            "retained_observations": len(observations),
            "native_observations": run["observations"]["count"],
            "policy": (
                "retain identity, mixed orientations, an antipode, a "
                "noncommuting composition, and four WGS84 origins"
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
    print(f"retained {len(observations)} Cesium converter witnesses at {DESTINATION}")


if __name__ == "__main__":
    main()
