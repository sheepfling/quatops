"""Retain selected Cesium converter outputs and pinned run provenance."""

from __future__ import annotations

from pathlib import Path

from scripts.evidence_json import (
    dump_evidence_json,
    read_evidence_document,
    read_evidence_rows,
    select_numeric_evidence,
)
from scripts.evidence_schema.converter_captures import CesiumContract
from scripts.evidence_schema.utilities import retention_paths, retention_provenance
from scripts.evidence_schema.utilities import sha256_file as sha256


def _vectors(value: dict, *, start: int = 0, stop: int | None = None) -> dict:
    return select_numeric_evidence(value, slice(start, stop))


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
    return retained


def main() -> None:
    capture, repository_root = retention_paths(__doc__)
    source_dir = repository_root / "experiments/converters/cesium/1.124.0"
    destination = repository_root / "tests/golden/transports/cesium/1.124.0"
    run = read_evidence_document(capture / "manifest.json")
    observations = read_evidence_rows(capture / "observations.json")
    inputs = read_evidence_document(source_dir / "converter_inputs.json")
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

    destination.mkdir(parents=True, exist_ok=True)
    witnesses_path = destination / "witnesses.json"
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
    if (
        run["target"].get("name") != "CesiumJS"
        or run["target"].get("package_version") != "1.124.0"
    ):
        raise ValueError("Cesium semantic contract does not match captured package")
    contract_path = source_dir / "semantic_contract.json"
    CesiumContract.model_validate(read_evidence_document(contract_path), strict=True)
    source_inputs = (
        repository_root / "experiments/interoperability/v1/canonical_inputs.json"
    )
    geo_inputs = (
        repository_root / "experiments/oracles/shared/geographic_inputs_v1.json"
    )
    native_numeric_helper_sha256 = run.get("numeric_helper_sha256")
    manifest = {
        "schema_version": "1.0",
        "semantic_contract": {
            "file": (source_dir / "semantic_contract.json")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha256(source_dir / "semantic_contract.json"),
        },
        "review_state": (
            "native capture retained; QuatOps replay covered by "
            "tests/experiments/converters/cesium/test_cesium_converter.py"
        ),
        "target": run["target"],
        "execution": run["execution"],
        "provenance": {
            "run_manifest_sha256": sha256(capture / "manifest.json"),
            "full_observation_sha256": sha256(capture / "observations.json"),
            "converter_input_sha256": sha256(source_dir / "converter_inputs.json"),
            "captured_converter_input_sha256": run["inputs"]["sha256"],
            "canonical_inputs_sha256": sha256(source_inputs),
            "geographic_inputs_sha256": sha256(geo_inputs),
            "native_capture_script_sha256": run["exporter"]["sha256"],
            "native_numeric_helper_sha256": native_numeric_helper_sha256,
            "native_package_lock_sha256": run["package_lock_sha256"],
            "witnesses_sha256": sha256(witnesses_path),
            "contract_sha256": sha256(contract_path),
            "input_serialization_note": (
                "JSON formatting changed only; parsed native input values are unchanged."
            ),
        },
        "reproduction": {
            "input_generator_sha256": sha256(source_dir / "create_inputs.py"),
            "input_builder_sha256": sha256(
                repository_root / "experiments/converters/cesium/inputs.py"
            ),
            "capture_script_sha256": sha256(source_dir / "capture_native.mjs"),
            "numeric_helper_sha256": sha256(
                repository_root / "experiments/converters/shared/numeric_evidence.mjs"
            ),
            **retention_provenance(repository_root, Path(__file__)),
            "package_lock_sha256": sha256(source_dir / "package-lock.json"),
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
                "round-trippable finite binary64 decimals, including signed zero"
            ),
        },
    }
    (destination / "manifest.json").write_text(
        dump_evidence_json(manifest), encoding="utf-8", newline="\n"
    )
    print(
        f"retained {len(selected_observations)} Cesium converter witnesses at {destination}"
    )


if __name__ == "__main__":
    main()
