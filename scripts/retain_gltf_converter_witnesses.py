"""Retain the minimal glTF asset and native Three.js loader observations."""

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
HERE = ROOT / "experiments/converters/gltf/2.0.0"
DESTINATION = ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter"
PACKAGE_LOCK = (
    ROOT / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_numeric(item: object, index: int | None = None) -> object:
    if not isinstance(item, dict) or "values" not in item or "dtype" not in item:
        raise TypeError("expected a typed native numeric value")
    values = item["values"]
    selected = values[index] if index is not None else values[:3]
    result = {"values": selected, "dtype": item["dtype"]}
    if "ieee754_hex_be" in item:
        bits = item["ieee754_hex_be"]
        result["ieee754_hex_be"] = bits[index] if index is not None else bits[:3]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    capture = args.capture_dir.resolve()
    native_manifest = read_evidence_document(capture / "manifest.json")
    observations = compact_numeric_evidence(
        read_evidence_rows(capture / "observations.json")
    )
    if len(observations) != 2 or {row["node"] for row in observations} != {
        "QuatOpsParent",
        "QuatOpsChild",
    }:
        raise ValueError(
            "expected native observations for the two converter-generated nodes"
        )
    native_entry_points = observations[0]["native_entry_points"]
    if any(row["native_entry_points"] != native_entry_points for row in observations):
        raise ValueError("native entry points should be shared by both node captures")
    observations = [
        {
            "node": row["node"],
            "outputs": {
                "local_quaternion_xyzw": row["outputs"]["local_quaternion_xyzw"],
                "local_basis_images_xyz": select_numeric(
                    row["outputs"]["local_basis_and_probe_images"]
                ),
                **(
                    {
                        "world_basis_images_xyz": select_numeric(
                            row["outputs"]["world_unit_directions_xyz"]
                        ),
                        "world_probe_point_xyz": select_numeric(
                            row["outputs"]["world_points_xyz"], -1
                        ),
                        "roundtrip_probe_point_xyz": select_numeric(
                            row["outputs"]["world_to_local_point_roundtrip_xyz"], -1
                        ),
                    }
                    if row["node"] == "QuatOpsChild"
                    else {}
                ),
            },
        }
        for row in observations
    ]

    DESTINATION.mkdir(parents=True, exist_ok=True)
    witnesses_path = DESTINATION / "witnesses.json"
    witnesses_path.write_text(
        dump_evidence_json(
            {
                "schema_version": "1.0",
                "evidence_kind": native_manifest["evidence_kind"],
                "native_entry_points": native_entry_points,
                "observations": observations,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    contract = {
        "contract_id": "quatops-gltf-static-hierarchy-v1",
        "target": "glTF 2.0 loaded by Three.js GLTFLoader 0.187.0",
        "algebra": "Hamilton active quaternion",
        "source_storage": "QuatOps WXYZ",
        "serialized_storage": "glTF XYZW",
        "canonical_frame": {
            "handedness": "right",
            "positive_x": "right",
            "positive_y": "forward",
            "positive_z": "up",
        },
        "asset_frame": {
            "handedness": "right",
            "positive_x": "right",
            "positive_y": "up",
            "positive_z": "backward",
            "forward": "-Z",
        },
        "destination_axes_from_source": ["x", "z", "-y"],
        "transform_semantics": "node local to parent; world = parent world * child local; column-vector action",
        "scope": [
            "static node TRS serialization",
            "hierarchy composition",
            "native local quaternion and basis actions",
            "native child world basis action and mixed-point action",
            "inverse point round trip",
        ],
        "excludes": [
            "animation",
            "mesh model-forward convention",
            "compression extensions",
            "other GLTFLoader versions",
        ],
    }
    contract_path = DESTINATION / "contract.json"
    contract_path.write_text(
        dump_evidence_json(contract), encoding="utf-8", newline="\n"
    )
    (DESTINATION / "manifest.json").write_text(
        dump_evidence_json(
            {
                "schema_version": "1.0",
                "review_state": "QuatOps-generated asset loaded by pinned native GLTFLoader; replay tests passed",
                "target": native_manifest["target"],
                "execution": native_manifest["execution"],
                "provenance": {
                    "input_sha256": sha256(HERE / "converter_inputs.json"),
                    "asset_sha256": sha256(HERE / "quatops_hierarchy.gltf"),
                    "native_observations_sha256": sha256(witnesses_path),
                    "full_run_manifest_sha256": sha256(capture / "manifest.json"),
                    "full_observation_sha256": sha256(capture / "observations.json"),
                    "native_loader_script_sha256": native_manifest["exporter"][
                        "sha256"
                    ],
                    "native_package_lock_sha256": native_manifest[
                        "package_lock_sha256"
                    ],
                    "native_numeric_helper_sha256": native_manifest[
                        "numeric_helper_sha256"
                    ],
                    "contract_sha256": sha256(contract_path),
                },
                "reproduction": {
                    "generator_sha256": sha256(HERE / "create_asset.py"),
                    "loader_script_sha256": sha256(HERE / "capture_threejs.mjs"),
                    "package_lock_sha256": sha256(PACKAGE_LOCK),
                    "numeric_helper_sha256": sha256(
                        ROOT / "experiments/converters/shared/numeric_evidence.mjs"
                    ),
                    "retention_script_sha256": sha256(Path(__file__)),
                    "retention_helper_sha256": sha256(
                        ROOT / "scripts/evidence_json.py"
                    ),
                    "retention_formatter_sha256": sha256(
                        ROOT / "scripts/evidence_format.py"
                    ),
                },
                "selection": {
                    "retained_observations": len(observations),
                    "native_observations": native_manifest["observations"]["count"],
                    "policy": (
                        "local quaternion and basis actions per node; one child world "
                        "basis, mixed-point, and inverse-point witness"
                    ),
                    "numeric_values": (
                        "round-trippable decimals and dtype; exact bits only for "
                        "signed-zero or nonfinite values"
                    ),
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"retained {len(observations)} Three.js loader witnesses at {DESTINATION}; "
        f"asset remains at {HERE / 'quatops_hierarchy.gltf'}"
    )


if __name__ == "__main__":
    main()
