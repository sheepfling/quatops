"""Retain the minimal glTF asset and native Three.js loader observations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scripts.evidence_json import (
    dump_evidence_json,
    read_evidence_document,
    read_evidence_rows,
    select_numeric_evidence,
)
from scripts.evidence_schema.converter_captures import ThreeGltfContract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_numeric(item: object, index: int | None = None) -> object:
    return select_numeric_evidence(item, index if index is not None else slice(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    source_dir = repository_root / "experiments/converters/gltf/2.0.0"
    destination = (
        repository_root / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter"
    )
    package_lock = (
        repository_root
        / "experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json"
    )
    capture = args.capture_dir.resolve()
    native_manifest = read_evidence_document(capture / "manifest.json")
    observations = read_evidence_rows(capture / "observations.json")
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

    destination.mkdir(parents=True, exist_ok=True)
    witnesses_path = destination / "witnesses.json"
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
    target = native_manifest["target"]
    if (
        target.get("name") != "Three.js GLTFLoader"
        or target.get("package_version") != "0.187.0"
    ):
        raise ValueError("glTF semantic contract does not match captured loader")
    contract_path = source_dir / "semantic_contract.json"
    ThreeGltfContract.model_validate(read_evidence_document(contract_path), strict=True)
    (destination / "manifest.json").write_text(
        dump_evidence_json(
            {
                "schema_version": "1.0",
                "semantic_contract": {
                    "file": (source_dir / "semantic_contract.json")
                    .relative_to(repository_root)
                    .as_posix(),
                    "sha256": sha256(source_dir / "semantic_contract.json"),
                },
                "review_state": "QuatOps-generated asset loaded by pinned native GLTFLoader; replay tests passed",
                "target": native_manifest["target"],
                "execution": native_manifest["execution"],
                "provenance": {
                    "input_sha256": sha256(source_dir / "converter_inputs.json"),
                    "captured_input_sha256": native_manifest["asset"]["input_sha256"],
                    "asset_sha256": sha256(source_dir / "quatops_hierarchy.gltf"),
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
                    "input_serialization_note": (
                        "JSON formatting changed only; parsed native input values are unchanged."
                    ),
                },
                "reproduction": {
                    "generator_sha256": sha256(source_dir / "create_asset.py"),
                    "input_builder_sha256": sha256(
                        repository_root / "experiments/converters/gltf/inputs.py"
                    ),
                    "loader_script_sha256": sha256(source_dir / "capture_threejs.mjs"),
                    "package_lock_sha256": sha256(package_lock),
                    "numeric_helper_sha256": sha256(
                        repository_root
                        / "experiments/converters/shared/numeric_evidence.mjs"
                    ),
                    "retention_script_sha256": sha256(Path(__file__)),
                    "retention_helper_sha256": sha256(
                        repository_root / "scripts/evidence_json.py"
                    ),
                    "retention_formatter_sha256": sha256(
                        repository_root / "scripts/evidence_format.py"
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
                        "round-trippable finite binary64 decimals, including signed zero"
                    ),
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"retained {len(observations)} Three.js loader witnesses at {destination}; "
        f"asset remains at {source_dir / 'quatops_hierarchy.gltf'}"
    )


if __name__ == "__main__":
    main()
