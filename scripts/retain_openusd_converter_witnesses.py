"""Retain a QuatOps-generated USDA stage and selected OpenUSD native results."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scripts.evidence_json import (
    dump_evidence_json,
    read_evidence_document,
    read_evidence_rows,
)
from scripts.evidence_schema.converter_captures import OpenUsdContract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    source_dir = repository_root / "experiments/converters/openusd/26.11"
    destination = repository_root / "tests/golden/transports/openusd/26.11/converter"
    capture = args.capture_dir.resolve()
    native = read_evidence_document(capture / "manifest.json")
    observations = read_evidence_rows(capture / "observations.json")
    if len(observations) != 3 or {row["node"] for row in observations} != {
        "parent",
        "child",
        "reset_child",
    }:
        raise ValueError(
            "expected parent, child, and reset-child native stage observations"
        )
    native_entry_points = observations[0]["native_entry_points"]
    ordered_xform_ops = observations[0]["outputs"]["ordered_xform_ops"]
    if any(row["native_entry_points"] != native_entry_points for row in observations):
        raise ValueError("native entry points should be shared by all node captures")
    if any(
        row["outputs"]["ordered_xform_ops"] != ordered_xform_ops for row in observations
    ):
        raise ValueError("all authored nodes should use the same xform-op order")
    observations = [
        {
            "node": row["node"],
            "prim_path": row["prim_path"],
            "outputs": {
                "orient_op_wxyz": row["outputs"]["orient_op_wxyz"],
                "resets_xform_stack": row["outputs"]["resets_xform_stack"],
                "world_basis_images_xyz": row["outputs"]["world_directions_xyz"][:3],
                "world_probe_point_xyz": row["outputs"]["world_points_xyz"][-1],
                **(
                    {
                        "roundtrip_probe_point_xyz": row["outputs"][
                            "world_to_local_point_roundtrip_xyz"
                        ][-1]
                    }
                    if row["node"] == "child"
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
                "evidence_kind": native["evidence_kind"],
                "native_entry_points": native_entry_points,
                "ordered_xform_ops": ordered_xform_ops,
                "observations": observations,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    target = native["target"]
    if target.get("name") != "OpenUSD" or target.get("distribution_version") != "26.11":
        raise ValueError("OpenUSD semantic contract does not match captured runtime")
    contract = OpenUsdContract.model_validate(
        read_evidence_document(source_dir / "semantic_contract.json"), strict=True
    )
    contract_path = destination / "contract.json"
    contract_path.write_text(
        dump_evidence_json(contract.model_dump(mode="json")),
        encoding="utf-8",
        newline="\n",
    )
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
                "review_state": "QuatOps-generated USDA reopened by pinned OpenUSD; native capture retained",
                "target": native["target"],
                "stage_metadata": native["stage_metadata"],
                "execution": native["execution"],
                "native_binaries_sha256": native["native_binaries_sha256"],
                "provenance": {
                    "stage_sha256": sha256(source_dir / "quatops_hierarchy.usda"),
                    "converter_input_sha256": sha256(
                        source_dir / "converter_inputs.json"
                    ),
                    "captured_converter_input_sha256": native["inputs"][
                        "converter_input_sha256"
                    ],
                    "native_observations_sha256": sha256(witnesses_path),
                    "native_run_manifest_sha256": sha256(capture / "manifest.json"),
                    "full_observation_sha256": sha256(capture / "observations.json"),
                    "generator_sha256": sha256(source_dir / "create_stage.py"),
                    "native_capture_script_sha256": sha256(
                        source_dir / "capture_native.py"
                    ),
                    "lockfile_sha256": native["lockfile_sha256"],
                    "contract_sha256": sha256(contract_path),
                },
                "reproduction": {
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
                    "native_observations": native["observations"]["count"],
                    "policy": (
                        "orient op and reset flag per role; native world basis and "
                        "mixed point actions; one inverse point round trip"
                    ),
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"retained {len(observations)} OpenUSD witnesses at {destination}; "
        f"stage remains at {source_dir / 'quatops_hierarchy.usda'}"
    )


if __name__ == "__main__":
    main()
