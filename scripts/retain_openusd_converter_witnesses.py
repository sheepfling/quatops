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

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/converters/openusd/26.11"
DESTINATION = ROOT / "tests/golden/transports/openusd/26.11/converter"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
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

    DESTINATION.mkdir(parents=True, exist_ok=True)
    witnesses_path = DESTINATION / "witnesses.json"
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
    contract = {
        "contract_id": "quatops-openusd-y-up-hierarchy-v1",
        "target": "OpenUSD usd-core 26.11; USDA reopened by pxr.Usd.Stage.Open",
        "source": {
            "algebra": "Hamilton active",
            "storage": "WXYZ",
            "frame": "right-handed; +X right, +Y forward, +Z up",
        },
        "stage": {
            "handedness": "right",
            "upAxis": "Y",
            "positive_x": "right",
            "positive_y": "up",
            "positive_z": "backward",
            "forward": "-Z",
            "metersPerUnit": 1.0,
        },
        "destination_axes_from_source": ["x", "z", "-y"],
        "serialized_orientation": "USDA quatd orient op, real-first WXYZ, converted by QuatOps QuaternionFrameMap",
        "matrix_action": (
            "OpenUSD Gf row vectors; child world row matrix = child local row "
            "matrix * parent world row matrix"
        ),
        "native_numeric_type": "binary64 Gf.Quatd and Gf.Matrix4d",
        "scope": [
            "file serialization and reopen",
            "quatd orient op readback",
            "translate+orient stacks",
            "parent/child composition",
            "resetXformStack",
            "native world basis directions and mixed-point actions",
            "one inverse point round trip",
        ],
        "excludes": [
            "other stage up-axis choices",
            "time samples",
            "shear/nonuniform scale decomposition",
            "renderer or file import pipeline",
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
                "review_state": "QuatOps-generated USDA reopened by pinned OpenUSD; native capture retained",
                "target": native["target"],
                "stage_metadata": native["stage_metadata"],
                "execution": native["execution"],
                "native_binaries_sha256": native["native_binaries_sha256"],
                "provenance": {
                    "stage_sha256": sha256(HERE / "quatops_hierarchy.usda"),
                    "converter_input_sha256": sha256(HERE / "converter_inputs.json"),
                    "captured_converter_input_sha256": native["inputs"][
                        "converter_input_sha256"
                    ],
                    "native_observations_sha256": sha256(witnesses_path),
                    "native_run_manifest_sha256": sha256(capture / "manifest.json"),
                    "full_observation_sha256": sha256(capture / "observations.json"),
                    "generator_sha256": sha256(HERE / "create_stage.py"),
                    "native_capture_script_sha256": sha256(HERE / "capture_native.py"),
                    "lockfile_sha256": native["lockfile_sha256"],
                    "contract_sha256": sha256(contract_path),
                },
                "reproduction": {
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
        f"retained {len(observations)} OpenUSD witnesses at {DESTINATION}; "
        f"stage remains at {HERE / 'quatops_hierarchy.usda'}"
    )


if __name__ == "__main__":
    main()
