"""Retain a QuatOps-generated USDA stage and selected OpenUSD native results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from evidence_json import read_evidence_document, read_evidence_rows

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

    DESTINATION.mkdir(parents=True, exist_ok=True)
    files = (
        (HERE / "converter_inputs.json", "inputs.json"),
        (HERE / "quatops_hierarchy.usda", "quatops_hierarchy.usda"),
    )
    for source, name in files:
        shutil.copyfile(source, DESTINATION / name)
    witnesses_path = DESTINATION / "witnesses.json"
    witnesses_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evidence_kind": native["evidence_kind"],
                "observations": observations,
            },
            indent=2,
        )
        + "\n",
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
        "scope": [
            "file serialization and reopen",
            "quatd orient op readback",
            "translate+orient stacks",
            "parent/child composition",
            "resetXformStack",
            "point and direction actions",
            "inverse point round trip",
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
        json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (DESTINATION / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "review_state": "QuatOps-generated USDA reopened by pinned OpenUSD; replay tests passed",
                "target": native["target"],
                "stage_metadata": native["stage_metadata"],
                "execution": native["execution"],
                "native_binaries_sha256": native["native_binaries_sha256"],
                "provenance": {
                    "stage_sha256": sha256(DESTINATION / "quatops_hierarchy.usda"),
                    "converter_input_sha256": sha256(DESTINATION / "inputs.json"),
                    "native_observations_sha256": sha256(witnesses_path),
                    "native_run_manifest_sha256": sha256(capture / "manifest.json"),
                    "full_observation_sha256": sha256(capture / "observations.json"),
                    "generator_sha256": sha256(HERE / "create_stage.py"),
                    "native_capture_script_sha256": sha256(HERE / "capture_native.py"),
                    "lockfile_sha256": native["lockfile_sha256"],
                    "contract_sha256": sha256(contract_path),
                },
                "selection": {
                    "retained_observations": len(observations),
                    "native_observations": native["observations"]["count"],
                    "policy": "one local/world stage observation for each authored hierarchy role",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"retained QuatOps USDA stage and {len(observations)} OpenUSD witnesses at {DESTINATION}"
    )


if __name__ == "__main__":
    main()
