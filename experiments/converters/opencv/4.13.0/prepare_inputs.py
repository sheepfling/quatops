"""Prepare canonical-to-Rodrigues boundary values with QuatOps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from quatops.python import hamilton

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    spec_path = HERE / "converter_inputs.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    shared_path = ROOT / spec["shared_canonical_inputs"]
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row for row in shared["orientations"]}
    inputs = {
        "schema_version": "1.0",
        "contract_id": spec["contract_id"],
        "canonical_input_sha256": sha256_file(shared_path),
        "preparation_spec_sha256": sha256_file(spec_path),
        "source_frame": shared["source_frame"],
        "probe_vector_xyz": shared["probe_vector_xyz"],
        "precision_profiles": spec["precision_profiles"],
        "orientations": [],
        "native_constructed_orientation": spec["native_constructed_orientation"],
        "composition": {
            "left_case_id": shared["composition"]["left_case_id"],
            "right_case_id": shared["composition"]["right_case_id"],
            "product_orders": [
                "left * right; right acts first",
                "right * left; left acts first",
            ],
        },
    }
    for case_id in spec["retained_case_ids"]:
        q = by_id[case_id]["canonical_wxyz"]
        inputs["orientations"].append(
            {
                "case_id": case_id,
                "canonical_wxyz": q,
                "quatops_rotation_vector_xyz_radians": list(
                    hamilton.to_rotation_vector(q)
                ),
            }
        )
    output_path = options.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inputs, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "orientations": len(inputs["orientations"]),
                "precision_profiles": inputs["precision_profiles"],
            }
        )
    )


if __name__ == "__main__":
    main()
