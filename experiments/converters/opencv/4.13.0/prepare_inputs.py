"""Prepare canonical-to-Rodrigues boundary values with QuatOps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from quatops.python import hamilton
from scripts.evidence_schema.primitives import SHA256, Quaternion4, Vector3
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    SharedFrameInput,
)
from scripts.evidence_schema.targets import NativeConstructedInput, OpenCVInputs


class PreparedOrientation(BaseModel):
    """One QuatOps source rotation supplied to native Rodrigues capture."""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str
    canonical_wxyz: Quaternion4
    quatops_rotation_vector_xyz_radians: Vector3


class PreparedComposition(BaseModel):
    """Noncommuting product order requested from the native matrix API."""

    model_config = ConfigDict(extra="forbid", strict=True)

    left_case_id: str
    right_case_id: str
    product_orders: list[str]


class PreparedOpenCVInputs(BaseModel):
    """Typed input document generated for the OpenCV capture environment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str
    contract_id: str
    canonical_input_sha256: SHA256
    preparation_spec_sha256: SHA256
    source_frame: SharedFrameInput
    probe_vector_xyz: Vector3
    precision_profiles: list[Literal["float32", "float64"]]
    orientations: list[PreparedOrientation]
    native_constructed_orientation: NativeConstructedInput
    composition: PreparedComposition


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    spec_path = (
        repository_root / "experiments/converters/opencv/4.13.0/converter_inputs.json"
    )
    spec = OpenCVInputs.model_validate_json(spec_path.read_bytes(), strict=True)
    shared_path = repository_root / spec.shared_canonical_inputs
    shared = SharedCanonicalInputCapture.model_validate_json(
        shared_path.read_bytes(), strict=True
    )
    by_id = {row.case_id: row for row in shared.orientations}
    orientations = [
        PreparedOrientation(
            case_id=case_id,
            canonical_wxyz=by_id[case_id].canonical_wxyz,
            quatops_rotation_vector_xyz_radians=list(
                hamilton.to_rotation_vector(by_id[case_id].canonical_wxyz)
            ),
        )
        for case_id in spec.retained_case_ids
    ]
    prepared = PreparedOpenCVInputs(
        schema_version="1.0",
        contract_id=spec.contract_id,
        canonical_input_sha256=sha256_file(shared_path),
        preparation_spec_sha256=sha256_file(spec_path),
        source_frame=shared.source_frame,
        probe_vector_xyz=shared.probe_vector_xyz,
        precision_profiles=spec.precision_profiles,
        orientations=orientations,
        native_constructed_orientation=spec.native_constructed_orientation,
        composition=PreparedComposition(
            left_case_id=shared.composition.left_case_id,
            right_case_id=shared.composition.right_case_id,
            product_orders=[
                "left * right; right acts first",
                "right * left; left acts first",
            ],
        ),
    )
    output_path = options.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(prepared.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "orientations": len(prepared.orientations),
                "precision_profiles": prepared.precision_profiles,
            }
        )
    )


if __name__ == "__main__":
    main()
