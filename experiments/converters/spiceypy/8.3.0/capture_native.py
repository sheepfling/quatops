"""Capture selected CSPICE quaternion-boundary witnesses; imports no QuatOps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

# Installed by this target's locked project; the root typecheck uses its stub.
import numpy as np
import spiceypy as spice  # pyright: ignore[reportMissingModuleSource]

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import (
    SPICEYPY_CONSTRUCTOR_CASE_ID,
    SpiceyPyCapture,
    SpiceyPyConstructedOrientation,
    SpiceyPyInputs,
    SpiceyPyObservations,
    SpiceyPyOrientation,
    TargetIdentity,
)
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    resolve_orientation_selection,
)
from scripts.evidence_schema.spiceypy_capture import (
    validate_spiceypy_capture_matches_inputs,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_quaternion(
    case_id: str, q: np.ndarray, probe: np.ndarray
) -> SpiceyPyOrientation:
    matrix = np.asarray(spice.q2m(q), dtype=np.float64)
    probe_action = (
        np.asarray(spice.mxv(matrix, probe), dtype=np.float64).tolist()
        if case_id == "mixed_a"
        else None
    )
    return SpiceyPyOrientation(
        case_id=case_id,
        native_input_wxyz=q.tolist(),
        native_mxv_basis_xyz=[
            np.asarray(spice.mxv(matrix, vector), dtype=np.float64).tolist()
            for vector in np.eye(3)
        ],
        native_mxv_probe_xyz=probe_action,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    target_dir = repository_root / "experiments/converters/spiceypy/8.3.0"
    shared_lock = repository_root / "experiments/converters/python/uv.lock"
    input_path = (options.inputs or target_dir / "converter_inputs.json").resolve()
    output_dir = options.output_dir or (
        repository_root / "tests/golden/oracles/spiceypy/8.3.0/converter"
    )
    spec = SpiceyPyInputs.model_validate(
        read_evidence_document(input_path), strict=True
    )
    shared_path = repository_root / spec.shared_canonical_inputs
    shared = SharedCanonicalInputCapture.model_validate(
        read_evidence_document(shared_path), strict=True
    )
    selected = resolve_orientation_selection(
        shared,
        spec.retained_case_ids,
        required_case_ids={"mixed_a", "mixed_b"},
    )
    probe = np.asarray(shared.probe_vector_xyz, dtype=np.float64)

    canonical_to_native = [
        capture_quaternion(
            row.case_id,
            np.asarray(row.canonical_wxyz, dtype=np.float64),
            probe,
        )
        for row in selected
        if row.case_id in NATIVE_RETAINED_CASE_IDS
    ]
    ctor = spec.native_constructed_orientation
    axis = np.asarray(ctor.axis_xyz, dtype=np.float64)
    matrix = np.asarray(spice.axisar(axis, float(ctor.angle_radians)), dtype=np.float64)
    qnative = np.asarray(spice.m2q(matrix), dtype=np.float64)
    q2m = np.asarray(spice.q2m(qnative), dtype=np.float64)
    native_created = [
        SpiceyPyConstructedOrientation(
            case_id=SPICEYPY_CONSTRUCTOR_CASE_ID,
            native_constructor="axisar(axis, angle) -> m2q(matrix)",
            native_m2q_wxyz=qnative.tolist(),
            native_mxv_basis_xyz=[
                np.asarray(spice.mxv(q2m, vector), dtype=np.float64).tolist()
                for vector in np.eye(3)
            ],
        )
    ]

    pair = {
        row.case_id: np.asarray(row.canonical_wxyz, dtype=np.float64)
        for row in selected
    }
    a, b = pair["mixed_a"], pair["mixed_b"]
    composition: list[SpiceyPyOrientation] = []
    for case_id, lhs, rhs in (
        ("mixed_a_times_mixed_b", a, b),
        ("mixed_b_times_mixed_a", b, a),
    ):
        product = np.asarray(spice.qxq(lhs, rhs), dtype=np.float64)
        orientation = capture_quaternion(case_id, product, probe)
        composition.append(
            SpiceyPyOrientation(
                case_id=orientation.case_id,
                native_input_wxyz=orientation.native_input_wxyz,
                native_mxv_basis_xyz=orientation.native_mxv_basis_xyz,
                native_mxv_probe_xyz=orientation.native_mxv_probe_xyz,
                native_lhs_wxyz=lhs.tolist(),
                native_rhs_wxyz=rhs.tolist(),
                product_order="lhs * rhs; rhs acts first",
            )
        )

    library_path = Path(str(resources.files("spiceypy"))) / "utils" / "cspice.dll"
    if not library_path.exists():
        raise FileNotFoundError(library_path)
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    witnesses_path = out / "witnesses.json"
    checked_witnesses = SpiceyPyCapture(
        schema_version="1.0",
        evidence_kind="native_runtime_selected_converter_witnesses",
        target=TargetIdentity(
            name="SpiceyPy / CSPICE",
            wrapper_version=spice.__version__,
            toolkit_version=spice.tkvrsn("TOOLKIT"),
        ),
        observations=SpiceyPyObservations(
            canonical_to_native=canonical_to_native,
            native_created=native_created,
            composition=composition,
        ),
    )
    validate_spiceypy_capture_matches_inputs(checked_witnesses, spec, shared)
    witnesses_path.write_text(
        dump_evidence_json(
            checked_witnesses.model_dump(mode="json", exclude_none=True)
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "1.0",
        "semantic_contract": {
            "file": (target_dir / "semantic_contract.json")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha256_file(target_dir / "semantic_contract.json"),
        },
        "review_state": "Pinned native capture only; replay acceptance is recorded by repository tests",
        "execution": {
            "was_run": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "target": {
            "name": "SpiceyPy / CSPICE",
            "wrapper_version": spice.__version__,
            "toolkit_version": spice.tkvrsn("TOOLKIT"),
            "native_scalar_dtype": "float64",
        },
        "selection": {
            "retained_orientation_cases": len(canonical_to_native),
            "native_created_orientations": 1,
            "noncommuting_product_orders": 2,
            "native_basis_images_per_orientation": 3,
            "native_mixed_probe_case_id": "mixed_a",
        },
        "provenance": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "reproduction_capture_script_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(target_dir / "semantic_contract.json"),
            "converter_input_spec_sha256": sha256_file(input_path),
            "selected_inputs_sha256": sha256_file(shared_path),
            "witnesses_sha256": sha256_file(witnesses_path),
            "pinned_lockfile_sha256": sha256_file(shared_lock),
            "reproduction_lockfile_sha256": sha256_file(shared_lock),
            "loaded_cspice_binary_path": str(library_path),
            "loaded_cspice_binary_sha256": sha256_file(library_path),
            "exact_invocation": (
                "uv run --locked --project experiments/converters/python --extra spiceypy "
                "--python 3.13.14 "
                "python experiments/converters/spiceypy/8.3.0/capture_native.py"
                " --repository-root <repository>"
            ),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "toolkit_version": manifest["target"]["toolkit_version"],
                "selection": manifest["selection"],
            }
        )
    )


if __name__ == "__main__":
    main()
