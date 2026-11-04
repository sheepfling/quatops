"""Capture selected CSPICE quaternion-boundary witnesses; imports no QuatOps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

# Installed by this target's locked project; the root typecheck uses its stub.
import spiceypy as spice  # pyright: ignore[reportMissingModuleSource]

from scripts.evidence_format import dump_evidence_json


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_quaternion(case_id: str, q: np.ndarray, probe: np.ndarray) -> dict:
    matrix = np.asarray(spice.q2m(q), dtype=np.float64)
    result = {
        "case_id": case_id,
        "native_input_wxyz": q.tolist(),
        "native_mxv_basis_xyz": [
            np.asarray(spice.mxv(matrix, vector), dtype=np.float64).tolist()
            for vector in np.eye(3)
        ],
    }
    if case_id == "mixed_a":
        result["native_mxv_probe_xyz"] = np.asarray(
            spice.mxv(matrix, probe), dtype=np.float64
        ).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "converter_inputs.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter",
    )
    options = parser.parse_args()
    input_path = options.inputs.resolve()
    spec_bytes = input_path.read_bytes()
    spec = json.loads(spec_bytes.decode("utf-8"))
    shared_path = ROOT / spec["shared_canonical_inputs"]
    shared_bytes = shared_path.read_bytes()
    shared = json.loads(shared_bytes.decode("utf-8"))
    by_id = {row["case_id"]: row for row in shared["orientations"]}
    selected = [by_id[case_id] for case_id in spec["retained_case_ids"]]
    probe = np.asarray(shared["probe_vector_xyz"], dtype=np.float64)

    observations = {
        "canonical_to_native": [
            capture_quaternion(
                row["case_id"],
                np.asarray(row["canonical_wxyz"], dtype=np.float64),
                probe,
            )
            for row in selected
        ],
        "native_created": [],
        "composition": [],
    }
    ctor = spec["native_constructed_orientation"]
    axis = np.asarray(ctor["axis_xyz"], dtype=np.float64)
    matrix = np.asarray(
        spice.axisar(axis, float(ctor["angle_radians"])), dtype=np.float64
    )
    qnative = np.asarray(spice.m2q(matrix), dtype=np.float64)
    q2m = np.asarray(spice.q2m(qnative), dtype=np.float64)
    observations["native_created"].append(
        {
            "case_id": ctor["case_id"],
            "native_constructor": "axisar(axis, angle) -> m2q(matrix)",
            "native_m2q_wxyz": qnative.tolist(),
            "native_mxv_basis_xyz": [
                np.asarray(spice.mxv(q2m, vector), dtype=np.float64).tolist()
                for vector in np.eye(3)
            ],
        }
    )

    pair = {
        row["case_id"]: np.asarray(row["canonical_wxyz"], dtype=np.float64)
        for row in selected
    }
    a, b = pair["mixed_a"], pair["mixed_b"]
    for case_id, lhs, rhs in (
        ("mixed_a_times_mixed_b", a, b),
        ("mixed_b_times_mixed_a", b, a),
    ):
        product = np.asarray(spice.qxq(lhs, rhs), dtype=np.float64)
        observations["composition"].append(
            capture_quaternion(case_id, product, probe)
            | {
                "native_lhs_wxyz": lhs.tolist(),
                "native_rhs_wxyz": rhs.tolist(),
                "product_order": "lhs * rhs; rhs acts first",
            }
        )

    library_path = Path(spice.__file__).resolve().parent / "utils" / "cspice.dll"
    if not library_path.exists():
        raise FileNotFoundError(library_path)
    out = options.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    witnesses_path = out / "witnesses.json"
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_runtime_selected_converter_witnesses",
        "target": {
            "name": "SpiceyPy / CSPICE",
            "wrapper_version": spice.__version__,
            "toolkit_version": spice.tkvrsn("TOOLKIT"),
        },
        "observations": observations,
    }
    witnesses_path.write_text(
        dump_evidence_json(witnesses), encoding="utf-8", newline="\n"
    )
    manifest = {
        "schema_version": "1.0",
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
            "retained_orientation_cases": len(selected),
            "native_created_orientations": 1,
            "noncommuting_product_orders": 2,
            "native_basis_images_per_orientation": 3,
            "native_mixed_probe_case_id": "mixed_a",
        },
        "provenance": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "reproduction_capture_script_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(out / "contract.json"),
            "converter_input_spec_sha256": sha256_file(input_path),
            "selected_inputs_sha256": sha256_file(shared_path),
            "witnesses_sha256": sha256_file(witnesses_path),
            "pinned_lockfile_sha256": sha256_file(HERE / "uv.lock"),
            "loaded_cspice_binary_path": str(library_path),
            "loaded_cspice_binary_sha256": sha256_file(library_path),
            "exact_invocation": (
                "uv run --locked --project experiments/converters/spiceypy/8.3.0 "
                "--python 3.10.11 "
                "python experiments/converters/spiceypy/8.3.0/capture_native.py"
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
