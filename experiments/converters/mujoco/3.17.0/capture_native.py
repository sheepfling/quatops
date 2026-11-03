"""Capture selected MuJoCo quaternion-boundary witnesses; imports no QuatOps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORACLE = ROOT / "tests/golden/oracles/mujoco/3.17.0"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_matrix(q: np.ndarray) -> np.ndarray:
    output = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(output, q)
    return output.reshape(3, 3)


def native_action(vector: np.ndarray, q: np.ndarray) -> np.ndarray:
    output = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(output, vector, q)
    return output


def capture_quaternion(case_id: str, q: np.ndarray, probe: np.ndarray) -> dict:
    matrix = native_matrix(q)
    probes = np.vstack((np.eye(3), probe))
    recovered = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(recovered, matrix.reshape(9))
    return {
        "case_id": case_id,
        "native_input_wxyz": q.tolist(),
        "native_mju_quat2Mat_rows": matrix.tolist(),
        "native_mju_rotVecQuat_basis_and_probe_xyz": [
            native_action(vector, q).tolist() for vector in probes
        ],
        "native_mju_mat2Quat_roundtrip_wxyz": recovered.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "converter_inputs.json")
    parser.add_argument("--output-dir", type=Path, default=ORACLE / "converter")
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
    qnative = np.empty(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(
        qnative,
        np.asarray(ctor["axis_xyz"], dtype=np.float64),
        float(ctor["angle_radians"]),
    )
    native_created = capture_quaternion(ctor["case_id"], qnative, probe)
    native_created["native_constructor"] = "mju_axisAngle2Quat(axis, angle)"
    native_created["native_mju_mat2Quat_from_constructor_matrix_wxyz"] = native_created[
        "native_mju_mat2Quat_roundtrip_wxyz"
    ]
    observations["native_created"].append(native_created)

    pair = {
        row["case_id"]: np.asarray(row["canonical_wxyz"], dtype=np.float64)
        for row in selected
    }
    a, b = pair["mixed_a"], pair["mixed_b"]
    for case_id, lhs, rhs in (
        ("mixed_a_times_mixed_b", a, b),
        ("mixed_b_times_mixed_a", b, a),
    ):
        product = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(product, lhs, rhs)
        observations["composition"].append(
            capture_quaternion(case_id, product, probe)
            | {
                "native_lhs_wxyz": lhs.tolist(),
                "native_rhs_wxyz": rhs.tolist(),
                "product_order": "lhs * rhs; rhs acts first",
            }
        )

    package_dir = Path(mujoco.__file__).resolve().parent
    native_files = []
    for pattern in ("*.dll", "*.pyd"):
        native_files.extend(
            {
                "path": candidate.relative_to(package_dir).as_posix(),
                "sha256": sha256_file(candidate),
            }
            for candidate in sorted(package_dir.rglob(pattern))
        )
    if not native_files:
        raise RuntimeError(f"No MuJoCo native DLL found under {package_dir}")
    out = options.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs.json").write_bytes(spec_bytes)
    witnesses_path = out / "witnesses.json"
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_runtime_selected_converter_witnesses",
        "target": {
            "name": "MuJoCo",
            "package_version": version("mujoco"),
            "engine_version": mujoco.mj_versionString(),
        },
        "observations": observations,
    }
    witnesses_path.write_text(
        json.dumps(witnesses, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
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
            "name": "MuJoCo",
            "package_version": version("mujoco"),
            "engine_version": mujoco.mj_versionString(),
            "numpy_version": np.__version__,
            "native_scalar_dtype": "float64",
        },
        "selection": {
            "retained_orientation_cases": len(selected),
            "native_created_orientations": 1,
            "noncommuting_product_orders": 2,
            "native_quaternion_actions_per_orientation": 4,
        },
        "native_binary_files": native_files,
        "provenance": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(out / "contract.json"),
            "converter_input_spec_sha256": sha256_file(input_path),
            "selected_inputs_sha256": sha256_file(shared_path),
            "witnesses_sha256": sha256_file(witnesses_path),
            "full_oracle_sha256": sha256_file(ORACLE / "native_rotation.json"),
            "pinned_lockfile_sha256": sha256_file(
                ROOT / "experiments/oracles/mujoco/3.17.0/uv.lock"
            ),
            "exact_invocation": (
                "experiments/oracles/mujoco/3.17.0/.venv/Scripts/python.exe "
                "experiments/converters/mujoco/3.17.0/capture_native.py"
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
                "engine_version": manifest["target"]["engine_version"],
                "selection": manifest["selection"],
            }
        )
    )


if __name__ == "__main__":
    main()
