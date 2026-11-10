"""Capture selected MuJoCo quaternion-boundary witnesses; imports no QuatOps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import resources
from importlib.metadata import version
from pathlib import Path

# Installed by this target's locked project; the root typecheck uses its stub.
import mujoco  # pyright: ignore[reportMissingModuleSource]
import numpy as np

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_schema.converter_captures import (
    MuJoCoCapture,
    MuJoCoInputs,
    MuJoCoObservations,
    MuJoCoOrientation,
    TargetIdentity,
)
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from scripts.evidence_schema.shared_inputs import SharedCanonicalInputCapture


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_action(vector: np.ndarray, q: np.ndarray) -> np.ndarray:
    output = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(output, vector, q)
    return output


def capture_quaternion(
    case_id: str,
    q: np.ndarray,
    probe: np.ndarray,
    *,
    native_constructor: str | None = None,
    lhs: np.ndarray | None = None,
    rhs: np.ndarray | None = None,
) -> MuJoCoOrientation:
    result: dict[str, object] = {
        "case_id": case_id,
        "native_input_wxyz": q.tolist(),
        "native_mju_rotVecQuat_basis_xyz": [
            native_action(vector, q).tolist() for vector in np.eye(3)
        ],
    }
    if case_id == "mixed_a":
        result["native_mju_rotVecQuat_probe_xyz"] = native_action(probe, q).tolist()
    if native_constructor is not None:
        result["native_constructor"] = native_constructor
    if lhs is not None and rhs is not None:
        result["native_lhs_wxyz"] = lhs.tolist()
        result["native_rhs_wxyz"] = rhs.tolist()
        result["product_order"] = "lhs * rhs; rhs acts first"
    return MuJoCoOrientation.model_validate(result, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    target_dir = repository_root / "experiments/converters/mujoco/3.17.0"
    shared_lock = repository_root / "experiments/converters/python/uv.lock"
    input_path = (options.inputs or target_dir / "converter_inputs.json").resolve()
    output_dir = options.output_dir or (
        repository_root / "tests/golden/oracles/mujoco/3.17.0/converter"
    )
    spec_bytes = input_path.read_bytes()
    spec = MuJoCoInputs.model_validate_json(spec_bytes, strict=True)
    shared_path = repository_root / spec.shared_canonical_inputs
    shared_bytes = shared_path.read_bytes()
    shared = SharedCanonicalInputCapture.model_validate_json(shared_bytes, strict=True)
    by_id = {row.case_id: row for row in shared.orientations}
    selected = [by_id[case_id] for case_id in spec.retained_case_ids]
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
    qnative = np.empty(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(
        qnative,
        np.asarray(ctor.axis_xyz, dtype=np.float64),
        ctor.angle_radians,
    )
    native_created = capture_quaternion(
        ctor.case_id or "native_axis_angle",
        qnative,
        probe,
        native_constructor="mju_axisAngle2Quat(axis, angle)",
    )

    pair = {
        row.case_id: np.asarray(row.canonical_wxyz, dtype=np.float64)
        for row in selected
    }
    a, b = pair["mixed_a"], pair["mixed_b"]

    def compose(case_id: str, lhs: np.ndarray, rhs: np.ndarray) -> MuJoCoOrientation:
        product = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(product, lhs, rhs)
        return capture_quaternion(case_id, product, probe, lhs=lhs, rhs=rhs)

    composition = [
        compose(case_id, lhs, rhs)
        for case_id, lhs, rhs in (
            ("mixed_a_times_mixed_b", a, b),
            ("mixed_b_times_mixed_a", b, a),
        )
    ]
    observations = MuJoCoObservations(
        canonical_to_native=canonical_to_native,
        native_created=[native_created],
        composition=composition,
    )

    package_dir = Path(str(resources.files("mujoco")))
    native_library = package_dir / "mujoco.dll"
    if not native_library.is_file():
        raise FileNotFoundError(f"MuJoCo native DLL not found: {native_library}")
    native_files = [{"path": "mujoco.dll", "sha256": sha256_file(native_library)}]
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    witnesses_path = out / "witnesses.json"
    witnesses = MuJoCoCapture(
        schema_version="1.0",
        evidence_kind="native_runtime_selected_converter_witnesses",
        target=TargetIdentity(
            name="MuJoCo",
            package_version=version("mujoco"),
            engine_version=mujoco.mj_versionString(),
        ),
        observations=observations,
    )
    witnesses_path.write_text(
        dump_evidence_json(witnesses.model_dump(mode="json", exclude_none=True)),
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
            "name": "MuJoCo",
            "package_version": version("mujoco"),
            "engine_version": mujoco.mj_versionString(),
            "numpy_version": np.__version__,
            "native_scalar_dtype": "float64",
        },
        "selection": {
            "retained_orientation_cases": len(canonical_to_native),
            "native_created_orientations": 1,
            "noncommuting_product_orders": 2,
            "native_basis_images_per_orientation": 3,
            "native_mixed_probe_case_id": "mixed_a",
        },
        "native_binary_files": native_files,
        "provenance": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "reproduction_capture_script_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(target_dir / "semantic_contract.json"),
            "converter_input_spec_sha256": sha256_file(input_path),
            "selected_inputs_sha256": sha256_file(shared_path),
            "witnesses_sha256": sha256_file(witnesses_path),
            "pinned_lockfile_sha256": sha256_file(shared_lock),
            "reproduction_lockfile_sha256": sha256_file(shared_lock),
            "exact_invocation": (
                "uv run --locked --project experiments/converters/python --extra mujoco "
                "--python 3.13.14 "
                "python experiments/converters/mujoco/3.17.0/capture_native.py "
                "--repository-root <repository>"
            ),
        },
    }
    (out / "manifest.json").write_text(
        dump_evidence_json(manifest),
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
