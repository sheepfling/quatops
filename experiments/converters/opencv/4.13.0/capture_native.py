"""Capture selected native OpenCV Rodrigues and matrix-action witnesses."""

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

import cv2
import numpy as np
from prepare_inputs import PreparedOpenCVInputs

from scripts.evidence_json import dump_evidence_json
from scripts.evidence_schema.converter_captures import (
    OpenCVCapture,
    OpenCVComposition,
    OpenCVObservations,
    OpenCVOrientation,
    TargetIdentity,
)

PROFILE_DTYPES = {"float32": np.float32, "float64": np.float64}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_action(matrix: np.ndarray, vector: np.ndarray) -> list[float]:
    vector_column = np.asarray(vector, dtype=matrix.dtype).reshape(3, 1)
    result = cv2.gemm(matrix, vector_column, 1.0, None, 0.0)
    return result.reshape(3).astype(np.float64).tolist()


def capture_matrix(
    case_id: str, dtype_label: str, rotation_vector: np.ndarray, probe: np.ndarray
) -> OpenCVOrientation:
    rotvec = np.asarray(rotation_vector, dtype=PROFILE_DTYPES[dtype_label]).reshape(
        3, 1
    )
    matrix, _ = cv2.Rodrigues(rotvec)
    matrix = np.asarray(matrix)
    reverse_rotvec, _ = cv2.Rodrigues(matrix)
    reverse_rotvec = np.asarray(reverse_rotvec)
    result = {
        "case_id": case_id,
        "dtype": dtype_label,
        "native_input_rotation_vector_xyz_radians": rotvec.reshape(3)
        .astype(np.float64)
        .tolist(),
        "native_gemm_basis_actions_xyz": [
            native_action(matrix, vector) for vector in np.eye(3)
        ],
        "native_matrix_to_rotation_vector_xyz_radians": reverse_rotvec.reshape(3)
        .astype(np.float64)
        .tolist(),
    }
    if case_id == "mixed_a":
        result["native_gemm_probe_action_xyz"] = native_action(matrix, probe)
    return OpenCVOrientation.model_validate(result, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    target_dir = repository_root / "experiments/converters/opencv/4.13.0"
    output_dir = options.output_dir or (
        repository_root / "tests/golden/oracles/opencv/4.13.0/converter"
    )
    input_path = options.inputs.resolve()
    inputs_bytes = input_path.read_bytes()
    inputs = PreparedOpenCVInputs.model_validate_json(inputs_bytes, strict=True)
    probe = np.asarray(inputs.probe_vector_xyz, dtype=np.float64)
    by_id = {row.case_id: row for row in inputs.orientations}

    canonical_to_native: list[OpenCVOrientation] = []
    native_created: list[OpenCVOrientation] = []
    composition: list[OpenCVComposition] = []
    for dtype_label in inputs.precision_profiles:
        dtype = PROFILE_DTYPES[dtype_label]
        canonical_to_native.extend(
            capture_matrix(
                row.case_id,
                dtype_label,
                np.asarray(row.quatops_rotation_vector_xyz_radians),
                probe,
            )
            for row in inputs.orientations
        )

        native_axis = inputs.native_constructed_orientation
        native_rotvec = (
            np.asarray(native_axis.axis_xyz, dtype=np.float64)
            * native_axis.angle_radians
        )
        native_created.append(
            capture_matrix(
                native_axis.case_id or "native_axis_angle",
                dtype_label,
                native_rotvec,
                probe,
            )
        )

        rotations = {}
        for case_id in (
            inputs.composition.left_case_id,
            inputs.composition.right_case_id,
        ):
            rotvec = np.asarray(
                by_id[case_id].quatops_rotation_vector_xyz_radians, dtype=dtype
            ).reshape(3, 1)
            rotations[case_id] = np.asarray(cv2.Rodrigues(rotvec)[0])
        left_id = inputs.composition.left_case_id
        right_id = inputs.composition.right_case_id
        for case_id, lhs_id, rhs_id in (
            (f"{left_id}_times_{right_id}", left_id, right_id),
            (f"{right_id}_times_{left_id}", right_id, left_id),
        ):
            product = cv2.gemm(rotations[lhs_id], rotations[rhs_id], 1.0, None, 0.0)
            reverse_rotvec, _ = cv2.Rodrigues(product)
            composition.append(
                OpenCVComposition.model_validate(
                    {
                        "case_id": case_id,
                        "dtype": dtype_label,
                        "native_lhs_case_id": lhs_id,
                        "native_rhs_case_id": rhs_id,
                        "product_order": "lhs matrix * rhs matrix; rhs acts first",
                        "native_gemm_basis_actions_xyz": [
                            native_action(product, vector) for vector in np.eye(3)
                        ],
                        "native_matrix_to_rotation_vector_xyz_radians": np.asarray(
                            reverse_rotvec
                        )
                        .reshape(3)
                        .astype(np.float64)
                        .tolist(),
                    },
                    strict=True,
                )
            )

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    semantic_contract_path = target_dir / "semantic_contract.json"
    (output / "contract.json").write_bytes(semantic_contract_path.read_bytes())
    witnesses_path = output / "witnesses.json"
    witnesses = OpenCVCapture(
        schema_version="1.0",
        evidence_kind="native_runtime_selected_converter_witnesses",
        target=TargetIdentity(
            name="OpenCV",
            version=cv2.__version__,
            distribution_version=version("opencv-python-headless"),
        ),
        observations=OpenCVObservations(
            canonical_to_native=canonical_to_native,
            native_created=native_created,
            composition=composition,
        ),
    )
    witnesses_path.write_text(
        dump_evidence_json(witnesses.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
        newline="\n",
    )
    cv2_path = Path(str(resources.files("cv2")))
    native_binaries = [
        {"path": path.relative_to(cv2_path).as_posix(), "sha256": sha256_file(path)}
        for pattern in ("*.pyd", "*.dll")
        for path in sorted(cv2_path.rglob(pattern))
    ]
    if not native_binaries:
        raise RuntimeError(f"No OpenCV native extension or DLL found under {cv2_path}")
    manifest = {
        "schema_version": "1.0",
        "semantic_contract": {
            "file": (target_dir / "semantic_contract.json")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha256_file(target_dir / "semantic_contract.json"),
        },
        "review_state": "Pinned native OpenCV converter capture; replay acceptance is recorded by repository tests",
        "execution": {
            "was_run": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "target": {
            "name": "OpenCV",
            "version": cv2.__version__,
            "distribution": "opencv-python-headless",
            "distribution_version": version("opencv-python-headless"),
            "numpy_version": np.__version__,
            "native_scalar_profiles": ["float32", "float64"],
        },
        "selection": {
            "orientation_cases_per_precision": len(inputs.orientations),
            "native_created_orientations_per_precision": 1,
            "noncommuting_product_orders_per_precision": 2,
            "native_basis_images_per_orientation": 3,
            "native_mixed_probe_case_id": "mixed_a",
        },
        "native_binaries": native_binaries,
        "provenance": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "prepared_inputs_sha256": sha256_file(input_path),
            "preparation_script_sha256": sha256_file(target_dir / "prepare_inputs.py"),
            "preparation_spec_sha256": sha256_file(
                target_dir / "converter_inputs.json"
            ),
            "shared_canonical_inputs_sha256": inputs.canonical_input_sha256,
            "witnesses_sha256": sha256_file(witnesses_path),
            "lockfile_sha256": sha256_file(target_dir / "uv.lock"),
            "cv2_extension_directory": "site-packages/cv2",
            "invocation_template": (
                "python experiments/converters/opencv/4.13.0/prepare_inputs.py "
                "--output <temporary inputs.json>; then run capture_native.py "
                "--inputs <temporary inputs.json>"
            ),
        },
        "reproduction": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "preparation_script_sha256": sha256_file(target_dir / "prepare_inputs.py"),
            "lockfile_sha256": sha256_file(target_dir / "uv.lock"),
        },
    }
    (output / "manifest.json").write_text(
        dump_evidence_json(manifest),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "opencv": cv2.__version__,
                "distribution": version("opencv-python-headless"),
                "selection": manifest["selection"],
            }
        )
    )


if __name__ == "__main__":
    main()
