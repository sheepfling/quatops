"""Capture selected native OpenCV Rodrigues and matrix-action witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_INPUT = ROOT / "tests/golden/oracles/opencv/4.13.0/converter/inputs.json"
DEFAULT_OUTPUT = ROOT / "tests/golden/oracles/opencv/4.13.0/converter"
PROFILE_DTYPES = {"float32": np.float32, "float64": np.float64}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_action(matrix: np.ndarray, vector: np.ndarray) -> list[float]:
    vector_column = np.asarray(vector, dtype=matrix.dtype).reshape(3, 1)
    result = cv2.gemm(matrix, vector_column, 1.0, None, 0.0)
    return result.reshape(3).astype(np.float64).tolist()


def capture_matrix(
    case_id: str, dtype_label: str, rotation_vector: np.ndarray, probes: np.ndarray
) -> dict:
    rotvec = np.asarray(rotation_vector, dtype=PROFILE_DTYPES[dtype_label]).reshape(
        3, 1
    )
    matrix, _ = cv2.Rodrigues(rotvec)
    matrix = np.asarray(matrix)
    reverse_rotvec, _ = cv2.Rodrigues(matrix)
    reverse_rotvec = np.asarray(reverse_rotvec)
    reconstructed, _ = cv2.Rodrigues(reverse_rotvec)
    return {
        "case_id": case_id,
        "dtype": dtype_label,
        "native_input_rotation_vector_xyz_radians": rotvec.reshape(3)
        .astype(np.float64)
        .tolist(),
        "native_rodrigues_matrix_rows": matrix.astype(np.float64).tolist(),
        "native_gemm_basis_and_probe_actions_xyz": [
            native_action(matrix, vector) for vector in probes
        ],
        "native_matrix_to_rotation_vector_xyz_radians": reverse_rotvec.reshape(3)
        .astype(np.float64)
        .tolist(),
        "native_reconstructed_matrix_rows": reconstructed.astype(np.float64).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    input_path = options.inputs.resolve()
    inputs_bytes = input_path.read_bytes()
    inputs = json.loads(inputs_bytes.decode("utf-8"))
    probes = np.vstack(
        (np.eye(3), np.asarray(inputs["probe_vector_xyz"], dtype=np.float64))
    )
    by_id = {row["case_id"]: row for row in inputs["orientations"]}

    observations = {"canonical_to_native": [], "native_created": [], "composition": []}
    for dtype_label in inputs["precision_profiles"]:
        dtype = PROFILE_DTYPES[dtype_label]
        for row in inputs["orientations"]:
            observations["canonical_to_native"].append(
                capture_matrix(
                    row["case_id"],
                    dtype_label,
                    np.asarray(row["quatops_rotation_vector_xyz_radians"]),
                    probes,
                )
            )

        native_axis = inputs["native_constructed_orientation"]
        native_rotvec = (
            np.asarray(native_axis["axis_xyz"], dtype=np.float64)
            * native_axis["angle_radians"]
        )
        observations["native_created"].append(
            capture_matrix(native_axis["case_id"], dtype_label, native_rotvec, probes)
            | {
                "constructor_input_axis_xyz": native_axis["axis_xyz"],
                "constructor_input_angle_radians": native_axis["angle_radians"],
            }
        )

        rotations = {}
        for case_id in (
            inputs["composition"]["left_case_id"],
            inputs["composition"]["right_case_id"],
        ):
            rotvec = np.asarray(
                by_id[case_id]["quatops_rotation_vector_xyz_radians"], dtype=dtype
            ).reshape(3, 1)
            rotations[case_id] = np.asarray(cv2.Rodrigues(rotvec)[0])
        left_id = inputs["composition"]["left_case_id"]
        right_id = inputs["composition"]["right_case_id"]
        for case_id, lhs_id, rhs_id in (
            (f"{left_id}_times_{right_id}", left_id, right_id),
            (f"{right_id}_times_{left_id}", right_id, left_id),
        ):
            product = cv2.gemm(rotations[lhs_id], rotations[rhs_id], 1.0, None, 0.0)
            reverse_rotvec, _ = cv2.Rodrigues(product)
            reconstructed, _ = cv2.Rodrigues(reverse_rotvec)
            observations["composition"].append(
                {
                    "case_id": case_id,
                    "dtype": dtype_label,
                    "native_lhs_case_id": lhs_id,
                    "native_rhs_case_id": rhs_id,
                    "product_order": "lhs matrix * rhs matrix; rhs acts first",
                    "native_gemm_product_matrix_rows": np.asarray(
                        product, dtype=np.float64
                    ).tolist(),
                    "native_gemm_basis_and_probe_actions_xyz": [
                        native_action(product, vector) for vector in probes
                    ],
                    "native_matrix_to_rotation_vector_xyz_radians": np.asarray(
                        reverse_rotvec
                    )
                    .reshape(3)
                    .astype(np.float64)
                    .tolist(),
                    "native_reconstructed_matrix_rows": np.asarray(
                        reconstructed, dtype=np.float64
                    ).tolist(),
                }
            )

    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    witnesses_path = output / "witnesses.json"
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_runtime_selected_converter_witnesses",
        "target": {
            "name": "OpenCV",
            "version": cv2.__version__,
            "distribution_version": version("opencv-python-headless"),
        },
        "observations": observations,
    }
    witnesses_path.write_text(
        json.dumps(witnesses, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    cv2_path = Path(cv2.__file__).resolve().parent
    native_binaries = [
        {"path": path.relative_to(cv2_path).as_posix(), "sha256": sha256_file(path)}
        for pattern in ("*.pyd", "*.dll")
        for path in sorted(cv2_path.rglob(pattern))
    ]
    if not native_binaries:
        raise RuntimeError(f"No OpenCV native extension or DLL found under {cv2_path}")
    manifest = {
        "schema_version": "1.0",
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
            "orientation_cases_per_precision": len(inputs["orientations"]),
            "native_created_orientations_per_precision": 1,
            "noncommuting_product_orders_per_precision": 2,
            "native_actions_per_orientation": 4,
        },
        "native_binaries": native_binaries,
        "provenance": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "converter_input_sha256": sha256_file(input_path),
            "preparation_script_sha256": sha256_file(HERE / "prepare_inputs.py"),
            "preparation_spec_sha256": sha256_file(HERE / "converter_inputs.json"),
            "shared_canonical_inputs_sha256": inputs["canonical_input_sha256"],
            "witnesses_sha256": sha256_file(witnesses_path),
            "requirements_sha256": sha256_file(HERE / "requirements-converter.txt"),
            "cv2_extension_directory": "site-packages/cv2",
            "invocation_template": (
                "python experiments/converters/opencv/4.13.0/capture_native.py "
                "--inputs tests/golden/oracles/opencv/4.13.0/converter/inputs.json"
            ),
        },
        "reproduction": {
            "capture_script_sha256": sha256_file(Path(__file__)),
            "preparation_script_sha256": sha256_file(HERE / "prepare_inputs.py"),
            "requirements_sha256": sha256_file(HERE / "requirements-converter.txt"),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
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
