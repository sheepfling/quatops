"""Compare the frozen native Sophus SO(3) output with separate SciPy formulas."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
from pydantic import JsonValue, RootModel
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
_TOLERANCE = 2e-14


class NativeCapture(RootModel[dict[str, JsonValue]]):
    """Pydantic-validated root of the frozen native capture JSON."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation_matrix(item: dict[str, Any]) -> np.ndarray:
    return np.asarray(item["matrix_rows"], dtype=np.float64)


def _native_rotvec(item: dict[str, Any]) -> np.ndarray:
    return np.asarray(item["log_of_exp_xyz"], dtype=np.float64)


def _quaternion_xyzw(item: dict[str, Any]) -> np.ndarray:
    values = item["unit_quaternion"]["coeffs_xyzw"]
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    input_path = options.capture.resolve()
    parsed = NativeCapture.model_validate_json(input_path.read_bytes(), strict=True)
    capture = cast(dict[str, Any], parsed.root)
    maxima = {
        "quaternion_construction_matrix": 0.0,
        "quaternion_construction_action": 0.0,
        "quaternion_storage_xyzw": 0.0,
        "quaternion_log_rotation": 0.0,
        "quaternion_log_rotvec": 0.0,
        "exp_matrix": 0.0,
        "log_rotvec": 0.0,
        "composition_matrix": 0.0,
        "inverse_matrix": 0.0,
        "perturbation_matrix": 0.0,
    }
    probes = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], capture["probe_xyz"]],
        dtype=np.float64,
    )

    for item in capture["quaternion_construction"]:
        raw_wxyz = np.asarray(item["input_wxyz"], dtype=np.float64)
        expected = Rotation.from_quat(raw_wxyz[[1, 2, 3, 0]])
        native = item["result"]
        matrix = _rotation_matrix(native)
        actions = np.asarray(native["native_vector_actions_xyz"], dtype=np.float64)
        maxima["quaternion_construction_matrix"] = max(
            maxima["quaternion_construction_matrix"],
            float(np.max(np.abs(matrix - expected.as_matrix()))),
        )
        maxima["quaternion_construction_action"] = max(
            maxima["quaternion_construction_action"],
            float(np.max(np.abs(actions - expected.apply(probes)))),
        )
        maxima["quaternion_storage_xyzw"] = max(
            maxima["quaternion_storage_xyzw"],
            float(np.max(np.abs(_quaternion_xyzw(native) - native["params_xyzw"]))),
        )
        native_log = np.asarray(item["log_xyz"], dtype=np.float64)
        expected_log = expected.as_rotvec()
        maxima["quaternion_log_rotation"] = max(
            maxima["quaternion_log_rotation"],
            float(
                np.max(
                    np.abs(
                        Rotation.from_rotvec(native_log).as_matrix()
                        - expected.as_matrix()
                    )
                )
            ),
        )
        if item["case_id"] != "half_turn_axis":
            maxima["quaternion_log_rotvec"] = max(
                maxima["quaternion_log_rotvec"],
                float(np.max(np.abs(native_log - expected_log))),
            )

    rotations_by_tangent: dict[str, Rotation] = {}
    for item in capture["exp_log"]:
        tangent = np.asarray(item["tangent_xyz"], dtype=np.float64)
        expected = Rotation.from_rotvec(tangent)
        native = item["exp_result"]
        maxima["exp_matrix"] = max(
            maxima["exp_matrix"],
            float(np.max(np.abs(_rotation_matrix(native) - expected.as_matrix()))),
        )
        maxima["log_rotvec"] = max(
            maxima["log_rotvec"],
            float(np.max(np.abs(_native_rotvec(item) - expected.as_rotvec()))),
        )
        rotations_by_tangent[item["case_id"]] = expected

    composition_cases = (
        (
            "mixed_a_times_mixed_b",
            rotations_by_tangent["mixed_a"] * rotations_by_tangent["mixed_b"],
        ),
        (
            "mixed_b_times_mixed_a",
            rotations_by_tangent["mixed_b"] * rotations_by_tangent["mixed_a"],
        ),
    )
    for item, (case_id, expected) in zip(capture["composition"], composition_cases):
        if item["case_id"] != case_id:
            raise ValueError(
                "native composition case order does not match the input contract"
            )
        maxima["composition_matrix"] = max(
            maxima["composition_matrix"],
            float(np.max(np.abs(_rotation_matrix(item) - expected.as_matrix()))),
        )

    expected_inverse = rotations_by_tangent["mixed_a"].inv().as_matrix()
    maxima["inverse_matrix"] = float(
        np.max(np.abs(_rotation_matrix(capture["inverse"]) - expected_inverse))
    )
    delta = rotations_by_tangent["perturbation"]
    base = rotations_by_tangent["mixed_a"]
    expected_perturbations = (delta * base, base * delta)
    for item, expected in zip(capture["perturbation_side"], expected_perturbations):
        maxima["perturbation_matrix"] = max(
            maxima["perturbation_matrix"],
            float(np.max(np.abs(_rotation_matrix(item) - expected.as_matrix()))),
        )

    if any(error > _TOLERANCE for error in maxima.values()):
        raise AssertionError(f"SciPy formula differences exceed {_TOLERANCE}: {maxima}")

    review = {
        "schema_version": "1.0",
        "evidence_kind": "derived_reference",
        "native_capture": input_path.name,
        "native_capture_sha256": _sha256(input_path),
        "reference_script": "verify_scipy.py",
        "reference_script_sha256": _sha256(Path(__file__)),
        "quatops_executed": False,
        "reference": {"library": "SciPy Rotation", "version": scipy.__version__},
        "checks": [
            "SO3 constructor normalization, matrix action, XYZW storage, and quaternion log",
            "SO3 exp/log including near-zero and greater-than-pi inputs",
            "noncommuting product orders, inverse, and left/right perturbations",
        ],
        "absolute_tolerance": _TOLERANCE,
        "max_abs_errors": maxima,
        "result": "pass",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if options.output is not None:
        output_path = options.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(review, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(review, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
