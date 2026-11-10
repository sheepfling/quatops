"""Compare the frozen native Sophus SO(3) output with separate SciPy formulas."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from retain_witnesses import (
    ExpLog,
    NativeCapture,
    NativeRotation,
    load_semantic_config,
    profile_for_capture,
)
from scipy.spatial.transform import Rotation

from scripts.evidence_schema.primitives import finite_max_abs

FloatArray = NDArray[np.float64]
StrictFiniteFloat = Annotated[FiniteFloat, Field(strict=True)]


class MaxAbsErrors(BaseModel):
    """Named SciPy comparison maxima, serialized as a stable evidence object."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    quaternion_construction_matrix: StrictFiniteFloat = 0.0
    rotation_vector_action: StrictFiniteFloat = 0.0
    quaternion_construction_input_norm: StrictFiniteFloat = 0.0
    rotation_quaternion_xyzw: StrictFiniteFloat = 0.0
    quaternion_log_rotation: StrictFiniteFloat = 0.0
    quaternion_log_rotvec: StrictFiniteFloat = 0.0
    exp_matrix: StrictFiniteFloat = 0.0
    log_rotvec: StrictFiniteFloat = 0.0
    composition_matrix: StrictFiniteFloat = 0.0
    inverse_matrix: StrictFiniteFloat = 0.0
    perturbation_matrix: StrictFiniteFloat = 0.0

    def exceeds(self, tolerance: float) -> bool:
        """Return whether any named matrix/action comparison exceeds tolerance."""
        return (
            max(
                self.quaternion_construction_matrix,
                self.rotation_vector_action,
                self.quaternion_construction_input_norm,
                self.rotation_quaternion_xyzw,
                self.quaternion_log_rotation,
                self.quaternion_log_rotvec,
                self.exp_matrix,
                self.log_rotvec,
                self.composition_matrix,
                self.inverse_matrix,
                self.perturbation_matrix,
            )
            > tolerance
        )


class ReferenceIdentity(BaseModel):
    """Version of the independent reference implementation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    library: Literal["SciPy Rotation"]
    version: str


class SciPyReview(BaseModel):
    """Typed SciPy-only comparison report for one frozen native capture."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    evidence_kind: Literal["derived_reference"]
    native_capture: str
    native_capture_sha256: str
    reference_script: Literal["verify_scipy.py"]
    reference_script_sha256: str
    quatops_executed: Literal[False]
    reference: ReferenceIdentity
    native_target: str
    checks: tuple[str, ...]
    absolute_tolerance: StrictFiniteFloat
    max_abs_errors: MaxAbsErrors
    result: Literal["pass"]
    generated_utc: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation_matrix(item: NativeRotation) -> FloatArray:
    return np.asarray(item.matrix_rows, dtype=np.float64)


def _native_rotvec(item: ExpLog) -> FloatArray:
    return np.asarray(item.log_of_exp_xyz, dtype=np.float64)


def _quaternion_xyzw(item: NativeRotation) -> FloatArray:
    return np.asarray(item.unit_quaternion.coeffs_xyzw, dtype=np.float64)


def _max_difference(left: FloatArray, right: FloatArray, label: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.subtract(left, right)
    return finite_max_abs(difference.flat, label)


def _rotation_differences(
    item: NativeRotation, expected: Rotation, probes: FloatArray, label: str
) -> tuple[float, float, float]:
    """Check matrix, quaternion meaning, and all captured vector actions."""
    matrix_error = _max_difference(
        _rotation_matrix(item), expected.as_matrix(), f"{label} matrix error"
    )
    actions = np.asarray(item.native_vector_actions_xyz, dtype=np.float64)
    action_error = _max_difference(
        actions, expected.apply(probes), f"{label} action error"
    )
    quaternion = _quaternion_xyzw(item)
    expected_quaternion = expected.as_quat()
    positive_error = _max_difference(
        quaternion, expected_quaternion, f"{label} quaternion error"
    )
    negative_error = _max_difference(
        quaternion, -expected_quaternion, f"{label} antipodal quaternion error"
    )
    return matrix_error, action_error, min(positive_error, negative_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    input_path = options.capture.resolve()
    capture = NativeCapture.model_validate_json(input_path.read_bytes(), strict=True)
    config = load_semantic_config()
    profile = profile_for_capture(config, capture)
    capture.validate_pinned_inputs(Path(__file__).with_name("capture_inputs.tsv"))
    tolerance = profile.scipy_tolerance
    maxima = MaxAbsErrors()
    probes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            config.probe_xyz,
        ],
        dtype=np.float64,
    )

    for item in capture.quaternion_construction:
        raw_wxyz = np.asarray(item.input_wxyz, dtype=np.float64)
        maxima.quaternion_construction_input_norm = max(
            maxima.quaternion_construction_input_norm,
            _max_difference(
                np.asarray([item.input_norm], dtype=np.float64),
                np.asarray([float(np.linalg.norm(raw_wxyz))], dtype=np.float64),
                f"constructor {item.case_id} input norm error",
            ),
        )
        expected = Rotation.from_quat(raw_wxyz[[1, 2, 3, 0]])
        native = item.result
        matrix_error, action_error, quaternion_error = _rotation_differences(
            native, expected, probes, f"constructor {item.case_id}"
        )
        maxima.quaternion_construction_matrix = max(
            maxima.quaternion_construction_matrix,
            matrix_error,
        )
        maxima.rotation_vector_action = max(
            maxima.rotation_vector_action,
            action_error,
        )
        maxima.rotation_quaternion_xyzw = max(
            maxima.rotation_quaternion_xyzw,
            quaternion_error,
        )
        native_log = np.asarray(item.log_xyz, dtype=np.float64)
        expected_log = expected.as_rotvec()
        maxima.quaternion_log_rotation = max(
            maxima.quaternion_log_rotation,
            _max_difference(
                Rotation.from_rotvec(native_log).as_matrix(),
                expected.as_matrix(),
                "constructor quaternion log rotation error",
            ),
        )
        if item.case_id != "half_turn_axis":
            maxima.quaternion_log_rotvec = max(
                maxima.quaternion_log_rotvec,
                _max_difference(
                    native_log, expected_log, "constructor log vector error"
                ),
            )

    rotations_by_tangent: dict[str, Rotation] = {}
    for item in capture.exp_log:
        tangent = np.asarray(item.tangent_xyz, dtype=np.float64)
        expected = Rotation.from_rotvec(tangent)
        matrix_error, action_error, quaternion_error = _rotation_differences(
            item.exp_result, expected, probes, f"exp/log {item.case_id}"
        )
        maxima.exp_matrix = max(
            maxima.exp_matrix,
            matrix_error,
        )
        maxima.rotation_vector_action = max(maxima.rotation_vector_action, action_error)
        maxima.rotation_quaternion_xyzw = max(
            maxima.rotation_quaternion_xyzw, quaternion_error
        )
        maxima.log_rotvec = max(
            maxima.log_rotvec,
            _max_difference(
                _native_rotvec(item), expected.as_rotvec(), "log vector error"
            ),
        )
        rotations_by_tangent[item.case_id] = expected

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
    for item, (case_id, expected) in zip(capture.composition, composition_cases):
        if item.case_id != case_id:
            raise ValueError(
                "native composition case order does not match the input contract"
            )
        matrix_error, action_error, quaternion_error = _rotation_differences(
            item, expected, probes, f"composition {case_id}"
        )
        maxima.composition_matrix = max(
            maxima.composition_matrix,
            matrix_error,
        )
        maxima.rotation_vector_action = max(maxima.rotation_vector_action, action_error)
        maxima.rotation_quaternion_xyzw = max(
            maxima.rotation_quaternion_xyzw, quaternion_error
        )

    inverse_expected_rotation = rotations_by_tangent["mixed_a"].inv()
    inverse_matrix, inverse_action, inverse_quaternion = _rotation_differences(
        capture.inverse, inverse_expected_rotation, probes, "inverse"
    )
    maxima.inverse_matrix = max(maxima.inverse_matrix, inverse_matrix)
    maxima.rotation_vector_action = max(maxima.rotation_vector_action, inverse_action)
    maxima.rotation_quaternion_xyzw = max(
        maxima.rotation_quaternion_xyzw, inverse_quaternion
    )
    delta = rotations_by_tangent["perturbation"]
    base = rotations_by_tangent["mixed_a"]
    expected_perturbations = (delta * base, base * delta)
    for item, expected in zip(capture.perturbation_side, expected_perturbations):
        matrix_error, action_error, quaternion_error = _rotation_differences(
            item, expected, probes, f"perturbation {item.case_id}"
        )
        maxima.perturbation_matrix = max(
            maxima.perturbation_matrix,
            matrix_error,
        )
        maxima.rotation_vector_action = max(maxima.rotation_vector_action, action_error)
        maxima.rotation_quaternion_xyzw = max(
            maxima.rotation_quaternion_xyzw, quaternion_error
        )

    if maxima.exceeds(tolerance):
        raise AssertionError(
            f"SciPy formula differences exceed {tolerance}: "
            f"{maxima.model_dump(mode='json')}"
        )

    review = SciPyReview(
        schema_version="1.0",
        evidence_kind="derived_reference",
        native_capture=input_path.name,
        native_capture_sha256=_sha256(input_path),
        reference_script="verify_scipy.py",
        reference_script_sha256=_sha256(Path(__file__)),
        quatops_executed=False,
        reference=ReferenceIdentity(
            library="SciPy Rotation", version=scipy.__version__
        ),
        native_target=capture.target,
        checks=(
            "SO3 constructor normalization, matrix action, XYZW storage, and quaternion log",
            "SO3 exp/log including near-zero and greater-than-pi inputs",
            "noncommuting product orders, inverse, and left/right perturbations",
        ),
        absolute_tolerance=tolerance,
        max_abs_errors=maxima,
        result="pass",
        generated_utc=datetime.now(timezone.utc).isoformat(),
    )
    serialized = review.model_dump(mode="json")
    if options.output is not None:
        output_path = options.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(serialized, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(serialized, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
