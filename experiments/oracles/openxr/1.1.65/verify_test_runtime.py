"""Check captured OpenXR test-runtime frame relations with SciPy formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic
import scipy
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import read_evidence_document

ObservationKey = Literal["A_from_B", "B_from_C", "A_from_C"]
OBSERVATION_CASES: tuple[tuple[ObservationKey, str, str], ...] = (
    ("A_from_B", "A", "B"),
    ("B_from_C", "B", "C"),
    ("A_from_C", "A", "C"),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Pose(StrictModel):
    orientation_xyzw: tuple[float, float, float, float]
    position_xyz_m: tuple[float, float, float]


class LocatedPose(StrictModel):
    target: str
    base: str
    location_flags: int
    pose: Pose


class Runtime(StrictModel):
    product: Literal["Khronos OpenXR-SDK-Source test_runtime"]
    headless_extension: Literal["XR_MND_headless"]
    headless_extension_advertised: bool
    hardware_runtime: Literal[False]


class Capture(StrictModel):
    evidence_kind: Literal["native_openxr_test_runtime_frame_chain"]
    runtime: Runtime
    api_version: tuple[int, int, int]
    locate_time_xr_time: int
    reference_space_type: Literal["LOCAL"]
    inputs_pose_in_reference_space: dict[str, Pose]
    observations: dict[ObservationKey, LocatedPose]
    quatops_executed: Literal[False]


def _native(capture: Capture, key: ObservationKey) -> tuple[Rotation, np.ndarray]:
    pose = capture.observations[key].pose
    return (
        Rotation.from_quat(pose.orientation_xyzw),
        np.asarray(pose.position_xyz_m, dtype=np.float64),
    )


def _input(capture: Capture, name: str) -> tuple[Rotation, np.ndarray]:
    pose = capture.inputs_pose_in_reference_space[name]
    return (
        Rotation.from_quat(pose.orientation_xyzw),
        np.asarray(pose.position_xyz_m, dtype=np.float64),
    )


def _compose(
    left: tuple[Rotation, np.ndarray], right: tuple[Rotation, np.ndarray]
) -> tuple[Rotation, np.ndarray]:
    left_rotation, left_translation = left
    right_rotation, right_translation = right
    return (
        left_rotation * right_rotation,
        left_rotation.apply(right_translation) + left_translation,
    )


def _relative(
    base: tuple[Rotation, np.ndarray], target: tuple[Rotation, np.ndarray]
) -> tuple[Rotation, np.ndarray]:
    base_rotation, base_translation = base
    target_rotation, target_translation = target
    inverse_base = base_rotation.inv()
    return (
        inverse_base * target_rotation,
        inverse_base.apply(target_translation - base_translation),
    )


def _check_unit_quaternion(
    quaternion: tuple[float, float, float, float], label: str
) -> None:
    norm = float(np.linalg.norm(np.asarray(quaternion, dtype=np.float64)))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{label} is not a unit quaternion: norm={norm}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--retain-to", type=Path)
    args = parser.parse_args()
    raw_capture = read_evidence_document(args.capture)
    capture = Capture.model_validate(raw_capture)
    if capture.api_version != (1, 1, 65):
        raise ValueError(f"unexpected test runtime API version: {capture.api_version}")
    if set(capture.inputs_pose_in_reference_space) != {"A", "B", "C"}:
        raise ValueError("fixture must contain exactly the declared A, B, and C inputs")
    if set(capture.observations) != {key for key, _, _ in OBSERVATION_CASES}:
        raise ValueError("fixture must contain exactly the three declared locate calls")
    if capture.runtime.hardware_runtime:
        raise ValueError(
            "fixture must identify the Khronos test runtime as non-hardware"
        )
    if not capture.runtime.headless_extension_advertised:
        raise ValueError("fixture does not record XR_MND_headless advertisement")
    if capture.locate_time_xr_time <= 0:
        raise ValueError("test runtime capture requires a positive synthetic XrTime")

    tolerance = 2e-6
    native_errors: dict[str, float] = {}
    for key, base_name, target_name in OBSERVATION_CASES:
        observation = capture.observations[key]
        if (observation.base, observation.target) != (base_name, target_name):
            raise ValueError(f"{key} does not preserve its target/base direction")
        if observation.location_flags != 15:
            raise ValueError(
                f"{key} did not return valid and tracked position/orientation"
            )
        _check_unit_quaternion(
            capture.inputs_pose_in_reference_space[base_name].orientation_xyzw,
            f"input {base_name}",
        )
        _check_unit_quaternion(
            capture.inputs_pose_in_reference_space[target_name].orientation_xyzw,
            f"input {target_name}",
        )
        _check_unit_quaternion(observation.pose.orientation_xyzw, f"observation {key}")
        expected = _relative(_input(capture, base_name), _input(capture, target_name))
        actual = _native(capture, key)
        rotation_error = float(
            np.max(np.abs(expected[0].as_matrix() - actual[0].as_matrix()))
        )
        translation_error = float(np.max(np.abs(expected[1] - actual[1])))
        native_errors[key] = max(rotation_error, translation_error)
        if native_errors[key] > tolerance:
            raise AssertionError(f"{key} differs from independent SciPy frame formula")

    composed = _compose(_native(capture, "A_from_B"), _native(capture, "B_from_C"))
    direct = _native(capture, "A_from_C")
    chain_rotation_error = float(
        np.max(np.abs(composed[0].as_matrix() - direct[0].as_matrix()))
    )
    chain_translation_error = float(np.max(np.abs(composed[1] - direct[1])))
    point_in_c = np.asarray([0.5, -1.25, 2.0], dtype=np.float64)
    point_in_a_via_chain = composed[0].apply(point_in_c) + composed[1]
    point_in_a_direct = direct[0].apply(point_in_c) + direct[1]
    point_error = float(np.max(np.abs(point_in_a_via_chain - point_in_a_direct)))
    if max(chain_rotation_error, chain_translation_error, point_error) > tolerance:
        raise AssertionError(
            "native frame-chain composition did not match the direct locate"
        )
    if args.retain_to is not None:
        args.retain_to.parent.mkdir(parents=True, exist_ok=True)
        args.retain_to.write_text(
            dump_evidence_json(raw_capture), encoding="utf-8", newline="\n"
        )

    print(
        json.dumps(
            {
                "evidence_kind": "derived_reference",
                "reference": "SciPy Rotation",
                "scipy_version": scipy.__version__,
                "numpy_version": np.__version__,
                "pydantic_version": pydantic.__version__,
                "quatops_executed": False,
                "tolerance": tolerance,
                "native_formula_max_errors": native_errors,
                "chain_rotation_matrix_max_abs_error": chain_rotation_error,
                "chain_translation_max_abs_error_m": chain_translation_error,
                "chain_point_max_abs_error_m": point_error,
                "result": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
