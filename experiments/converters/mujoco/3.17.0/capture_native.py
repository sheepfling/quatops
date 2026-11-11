"""Capture selected MuJoCo quaternion-boundary witnesses; imports no QuatOps."""

from __future__ import annotations

from importlib import resources
from importlib.metadata import version
from pathlib import Path

import mujoco  # pyright: ignore[reportMissingModuleSource]
import numpy as np

from scripts.direct_quaternion_capture import DirectCaptureResult, run_direct_capture
from scripts.evidence_schema.converter_captures import (
    MuJoCoCapture,
    MuJoCoInputs,
    MuJoCoObservations,
    MuJoCoOrientation,
    TargetIdentity,
)
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    SharedQuaternionInput,
)


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
    return MuJoCoOrientation(
        case_id=case_id,
        native_input_wxyz=q.tolist(),
        native_mju_rotVecQuat_basis_xyz=[
            native_action(vector, q).tolist() for vector in np.eye(3)
        ],
        native_mju_rotVecQuat_probe_xyz=(
            native_action(probe, q).tolist() if case_id == "mixed_a" else None
        ),
        native_constructor=native_constructor,
        native_lhs_wxyz=lhs.tolist() if lhs is not None else None,
        native_rhs_wxyz=rhs.tolist() if rhs is not None else None,
        product_order="lhs * rhs; rhs acts first" if lhs is not None else None,
    )


def capture(
    spec: MuJoCoInputs,
    shared: SharedCanonicalInputCapture,
    selected: list[SharedQuaternionInput],
) -> DirectCaptureResult:
    probe = np.asarray(shared.probe_vector_xyz, dtype=np.float64)
    canonical_to_native = [
        capture_quaternion(
            row.case_id, np.asarray(row.canonical_wxyz, dtype=np.float64), probe
        )
        for row in selected
    ]
    ctor = spec.native_constructed_orientation
    qnative = np.empty(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(
        qnative, np.asarray(ctor.axis_xyz, dtype=np.float64), ctor.angle_radians
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

    witnesses = MuJoCoCapture(
        schema_version="1.0",
        evidence_kind="native_runtime_selected_converter_witnesses",
        target=TargetIdentity(
            name="MuJoCo",
            package_version=version("mujoco"),
            engine_version=mujoco.mj_versionString(),
        ),
        observations=MuJoCoObservations(
            canonical_to_native=canonical_to_native,
            native_created=[native_created],
            composition=[
                compose("mixed_a_times_mixed_b", a, b),
                compose("mixed_b_times_mixed_a", b, a),
            ],
        ),
    )
    library = Path(str(resources.files("mujoco"))) / "mujoco.dll"
    return DirectCaptureResult(witnesses, library)


if __name__ == "__main__":
    run_direct_capture(Path(__file__), MuJoCoInputs, capture)
