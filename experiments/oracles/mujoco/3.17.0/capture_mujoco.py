"""Capture MuJoCo's low-level quaternion routines through pinned Python bindings."""

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

HERE = Path(__file__).parent
ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=HERE.parents[1] / "shared" / "orientation_inputs_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=HERE.parents[3]
        / "tests"
        / "golden"
        / "oracles"
        / "mujoco"
        / "3.17.0"
        / "native_rotation.json",
        type=Path,
    )
    return parser.parse_args()


def native_quaternion_call(name, *args):
    result = np.empty(4, dtype=np.float64)
    getattr(mujoco, name)(result, *args)
    return result


def matrix_from_quaternion(q):
    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, np.asarray(q, dtype=np.float64))
    return result.reshape(3, 3)


def rotation_vector_from_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q[1:])
    if norm == 0.0:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(norm, q[0])
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return q[1:] * (angle / norm)


def quaternion_from_rotation_vector(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle == 0.0:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    return np.r_[np.cos(angle / 2.0), rotvec * (np.sin(angle / 2.0) / angle)]


def native_sub_quat(qa, qb):
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_subQuat(
        result,
        np.asarray(qa, dtype=np.float64),
        np.asarray(qb, dtype=np.float64),
    )
    return result


def native_quat2vel(q, dt):
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_quat2Vel(result, np.asarray(q, dtype=np.float64), dt)
    return result


def native_euler(case_id, xyz, order, sequence_case, source):
    xyz = np.asarray(xyz, dtype=np.float64)
    if source["units"] == "degrees":
        xyz = np.deg2rad(xyz)
    ordered = [float(xyz["XYZ".index(axis)]) for axis in order]
    sequence = order.lower() if sequence_case == "lowercase" else order
    q = np.empty(4, dtype=np.float64)
    mujoco.mju_euler2Quat(q, np.asarray(ordered, dtype=np.float64), sequence)
    return {
        "case_id": case_id,
        "order_label": order,
        "sequence": sequence,
        "sequence_case": sequence_case,
        "input_units": source["units"],
        "input_radians_xyz": xyz.tolist(),
        "input_angles_in_sequence_order_radians": ordered,
        "input_source": source,
        "quaternion_wxyz": q.tolist(),
        "matrix_rows": matrix_from_quaternion(q).tolist(),
    }


def finite_difference_quat_derivative(q, vel, step):
    plus = np.asarray(q, dtype=np.float64).copy()
    minus = np.asarray(q, dtype=np.float64).copy()
    mujoco.mju_quatIntegrate(plus, np.asarray(vel, dtype=np.float64), step)
    mujoco.mju_quatIntegrate(minus, np.asarray(vel, dtype=np.float64), -step)
    return (plus - minus) / (2.0 * step)


def integration_jacobians(vel, scale):
    dquat = np.empty(9, dtype=np.float64)
    dvel = np.empty(9, dtype=np.float64)
    dscale = np.empty(3, dtype=np.float64)
    mujoco.mjd_quatIntegrate(
        np.asarray(vel, dtype=np.float64), scale, dquat, dvel, dscale
    )
    return {
        "Dquat_flat_row_major": dquat.tolist(),
        "Dvel_flat_row_major": dvel.tolist(),
        "Dscale_xyz": dscale.tolist(),
    }


def main():
    options = parse_args()
    corpus_bytes = options.corpus.read_bytes()
    corpus = json.loads(corpus_bytes.decode("utf-8"))
    euler_orders = []
    for case in corpus["euler_inputs"]:
        for order in ORDERS:
            source = {
                "units": case["units"],
                "numeric_format": case["numeric_format"],
                "actual_bits_hex": case["actual_bits_hex"],
            }
            for sequence_case in ("lowercase", "uppercase"):
                euler_orders.append(
                    native_euler(
                        case["case_id"],
                        case["xyz"],
                        order,
                        sequence_case,
                        source,
                    )
                )

    euler_boundaries = []
    for case in corpus["euler_boundaries"]:
        source = {
            "units": case["units"],
            "numeric_format": case["numeric_format"],
            "actual_bits_hex": case["actual_bits_hex"],
            "input_xyz": case["xyz"],
        }
        euler_boundaries.append(
            native_euler(
                case["case_id"],
                case["xyz"],
                case["order_label"],
                "lowercase",
                source,
            )
        )

    multiplication = []
    for case in corpus["multiplication_pairs"]:
        a = np.asarray(case["a_wxyz"], dtype=np.float64)
        b = np.asarray(case["b_wxyz"], dtype=np.float64)
        ab = native_quaternion_call("mju_mulQuat", a, b)
        ba = native_quaternion_call("mju_mulQuat", b, a)
        conjugate = native_quaternion_call("mju_negQuat", a)
        multiplication.append(
            {
                "case_id": case["case_id"],
                "a_input_wxyz": a.tolist(),
                "b_input_wxyz": b.tolist(),
                "mju_mulQuat_a_b_wxyz": ab.tolist(),
                "mju_mulQuat_b_a_wxyz": ba.tolist(),
                "mju_negQuat_a_wxyz": conjugate.tolist(),
                "mju_quat2Mat_a_rows": matrix_from_quaternion(a).tolist(),
            }
        )

    axis_angle = []
    for case in corpus["axis_angle_inputs"]:
        q = np.empty(4, dtype=np.float64)
        axis = np.asarray(case["axis_xyz"], dtype=np.float64)
        mujoco.mju_axisAngle2Quat(q, axis, float(case["angle_radians"]))
        axis_angle.append(
            {
                "case_id": case["case_id"],
                "input_axis_xyz": axis.tolist(),
                "input_angle_radians": float(case["angle_radians"]),
                "quaternion_wxyz": q.tolist(),
                "matrix_rows": matrix_from_quaternion(q).tolist(),
            }
        )

    matrices = []
    for case in corpus["matrix_inputs"]:
        if case["domain"] != "proper_rotation":
            continue
        source_matrix = np.asarray(case["rows"], dtype=np.float64)
        q = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, source_matrix.reshape(9))
        matrices.append(
            {
                "case_id": case["case_id"],
                "input_matrix_rows": source_matrix.tolist(),
                "mju_mat2Quat_wxyz": q.tolist(),
                "mju_quat2Mat_rows": matrix_from_quaternion(q).tolist(),
            }
        )

    integration = []
    for index, case in enumerate(corpus["integration_inputs"]):
        q = np.asarray(case["initial_wxyz"], dtype=np.float64)
        omega_body = np.asarray(case["omega_xyz"], dtype=np.float64)
        scale = float(case["dt_seconds"])
        integrated = q.copy()
        mujoco.mju_quatIntegrate(integrated, omega_body, scale)
        derivative_world = np.empty(4, dtype=np.float64)
        mujoco.mju_derivQuat(derivative_world, q, omega_body)
        derivative_local = np.empty(4, dtype=np.float64)
        omega_world = matrix_from_quaternion(q) @ omega_body
        mujoco.mju_derivQuat(derivative_local, q, omega_world)
        finite_difference = finite_difference_quat_derivative(q, omega_body, 1e-6)
        normalized_q = q / np.linalg.norm(q)
        q_inverse = native_quaternion_call("mju_negQuat", normalized_q)
        integrated_relative = native_quaternion_call(
            "mju_mulQuat", q_inverse, integrated
        )
        quat2vel = (
            {
                "status": "captured",
                "velocity_xyz": native_quat2vel(integrated_relative, scale).tolist(),
            }
            if scale != 0.0
            else {"status": "not_called_zero_scale", "velocity_xyz": None}
        )
        integration.append(
            {
                "case_id": case["case_id"],
                "input_q_wxyz": q.tolist(),
                "input_omega_xyz": omega_body.tolist(),
                "input_omega_frame": case["omega_frame"],
                "mju_quatIntegrate_velocity_frame": "body_local",
                "mju_derivQuat_velocity_frame": "world_expressed",
                "dt_seconds": scale,
                "integrated_q_wxyz": integrated.tolist(),
                "normalized_input_q_wxyz": (q / np.linalg.norm(q)).tolist()
                if np.linalg.norm(q) > 0.0
                else q.tolist(),
                "mju_derivQuat_same_numeric_omega_wxyz_per_second": derivative_world.tolist(),
                "world_omega_from_integration_input_xyz": omega_world.tolist(),
                "mju_derivQuat_world_frame_wxyz_per_second": derivative_local.tolist(),
                "integrator_finite_difference_wxyz_per_second": finite_difference.tolist(),
                "mju_quat2Vel_from_integrated_relative_quaternion": quat2vel,
                "mjd_quatIntegrate": integration_jacobians(omega_body, scale),
            }
        )

    sub_quat = []
    safe_rotations = []
    for case in corpus["axis_angle_inputs"]:
        angle = float(case["angle_radians"])
        if np.isclose(angle, np.pi / 2.0, rtol=0.0, atol=1e-14):
            q = np.empty(4, dtype=np.float64)
            mujoco.mju_axisAngle2Quat(
                q,
                np.asarray(case["axis_xyz"], dtype=np.float64),
                angle,
            )
            safe_rotations.append((case["case_id"], q))
    if len(safe_rotations) != 4:
        raise RuntimeError(f"Expected four positive quarter-turn probes; got {len(safe_rotations)}")
    for index, (case_id, qa) in enumerate(safe_rotations):
        next_case_id, qb = safe_rotations[(index + 1) % len(safe_rotations)]
        difference = native_sub_quat(qa, qb)
        qneg_b = native_quaternion_call("mju_negQuat", qb)
        relative_q = native_quaternion_call("mju_mulQuat", qneg_b, qa)
        da = np.empty(9, dtype=np.float64)
        db = np.empty(9, dtype=np.float64)
        mujoco.mjd_subQuat(qa, qb, da, db)
        sub_quat.append(
            {
                "case_id": f"{case_id}__minus__{next_case_id}",
                "qa_wxyz": qa.tolist(),
                "qb_wxyz": qb.tolist(),
                "mju_subQuat_qa_qb_rotvec_xyz": difference.tolist(),
                "mju_quat2Vel_relative_q_dt1_xyz": native_quat2vel(relative_q, 1.0).tolist(),
                "mjd_subQuat_Da_flat_row_major": da.tolist(),
                "mjd_subQuat_Db_flat_row_major": db.tolist(),
            }
        )

    package_dir = Path(mujoco.__file__).parent
    binary_files = []
    for pattern in ("*.dll", "*.pyd"):
        for path in sorted(package_dir.rglob(pattern)):
            binary_files.append(
                {"path": path.relative_to(package_dir).as_posix(), "sha256": sha256_file(path)}
            )
    if not binary_files:
        raise RuntimeError(f"No MuJoCo native binary found under {package_dir}")

    lock_path = HERE / "uv.lock"
    output = {
        "schema_version": "1.0",
        "evidence_kind": "native_mujoco_runtime_capture",
        "runtime": {
            "package": "mujoco",
            "package_version": version("mujoco"),
            "engine_version": mujoco.mj_versionString(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "binding_scalar_dtype": "float64",
            "native_binary_files_under_package": binary_files,
        },
        "provenance": {
            "input_corpus_sha256": sha256(corpus_bytes),
            "lockfile_sha256": sha256_file(lock_path),
            "exporter_sha256": sha256_file(Path(__file__)),
            "exact_invocation": "uv run --locked python capture_mujoco.py --corpus <orientation_inputs_v1.json> --output <native_rotation.json>",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "euler_orders": len(euler_orders),
            "euler_boundaries": len(euler_boundaries),
            "multiplication": len(multiplication),
            "axis_angle": len(axis_angle),
            "proper_matrices": len(matrices),
            "integration_and_derivatives": len(integration),
            "quaternion_differences": len(sub_quat),
        },
        "observations": {
            "euler_orders": euler_orders,
            "euler_boundaries": euler_boundaries,
            "multiplication": multiplication,
            "axis_angle": axis_angle,
            "proper_matrices": matrices,
            "integration": integration,
            "sub_quat": sub_quat,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
