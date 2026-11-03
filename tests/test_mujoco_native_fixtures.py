"""Tests for the frozen MuJoCo 3.17.0 low-level capture."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parent.parent
FIXTURE_PATH = (
    ROOT / "tests" / "golden" / "oracles" / "mujoco" / "3.17.0" / "native_rotation.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
OBS = FIXTURE["observations"]
CORPUS_SHA256 = "07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f"
TOLERANCE = 3e-12


def _wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return q[[1, 2, 3, 0]]


def _xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    return q[[3, 0, 1, 2]]


def _hamilton(lhs, rhs):
    aw, ax, ay, az = lhs
    bw, bx, by, bz = rhs
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=float,
    )


def _conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.r_[q[0], -q[1:]]


def _exp_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle == 0:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    return np.r_[np.cos(angle / 2), rotvec * (np.sin(angle / 2) / angle)]


def _rotvec(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    vector_norm = np.linalg.norm(q[1:])
    if vector_norm == 0:
        return np.zeros(3)
    return q[1:] * (2 * np.arctan2(vector_norm, q[0]) / vector_norm)


def _mju_subquat(qa, qb):
    return _rotvec(_hamilton(_conjugate(qb), qa))


def _mju_integrate(q, vel, scale):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return _hamilton(q, _exp_rotvec(np.asarray(vel, dtype=float) * scale))


def _rotation_error(lhs_wxyz, rhs_wxyz):
    lhs = Rotation.from_quat(_wxyz_to_xyzw(lhs_wxyz)).as_matrix()
    rhs = Rotation.from_quat(_wxyz_to_xyzw(rhs_wxyz)).as_matrix()
    return float(np.max(np.abs(lhs - rhs)))


def _finite_difference_jacobian(function, base, epsilon=1e-6):
    result = np.empty((3, 3), dtype=float)
    for column in range(3):
        delta = np.zeros(3)
        delta[column] = epsilon
        result[:, column] = (function(delta) - function(-delta)) / (2 * epsilon)
    return result


def test_mujoco_fixture_records_pinned_binary_dtype_and_capture_counts():
    assert FIXTURE["evidence_kind"] == "native_mujoco_runtime_capture"
    assert FIXTURE["runtime"]["package_version"] == "3.17.0"
    assert FIXTURE["runtime"]["engine_version"] == "3.17.0"
    assert FIXTURE["runtime"]["binding_scalar_dtype"] == "float64"
    assert FIXTURE["runtime"]["python_version"].startswith("3.10.11 ")
    assert any(
        row["path"] == "mujoco.dll"
        for row in FIXTURE["runtime"]["native_binary_files_under_package"]
    )
    assert all(
        row["sha256"] for row in FIXTURE["runtime"]["native_binary_files_under_package"]
    )
    assert FIXTURE["provenance"]["input_corpus_sha256"] == CORPUS_SHA256
    assert FIXTURE["counts"] == {
        "euler_orders": 12,
        "euler_boundaries": 6,
        "multiplication": 3,
        "axis_angle": 5,
        "proper_matrices": 3,
        "integration_and_derivatives": 4,
        "quaternion_differences": 2,
    }


def test_mujoco_euler_case_casing_maps_to_scipy_intrinsic_and_extrinsic_routes():
    for row in OBS["euler_orders"]:
        scipy_order = (
            row["order_label"].upper()
            if row["sequence_case"] == "lowercase"
            else row["order_label"].lower()
        )
        expected = Rotation.from_euler(
            scipy_order, row["input_angles_in_sequence_order_radians"]
        )
        assert (
            _rotation_error(row["quaternion_wxyz"], _xyzw_to_wxyz(expected.as_quat()))
            <= TOLERANCE
        ), (row["case_id"], row["sequence"])
        np.testing.assert_allclose(
            row["matrix_rows"], expected.as_matrix(), rtol=0.0, atol=TOLERANCE
        )

    for row in OBS["euler_boundaries"]:
        expected = Rotation.from_euler(
            row["order_label"].upper(), row["input_angles_in_sequence_order_radians"]
        )
        assert (
            _rotation_error(row["quaternion_wxyz"], _xyzw_to_wxyz(expected.as_quat()))
            <= 3e-7
        )


def test_mujoco_products_axis_angle_and_matrix_routes_match_independent_formulas():
    for row in OBS["multiplication"]:
        a = np.asarray(row["a_input_wxyz"], dtype=float)
        b = np.asarray(row["b_input_wxyz"], dtype=float)
        np.testing.assert_allclose(
            row["mju_mulQuat_a_b_wxyz"], _hamilton(a, b), rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            row["mju_mulQuat_b_a_wxyz"], _hamilton(b, a), rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            row["mju_negQuat_a_wxyz"], _conjugate(a), rtol=0.0, atol=0.0
        )
        w, x, y, z = a
        expected = np.asarray(
            (
                (
                    w * w + x * x - y * y - z * z,
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ),
                (
                    2 * (x * y + w * z),
                    w * w - x * x + y * y - z * z,
                    2 * (y * z - w * x),
                ),
                (
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    w * w - x * x - y * y + z * z,
                ),
            )
        )
        np.testing.assert_allclose(
            row["mju_quat2Mat_a_rows"], expected, rtol=0.0, atol=TOLERANCE
        )

    for row in OBS["axis_angle"]:
        expected = Rotation.from_rotvec(
            np.asarray(row["input_axis_xyz"], dtype=float) * row["input_angle_radians"]
        )
        assert (
            _rotation_error(row["quaternion_wxyz"], _xyzw_to_wxyz(expected.as_quat()))
            <= TOLERANCE
        )
        np.testing.assert_allclose(
            row["matrix_rows"], expected.as_matrix(), rtol=0.0, atol=TOLERANCE
        )

    for row in OBS["proper_matrices"]:
        expected = np.asarray(row["input_matrix_rows"], dtype=float)
        np.testing.assert_allclose(
            row["mju_quat2Mat_rows"], expected, rtol=0.0, atol=TOLERANCE
        )
        assert (
            _rotation_error(
                row["mju_mat2Quat_wxyz"],
                _xyzw_to_wxyz(Rotation.from_matrix(expected).as_quat()),
            )
            <= TOLERANCE
        )


def test_mujoco_integration_and_derivative_use_explicit_body_and_world_rates():
    for row in OBS["integration"]:
        q = np.asarray(row["input_q_wxyz"], dtype=float)
        omega_body = np.asarray(row["input_omega_xyz"], dtype=float)
        dt = row["dt_seconds"]
        expected_q = _mju_integrate(q, omega_body, dt)
        assert _rotation_error(row["integrated_q_wxyz"], expected_q) <= 2e-12

        derivative_world = 0.5 * _hamilton([0.0, *omega_body], q)
        np.testing.assert_allclose(
            row["mju_derivQuat_same_numeric_omega_wxyz_per_second"],
            derivative_world,
            rtol=0.0,
            atol=2e-14,
        )
        rotation = Rotation.from_quat(_wxyz_to_xyzw(q))
        omega_world = rotation.apply(omega_body)
        np.testing.assert_allclose(
            row["world_omega_from_integration_input_xyz"],
            omega_world,
            rtol=0.0,
            atol=TOLERANCE,
        )
        derivative_local = 0.5 * _hamilton(q, [0.0, *omega_body])
        np.testing.assert_allclose(
            row["mju_derivQuat_world_frame_wxyz_per_second"],
            derivative_local,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            row["integrator_finite_difference_wxyz_per_second"],
            derivative_local,
            rtol=0.0,
            atol=2e-9,
        )
        native_quat2vel = row["mju_quat2Vel_from_integrated_relative_quaternion"]
        if dt == 0.0:
            assert native_quat2vel == {
                "status": "not_called_zero_scale",
                "velocity_xyz": None,
            }
        else:
            assert native_quat2vel["status"] == "captured"
            np.testing.assert_allclose(
                native_quat2vel["velocity_xyz"], omega_body, rtol=0.0, atol=3e-9
            )


def test_mujoco_quat_integrate_jacobians_match_finite_differences_in_recorded_charts():
    epsilon = 1e-6
    for row in OBS["integration"]:
        q = np.asarray(row["input_q_wxyz"], dtype=float)
        vel = np.asarray(row["input_omega_xyz"], dtype=float)
        scale = float(row["dt_seconds"])
        base_output = _mju_integrate(q, vel, scale)
        jacobian = row["mjd_quatIntegrate"]
        native_dquat = np.asarray(jacobian["Dquat_flat_row_major"]).reshape(3, 3)
        native_dvel = np.asarray(jacobian["Dvel_flat_row_major"]).reshape(3, 3)
        native_dscale = np.asarray(jacobian["Dscale_xyz"])

        def with_q_delta(delta, q=q, vel=vel, scale=scale, base_output=base_output):
            q_perturbed = _hamilton(q, _exp_rotvec(delta))
            out = _mju_integrate(q_perturbed, vel, scale)
            return _mju_subquat(out, base_output)

        finite_dquat = _finite_difference_jacobian(with_q_delta, np.zeros(3), epsilon)
        np.testing.assert_allclose(native_dquat, finite_dquat, rtol=0.0, atol=3e-7)

        scaled_velocity = vel * scale

        def with_scaled_velocity_delta(
            delta, q=q, scaled_velocity=scaled_velocity, base_output=base_output
        ):
            out = _mju_integrate(q, scaled_velocity + delta, 1.0)
            return _mju_subquat(out, base_output)

        finite_dvel = _finite_difference_jacobian(
            with_scaled_velocity_delta, np.zeros(3), epsilon
        )
        np.testing.assert_allclose(native_dvel, finite_dvel, rtol=0.0, atol=3e-7)
        np.testing.assert_allclose(
            native_dscale, native_dvel @ vel, rtol=0.0, atol=2e-12
        )


def test_mujoco_subquat_derivative_matrices_match_the_native_local_perturbation_chart():
    epsilon = 1e-6
    for row in OBS["sub_quat"]:
        qa = np.asarray(row["qa_wxyz"], dtype=float)
        qb = np.asarray(row["qb_wxyz"], dtype=float)
        expected = _mju_subquat(qa, qb)
        np.testing.assert_allclose(
            row["mju_subQuat_qa_qb_rotvec_xyz"], expected, rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            row["mju_quat2Vel_relative_q_dt1_xyz"], expected, rtol=0.0, atol=2e-12
        )
        native_da = np.asarray(row["mjd_subQuat_Da_flat_row_major"]).reshape(3, 3)
        native_db = np.asarray(row["mjd_subQuat_Db_flat_row_major"]).reshape(3, 3)

        def perturb_a(delta, qa=qa, qb=qb):
            return _mju_subquat(_hamilton(qa, _exp_rotvec(delta)), qb)

        def perturb_b(delta, qa=qa, qb=qb):
            return _mju_subquat(qa, _hamilton(qb, _exp_rotvec(delta)))

        np.testing.assert_allclose(
            native_da,
            _finite_difference_jacobian(perturb_a, np.zeros(3), epsilon),
            rtol=0.0,
            atol=4e-6,
        )
        np.testing.assert_allclose(
            native_db,
            _finite_difference_jacobian(perturb_b, np.zeros(3), epsilon),
            rtol=0.0,
            atol=4e-6,
        )
