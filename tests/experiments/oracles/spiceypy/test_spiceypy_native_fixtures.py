"""CSPICE 67 behavior captured through the pinned SpiceyPy 8.3.0 binding."""

import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.support.native_engine_support import (
    from_euler,
    quaternion_coefficient_error,
    to_hamilton_xyzw,
)
from tests.support.oracle_support import BACKENDS
from tests.support.paths import REPOSITORY_ROOT, TESTS_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE_PATH = (
    TESTS_ROOT / "golden" / "oracles" / "spiceypy" / "8.3.0" / "native_rotation.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
OBS = FIXTURE["observations"]
ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
CORPUS_SHA256 = "07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f"
TOLERANCE = 3e-12


def _ordered(xyz, order):
    return tuple(float(xyz["XYZ".index(axis)]) for axis in order)


def _rotation_error(lhs_wxyz, rhs_wxyz):
    lhs = np.asarray(lhs_wxyz, dtype=float)[[1, 2, 3, 0]]
    rhs = np.asarray(rhs_wxyz, dtype=float)[[1, 2, 3, 0]]
    return float(
        np.max(
            np.abs(
                Rotation.from_quat(lhs).as_matrix()
                - Rotation.from_quat(rhs).as_matrix()
            )
        )
    )


def _hamilton_product(lhs, rhs):
    aw, ax, ay, az = lhs
    bw, bx, by, bz = rhs
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def test_cspice_fixture_records_both_pinned_wrapper_and_loaded_toolkit():
    assert FIXTURE["evidence_kind"] == "native_cspice_runtime_capture_via_spiceypy"
    assert FIXTURE["runtime"]["wrapper"] == "SpiceyPy"
    assert FIXTURE["runtime"]["wrapper_version"] == "8.3.0"
    assert FIXTURE["runtime"]["toolkit_version"] == "CSPICE_N0067"
    assert FIXTURE["runtime"]["loaded_library_path"].lower().endswith("cspice.dll")
    assert FIXTURE["runtime"]["loaded_library_sha256"]
    assert FIXTURE["provenance"]["input_corpus_sha256"] == CORPUS_SHA256
    assert FIXTURE["counts"] == {
        "euler_orders": 6,
        "euler_boundaries": 6,
        "axis_angle": 5,
        "multiplication": 3,
        "proper_matrices": 3,
        "qdq2av": 2,
    }


def test_cspice_euler_and_matrix_direction_match_scipy_and_read_back():
    for group in ("euler_orders", "euler_boundaries"):
        for row in OBS[group]:
            expected_active = Rotation.from_euler(
                row["order"], _ordered(row["input_radians_xyz"], row["order"])
            ).as_matrix()
            native_frame_matrix = np.asarray(row["eul2m_matrix_rows"], dtype=float)
            np.testing.assert_allclose(
                native_frame_matrix,
                expected_active.T,
                rtol=0.0,
                atol=TOLERANCE,
                err_msg=f"CSPICE eul2m {row['case_id']} {row['order']}",
            )
            np.testing.assert_allclose(
                row["q2m_from_m2q_rows"],
                native_frame_matrix,
                rtol=0.0,
                atol=TOLERANCE,
            )
            readback = Rotation.from_euler(
                row["order"], _ordered(row["m2eul_readback_radians_xyz"], row["order"])
            ).as_matrix()
            np.testing.assert_allclose(
                readback.T,
                native_frame_matrix,
                rtol=0.0,
                atol=2e-8 if group == "euler_boundaries" else 8e-14,
                err_msg=f"CSPICE m2eul {row['case_id']} {row['order']}",
            )
            native_quaternion = np.asarray(row["m2q_wxyz"], dtype=float)
            scipy_quaternion = Rotation.from_matrix(native_frame_matrix).as_quat()[
                [3, 0, 1, 2]
            ]
            assert (
                quaternion_coefficient_error(native_quaternion, scipy_quaternion)
                <= 2e-12
            )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_cspice_euler_quaternions_match_all_quatops_backends(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for group in ("euler_orders", "euler_boundaries"):
        for row in OBS[group]:
            actual = from_euler(
                module,
                _ordered(row["input_radians_xyz"], row["order"]),
                row["order"],
                False,
                component_first,
            )
            actual_xyzw = to_hamilton_xyzw(
                module, actual, is_jpl, is_numpy, component_first
            )
            native_wxyz = np.asarray(row["m2q_wxyz"], dtype=float)
            # eul2m/m2q expose a frame matrix opposite to QuatOps' active
            # Euler rotation; conjugate only in this explicit comparison map.
            native_xyzw = np.r_[-native_wxyz[[1, 2, 3]], native_wxyz[0]]
            assert quaternion_coefficient_error(actual_xyzw, native_xyzw) <= 2e-11, (
                f"{backend_name}: CSPICE {row['case_id']} {row['order']}"
            )


def test_cspice_products_axis_angle_and_m2q_proper_matrices_match_formulas():
    for row in OBS["multiplication"]:
        actual = np.asarray(row["qxq_a_b_wxyz"], dtype=float)
        expected = _hamilton_product(row["a_input_wxyz"], row["b_input_wxyz"])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(
            row["qxq_b_a_wxyz"],
            _hamilton_product(row["b_input_wxyz"], row["a_input_wxyz"]),
            rtol=0.0,
            atol=2e-14,
        )
        lhs = np.asarray(row["q2m_a_rows"])
        rhs = np.asarray(row["q2m_b_rows"])
        product = np.asarray(row["q2m_product_rows"])
        np.testing.assert_allclose(product, lhs @ rhs, rtol=0.0, atol=3e-14)

    for row in OBS["axis_angle"]:
        expected = Rotation.from_rotvec(
            np.asarray(row["input_axis_xyz"]) * row["input_angle_radians"]
        )
        native_matrix = np.asarray(row["axisar_matrix_rows"], dtype=float)
        np.testing.assert_allclose(
            native_matrix, expected.as_matrix(), rtol=0.0, atol=TOLERANCE
        )
        native_q = np.asarray(row["axisar_wxyz"], dtype=float)
        assert _rotation_error(native_q, expected.as_quat()[[3, 0, 1, 2]]) <= TOLERANCE

    for row in OBS["proper_matrices"]:
        matrix = np.asarray(row["input_matrix_rows"], dtype=float)
        native_q = np.asarray(row["m2q_wxyz"], dtype=float)
        np.testing.assert_allclose(row["q2m_rows"], matrix, rtol=0.0, atol=TOLERANCE)
        q_as_active = Rotation.from_quat(native_q[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(q_as_active, matrix, rtol=0.0, atol=TOLERANCE)


def test_cspice_qdq2av_identifies_the_native_frame_and_derivative_side():
    for row in OBS["qdq2av"]:
        q = np.asarray(row["input_q_wxyz"], dtype=float)
        omega = np.asarray(row["input_omega_xyz"], dtype=float)
        native_passive_matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T
        qdot_body = 0.5 * _hamilton_product(q, [0.0, *omega])
        qdot_left = 0.5 * _hamilton_product([0.0, *omega], q)
        probes = {
            probe["derivative_construction"]: probe for probe in row["qdq2av_probes"]
        }
        np.testing.assert_allclose(
            probes["q_then_omega"]["input_qdot_wxyz_per_second"],
            qdot_body,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            probes["omega_then_q"]["input_qdot_wxyz_per_second"],
            qdot_left,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            probes["q_then_omega"]["qdq2av_omega_xyz_per_second"],
            -omega,
            rtol=0.0,
            atol=3e-14,
        )
        np.testing.assert_allclose(
            probes["omega_then_q"]["qdq2av_omega_xyz_per_second"],
            -native_passive_matrix @ omega,
            rtol=0.0,
            atol=3e-14,
        )


def test_spiceypy_capture_script_and_lockfile_hashes_are_frozen():
    capture_path = (
        ROOT / "experiments" / "oracles" / "spiceypy" / "8.3.0" / "capture_spice.py"
    )
    lock_path = capture_path.with_name("uv.lock")
    assert (
        hashlib.sha256(capture_path.read_bytes()).hexdigest()
        == FIXTURE["provenance"]["exporter_sha256"]
    )
    assert (
        hashlib.sha256(lock_path.read_bytes()).hexdigest()
        == FIXTURE["provenance"]["lockfile_sha256"]
    )
