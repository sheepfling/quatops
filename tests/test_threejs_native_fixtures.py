"""Three.js 0.187.0 native Node capture checked against SciPy and QuatOps."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from tests.native_engine_support import (
    from_engine_xyzw,
    from_euler,
    quaternion_coefficient_error,
    quaternion_xyzw,
    to_hamilton_xyzw,
)
from tests.oracle_support import BACKENDS, orient_matrices, orient_vectors


FIXTURE_PATH = (
    Path(__file__).parent
    / "golden"
    / "oracles"
    / "threejs"
    / "0.187.0"
    / "native_rotation.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
OBS = FIXTURE["observations"]
ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
CORPUS_SHA256 = "07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f"
TOLERANCE = 3e-12


def _rotation(xyzw):
    return Rotation.from_quat(np.asarray(xyzw, dtype=float))


def _orientation_error(lhs, rhs):
    return float(np.max(np.abs(_rotation(lhs).as_matrix() - _rotation(rhs).as_matrix())))


def _shortest_slerp_xyzw(q0, q1, t):
    """Independent unit-quaternion geodesic formula, including extrapolation."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        result = (1.0 - t) * a + t * b
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    result = (np.sin((1.0 - t) * theta) / sine) * a
    result += (np.sin(t * theta) / sine) * b
    return result / np.linalg.norm(result)


def _matrix4_from_column_major(values):
    return np.asarray(values, dtype=float).reshape((4, 4), order="F")


def _ordered_angles(xyz, order):
    return tuple(float(xyz["XYZ".index(axis)]) for axis in order)


def test_threejs_fixture_has_pinned_runtime_and_input_provenance():
    assert FIXTURE["evidence_kind"] == "native_threejs_node_runtime_capture"
    assert FIXTURE["target"]["name"] == "Three.js"
    assert FIXTURE["target"]["package_version"] == "0.187.0"
    assert FIXTURE["target"]["revision"] == "187"
    assert FIXTURE["target"]["numeric_representation"] == "JavaScript Number (binary64)"
    assert FIXTURE["provenance"]["corpus_sha256"] == CORPUS_SHA256
    assert FIXTURE["provenance"]["package_lock_sha256"]
    assert FIXTURE["provenance"]["exporter_sha256"]
    assert FIXTURE["target"]["node_version"] == "v24.21.0"
    assert FIXTURE["counts"] == {
        "euler_orders": 534,
        "euler_boundaries": 540,
        "axis_angle": 60,
        "quaternion_normalization": 140,
        "multiplication": 3,
        "shortest_slerp": 7,
        "independent_matrices": 11,
        "two_vector_alignment": 12,
    }


def test_threejs_native_euler_outputs_match_scipy_intrinsic_order_contract():
    rows = OBS["euler_orders"]
    assert len(rows) == 89 * 6
    assert {row["order"] for row in rows} == set(ORDERS)
    for row in rows:
        angles = _ordered_angles(row["input_radians_xyz"], row["order"])
        expected = Rotation.from_euler(row["order"], angles).as_matrix()
        native = _rotation(row["euler_quaternion_xyzw"]).as_matrix()
        np.testing.assert_allclose(native, expected, rtol=0.0, atol=TOLERANCE)
        np.testing.assert_allclose(row["euler_matrix_rows"], expected, rtol=0.0, atol=TOLERANCE)
        np.testing.assert_allclose(
            np.asarray(row["quaternion_rotated_positive_basis_vectors_xyz"]),
            expected.T,
            rtol=0.0,
            atol=TOLERANCE,
        )
        np.testing.assert_allclose(
            np.asarray(row["matrix_rotated_positive_basis_vectors_xyz"]),
            expected.T,
            rtol=0.0,
            atol=TOLERANCE,
        )
        assert _orientation_error(
            row["matrix_quaternion_xyzw"], row["euler_quaternion_xyzw"]
        ) <= TOLERANCE
        # The shared general inputs are well-conditioned for Three.js readback.
        assert _orientation_error(
            row["quaternion_readback_roundtrip_xyzw"], row["euler_quaternion_xyzw"]
        ) <= 3e-12


def test_threejs_native_euler_boundaries_keep_measured_near_lock_readback_behavior():
    rows = OBS["euler_boundaries"]
    assert len(rows) == 540
    for row in rows:
        expected = Rotation.from_euler(
            row["order"], _ordered_angles(row["input_radians_xyz"], row["order"])
        ).as_matrix()
        np.testing.assert_allclose(
            _rotation(row["euler_quaternion_xyzw"]).as_matrix(),
            expected,
            rtol=0.0,
            atol=TOLERANCE,
        )
        np.testing.assert_allclose(row["euler_matrix_rows"], expected, rtol=0.0, atol=TOLERANCE)
        assert row["input_source"]["actual_bits_hex"]

    # Three.js 0.187.0's Euler readback branch loses a small but observable
    # amount of orientation for selected cases just beyond gimbal lock. Keep
    # this native diagnostic instead of widening every conversion tolerance.
    near_lock = next(
        row
        for row in rows
        if row["case_id"] == "boundary-offset-ZYX-+1-+1e-04"
    )
    residual = _orientation_error(
        near_lock["quaternion_readback_roundtrip_xyzw"],
        near_lock["euler_quaternion_xyzw"],
    )
    assert residual > 1e-4


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_threejs_euler_orders_and_boundaries_match_all_quatops_backends(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for row in (*OBS["euler_orders"], *OBS["euler_boundaries"]):
        order = row["order"]
        xyz = row["input_radians_xyz"]
        expected = row["euler_quaternion_xyzw"]
        actual = from_euler(
            module,
            _ordered_angles(xyz, order),
            order,
            False,
            component_first,
        )
        actual_xyzw = to_hamilton_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        assert quaternion_coefficient_error(actual_xyzw, expected) <= 2e-11, (
            f"{backend_name}: Three.js {order} {row['case_id']}"
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_threejs_axis_angle_matrix_and_two_vector_routes_match_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for row in OBS["axis_angle"]:
        axis = np.asarray(row["input_axis_xyz"], dtype=float)
        angle = float(row["input_angle_radians"])
        scipy_matrix = Rotation.from_rotvec(axis * angle).as_matrix()
        native = row["quaternion_xyzw"]
        np.testing.assert_allclose(
            _rotation(native).as_matrix(), scipy_matrix, rtol=0.0, atol=TOLERANCE
        )
        actual = module.from_axis_angle(
            orient_vectors(np.r_[axis, angle], component_first)
        )
        actual_xyzw = to_hamilton_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        assert quaternion_coefficient_error(actual_xyzw, native) <= TOLERANCE, (
            f"{backend_name}: Three.js axis-angle {row['case_id']}"
        )

    for row in OBS["independent_matrices"]:
        if row["domain"] != "proper_rotation":
            # Three.js accepts the call but documents a pure rotation matrix
            # as a precondition; retain these raw diagnostics without making
            # them a conversion guarantee.
            continue
        expected = Rotation.from_matrix(np.asarray(row["input_matrix_rows"])).as_matrix()
        native = row["matrix_to_quaternion_xyzw"]
        np.testing.assert_allclose(_rotation(native).as_matrix(), expected, rtol=0.0, atol=TOLERANCE)
        np.testing.assert_allclose(
            row["quaternion_reconstructed_matrix_rows"], expected, rtol=0.0, atol=TOLERANCE
        )
        actual = module.from_rotation_matrix(
            orient_matrices(np.asarray(row["input_matrix_rows"]), component_first),
            atol=TOLERANCE,
        )
        actual_xyzw = to_hamilton_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        assert quaternion_coefficient_error(actual_xyzw, native) <= TOLERANCE, (
            f"{backend_name}: Three.js matrix {row['case_id']}"
        )

    for row in OBS["two_vector_alignment"]:
        from_vector = np.asarray(row["from_unit_xyz"], dtype=float)
        to_vector = np.asarray(row["to_unit_xyz"], dtype=float)
        q = from_engine_xyzw(
            module, row["quaternion_xyzw"], is_jpl, is_numpy, component_first
        )
        actual_vector = np.asarray(module.rotate(q, from_vector), dtype=float)
        np.testing.assert_allclose(
            actual_vector, to_vector, rtol=0.0, atol=TOLERANCE,
            err_msg=f"{backend_name}: Three.js two-vector alignment",
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_threejs_products_shortest_slerp_and_normalization_match_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for row in OBS["multiplication"]:
        a_xyzw = np.asarray(row["a_input_wxyz"])[[1, 2, 3, 0]]
        b_xyzw = np.asarray(row["b_input_wxyz"])[[1, 2, 3, 0]]
        a = from_engine_xyzw(module, a_xyzw, is_jpl, is_numpy, component_first)
        b = from_engine_xyzw(module, b_xyzw, is_jpl, is_numpy, component_first)
        for lhs, rhs, expected in (
            (a, b, row["a_multiply_b_xyzw"]),
            (b, a, row["b_multiply_a_xyzw"]),
        ):
            product = module.multiply(lhs, rhs)
            actual = to_hamilton_xyzw(
                module, product, is_jpl, is_numpy, component_first
            )
            np.testing.assert_allclose(
                actual, expected, rtol=0.0, atol=2e-13,
                err_msg=f"{backend_name}: Three.js product {row['case_id']}",
            )

    for row in OBS["shortest_slerp"]:
        start_xyzw = np.asarray(row["a_input_wxyz"])[[1, 2, 3, 0]]
        end_xyzw = np.asarray(row["b_input_wxyz"])[[1, 2, 3, 0]]
        start = from_engine_xyzw(
            module, start_xyzw, is_jpl, is_numpy, component_first
        )
        end = from_engine_xyzw(module, end_xyzw, is_jpl, is_numpy, component_first)
        for sample in row["samples"]:
            native = sample["quaternion_xyzw"]
            factor = sample["t"]
            expected = _shortest_slerp_xyzw(start_xyzw, end_xyzw, factor)
            assert _orientation_error(native, expected) <= 3e-12, (
                f"Three.js shortest SLERP {row['case_id']} t={factor}"
            )
            actual = module.slerp(start, end, factor, shortest=True)
            actual_xyzw = to_hamilton_xyzw(
                module, actual, is_jpl, is_numpy, component_first
            )
            assert _orientation_error(actual_xyzw, native) <= 3e-12, (
                f"{backend_name}: Three.js SLERP {row['case_id']} t={factor}"
            )

    for row in OBS["quaternion_normalization"]:
        source = np.asarray(row["input_xyzw"], dtype=float)
        if row["length"] == 0.0:
            # Three.js maps a zero quaternion to identity in normalize(); this
            # is recorded as native behavior, not asserted as QuatOps parity.
            assert row["normalized_xyzw"] == [0.0, 0.0, 0.0, 1.0]
            continue
        value = from_engine_xyzw(
            module, source, is_jpl, is_numpy, component_first
        )
        normalized = module.normalize(value)
        actual_xyzw = to_hamilton_xyzw(
            module, normalized, is_jpl, is_numpy, component_first
        )
        expected_xyzw = row["normalized_xyzw"]
        np.testing.assert_allclose(
            actual_xyzw, expected_xyzw, rtol=0.0, atol=TOLERANCE,
            err_msg=f"{backend_name}: Three.js normalize {row['case_id']}",
        )


def test_threejs_reorder_discards_turn_count_and_object3d_matrix_layout_is_explicit():
    reorder = OBS["euler_reorder"]
    assert reorder["source_order"] == "XYZ"
    assert reorder["target_order"] == "ZYX"
    assert _orientation_error(
        reorder["source_quaternion_xyzw"], reorder["reordered_quaternion_xyzw"]
    ) <= TOLERANCE
    assert abs(reorder["input_radians_xyz"][0]) > 2 * np.pi
    assert abs(reorder["reordered_euler_radians_xyz"][0]) < 2 * np.pi

    hierarchy = OBS["object3d_hierarchy"]
    parent = _matrix4_from_column_major(
        hierarchy["parent_local_matrix4_elements_column_major"]
    )
    child = _matrix4_from_column_major(
        hierarchy["child_local_matrix4_elements_column_major"]
    )
    world = _matrix4_from_column_major(
        hierarchy["child_world_matrix4_elements_column_major"]
    )
    np.testing.assert_allclose(world, parent @ child, rtol=0.0, atol=2e-14)


def test_threejs_fixture_hash_fields_match_checked_in_sources():
    script_path = (
        Path(__file__).parent.parent
        / "experiments"
        / "oracles"
        / "threejs"
        / "0.187.0"
        / "capture_three.mjs"
    )
    lock_path = script_path.with_name("package-lock.json")
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == FIXTURE["provenance"]["exporter_sha256"]
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == FIXTURE["provenance"]["package_lock_sha256"]
