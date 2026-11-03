"""Blender 5.1 native rotation captures checked against QuatOps and SciPy."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from quatops.engines import SignedAxisMap
from tests.native_engine_support import (
    basis_images,
    from_engine_xyzw,
    from_euler,
    quaternion_coefficient_error,
    quaternion_xyzw,
    to_hamilton_xyzw,
    vector_xyz,
)
from tests.oracle_support import BACKENDS, orient_matrices, orient_vectors


FIXTURE_PATH = Path(__file__).parent / "golden" / "blender" / "5.1.2" / "native_rotation.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
EXTENDED = FIXTURE["extended_conversions"]
ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
TOLERANCE = 5.0e-5
READBACK_TOLERANCE = 3.0e-4


def _angles_xyz(row, key="input_degrees_xyz"):
    values = row[key]
    return {axis: float(values[axis]) for axis in "xyz"}


def _ordered(angles_xyz, order):
    return tuple(angles_xyz[axis.lower()] for axis in order)


def _ordered_readback(values_xyz, order):
    if not hasattr(values_xyz, "keys"):
        values_xyz = dict(zip("xyz", values_xyz))
    return tuple(float(values_xyz[axis.lower()]) for axis in order)


def _rotation(xyzw):
    return Rotation.from_quat(np.asarray(xyzw, dtype=float))


def _same_orientation(lhs_xyzw, rhs_xyzw, tolerance, label):
    np.testing.assert_allclose(
        _rotation(lhs_xyzw).as_matrix(),
        _rotation(rhs_xyzw).as_matrix(),
        rtol=0.0,
        atol=tolerance,
        err_msg=label,
    )


def _backend_xyzw(module, value, is_jpl, is_numpy, component_first):
    return to_hamilton_xyzw(module, value, is_jpl, is_numpy, component_first)


def test_blender_fixture_records_its_pinned_runtime_and_explicit_frame():
    assert FIXTURE["evidence_kind"] == "native_blender_runtime_capture"
    assert FIXTURE["engine"]["name"] == "Blender"
    assert FIXTURE["engine"]["version"] == "5.1.2"
    assert FIXTURE["engine"]["build_hash"] == "ec6e62d40fa9"
    assert FIXTURE["engine"]["python_version"] == "3.13.9"
    assert FIXTURE["engine"]["precision_probe"]["euler_matches_float32_rounding"]
    assert FIXTURE["engine"]["precision_probe"]["quaternion_matches_float32_rounding"]
    assert FIXTURE["source_frame"]["handedness"] == "right-handed"
    assert FIXTURE["source_frame"]["positive_x"] == "right"
    assert FIXTURE["source_frame"]["positive_y"] == "forward"
    assert FIXTURE["source_frame"]["positive_z"] == "up"
    assert "not inferred" in FIXTURE["source_frame"]["declaration"]
    assert (
        FIXTURE["blender_world_frame_probe"]["semantic_imported_object_forward_axis"]
        == "not assumed"
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_blender_all_six_euler_orders_match_native_scipy_and_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    rows = EXTENDED["euler_orders"]
    assert len(rows) == 6 * 87
    assert {row["order"] for row in rows} == set(ORDERS)

    intrinsic_errors = {order: [] for order in ORDERS}
    extrinsic_errors = {order: [] for order in ORDERS}
    for row in rows:
        order = row["order"]
        angles_xyz = _angles_xyz(row)
        ordered = _ordered(angles_xyz, order)
        native_xyzw = quaternion_xyzw(row["euler_quaternion"])
        native_matrix = _rotation(native_xyzw).as_matrix()

        # The runtime data determines the case: Blender's uppercase order
        # labels match SciPy's extrinsic lowercase sequence for these probes.
        scipy_extrinsic = Rotation.from_euler(order.lower(), ordered, degrees=True)
        scipy_intrinsic = Rotation.from_euler(order, ordered, degrees=True)
        ext_error = np.max(np.abs(native_matrix - scipy_extrinsic.as_matrix()))
        int_error = np.max(np.abs(native_matrix - scipy_intrinsic.as_matrix()))
        extrinsic_errors[order].append(ext_error)
        intrinsic_errors[order].append(int_error)
        np.testing.assert_allclose(
            native_matrix,
            scipy_extrinsic.as_matrix(),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"Blender {order} {row['case_id']} vs SciPy extrinsic route",
        )
        np.testing.assert_allclose(
            np.asarray(row["euler_matrix_rows"], dtype=float),
            scipy_extrinsic.as_matrix(),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"Blender {order} {row['case_id']} native Euler matrix",
        )
        np.testing.assert_allclose(
            basis_images(row["quaternion_rotated_positive_basis_vectors_xyz"]),
            scipy_extrinsic.as_matrix().T,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"Blender {order} {row['case_id']} native quaternion vector action",
        )
        np.testing.assert_allclose(
            basis_images(row["matrix_rotated_positive_basis_vectors_xyz"]),
            scipy_extrinsic.as_matrix().T,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"Blender {order} {row['case_id']} native matrix vector action",
        )

        actual = from_euler(module, ordered, order.lower(), True, component_first)
        actual_xyzw = _backend_xyzw(module, actual, is_jpl, is_numpy, component_first)
        assert quaternion_coefficient_error(actual_xyzw, native_xyzw) <= TOLERANCE, (
            f"{backend_name}: Blender {order} {row['case_id']} quaternion coefficients"
        )
        np.testing.assert_allclose(
            _rotation(actual_xyzw).as_matrix(),
            scipy_extrinsic.as_matrix(),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{backend_name}: Blender {order} {row['case_id']} QuatOps Euler",
        )

        matrix_native_quaternion = quaternion_xyzw(row["matrix_quaternion"])
        _same_orientation(
            matrix_native_quaternion,
            native_xyzw,
            TOLERANCE,
            f"Blender {order} {row['case_id']} matrix-to-quaternion",
        )
        actual_from_matrix = module.from_rotation_matrix(
            orient_matrices(np.asarray(row["euler_matrix_rows"], dtype=float), component_first),
            atol=TOLERANCE,
        )
        actual_from_matrix_xyzw = _backend_xyzw(
            module, actual_from_matrix, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            actual_from_matrix_xyzw,
            matrix_native_quaternion,
            TOLERANCE,
            f"{backend_name}: Blender {order} {row['case_id']} QuatOps matrix route",
        )

        for readback_key, roundtrip_key in (
            ("quaternion_to_euler_radians_xyz", "quaternion_readback_roundtrip"),
            ("matrix_to_euler_radians_xyz", "matrix_readback_roundtrip"),
        ):
            readback = vector_xyz(row[readback_key])
            rebuilt = from_euler(
                module,
                _ordered_readback(readback, order),
                order.lower(),
                False,
                component_first,
            )
            rebuilt_xyzw = _backend_xyzw(
                module, rebuilt, is_jpl, is_numpy, component_first
            )
            _same_orientation(
                rebuilt_xyzw,
                quaternion_xyzw(row[roundtrip_key]),
                READBACK_TOLERANCE,
                f"{backend_name}: Blender {order} {row['case_id']} {readback_key}",
            )

    for order in ORDERS:
        assert max(extrinsic_errors[order]) <= TOLERANCE
        assert max(intrinsic_errors[order]) > 0.01


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_blender_native_composition_and_basis_action_match_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    frame_probe = FIXTURE["blender_world_frame_probe"]
    np.testing.assert_array_equal(frame_probe["positive_x_cross_positive_y"], (0.0, 0.0, 1.0))
    native_map = FIXTURE["source_to_blender_axis_map"]
    axis_map = SignedAxisMap(tuple(native_map["destination_axes_from_source"]))
    assert axis_map.destination_axes == ("x", "y", "z")
    assert axis_map.determinant == native_map["determinant"] == 1
    for probe in native_map["principal_rotation_probes"]:
        mapped = axis_map.map_hamilton_xyzw(
            quaternion_xyzw(probe["source_quaternion"])
        )
        expected = quaternion_xyzw(probe["blender_quaternion"])
        np.testing.assert_allclose(mapped, expected, rtol=0.0, atol=TOLERANCE)
        np.testing.assert_allclose(
            basis_images(probe["blender_rotated_positive_basis_vectors_xyz"]),
            _rotation(mapped).as_matrix().T,
            rtol=0.0,
            atol=TOLERANCE,
        )
    for row in FIXTURE["cross_products"]:
        expected = {
            "x_cross_y": (0.0, 0.0, 1.0),
            "y_cross_z": (1.0, 0.0, 0.0),
            "z_cross_x": (0.0, 1.0, 0.0),
        }[row["case_id"]]
        np.testing.assert_array_equal(row["result_xyz"], expected)
        np.testing.assert_array_equal(row["reversed_result_xyz"], -np.asarray(expected))

    case = FIXTURE["quaternion_composition"]
    lhs = from_engine_xyzw(
        module, quaternion_xyzw(case["lhs_quaternion"]), is_jpl, is_numpy, component_first
    )
    rhs = from_engine_xyzw(
        module, quaternion_xyzw(case["rhs_quaternion"]), is_jpl, is_numpy, component_first
    )
    product = module.multiply(lhs, rhs)
    product_image = np.asarray(
        module.rotate(product, vector_xyz(case["probe_vector_xyz"])), dtype=float
    )
    np.testing.assert_allclose(
        product_image,
        vector_xyz(case["product_rotated_vector_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: Blender quaternion composition action",
    )
    np.testing.assert_allclose(
        vector_xyz(case["product_rotated_vector_xyz"]),
        vector_xyz(case["rhs_then_lhs_rotated_vector_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
    )
    assert case["product_matches_rhs_then_lhs"]
    assert not case["product_matches_lhs_then_rhs"]


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_blender_matrix_axis_angle_and_rotation_vector_routes_match_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for case in EXTENDED["matrix_to_quaternion"]:
        matrix = np.asarray(case["matrix_rows"], dtype=float)
        scipy_matrix = Rotation.from_quat(
            quaternion_xyzw(case["expected_quaternion"])
        ).as_matrix()
        np.testing.assert_allclose(matrix, scipy_matrix, rtol=0.0, atol=TOLERANCE)
        actual = module.from_rotation_matrix(
            orient_matrices(matrix, component_first), atol=TOLERANCE
        )
        actual_xyzw = _backend_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            actual_xyzw,
            quaternion_xyzw(case["matrix_quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender matrix {case['case_id']}",
        )

    for case in EXTENDED["axis_angle"]:
        axis = np.asarray(case["input_axis_xyz"], dtype=float)
        angle = float(case["input_angle_radians"])
        actual = module.from_axis_angle(orient_vectors(np.r_[axis, angle], component_first))
        actual_xyzw = _backend_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            actual_xyzw,
            quaternion_xyzw(case["quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender axis-angle {case['case_id']} {case['input_sign']}",
        )
        read_axis = np.asarray(case["readback_axis_xyz"], dtype=float)
        read_angle = float(case["readback_angle_radians"])
        readback = module.from_axis_angle(
            orient_vectors(np.r_[read_axis, read_angle], component_first)
        )
        readback_xyzw = _backend_xyzw(
            module, readback, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            readback_xyzw,
            quaternion_xyzw(case["quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender axis-angle readback {case['case_id']} {case['input_sign']}",
        )

    for case in EXTENDED["rotation_vector"]:
        actual = module.from_rotvec(
            orient_vectors(np.asarray(case["input_rotation_vector_xyz"], dtype=float), component_first)
        )
        actual_xyzw = _backend_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            actual_xyzw,
            quaternion_xyzw(case["constructed_quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender rotvec constructor {case['case_id']}",
        )
        readback = module.from_rotvec(
            orient_vectors(np.asarray(case["readback_rotation_vector_xyz"], dtype=float), component_first)
        )
        readback_xyzw = _backend_xyzw(
            module, readback, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            readback_xyzw,
            quaternion_xyzw(case["readback_quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender rotvec readback {case['case_id']}",
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_blender_slerp_normalization_and_rotation_difference_match_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    by_case = {}
    for case in EXTENDED["slerp"]:
        by_case.setdefault(case["case_id"], []).append(case)
    for case_id, rows in by_case.items():
        start_xyzw = quaternion_xyzw(rows[0]["start_quaternion"])
        end_xyzw = quaternion_xyzw(rows[0]["end_quaternion"])
        scipy_slerp = Slerp(
            (0.0, 1.0),
            Rotation.from_quat(np.stack((start_xyzw, end_xyzw))),
        )
        start = from_engine_xyzw(
            module, start_xyzw, is_jpl, is_numpy, component_first
        )
        end = from_engine_xyzw(
            module, end_xyzw, is_jpl, is_numpy, component_first
        )
        for case in rows:
            factor = float(case["factor"])
            actual = module.slerp(start, end, factor, shortest=True)
            actual_xyzw = _backend_xyzw(
                module, actual, is_jpl, is_numpy, component_first
            )
            native_xyzw = quaternion_xyzw(case["interpolated_quaternion"])
            _same_orientation(
                actual_xyzw,
                native_xyzw,
                TOLERANCE,
                f"{backend_name}: Blender slerp {case_id} t={factor}",
            )
            _same_orientation(
                native_xyzw,
                scipy_slerp([factor]).as_quat()[0],
                TOLERANCE,
                f"Blender slerp {case_id} t={factor} vs SciPy",
            )

    for case in EXTENDED["normalization"]:
        if case["error"] is not None:
            continue
        if case["case_id"] == "zero":
            assert quaternion_xyzw(case["normalized_quaternion"]).tolist() == [
                1.0, 0.0, 0.0, 0.0
            ]
            continue
        input_wxyz = np.asarray(case["input_wxyz"], dtype=float)
        input_xyzw = input_wxyz[[1, 2, 3, 0]]
        value = from_engine_xyzw(
            module, input_xyzw, is_jpl, is_numpy, component_first
        )
        normalized = module.normalize(value)
        normalized_xyzw = _backend_xyzw(
            module, normalized, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            normalized_xyzw,
            quaternion_xyzw(case["normalized_quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender normalize {case['case_id']}",
        )

    for case in EXTENDED["rotation_difference"]:
        start = from_engine_xyzw(
            module, quaternion_xyzw(case["start_quaternion"]), is_jpl, is_numpy, component_first
        )
        end = from_engine_xyzw(
            module, quaternion_xyzw(case["end_quaternion"]), is_jpl, is_numpy, component_first
        )
        relative = module.multiply(module.inverse(start), end)
        relative_xyzw = _backend_xyzw(
            module, relative, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            relative_xyzw,
            quaternion_xyzw(case["difference_quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender rotation_difference {case['case_id']}",
        )
        assert case["matches_start_inverse_then_end"]
        assert not case["matches_end_then_start_inverse"] or case["case_id"] == "same_orientation_sign_pair"


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_blender_swing_twist_decomposition_reconstructs_native_rotations(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for case in EXTENDED["swing_twist"]:
        axis = case["twist_axis"].lower()
        axis_xyz = np.eye(3)["xyz".index(axis)]
        swing = from_engine_xyzw(
            module, quaternion_xyzw(case["swing_quaternion"]), is_jpl, is_numpy, component_first
        )
        twist = module.from_axis_angle(
            orient_vectors(
                np.r_[axis_xyz, float(case["twist_angle_radians"])], component_first
            )
        )
        reconstructed = module.multiply(swing, twist)
        reconstructed_xyzw = _backend_xyzw(
            module, reconstructed, is_jpl, is_numpy, component_first
        )
        _same_orientation(
            reconstructed_xyzw,
            quaternion_xyzw(case["quaternion"]),
            TOLERANCE,
            f"{backend_name}: Blender swing-twist {case['case_id']} about {axis}",
        )
        assert case["swing_then_twist_matches"]
        if case["case_id"] == "mixed_rotation":
            assert not case["twist_then_swing_matches"]


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_blender_euler_compatibility_trajectories_match_quatops(
    backend_name, module, is_jpl, is_numpy, component_first
):
    for trajectory in EXTENDED["euler_compat_trajectories"]:
        order = trajectory["order"]
        frames = trajectory["frames"]
        for frame in frames:
            native_xyzw = quaternion_xyzw(frame["quaternion"])
            compatible = vector_xyz(frame["compatible_to_euler_radians_xyz"])
            rebuilt = from_euler(
                module,
                _ordered_readback(compatible, order),
                order.lower(),
                False,
                component_first,
            )
            rebuilt_xyzw = _backend_xyzw(
                module, rebuilt, is_jpl, is_numpy, component_first
            )
            _same_orientation(
                rebuilt_xyzw,
                native_xyzw,
                READBACK_TOLERANCE,
                f"{backend_name}: Blender {order} {trajectory['trajectory_id']} frame {frame['frame']}",
            )

        if trajectory["trajectory_id"] in {
            "x_crosses_positive_180",
            "x_crosses_negative_180",
        }:
            ordinary = np.asarray(
                [
                    [frame["ordinary_to_euler_radians_xyz"][axis] for axis in "xyz"]
                    for frame in frames
                ]
            )
            compatible = np.asarray(
                [
                    [frame["compatible_to_euler_radians_xyz"][axis] for axis in "xyz"]
                    for frame in frames
                ]
            )
            ordinary_max_jump = np.max(np.abs(np.diff(ordinary, axis=0)))
            compatible_max_jump = np.max(np.abs(np.diff(compatible, axis=0)))
            assert ordinary_max_jump > 6.0
            assert compatible_max_jump < 0.1


def test_blender_scene_object_rotation_modes_match_mathutils_routes():
    case = FIXTURE["scene_object_rotation"]
    euler_mode = case["euler_mode"]
    quaternion_mode = case["quaternion_mode"]
    assert euler_mode["rotation_mode"] == "XYZ"
    assert quaternion_mode["rotation_mode"] == "QUATERNION"
    assert len(FIXTURE["cases"]) == 15
    expected = Rotation.from_euler(
        "xyz",
        tuple(float(case["input_euler_radians_xyz"][axis]) for axis in "xyz"),
    ).as_matrix()
    np.testing.assert_allclose(
        np.asarray(euler_mode["matrix_world_3x3_rows"], dtype=float),
        expected,
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(quaternion_mode["matrix_world_3x3_rows"], dtype=float),
        expected,
        rtol=0.0,
        atol=TOLERANCE,
    )
    _same_orientation(
        quaternion_xyzw(euler_mode["matrix_world_quaternion"]),
        quaternion_xyzw(quaternion_mode["matrix_world_quaternion"]),
        TOLERANCE,
        "Blender Object Euler and quaternion rotation modes",
    )
