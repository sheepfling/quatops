"""Unity Editor runtime captures checked against QuatOps representations."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quatops.engines import SignedAxisMap
from tests.native_engine_support import (
    basis_images,
    from_engine_xyzw,
    from_euler,
    matrix_rows,
    quaternion_coefficient_error,
    quaternion_xyzw,
    rotate_basis,
    to_hamilton_xyzw,
    vector_xyz,
)
from tests.oracle_support import BACKENDS

FIXTURE_PATH = Path(__file__).parent / "golden" / "unity" / "6000.5.2f1" / "native_rotation.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
NATIVE = FIXTURE
STATIC_FIXTURES = json.loads(
    (Path(__file__).parent / "golden" / "engine_euler_fixtures.json").read_text(encoding="utf-8")
)["cases"]
TOLERANCE = 4e-6


def _input_angles(case):
    value = case["input_degrees_xyz"]
    return {"x": value["x"], "y": value["y"], "z": value["z"]}


def _ordered_angles(angles_xyz, order):
    return tuple(angles_xyz[axis] for axis in order.lower())


def _api_case(case, api_name):
    angles = _input_angles(case)
    if api_name == "unity_engine":
        return (
            angles,
            "zxy",
            case["unity_engine_quaternion_xyzw"],
            case["unity_engine_rotated_positive_basis_vectors_xyz"],
        )
    native = case["unity_mathematics_euler_xyz"]
    return (
        angles,
        "xyz",
        native["quaternion_xyzw"],
        native["rotated_positive_basis_vectors_xyz"],
    )


@pytest.mark.parametrize("api_name", ("unity_engine", "unity_mathematics"))
@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_native_euler_rotations_and_basis_images_match_quatops(
    api_name, case, backend_name, module, is_jpl, is_numpy, component_first
):
    angles_xyz, sequence, native_quat, native_axes = _api_case(case, api_name)
    input_angles = _ordered_angles(angles_xyz, sequence)
    actual = from_euler(module, input_angles, sequence, True, component_first)
    actual_xyzw = to_hamilton_xyzw(module, actual, is_jpl, is_numpy, component_first)
    native_xyzw = quaternion_xyzw(native_quat)
    assert quaternion_coefficient_error(actual_xyzw, native_xyzw) <= TOLERANCE, (
        f"{backend_name}: {case['case_id']} {api_name} quaternion coefficients"
    )
    expected_matrix = Rotation.from_quat(native_xyzw).as_matrix()
    actual_matrix = Rotation.from_quat(actual_xyzw).as_matrix()

    np.testing.assert_allclose(
        actual_matrix,
        expected_matrix,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: {case['case_id']} {api_name} orientation",
    )
    native_basis = basis_images(native_axes)
    np.testing.assert_allclose(
        native_basis,
        expected_matrix.T,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{case['case_id']} {api_name} native vector images",
    )
    np.testing.assert_allclose(
        rotate_basis(module, actual),
        native_basis,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: {case['case_id']} {api_name} basis images",
    )

    if api_name == "unity_engine":
        np.testing.assert_allclose(
            matrix_rows(case["unity_engine_matrix_rows"]),
            expected_matrix,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{case['case_id']} UnityEngine Matrix4x4 rows",
        )
        readback_angles = vector_xyz(case["unity_engine_euler_readback_degrees"])
        readback = from_euler(
            module,
            _ordered_angles(dict(zip("xyz", readback_angles)), sequence),
            sequence,
            True,
            component_first,
        )
        readback_xyzw = to_hamilton_xyzw(
            module, readback, is_jpl, is_numpy, component_first
        )
        native_readback_xyzw = quaternion_xyzw(
            case["unity_engine_euler_readback_roundtrip_quaternion_xyzw"]
        )
        np.testing.assert_allclose(
            Rotation.from_quat(readback_xyzw).as_matrix(),
            Rotation.from_quat(native_readback_xyzw).as_matrix(),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{backend_name}: {case['case_id']} Unity Euler readback roundtrip",
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("order", ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"))
def test_unity_mathematics_all_euler_orders_and_inverse_readbacks_match_quatops(
    order, backend_name, module, is_jpl, is_numpy, component_first
):
    rows = [
        row for row in FIXTURE["extended_conversions"]["unity_mathematics_euler_orders"]
        if row["order"] == order
    ]
    assert len(rows) == 37
    sequence = order.lower()
    for case in rows:
        input_xyz = _input_angles(case)
        ordered = _ordered_angles(input_xyz, sequence)
        actual = from_euler(module, ordered, sequence, True, component_first)
        actual_xyzw = to_hamilton_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        native_xyzw = quaternion_xyzw(case["quaternion_xyzw"])
        assert quaternion_coefficient_error(actual_xyzw, native_xyzw) <= TOLERANCE, (
            f"{backend_name}: Unity.Mathematics Euler{order} {case['case_id']} coefficients"
        )

        scipy_matrix = Rotation.from_euler(
            sequence, ordered, degrees=True
        ).as_matrix()
        native_matrix = Rotation.from_quat(native_xyzw).as_matrix()
        actual_matrix = Rotation.from_quat(actual_xyzw).as_matrix()
        np.testing.assert_allclose(
            native_matrix, scipy_matrix, rtol=0.0, atol=TOLERANCE,
            err_msg=f"Unity.Mathematics Euler{order} {case['case_id']} vs SciPy",
        )
        np.testing.assert_allclose(
            actual_matrix, scipy_matrix, rtol=0.0, atol=TOLERANCE,
            err_msg=f"{backend_name}: Euler{order} {case['case_id']} vs SciPy",
        )
        native_basis = basis_images(case["rotated_positive_basis_vectors_xyz"])
        np.testing.assert_allclose(native_basis, scipy_matrix.T, rtol=0.0, atol=TOLERANCE)
        np.testing.assert_allclose(
            rotate_basis(module, actual), native_basis, rtol=0.0, atol=TOLERANCE,
            err_msg=f"{backend_name}: Euler{order} {case['case_id']} basis images",
        )

        readback_xyz = vector_xyz(case["readback_degrees_xyz"])
        readback = from_euler(
            module,
            _ordered_angles(dict(zip("xyz", readback_xyz)), sequence),
            sequence,
            True,
            component_first,
        )
        readback_xyzw = to_hamilton_xyzw(
            module, readback, is_jpl, is_numpy, component_first
        )
        native_roundtrip = quaternion_xyzw(
            case["readback_roundtrip_quaternion_xyzw"]
        )
        np.testing.assert_allclose(
            Rotation.from_quat(readback_xyzw).as_matrix(),
            Rotation.from_quat(native_roundtrip).as_matrix(),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{backend_name}: Euler{order} {case['case_id']} inverse readback roundtrip",
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_euler_xyz_is_extrinsic_and_distinct_from_intrinsic_xyz(
    backend_name, module, is_jpl, is_numpy, component_first
):
    case = next(
        row for row in FIXTURE["extended_conversions"]["unity_mathematics_euler_orders"]
        if row["case_id"] == "diagnostic_10_20_30" and row["order"] == "XYZ"
    )
    expected = Rotation.from_euler("xyz", (10, 20, 30), degrees=True)
    native = Rotation.from_quat(
        quaternion_xyzw(case["quaternion_xyzw"])
    )
    extrinsic = from_euler(module, (10, 20, 30), "xyz", True, component_first)
    extrinsic_xyzw = to_hamilton_xyzw(
        module, extrinsic, is_jpl, is_numpy, component_first
    )
    intrinsic = from_euler(module, (10, 20, 30), "XYZ", True, component_first)
    intrinsic_xyzw = to_hamilton_xyzw(
        module, intrinsic, is_jpl, is_numpy, component_first
    )
    np.testing.assert_allclose(native.as_matrix(), expected.as_matrix(), rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(
        Rotation.from_quat(extrinsic_xyzw).as_matrix(),
        expected.as_matrix(),
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: lowercase xyz matches Unity EulerXYZ",
    )
    assert np.max(
        np.abs(Rotation.from_quat(intrinsic_xyzw).as_matrix() - expected.as_matrix())
    ) > 0.1


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("api_name", ("unity_engine", "unity_mathematics"))
def test_unity_quaternion_product_order_matches_native_runtime(
    api_name, backend_name, module, is_jpl, is_numpy, component_first
):
    native = FIXTURE["quaternion_composition"][api_name]
    lhs = from_engine_xyzw(
        module, quaternion_xyzw(native["lhs_xyzw"]), is_jpl, is_numpy, component_first
    )
    rhs = from_engine_xyzw(
        module, quaternion_xyzw(native["rhs_xyzw"]), is_jpl, is_numpy, component_first
    )
    product = module.multiply(lhs, rhs)
    image = module.rotate(product, vector_xyz(native["probe_vector_xyz"]))
    np.testing.assert_allclose(
        np.asarray(image, dtype=float),
        vector_xyz(native["product_rotated_vector_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: {api_name} product vector",
    )
    np.testing.assert_allclose(
        vector_xyz(native["product_rotated_vector_xyz"]),
        vector_xyz(native["rhs_then_lhs_rotated_vector_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
    )
    assert native["product_matches_rhs_then_lhs"]
    assert not np.allclose(
        vector_xyz(native["product_rotated_vector_xyz"]),
        vector_xyz(native["lhs_then_rhs_rotated_vector_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
    )


def test_unity_transform_parent_child_order_matches_native_runtime():
    native = FIXTURE["transform_composition"]
    np.testing.assert_allclose(
        vector_xyz(native["parent_child_transform_position_xyz"]),
        vector_xyz(native["child_local_then_parent_world_position_xyz"]),
        rtol=0.0,
        atol=8e-6,
    )
    assert native["parent_child_matches_child_then_parent"]
    assert not np.allclose(
        vector_xyz(native["parent_child_transform_position_xyz"]),
        vector_xyz(native["parent_world_then_child_local_position_xyz"]),
        rtol=0.0,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_transform_world_rotation_matches_parent_times_local(
    backend_name, module, is_jpl, is_numpy, component_first
):
    native = FIXTURE["transform_composition"]
    parent = from_engine_xyzw(
        module,
        quaternion_xyzw(native["parent_rotation_xyzw"]),
        is_jpl,
        is_numpy,
        component_first,
    )
    child_local = from_engine_xyzw(
        module,
        quaternion_xyzw(native["child_local_rotation_xyzw"]),
        is_jpl,
        is_numpy,
        component_first,
    )
    actual_xyzw = to_hamilton_xyzw(
        module, module.multiply(parent, child_local), is_jpl, is_numpy, component_first
    )
    expected_xyzw = quaternion_xyzw(native["child_world_rotation_xyzw"])
    np.testing.assert_allclose(
        Rotation.from_quat(actual_xyzw).as_matrix(),
        Rotation.from_quat(expected_xyzw).as_matrix(),
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: Unity child world rotation",
    )


def test_unity_euler_readback_roundtrip_residual_matches_native_error_metric():
    recorded_errors = []
    for case in CASES:
        source = quaternion_xyzw(case["unity_engine_quaternion_xyzw"])
        roundtrip = quaternion_xyzw(
            case["unity_engine_euler_readback_roundtrip_quaternion_xyzw"]
        )
        scipy_error = np.rad2deg(
            (Rotation.from_quat(source).inv() * Rotation.from_quat(roundtrip)).magnitude()
        )
        recorded = case["unity_engine_euler_readback_roundtrip_angle_error_degrees"]
        assert abs(recorded - scipy_error) <= 0.002
        recorded_errors.append(recorded)

    near_gimbal = CASES[next(i for i, case in enumerate(CASES) if case["case_id"] == "near_gimbal")]
    assert near_gimbal["unity_engine_euler_readback_roundtrip_angle_error_degrees"] > 0.05
    assert max(recorded_errors) > 0.05


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_cross_products_match_native_axis_convention(
    backend_name, module, is_jpl, is_numpy, component_first
):
    expected = {
        "x_cross_y": (0.0, 0.0, 1.0),
        "y_cross_z": (1.0, 0.0, 0.0),
        "z_cross_x": (0.0, 1.0, 0.0),
    }
    for case in FIXTURE["cross_products"]:
        value = expected[case["case_id"]]
        np.testing.assert_array_equal(vector_xyz(case["unity_engine_result_xyz"]), value)
        np.testing.assert_array_equal(vector_xyz(case["unity_mathematics_result_xyz"]), value)
        np.testing.assert_array_equal(vector_xyz(case["unity_engine_reversed_result_xyz"]), -np.asarray(value))
        np.testing.assert_array_equal(vector_xyz(case["unity_mathematics_reversed_result_xyz"]), -np.asarray(value))


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_explicit_source_mapping_matches_native_quaternion_actions(
    backend_name, module, is_jpl, is_numpy, component_first
):
    assert FIXTURE["source_frame"] == {
        "handedness": "right-handed",
        "positive_x": "right",
        "positive_y": "forward",
        "positive_z": "up",
    }
    assert FIXTURE["unity_world_frame"] == {
        "handedness": "left-handed",
        "positive_x": "right",
        "positive_y": "up",
        "positive_z": "forward",
    }
    native_map = FIXTURE["source_to_unity_axis_map"]
    axis_map = SignedAxisMap(tuple(native_map["destination_axes_from_source"]))
    assert axis_map.destination_axes == ("x", "z", "y")
    assert axis_map.determinant == native_map["determinant"] == -1
    for probe in native_map["principal_rotation_probes"]:
        mapped = axis_map.map_hamilton_xyzw(quaternion_xyzw(probe["source_quaternion_xyzw"]))
        expected = quaternion_xyzw(probe["unity_quaternion_xyzw"])
        np.testing.assert_allclose(mapped, expected, rtol=0.0, atol=TOLERANCE)
        value = from_engine_xyzw(module, mapped, is_jpl, is_numpy, component_first)
        np.testing.assert_allclose(
            rotate_basis(module, value),
            basis_images(probe["unity_rotated_positive_basis_vectors_xyz"]),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{backend_name}: {probe['source_rotation_axis']} axis map",
        )
        assert probe["preserves_mapped_rotation_action"]


def test_unity_native_math_fixture_agrees_with_frozen_upstream_case():
    static = next(case for case in STATIC_FIXTURES if case["case_id"] == "unity_mathematics_XYZ")
    native = next(case for case in CASES if case["case_id"] == "mixed_negative")
    native_api = native["unity_mathematics_euler_xyz"]
    native_xyzw = quaternion_xyzw(native_api["quaternion_xyzw"])
    expected_xyzw = np.asarray(static["expected_quaternion_xyzw"], dtype=float)
    orientation_error = min(
        np.linalg.norm(native_xyzw - expected_xyzw),
        np.linalg.norm(native_xyzw + expected_xyzw),
    )
    assert orientation_error <= 1e-6
    np.testing.assert_allclose(
        Rotation.from_quat(native_xyzw).as_matrix(),
        np.asarray(static["expected_matrix_rows"], dtype=float),
        rtol=0.0,
        atol=1e-6,
    )


def test_unity_fixture_records_pinned_editor_and_math_module_versions():
    assert FIXTURE["evidence_kind"] == "native_unity_editor_runtime_capture"
    assert FIXTURE["quaternion_algebra"] == "Hamilton"
    assert FIXTURE["quaternion_order"] == "xyzw"
    assert FIXTURE["rotation_action"] == "active rotation applied to vectors"
    assert FIXTURE["engine"]["version"] == "6000.5.2f1"
    assert FIXTURE["engine"]["platform"] == "WindowsEditor"
    assert FIXTURE["engine"]["precision"] == "float32"
    assert FIXTURE["engine"]["mathematics_package"] == (
        "com.unity.mathematics 1.4.0 (built-in)"
    )
