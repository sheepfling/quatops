"""Godot headless runtime captures checked against QuatOps representations."""

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

FIXTURE_PATH = Path(__file__).parent / "golden" / "godot" / "4.8-dev1" / "native_rotation.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
GODOT_FIXTURES = {
    version: json.loads(
        (Path(__file__).parent / "golden" / "godot" / version / "native_rotation.json").read_text(
            encoding="utf-8"
        )
    )
    for version in ("4.7-stable", "4.8-dev1")
}
STATIC_FIXTURES = json.loads(
    (Path(__file__).parent / "golden" / "engine_euler_fixtures.json").read_text(encoding="utf-8")
)["cases"]
TOLERANCE = 3e-6
READBACK_TOLERANCE = 2e-5


def _angles(case):
    value = case["input_radians_xyz"]
    by_axis = {"x": value["x"], "y": value["y"], "z": value["z"]}
    return tuple(by_axis[axis.lower()] for axis in "YXZ")


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_godot_native_quaternion_and_basis_euler_match_quatops(
    case, version, backend_name, module, is_jpl, is_numpy, component_first
):
    case = next(
        row for row in GODOT_FIXTURES[version]["cases"]
        if row["case_id"] == case["case_id"]
    )
    native_xyzw = quaternion_xyzw(case["quaternion_from_euler_xyzw"])
    expected_matrix = Rotation.from_quat(native_xyzw).as_matrix()
    actual = from_euler(module, _angles(case), "YXZ", False, component_first)
    actual_xyzw = to_hamilton_xyzw(module, actual, is_jpl, is_numpy, component_first)
    assert quaternion_coefficient_error(actual_xyzw, native_xyzw) <= TOLERANCE, (
        f"{backend_name}: {case['case_id']} quaternion coefficients"
    )
    actual_matrix = Rotation.from_quat(actual_xyzw).as_matrix()

    np.testing.assert_allclose(
        actual_matrix,
        expected_matrix,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: {case['case_id']} Quaternion.from_euler orientation",
    )
    native_images = basis_images(case["quaternion_rotated_positive_basis_vectors_xyz"])
    np.testing.assert_allclose(
        native_images,
        expected_matrix.T,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{case['case_id']} Quaternion native basis images",
    )
    np.testing.assert_allclose(
        matrix_rows(case["basis_from_euler_rows"]),
        expected_matrix,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{case['case_id']} Basis.from_euler matrix rows",
    )
    np.testing.assert_allclose(
        basis_images(case["basis_from_euler_rotated_positive_basis_vectors_xyz"]),
        native_images,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{case['case_id']} Basis.from_euler vector images",
    )
    basis_quaternion = quaternion_xyzw(case["basis_quaternion_xyzw"])
    assert quaternion_coefficient_error(basis_quaternion, native_xyzw) <= TOLERANCE
    quaternion_basis_images = basis_images(case["quaternion_basis_columns_xyz"])
    np.testing.assert_allclose(
        quaternion_basis_images,
        native_images,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{case['case_id']} Basis(Quaternion) columns",
    )
    np.testing.assert_allclose(
        rotate_basis(module, actual),
        native_images,
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: {case['case_id']} QuatOps basis images",
    )

    readback = vector_xyz(case["quaternion_get_euler_radians_xyz"])
    readback_by_axis = dict(zip("xyz", readback))
    reconstructed = from_euler(
        module,
        tuple(readback_by_axis[axis.lower()] for axis in "YXZ"),
        "YXZ",
        False,
        component_first,
    )
    readback_xyzw = to_hamilton_xyzw(
        module, reconstructed, is_jpl, is_numpy, component_first
    )
    native_readback_xyzw = quaternion_xyzw(
        case["quaternion_euler_readback_roundtrip_xyzw"]
    )
    np.testing.assert_allclose(
        Rotation.from_quat(readback_xyzw).as_matrix(),
        Rotation.from_quat(native_readback_xyzw).as_matrix(),
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: {case['case_id']} Godot YXZ Euler readback roundtrip",
    )


@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_godot_all_six_basis_euler_orders_match_quatops_and_scipy(
    version, backend_name, module, is_jpl, is_numpy, component_first
):
    fixture = GODOT_FIXTURES[version]
    native_rows = fixture["extended_conversions"]["euler_orders"]
    assert {row["order"] for row in native_rows} == {
        "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
    }
    assert len(native_rows) == 67 * 6

    for row in native_rows:
        order = row["order"]
        input_degrees = vector_xyz(row["input_degrees_xyz"])
        ordered_degrees = tuple(input_degrees["XYZ".index(axis)] for axis in order)
        scipy_matrix = Rotation.from_euler(order, ordered_degrees, degrees=True).as_matrix()
        native_matrix = matrix_rows(row["basis_from_euler_rows"])
        np.testing.assert_allclose(
            native_matrix,
            scipy_matrix,
            rtol=0.0,
            atol=3e-6,
            err_msg=f"Godot {version} {row['case_id']} {order}: native Basis.from_euler vs SciPy",
        )
        np.testing.assert_allclose(
            basis_images(row["basis_from_euler_columns_xyz"]).T,
            scipy_matrix,
            rtol=0.0,
            atol=3e-6,
            err_msg=f"Godot {version} {row['case_id']} {order}: Basis columns",
        )
        np.testing.assert_allclose(
            basis_images(row["basis_from_euler_rotated_positive_basis_vectors_xyz"]),
            scipy_matrix.T,
            rtol=0.0,
            atol=3e-6,
            err_msg=f"Godot {version} {row['case_id']} {order}: Basis vector images",
        )

        actual = from_euler(
            module, ordered_degrees, order, True, component_first
        )
        actual_xyzw = to_hamilton_xyzw(
            module, actual, is_jpl, is_numpy, component_first
        )
        np.testing.assert_allclose(
            Rotation.from_quat(actual_xyzw).as_matrix(),
            scipy_matrix,
            rtol=0.0,
            atol=3e-6,
            err_msg=f"{backend_name}: Godot {version} {row['case_id']} {order} QuatOps Euler",
        )

        for route in (
            "basis_get_rotation_quaternion_xyzw",
            "quaternion_basis_constructor_xyzw",
        ):
            native_xyzw = quaternion_xyzw(row[route])
            np.testing.assert_allclose(
                Rotation.from_quat(native_xyzw).as_matrix(),
                scipy_matrix,
                rtol=0.0,
                atol=3e-6,
                err_msg=f"Godot {version} {row['case_id']} {order}: {route}",
            )
        assert quaternion_coefficient_error(
            quaternion_xyzw(row["basis_get_rotation_quaternion_xyzw"]),
            quaternion_xyzw(row["quaternion_basis_constructor_xyzw"]),
        ) <= 3e-6

        for readback_key, roundtrip_key in (
            ("basis_get_euler_radians_xyz", "basis_readback_roundtrip_quaternion_xyzw"),
            ("quaternion_get_euler_radians_xyz", "quaternion_readback_roundtrip_quaternion_xyzw"),
        ):
            readback_xyz = vector_xyz(row[readback_key])
            readback_ordered = tuple(readback_xyz["XYZ".index(axis)] for axis in order)
            rebuilt = from_euler(
                module, readback_ordered, order, False, component_first
            )
            rebuilt_xyzw = to_hamilton_xyzw(
                module, rebuilt, is_jpl, is_numpy, component_first
            )
            native_roundtrip = quaternion_xyzw(row[roundtrip_key])
            np.testing.assert_allclose(
                Rotation.from_quat(rebuilt_xyzw).as_matrix(),
                Rotation.from_quat(native_roundtrip).as_matrix(),
                rtol=0.0,
                atol=3e-6,
                err_msg=f"{backend_name}: Godot {version} {row['case_id']} {order} {readback_key} roundtrip",
            )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_quaternion_product_order_matches_native_runtime(
    version, backend_name, module, is_jpl, is_numpy, component_first
):
    native = GODOT_FIXTURES[version]["quaternion_composition"]
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
        err_msg=f"{backend_name}: Godot quaternion product vector",
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


@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_transform_product_order_matches_native_runtime(version):
    native = GODOT_FIXTURES[version]["transform_composition"]
    np.testing.assert_allclose(
        vector_xyz(native["lhs_times_rhs_position_xyz"]),
        vector_xyz(native["rhs_then_lhs_position_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
    )
    assert native["product_matches_rhs_then_lhs"]
    assert native["scale_xyz"] == {"x": 1.0, "y": 1.0, "z": 1.0}
    assert not np.allclose(
        vector_xyz(native["lhs_times_rhs_position_xyz"]),
        vector_xyz(native["lhs_then_rhs_position_xyz"]),
        rtol=0.0,
        atol=TOLERANCE,
    )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_transform_product_rotation_matches_native_runtime(
    version, backend_name, module, is_jpl, is_numpy, component_first
):
    native = GODOT_FIXTURES[version]["transform_composition"]
    assert native["product_rotation_matches_rhs_then_lhs"]
    lhs = from_engine_xyzw(
        module, quaternion_xyzw(native["lhs_rotation_xyzw"]), is_jpl, is_numpy, component_first
    )
    rhs = from_engine_xyzw(
        module, quaternion_xyzw(native["rhs_rotation_xyzw"]), is_jpl, is_numpy, component_first
    )
    actual_xyzw = to_hamilton_xyzw(
        module, module.multiply(lhs, rhs), is_jpl, is_numpy, component_first
    )
    expected_xyzw = quaternion_xyzw(native["lhs_times_rhs_rotation_xyzw"])
    np.testing.assert_allclose(
        Rotation.from_quat(actual_xyzw).as_matrix(),
        Rotation.from_quat(expected_xyzw).as_matrix(),
        rtol=0.0,
        atol=TOLERANCE,
        err_msg=f"{backend_name}: Godot Transform3D product rotation",
    )


@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_euler_readback_roundtrip_residual_matches_native_error_metric(version):
    cases = GODOT_FIXTURES[version]["cases"]
    recorded_errors = []
    for case in cases:
        source = quaternion_xyzw(case["quaternion_from_euler_xyzw"])
        roundtrip = quaternion_xyzw(case["quaternion_euler_readback_roundtrip_xyzw"])
        scipy_error = np.rad2deg(
            (Rotation.from_quat(source).inv() * Rotation.from_quat(roundtrip)).magnitude()
        )
        recorded = case["quaternion_euler_readback_roundtrip_angle_error_degrees"]
        assert abs(recorded - scipy_error) <= 0.002
        recorded_errors.append(recorded)

    near_gimbal = next(case for case in cases if case["case_id"] == "near_gimbal")
    assert near_gimbal["quaternion_euler_readback_roundtrip_angle_error_degrees"] > 0.0005
    assert max(recorded_errors) > 0.0005


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_cross_products_match_native_world_axis_convention(
    version, backend_name, module, is_jpl, is_numpy, component_first
):
    expected = {
        "x_cross_y": (0.0, 0.0, 1.0),
        "y_cross_z": (1.0, 0.0, 0.0),
        "z_cross_x": (0.0, 1.0, 0.0),
    }
    for case in GODOT_FIXTURES[version]["cross_products"]:
        value = np.asarray(expected[case["case_id"]])
        np.testing.assert_array_equal(vector_xyz(case["result_xyz"]), value)
        np.testing.assert_array_equal(vector_xyz(case["reversed_result_xyz"]), -value)


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_explicit_source_mapping_matches_native_quaternion_actions(
    version, backend_name, module, is_jpl, is_numpy, component_first
):
    fixture = GODOT_FIXTURES[version]
    assert fixture["source_frame"] == {
        "handedness": "right-handed",
        "positive_x": "right",
        "positive_y": "forward",
        "positive_z": "up",
    }
    assert fixture["godot_world_frame"] == {
        "handedness": "right-handed",
        "positive_x": "right",
        "positive_y": "up",
        "positive_z": "backward",
        "forward_direction": "negative_z",
    }
    native_map = fixture["source_to_godot_axis_map"]
    axis_map = SignedAxisMap(tuple(native_map["destination_axes_from_source"]))
    assert axis_map.destination_axes == ("x", "z", "-y")
    assert axis_map.determinant == native_map["determinant"] == 1
    for probe in native_map["principal_rotation_probes"]:
        mapped = axis_map.map_hamilton_xyzw(
            quaternion_xyzw(probe["source_quaternion_xyzw"])
        )
        expected = quaternion_xyzw(probe["godot_quaternion_xyzw"])
        np.testing.assert_allclose(mapped, expected, rtol=0.0, atol=TOLERANCE)
        value = from_engine_xyzw(module, mapped, is_jpl, is_numpy, component_first)
        np.testing.assert_allclose(
            rotate_basis(module, value),
            basis_images(probe["godot_rotated_positive_basis_vectors_xyz"]),
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{backend_name}: {probe['source_rotation_axis']} axis map",
        )
        assert probe["preserves_mapped_rotation_action"]


@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_native_basis_fixture_agrees_with_frozen_upstream_case(version):
    static = next(case for case in STATIC_FIXTURES if case["case_id"] == "godot_construct_basis_axes")
    native = next(
        case for case in GODOT_FIXTURES[version]["cases"]
        if case["case_id"] == "mixed_positive"
    )
    native_xyzw = quaternion_xyzw(native["basis_quaternion_xyzw"])
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


@pytest.mark.parametrize("version", tuple(GODOT_FIXTURES))
def test_godot_fixture_records_the_pinned_runtime_build(version):
    fixture = GODOT_FIXTURES[version]
    expected_builds = {
        "4.7-stable": ("4.7-stable (official)", "5b4e0cb0fd279832bbdd69fed5354d4e5ad26f88"),
        "4.8-dev1": ("4.8-dev1 (official)", "ebbf577a041d1ab8d7824b6f90e7d0575461ca45"),
    }
    assert fixture["evidence_kind"] == "native_godot_runtime_capture"
    assert fixture["quaternion_algebra"] == "Hamilton"
    assert fixture["quaternion_order"].startswith("xyzw")
    assert fixture["rotation_action"] == "active rotation applied to vectors"
    assert (fixture["engine"]["version"], fixture["engine"]["hash"]) == expected_builds[version]
    assert fixture["engine"]["platform"] == "Windows"
    assert fixture["engine"]["precision"] == "float32"
