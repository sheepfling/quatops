"""Native Unreal experiments for coordinate and composition semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from quatops.engines import SignedAxisMap
from tests.oracle_support import BACKENDS, COEFFICIENT_SIGN, from_wxyz

FIXTURE_PATH = Path(__file__).parent / "golden" / "unreal" / "5.7.4" / "native_rotation.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
NATIVE = FIXTURE["native_semantics"]
PRECISIONS = ("float32", "float64")
ATOL = {"float32": 2e-6, "float64": 2e-14}
CROSS_PRODUCTS = {
    "x_cross_y": (0.0, 0.0, 1.0),
    "y_cross_z": (1.0, 0.0, 0.0),
    "z_cross_x": (0.0, 1.0, 0.0),
}


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unreal_native_cross_products_match_numeric_basis_identity(precision):
    for case in NATIVE[f"cross_products_{precision}"]:
        expected = np.asarray(CROSS_PRODUCTS[case["case_id"]])
        np.testing.assert_allclose(case["result_xyz"], expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            case["reversed_result_xyz"], -expected, rtol=0.0, atol=0.0
        )
        assert case["matches_numeric_basis_identity"]
        assert case["reversed_is_negative"]


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unreal_quaternion_and_transform_composition_orders(precision):
    quat = NATIVE[f"quaternion_composition_{precision}"]
    np.testing.assert_allclose(
        quat["lhs_times_rhs_rotated_vector_xyz"],
        quat["rhs_then_lhs_rotated_vector_xyz"],
        rtol=0.0,
        atol=ATOL[precision],
    )
    assert quat["product_matches_rhs_then_lhs"]
    assert not quat["product_matches_lhs_then_rhs"]
    assert not np.allclose(
        quat["lhs_then_rhs_rotated_vector_xyz"],
        quat["rhs_then_lhs_rotated_vector_xyz"],
        rtol=0.0,
        atol=ATOL[precision],
    )

    transform = NATIVE[f"transform_composition_{precision}"]
    np.testing.assert_allclose(
        transform["lhs_times_rhs_position_xyz"],
        transform["lhs_then_rhs_position_xyz"],
        rtol=0.0,
        atol=ATOL[precision],
    )
    assert transform["product_matches_lhs_then_rhs"]
    assert not transform["product_matches_rhs_then_lhs"]
    assert transform["scale_xyz"] == [1.0, 1.0, 1.0]
    assert not np.allclose(
        transform["lhs_then_rhs_position_xyz"],
        transform["rhs_then_lhs_position_xyz"],
        rtol=0.0,
        atol=ATOL[precision],
    )


def _backend_xyzw_to_value(module, xyzw, is_jpl, is_numpy, component_first):
    wxyz = np.asarray(xyzw, dtype=float)[[3, 0, 1, 2]]
    if is_jpl:
        wxyz = wxyz * COEFFICIENT_SIGN
    return from_wxyz(module, wxyz, is_numpy, component_first)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_explicit_rh_source_map_matches_native_unreal_rotations(
    precision, backend_name, module, is_jpl, is_numpy, component_first
):
    source = NATIVE["source_frame"]
    target = NATIVE["unreal_world_frame"]
    assert source == {
        "handedness": "right-handed",
        "positive_x": "right",
        "positive_y": "forward",
        "positive_z": "up",
    }
    assert target == {
        "handedness": "left-handed",
        "positive_x": "forward",
        "positive_y": "right",
        "positive_z": "up",
    }

    native_map = NATIVE["source_to_unreal_axis_map"]
    axis_map = SignedAxisMap(tuple(native_map["destination_axes_from_source"]))
    assert axis_map.destination_axes == ("y", "x", "z")
    assert axis_map.determinant == native_map["determinant"] == -1

    probes = native_map[f"mapped_source_principal_rotations_{precision}"]
    for native in probes:
        mapped_xyzw = np.asarray(
            axis_map.map_hamilton_xyzw(native["source_quaternion_xyzw"]), dtype=float
        )
        expected_xyzw = np.asarray(native["unreal_quaternion_xyzw"], dtype=float)
        np.testing.assert_allclose(
            mapped_xyzw,
            expected_xyzw,
            rtol=0.0,
            atol=ATOL[precision],
            err_msg=f"{backend_name}: {native['source_rotation_axis']} mapped XYZW",
        )

        quat = _backend_xyzw_to_value(
            module, mapped_xyzw, is_jpl, is_numpy, component_first
        )
        basis_images = np.stack(
            [
                np.asarray(module.rotate(quat, tuple(axis)), dtype=float).reshape(3)
                for axis in np.eye(3)
            ]
        )
        np.testing.assert_allclose(
            basis_images,
            native["unreal_rotated_positive_basis_vectors_xyz"],
            rtol=0.0,
            atol=ATOL[precision],
            err_msg=(
                f"{backend_name}: {precision}: "
                f"{native['source_rotation_axis']} native Unreal basis images"
            ),
        )
        assert native["preserves_mapped_rotation_action"]


def test_unreal_semantics_fixture_records_pinned_runtime_provenance():
    assert FIXTURE["evidence_kind"] == "native_unreal_runtime_capture"
    assert FIXTURE["quaternion_multiplication_order"] == (
        "lhs * rhs; Unreal applies rhs first, then lhs"
    )
    assert FIXTURE["engine"]["version"] == "5.7.4-51494982+++UE5+Release-5.7"
    assert FIXTURE["engine"]["platform"] == "WindowsEditor"
    assert FIXTURE["engine"]["platform_architecture"] == "x64"
