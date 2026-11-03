"""QuatOps conformance checks against a frozen Unreal Engine runtime capture."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.oracle_support import (
    BACKENDS,
    COEFFICIENT_SIGN,
    as_hamilton_coefficients,
    from_wxyz,
    orient_vectors,
    to_wxyz,
)

FIXTURE_PATH = Path(__file__).parent / "golden" / "unreal" / "5.7.4" / "native_rotation.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
PRECISIONS = ("float32", "float64")
QUATERNION_ATOL = {"float32": 2e-6, "float64": 2e-14}
MATRIX_ATOL = {"float32": 2e-6, "float64": 2e-14}
READBACK_ORIENTATION_ATOL = {"float32": 5e-5, "float64": 1e-12}


def _engine_euler_input(rotator_fields):
    """Map named Unreal Pitch/Yaw/Roll fields to the captured xyz sequence."""
    return np.asarray(
        (-rotator_fields["roll"], -rotator_fields["pitch"], rotator_fields["yaw"]),
        dtype=float,
    )


def _native_xyzw_to_module_wxyz(values, is_jpl):
    wxyz = np.asarray(values, dtype=float)[[3, 0, 1, 2]]
    # The sign map is its own inverse, so it converts these Hamilton values to
    # the equivalent JPL coefficient representation when required.
    return wxyz * COEFFICIENT_SIGN if is_jpl else wxyz


def _from_engine_euler(module, fields, component_first):
    angles = _engine_euler_input(fields)
    angle_input = orient_vectors(angles, component_first)
    return module.from_euler(angle_input, "xyz", degrees=True)


def _as_hamilton_xyzw(module, value, is_jpl, is_numpy, component_first):
    wxyz = np.asarray(to_wxyz(module, value, is_numpy, component_first), dtype=float)
    wxyz = wxyz.reshape((-1, 4))[0]
    hamilton_wxyz = as_hamilton_coefficients(wxyz, is_jpl)
    return hamilton_wxyz[[1, 2, 3, 0]]


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unreal_rotator_quat_and_axes_match_quatops(
    case, precision, backend_name, module, is_jpl, is_numpy, component_first
):
    native = case[precision]
    expected_xyzw = np.asarray(native["rotator_quaternion_xyzw"], dtype=float)
    actual = _from_engine_euler(
        module,
        case["input_rotator_degrees"],
        component_first,
    )
    actual_xyzw = _as_hamilton_xyzw(
        module, actual, is_jpl, is_numpy, component_first
    )

    # Unreal's constructor and QuatOps may choose opposite quaternion signs for
    # the same rotation, so compare the represented orientation here.
    coefficient_error = min(
        np.linalg.norm(actual_xyzw - expected_xyzw),
        np.linalg.norm(actual_xyzw + expected_xyzw),
    )
    assert coefficient_error <= QUATERNION_ATOL[precision], backend_name

    expected_make_from_euler_input = np.asarray(
        (
            case["input_rotator_degrees"]["roll"],
            case["input_rotator_degrees"]["pitch"],
            case["input_rotator_degrees"]["yaw"],
        ),
        dtype=float,
    )
    np.testing.assert_allclose(
        native["make_from_euler_input_xyz_degrees"],
        expected_make_from_euler_input,
        rtol=0.0,
        atol=5e-6 if precision == "float32" else 2e-14,
        err_msg=f"{precision}: {case['case_id']} Unreal MakeFromEuler input order",
    )
    assert native["rotator_make_from_euler_matches_named_fields"]
    assert native["quat_make_from_euler_matches_named_fields"]
    for api_name in (
        "rotator_make_from_euler_quaternion_xyzw",
        "quat_make_from_euler_xyzw",
    ):
        api_xyzw = np.asarray(native[api_name], dtype=float)
        api_error = min(
            np.linalg.norm(actual_xyzw - api_xyzw),
            np.linalg.norm(actual_xyzw + api_xyzw),
        )
        assert api_error <= QUATERNION_ATOL[precision], (
            f"{backend_name}: {precision}: {case['case_id']} {api_name}"
        )

    expected_matrix = Rotation.from_quat(expected_xyzw).as_matrix()
    actual_matrix = Rotation.from_quat(actual_xyzw).as_matrix()
    native_matrix_rows = np.asarray(native["rotation_matrix_3x3_rows"], dtype=float)
    native_axis_images = np.asarray(
        native["rotated_positive_basis_vectors_xyz"], dtype=float
    )
    tolerance = MATRIX_ATOL[precision]

    # Unreal stores TQuat::ToMatrix in its row-vector layout. Its rows are the
    # RotateVector images of +X, +Y, and +Z, hence they match the transpose of
    # SciPy's conventional column-vector rotation matrix.
    np.testing.assert_allclose(
        native_matrix_rows, expected_matrix.T, rtol=0.0, atol=tolerance,
        err_msg=f"{precision}: {case['case_id']} native ToMatrix",
    )
    np.testing.assert_allclose(
        native_axis_images, expected_matrix.T, rtol=0.0, atol=tolerance,
        err_msg=f"{precision}: {case['case_id']} native RotateVector axes",
    )
    np.testing.assert_allclose(
        actual_matrix, expected_matrix, rtol=0.0, atol=tolerance,
        err_msg=f"{backend_name}: {precision}: {case['case_id']} matrix",
    )
    np.testing.assert_allclose(
        actual_matrix.T, native_axis_images, rtol=0.0, atol=tolerance,
        err_msg=f"{backend_name}: {precision}: {case['case_id']} basis images",
    )

    # Read both native Euler APIs back into QuatOps and compare by orientation;
    # Euler branch choices are pinned independently below.
    rotator_readback = native["quat_rotator_readback_degrees"]
    from_rotator_readback = _from_engine_euler(
        module, rotator_readback, component_first
    )
    euler_xyz = native["quat_euler_readback_degrees_xyz"]
    euler_fields = {"roll": euler_xyz[0], "pitch": euler_xyz[1], "yaw": euler_xyz[2]}
    from_euler_readback = _from_engine_euler(
        module, euler_fields, component_first
    )
    for readback_name, readback in (
        ("Rotator", from_rotator_readback),
        ("Euler", from_euler_readback),
    ):
        readback_xyzw = _as_hamilton_xyzw(
            module, readback, is_jpl, is_numpy, component_first
        )
        readback_matrix = Rotation.from_quat(readback_xyzw).as_matrix()
        readback_tolerance = (
            2e-3
            if case["case_id"].startswith("gimbal_threshold_")
            else READBACK_ORIENTATION_ATOL[precision]
        )
        np.testing.assert_allclose(
            readback_matrix,
            expected_matrix,
            rtol=0.0,
            atol=readback_tolerance,
            err_msg=(
                f"{backend_name}: {precision}: {case['case_id']} "
                f"{readback_name} readback orientation"
            ),
        )

    # The package's quaternion-to-Euler output must also reconstruct the same
    # orientation, without requiring Unreal's exact branch representation.
    quatops_euler = module.to_euler(actual, "xyz", degrees=True)
    quatops_euler = np.asarray(tuple(quatops_euler), dtype=float).reshape(-1, 3)[0]
    reconstructed = module.from_euler(quatops_euler, "xyz", degrees=True)
    reconstructed_xyzw = _as_hamilton_xyzw(
        module, reconstructed, is_jpl, is_numpy, component_first
    )
    np.testing.assert_allclose(
        Rotation.from_quat(reconstructed_xyzw).as_matrix(),
        expected_matrix,
        rtol=0.0,
        atol=tolerance,
        err_msg=f"{backend_name}: {precision}: {case['case_id']} QuatOps to_euler",
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unreal_quaternion_products_preserve_coefficients_and_signs(
    precision, backend_name, module, is_jpl, is_numpy, component_first
):
    native_products = FIXTURE[f"multiplication_cases_{precision}"]
    for native in native_products:
        lhs_wxyz = _native_xyzw_to_module_wxyz(native["lhs_xyzw"], is_jpl)
        rhs_wxyz = _native_xyzw_to_module_wxyz(native["rhs_xyzw"], is_jpl)
        lhs = from_wxyz(module, lhs_wxyz, is_numpy, component_first)
        rhs = from_wxyz(module, rhs_wxyz, is_numpy, component_first)
        product = module.multiply(lhs, rhs)
        actual_xyzw = _as_hamilton_xyzw(
            module, product, is_jpl, is_numpy, component_first
        )
        expected_xyzw = np.asarray(native["product_xyzw"], dtype=float)
        expected_sign_bits = np.asarray(native["product_xyzw_sign_bits"], dtype=bool)

        np.testing.assert_allclose(
            actual_xyzw,
            expected_xyzw,
            rtol=0.0,
            atol=QUATERNION_ATOL[precision],
            err_msg=(
                f"{backend_name}: {precision}: "
                f"{native['lhs_case_id']} * {native['rhs_case_id']}"
            ),
        )
        np.testing.assert_array_equal(np.signbit(expected_xyzw), expected_sign_bits)
        np.testing.assert_array_equal(
            np.signbit(actual_xyzw),
            expected_sign_bits,
            err_msg=f"{backend_name}: {precision}: multiplication coefficient signs",
        )


BRANCH_CONTRACT = {
    "float32": {
        "gimbal": ((90.0, 68.0, 0.0), (0.0, 90.0, 68.0)),
        "pitch_over_90": ((75.0, -157.0, 163.0), (163.0, 75.0, -157.0)),
        "yaw_positive_180": ((0.0, -180.0, 0.0), (0.0, 0.0, -180.0)),
        "yaw_negative_180": ((0.0, 180.0, 0.0), (0.0, 0.0, 180.0)),
    },
    "float64": {
        "gimbal": ((90.0, 68.0, 0.0), (0.0, 90.0, 68.0)),
        "pitch_over_90": ((75.0, -157.0, 163.0), (163.0, 75.0, -157.0)),
        "yaw_positive_180": ((0.0, 180.0, 0.0), (0.0, 0.0, 180.0)),
        "yaw_negative_180": ((0.0, -180.0, 0.0), (0.0, 0.0, -180.0)),
    },
}


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unreal_rotator_branch_and_wrapping_contract(precision):
    cases_by_id = {case["case_id"]: case for case in CASES}
    tolerance = 1e-4 if precision == "float32" else 2e-13
    for case_id, (expected_rotator, expected_euler) in BRANCH_CONTRACT[precision].items():
        native = cases_by_id[case_id][precision]
        rotator = native["quat_rotator_readback_degrees"]
        actual_rotator = (rotator["pitch"], rotator["yaw"], rotator["roll"])
        np.testing.assert_allclose(
            actual_rotator,
            expected_rotator,
            rtol=0.0,
            atol=tolerance,
            err_msg=f"{precision}: {case_id} FQuat.Rotator branch",
        )
        np.testing.assert_allclose(
            native["quat_euler_readback_degrees_xyz"],
            expected_euler,
            rtol=0.0,
            atol=tolerance,
            err_msg=f"{precision}: {case_id} FQuat.Euler branch",
        )


@pytest.mark.parametrize(
    "precision,expected_pitch",
    [
        ("float32", (89.9072036743164, 90.0, 90.0, 89.9072036743164)),
        ("float64", (89.9100000000077, 90.0, 90.0, 89.9100000000077)),
    ],
)
def test_unreal_rotator_readback_neighbors_pin_singularity_threshold(
    precision, expected_pitch
):
    cases_by_id = {case["case_id"]: case for case in CASES}
    neighbor_ids = (
        "gimbal_threshold_below_lower",
        "gimbal_threshold_above_lower",
        "gimbal_threshold_below_upper",
        "gimbal_threshold_above_upper",
    )
    actual_pitch = []
    for case_id, expected in zip(neighbor_ids, expected_pitch):
        native = cases_by_id[case_id][precision]
        rotator = native["quat_rotator_readback_degrees"]
        tolerance = 5e-3 if precision == "float32" else 2e-10
        assert rotator["pitch"] == pytest.approx(expected, abs=tolerance)
        actual_pitch.append(rotator["pitch"])

    # The 89.91/90.09 degree samples land on opposite sides of the branch
    # threshold; the two samples around 90 degrees enter the singular branch.
    assert actual_pitch[0] < 89.95
    assert actual_pitch[1] == pytest.approx(90.0, abs=1e-10)
    assert actual_pitch[2] == pytest.approx(90.0, abs=1e-10)
    assert actual_pitch[3] < 89.95
    upper = cases_by_id[neighbor_ids[3]][precision]["quat_rotator_readback_degrees"]
    assert upper["yaw"] == pytest.approx(180.0, abs=1e-10)
    assert upper["roll"] == pytest.approx(180.0, abs=1e-10)


def test_unreal_fixture_records_pinned_runtime_provenance():
    assert FIXTURE["evidence_kind"] == "native_unreal_runtime_capture"
    assert FIXTURE["engine"]["version"] == "5.7.4-51494982+++UE5+Release-5.7"
    assert FIXTURE["engine"]["platform"] == "WindowsEditor"
    assert FIXTURE["engine"]["platform_architecture"] == "x64"
    assert FIXTURE["engine"]["precision_types"] == "FQuat4f and FQuat4d"
