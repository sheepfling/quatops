"""QuatOps checks against native Unreal quaternion conversion captures."""

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
    orient_matrices,
    orient_vectors,
    to_wxyz,
)

FIXTURE_PATH = Path(__file__).parent / "golden" / "unreal" / "5.7.4" / "native_rotation.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
EXTENDED = FIXTURE["extended_conversions"]
PRECISIONS = ("float32", "float64")
ATOL = {"float32": 2e-6, "float64": 2e-14}


def _backend_from_engine_xyzw(module, xyzw, is_jpl, is_numpy, component_first):
    wxyz = np.asarray(xyzw, dtype=float)[[3, 0, 1, 2]]
    if is_jpl:
        wxyz = wxyz * COEFFICIENT_SIGN
    return from_wxyz(module, wxyz, is_numpy, component_first)


def _engine_xyzw_from_backend(module, value, is_jpl, is_numpy, component_first):
    wxyz = np.asarray(to_wxyz(module, value, is_numpy, component_first), dtype=float)
    wxyz = wxyz.reshape((-1, 4))[0]
    hamilton_wxyz = as_hamilton_coefficients(wxyz, is_jpl)
    return hamilton_wxyz[[1, 2, 3, 0]]


def _assert_same_rotation(actual_xyzw, expected_xyzw, atol, message):
    actual = Rotation.from_quat(np.asarray(actual_xyzw, dtype=float))
    expected = Rotation.from_quat(np.asarray(expected_xyzw, dtype=float))
    np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), rtol=0.0, atol=atol, err_msg=message)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unreal_matrix_to_quaternion_captures(precision):
    cases = EXTENDED[f"matrix_to_quaternion_{precision}"]
    assert len([case for case in cases if not case.get("diagnostic_only")]) >= 9
    for case in cases:
        if case.get("diagnostic_only"):
            # Unreal captures its native response for malformed/nonrotation
            # matrices. QuatOps intentionally validates proper rotation input.
            assert np.isfinite(case["constructed_quaternion_xyzw"]).all()
            continue
        matrix_rows = np.asarray(case["matrix_rows_native"], dtype=float)
        active_column_matrix = matrix_rows.T
        expected_xyzw = np.asarray(case["constructed_quaternion_xyzw"], dtype=float)
        assert case["same_rotation"]
        np.testing.assert_allclose(
            Rotation.from_quat(expected_xyzw).as_matrix(),
            active_column_matrix,
            rtol=0.0,
            atol=ATOL[precision],
            err_msg=f"{precision}: {case['case_id']} Unreal matrix storage",
        )

        for backend_name, module, is_jpl, is_numpy, component_first in BACKENDS:
            matrix_input = orient_matrices(active_column_matrix, component_first)
            actual = module.from_rotation_matrix(matrix_input, atol=ATOL[precision])
            actual_xyzw = _engine_xyzw_from_backend(
                module, actual, is_jpl, is_numpy, component_first
            )
            _assert_same_rotation(
                actual_xyzw,
                expected_xyzw,
                ATOL[precision],
                f"{backend_name}: {precision}: {case['case_id']} TQuat(matrix)",
            )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unreal_axis_angle_construction_and_branch_readbacks(
    precision, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = ATOL[precision]
    for case in EXTENDED[f"axis_angle_{precision}"]:
        case_tolerance = max(tolerance, 2e-7) if case["case_id"] == "tiny_angle" else tolerance
        axis = np.asarray(case["input_axis_xyz"], dtype=float)
        angle = case["input_angle_radians"]
        expected_xyzw = np.asarray(case["constructed_quaternion_xyzw"], dtype=float)
        module_input = orient_vectors(np.r_[axis, angle], component_first)
        actual = module.from_axis_angle(module_input)
        actual_xyzw = _engine_xyzw_from_backend(
            module, actual, is_jpl, is_numpy, component_first
        )
        _assert_same_rotation(
            actual_xyzw,
            expected_xyzw,
            case_tolerance,
            f"{backend_name}: {precision}: {case['case_id']} from_axis_angle",
        )

        negative_xyzw = np.asarray(case["negative_quaternion_xyzw"], dtype=float)
        assert np.array_equal(negative_xyzw, -expected_xyzw)
        native_representations = (
            (
                case["axis_angle_readback_axis_xyz"],
                case["axis_angle_readback_angle_radians"],
                expected_xyzw,
                case["rotation_vector_readback_xyz"],
                "axis-angle q",
            ),
            (
                case["negative_axis_angle_readback_axis_xyz"],
                case["negative_axis_angle_readback_angle_radians"],
                negative_xyzw,
                case["negative_rotation_vector_readback_xyz"],
                "axis-angle -q",
            ),
        )
        for read_axis, read_angle, source_xyzw, native_rotvec, label in native_representations:
            rebuilt = module.from_axis_angle(
                orient_vectors(np.r_[read_axis, read_angle], component_first)
            )
            rebuilt_xyzw = _engine_xyzw_from_backend(
                module, rebuilt, is_jpl, is_numpy, component_first
            )
            _assert_same_rotation(
                rebuilt_xyzw,
                source_xyzw,
                case_tolerance,
                f"{backend_name}: {precision}: {case['case_id']} {label} reconstruction",
            )
            rotvec = module.to_rotvec(
                _backend_from_engine_xyzw(
                    module, source_xyzw, is_jpl, is_numpy, component_first
                )
            )
            rotvec_q = module.from_rotvec(rotvec)
            rotvec_xyzw = _engine_xyzw_from_backend(
                module, rotvec_q, is_jpl, is_numpy, component_first
            )
            _assert_same_rotation(
                rotvec_xyzw,
                source_xyzw,
                case_tolerance,
                f"{backend_name}: {precision}: {case['case_id']} QuatOps rotvec reconstruction",
            )

            engine_rotvec_q = module.from_rotvec(
                orient_vectors(native_rotvec, component_first)
            )
            engine_rotvec_xyzw = _engine_xyzw_from_backend(
                module, engine_rotvec_q, is_jpl, is_numpy, component_first
            )
            if (
                precision == "float64"
                and case["case_id"] == "tiny_angle"
                and label == "axis-angle -q"
            ):
                # In this pinned build, Log() suffers cancellation near -1 for
                # this tiny double quaternion and does not round-trip -q.
                assert np.linalg.norm(native_rotvec) == pytest.approx(
                    5.847336346245288, abs=1e-12
                )
                source_rotation = Rotation.from_quat(source_xyzw)
                reconstructed = Rotation.from_rotvec(native_rotvec)
                assert (reconstructed.inv() * source_rotation).magnitude() > 0.4
            else:
                _assert_same_rotation(
                    engine_rotvec_xyzw,
                    source_xyzw,
                    case_tolerance,
                    f"{backend_name}: {precision}: {case['case_id']} {label} native ToRotationVector",
                )

        angle_q = case["axis_angle_readback_angle_radians"]
        angle_minus_q = case["negative_axis_angle_readback_angle_radians"]
        np.testing.assert_allclose(
            angle_q + angle_minus_q,
            2.0 * np.pi,
            rtol=0.0,
            atol=max(case_tolerance, 2e-7),
            err_msg=f"{precision}: {case['case_id']} Unreal q/-q angle branches",
        )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unreal_axis_angle_and_rotation_vector_readbacks_pin_native_branches(precision):
    tolerance = max(ATOL[precision], 2e-7)
    for case in EXTENDED[f"axis_angle_{precision}"]:
        case_id = case["case_id"]
        axis = np.asarray(case["input_axis_xyz"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        angle = float(case["input_angle_radians"])
        small_vector_fallback = case_id in {"zero_angle", "tiny_angle"}
        expected_axis = np.array((1.0, 0.0, 0.0)) if small_vector_fallback else axis
        expected_negative_axis = (
            expected_axis if small_vector_fallback else -expected_axis
        )

        np.testing.assert_allclose(
            case["axis_angle_readback_axis_xyz"], expected_axis,
            rtol=0.0, atol=tolerance,
            err_msg=f"{precision}: {case_id} ToAxisAndAngle(q) axis branch",
        )
        np.testing.assert_allclose(
            case["negative_axis_angle_readback_axis_xyz"], expected_negative_axis,
            rtol=0.0, atol=tolerance,
            err_msg=f"{precision}: {case_id} ToAxisAndAngle(-q) axis branch",
        )
        assert case["axis_angle_readback_angle_radians"] == pytest.approx(
            angle, abs=tolerance
        )
        assert case["negative_axis_angle_readback_angle_radians"] == pytest.approx(
            2.0 * np.pi - angle, abs=tolerance
        )

        expected_rotvec = axis * angle
        if case_id == "zero_angle":
            expected_negative_rotvec = np.zeros(3)
        elif case_id == "tiny_angle" and precision == "float64":
            # Pin Unreal's observed double-precision ToRotationVector(-q)
            # cancellation branch, including its direction.
            expected_negative_rotvec = -axis * 5.847336346245288
        elif case_id == "tiny_angle":
            expected_negative_rotvec = -axis * angle
        else:
            expected_negative_rotvec = -axis * (2.0 * np.pi - angle)

        np.testing.assert_allclose(
            case["rotation_vector_readback_xyz"], expected_rotvec,
            rtol=0.0, atol=tolerance,
            err_msg=f"{precision}: {case_id} ToRotationVector(q) branch",
        )
        np.testing.assert_allclose(
            case["negative_rotation_vector_readback_xyz"], expected_negative_rotvec,
            rtol=0.0, atol=tolerance,
            err_msg=f"{precision}: {case_id} ToRotationVector(-q) branch",
        )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unreal_rotation_vector_construction_and_readback(
    precision, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = ATOL[precision]
    for case in EXTENDED[f"rotation_vector_{precision}"]:
        vector = np.asarray(case["input_rotation_vector_xyz"], dtype=float)
        expected_xyzw = np.asarray(case["constructed_quaternion_xyzw"], dtype=float)
        np.testing.assert_allclose(
            case["rotation_vector_readback_xyz"],
            vector,
            rtol=0.0,
            atol=tolerance,
            err_msg=f"{precision}: {case['case_id']} native ToRotationVector branch",
        )
        actual = module.from_rotvec(orient_vectors(vector, component_first))
        actual_xyzw = _engine_xyzw_from_backend(
            module, actual, is_jpl, is_numpy, component_first
        )
        _assert_same_rotation(
            actual_xyzw,
            expected_xyzw,
            tolerance,
            f"{backend_name}: {precision}: {case['case_id']} MakeFromRotationVector",
        )

        readback = module.from_rotvec(
            orient_vectors(case["rotation_vector_readback_xyz"], component_first)
        )
        readback_xyzw = _engine_xyzw_from_backend(
            module, readback, is_jpl, is_numpy, component_first
        )
        _assert_same_rotation(
            readback_xyzw,
            expected_xyzw,
            tolerance,
            f"{backend_name}: {precision}: {case['case_id']} ToRotationVector reconstruction",
        )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unreal_slerp_native_samples(precision, backend_name, module, is_jpl, is_numpy, component_first):
    cases = EXTENDED[f"slerp_{precision}"]
    for case in cases:
        lhs = _backend_from_engine_xyzw(
            module, case["lhs_xyzw"], is_jpl, is_numpy, component_first
        )
        rhs = _backend_from_engine_xyzw(
            module, case["rhs_xyzw"], is_jpl, is_numpy, component_first
        )
        actual = module.slerp(lhs, rhs, float(case["alpha"]))
        actual_xyzw = _engine_xyzw_from_backend(
            module, actual, is_jpl, is_numpy, component_first
        )
        expected_xyzw = np.asarray(case["slerp_xyzw"], dtype=float)
        np.testing.assert_allclose(
            actual_xyzw,
            expected_xyzw,
            rtol=0.0,
            atol=ATOL[precision],
            err_msg=f"{backend_name}: {precision}: {case['case_id']} Slerp t={case['alpha']}",
        )
        assert case["shortest_result_is_unit"]

        full_path = np.asarray(case["slerp_full_path_xyzw"], dtype=float)
        if case["case_id"] == "antipodal":
            # Unreal returns a normalized result for this nonunique path;
            # QuatOps reports that the full path is undefined.
            assert case["full_path_result_is_unit"]
            with pytest.raises(ValueError, match="not unique"):
                module.slerp(lhs, rhs, float(case["alpha"]), shortest=False)
            continue
        actual_full = module.slerp(lhs, rhs, float(case["alpha"]), shortest=False)
        actual_full_xyzw = _engine_xyzw_from_backend(
            module, actual_full, is_jpl, is_numpy, component_first
        )
        np.testing.assert_allclose(
            actual_full_xyzw,
            full_path,
            rtol=0.0,
            atol=ATOL[precision],
            err_msg=(
                f"{backend_name}: {precision}: {case['case_id']} "
                f"SlerpFullPath t={case['alpha']}"
            ),
        )
        assert case["full_path_result_is_unit"]
        if case["case_id"] == "long_path" and case["alpha"] == 0.5:
            shortest_matrix = Rotation.from_quat(actual_xyzw).as_matrix()
            full_path_matrix = Rotation.from_quat(actual_full_xyzw).as_matrix()
            assert not np.allclose(
                shortest_matrix, full_path_matrix, rtol=0.0, atol=0.1
            ), f"{backend_name}: shortest and full-path midpoint should differ"


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unreal_normalization_threshold_and_degenerate_behavior(
    precision, backend_name, module, is_jpl, is_numpy, component_first
):
    cases = {case["case_id"]: case for case in EXTENDED[f"normalization_{precision}"]}
    for case_id in ("identity", "above_default_squared_tolerance", "nonunit"):
        case = cases[case_id]
        value = _backend_from_engine_xyzw(
            module, case["input_xyzw"], is_jpl, is_numpy, component_first
        )
        actual = module.normalize(value)
        actual_xyzw = _engine_xyzw_from_backend(
            module, actual, is_jpl, is_numpy, component_first
        )
        np.testing.assert_allclose(
            actual_xyzw,
            case["normalized_xyzw"],
            rtol=0.0,
            atol=ATOL[precision],
            err_msg=f"{backend_name}: {precision}: {case_id} normalization",
        )

    zero = cases["zero"]
    assert zero["input_norm_squared"] == 0.0
    assert zero["normalized_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    value = _backend_from_engine_xyzw(
        module, zero["input_xyzw"], is_jpl, is_numpy, component_first
    )
    with pytest.raises(ValueError, match="zero quaternion"):
        module.normalize(value)

    below = cases["below_default_squared_tolerance"]
    assert below["input_norm_squared"] < EXTENDED["normalization_squared_tolerance"]
    assert below["normalized_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    below_value = _backend_from_engine_xyzw(
        module, below["input_xyzw"], is_jpl, is_numpy, component_first
    )
    below_normalized = module.normalize(below_value)
    below_xyzw = _engine_xyzw_from_backend(
        module, below_normalized, is_jpl, is_numpy, component_first
    )
    # Unreal falls back to identity below its squared-norm threshold. QuatOps
    # normalizes every finite nonzero input, preserving this tiny X axis.
    np.testing.assert_allclose(
        below_xyzw,
        (1.0, 0.0, 0.0, 0.0),
        rtol=0.0,
        atol=ATOL[precision],
        err_msg=f"{backend_name}: {precision}: QuatOps normalizes tiny nonzero input",
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_unreal_extended_capture_metadata_and_native_branch_proofs(precision):
    assert FIXTURE["quaternion_order"] == "xyzw"
    assert FIXTURE["quaternion_algebra"] == "Hamilton"
    assert FIXTURE["rotation_action"] == "active"
    assert EXTENDED["rotation_action"] == "active; quaternion coefficient order xyzw"

    axis_cases = EXTENDED[f"axis_angle_{precision}"]
    for case in axis_cases:
        assert case["constructor_norm_is_unit"]
        assert case["axis_angle_readback_angle_radians"] + case[
            "negative_axis_angle_readback_angle_radians"
        ] == pytest.approx(2.0 * np.pi, abs=max(ATOL[precision], 2e-7))

    assert any(case["case_id"] == "antipodal" for case in EXTENDED[f"slerp_{precision}"])
    assert EXTENDED["normalization_squared_tolerance"] == pytest.approx(1e-8)
