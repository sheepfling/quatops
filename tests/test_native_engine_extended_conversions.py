"""Compare Unity and Godot native conversion captures with QuatOps and SciPy."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.native_engine_support import (
    from_engine_xyzw,
    matrix_rows,
    quaternion_xyzw,
    to_hamilton_xyzw,
    vector_xyz,
)
from tests.oracle_support import BACKENDS, orient_matrices, orient_vectors

ROOT = Path(__file__).parent
FIXTURES = {
    "godot": json.loads(
        (ROOT / "golden" / "godot" / "4.8-dev1" / "native_rotation.json").read_text(encoding="utf-8")
    ),
    "godot_4_7": json.loads(
        (ROOT / "golden" / "godot" / "4.7-stable" / "native_rotation.json").read_text(encoding="utf-8")
    ),
    "unity": json.loads(
        (ROOT / "golden" / "unity" / "6000.5.2f1" / "native_rotation.json").read_text(encoding="utf-8")
    ),
}
TOLERANCES = {"godot": 4e-6, "godot_4_7": 4e-6, "unity": 5e-6}
GODOT_FIXTURE_IDS = ("godot", "godot_4_7")


def _same_rotation(actual_xyzw, expected_xyzw, tolerance, label):
    actual = Rotation.from_quat(np.asarray(actual_xyzw, dtype=float)).as_matrix()
    expected = Rotation.from_quat(np.asarray(expected_xyzw, dtype=float)).as_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance, err_msg=label)


def _backend_q(module, xyzw, is_jpl, is_numpy, component_first):
    return from_engine_xyzw(module, xyzw, is_jpl, is_numpy, component_first)


def _native_xyzw(module, value, is_jpl, is_numpy, component_first):
    return to_hamilton_xyzw(module, value, is_jpl, is_numpy, component_first)


def _matrix_cases(engine):
    for case in FIXTURES[engine]["extended_conversions"]["matrix_to_quaternion"]:
        if engine.startswith("godot"):
            quaternions = (
                case["basis_get_rotation_quaternion_xyzw"],
                case["quaternion_basis_constructor_xyzw"],
            )
        else:
            quaternions = (
                case["unity_engine_matrix_rotation_xyzw"],
                case["unity_mathematics_matrix_constructor_xyzw"],
            )
        yield case["case_id"], matrix_rows(case["matrix_rows_native"]), [
            quaternion_xyzw(q) for q in quaternions
        ]


def _axis_angle_cases(engine):
    for case in FIXTURES[engine]["extended_conversions"]["axis_angle"]:
        if engine.startswith("godot"):
            quaternions = (case["constructed_quaternion_xyzw"],)
        else:
            quaternions = (
                case["unity_engine_constructed_quaternion_xyzw"],
                case["unity_mathematics_constructed_quaternion_xyzw"],
            )
        yield case, [quaternion_xyzw(q) for q in quaternions]


def _rotation_vector_cases(engine):
    for case in FIXTURES[engine]["extended_conversions"]["rotation_vector"]:
        if engine.startswith("godot"):
            quaternions = (case["constructed_quaternion_xyzw"],)
        else:
            quaternions = (
                case["unity_engine_constructed_quaternion_xyzw"],
                case["unity_mathematics_constructed_quaternion_xyzw"],
            )
        yield case, [quaternion_xyzw(q) for q in quaternions]


@pytest.mark.parametrize("engine", (*GODOT_FIXTURE_IDS, "unity"))
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_native_matrix_to_quaternion_routes_match_quatops_and_scipy(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    for case_id, matrix_native, native_quaternions in _matrix_cases(engine):
        scipy_matrix = Rotation.from_matrix(matrix_native).as_matrix()
        np.testing.assert_allclose(
            scipy_matrix, matrix_native, rtol=0.0, atol=tolerance,
            err_msg=f"{engine} {case_id}: captured matrix is a proper rotation",
        )
        for route_index, native_xyzw in enumerate(native_quaternions):
            _same_rotation(
                native_xyzw,
                Rotation.from_matrix(matrix_native).as_quat(),
                tolerance,
                f"{engine} {case_id}: native matrix route {route_index}",
            )
            actual = module.from_rotation_matrix(
                orient_matrices(matrix_native, component_first), atol=tolerance
            )
            actual_xyzw = _native_xyzw(
                module, actual, is_jpl, is_numpy, component_first
            )
            _same_rotation(
                actual_xyzw, native_xyzw, tolerance,
                f"{backend_name}: {engine} {case_id} matrix route {route_index}",
            )


@pytest.mark.parametrize("engine", GODOT_FIXTURE_IDS)
def test_godot_near_degenerate_basis_results_remain_native_diagnostics(engine):
    cases = FIXTURES[engine]["extended_conversions"]["matrix_conversion_diagnostics"]
    expected_determinants = (1.0e-2, 1.0e-4, 1.0e-6)
    identity_xyzw = np.array((0.0, 0.0, 0.0, 1.0))
    assert len(cases) == len(expected_determinants)

    for case, expected_determinant in zip(cases, expected_determinants):
        assert case["diagnostic_only"]
        matrix = matrix_rows(case["matrix_rows_native"])
        np.testing.assert_allclose(
            np.linalg.det(matrix), expected_determinant, rtol=0.0, atol=2e-7
        )
        assert not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=1e-5)

        for route, finite_key, norm_key in (
            (
                "basis_get_rotation_quaternion_xyzw",
                "basis_get_rotation_quaternion_is_finite",
                "basis_get_rotation_quaternion_norm",
            ),
            (
                "quaternion_basis_constructor_xyzw",
                "quaternion_basis_constructor_is_finite",
                "quaternion_basis_constructor_norm",
            ),
        ):
            assert case[finite_key]
            assert case[norm_key] == pytest.approx(1.0, abs=1e-7)
            np.testing.assert_allclose(
                quaternion_xyzw(case[route]),
                identity_xyzw,
                rtol=0.0,
                atol=1e-7,
                err_msg=f"{engine} {case['case_id']} {route} diagnostic response",
            )


@pytest.mark.parametrize("engine", (*GODOT_FIXTURE_IDS, "unity"))
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_native_axis_angle_constructors_and_q_sign_readbacks_match_quatops(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    for case, native_quaternions in _axis_angle_cases(engine):
        axis = vector_xyz(case["input_axis_xyz"])
        angle = case["input_angle_radians"]
        actual = module.from_axis_angle(
            orient_vectors(np.r_[axis, angle], component_first)
        )
        actual_xyzw = _native_xyzw(module, actual, is_jpl, is_numpy, component_first)
        for route_index, native_xyzw in enumerate(native_quaternions):
            _same_rotation(
                actual_xyzw, native_xyzw, tolerance,
                f"{backend_name}: {engine} {case['case_id']} axis-angle route {route_index}",
            )

        negative_xyzw = (
            quaternion_xyzw(case["negative_quaternion_xyzw"])
            if engine.startswith("godot")
            else -native_quaternions[0]
        )
        for read_axis_key, read_angle_key, expected_xyzw in (
            ("axis_angle_readback_axis_xyz", "axis_angle_readback_angle_radians", native_quaternions[0]),
            ("negative_axis_angle_readback_axis_xyz", "negative_axis_angle_readback_angle_radians", negative_xyzw),
        ):
            read_axis = vector_xyz(case[read_axis_key])
            read_angle = float(case[read_angle_key])
            if np.linalg.norm(read_axis) == 0.0:
                assert np.linalg.norm(Rotation.from_quat(expected_xyzw).as_matrix() - np.eye(3)) <= tolerance
                assert abs(np.sin(0.5 * read_angle)) <= tolerance
                continue
            rebuilt = module.from_axis_angle(
                orient_vectors(
                    np.r_[read_axis, read_angle],
                    component_first,
                )
            )
            rebuilt_xyzw = _native_xyzw(
                module, rebuilt, is_jpl, is_numpy, component_first
            )
            _same_rotation(
                rebuilt_xyzw, expected_xyzw, tolerance,
                f"{backend_name}: {engine} {case['case_id']} {read_angle_key}",
            )


@pytest.mark.parametrize("engine", (*GODOT_FIXTURE_IDS, "unity"))
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_native_rotation_vector_routes_match_quatops(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    for case, native_quaternions in _rotation_vector_cases(engine):
        vector = vector_xyz(case["input_rotation_vector_xyz"])
        actual = module.from_rotvec(orient_vectors(vector, component_first))
        actual_xyzw = _native_xyzw(module, actual, is_jpl, is_numpy, component_first)
        for route_index, native_xyzw in enumerate(native_quaternions):
            _same_rotation(
                actual_xyzw, native_xyzw, tolerance,
                f"{backend_name}: {engine} {case['case_id']} rotvec route {route_index}",
            )
        readback = module.from_rotvec(
            orient_vectors(vector_xyz(case["rotation_vector_readback_xyz"]), component_first)
        )
        readback_xyzw = _native_xyzw(
            module, readback, is_jpl, is_numpy, component_first
        )
        _same_rotation(
            readback_xyzw, native_quaternions[0], tolerance,
            f"{backend_name}: {engine} {case['case_id']} native rotvec readback",
        )


@pytest.mark.parametrize("engine", GODOT_FIXTURE_IDS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_godot_rotation_exp_log_matches_quatops_rotation_vectors(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    cases = FIXTURES[engine]["extended_conversions"]["rotation_exp_log"]
    for case in cases["exp"]:
        rotation_vector = vector_xyz(case["input_rotation_vector_xyz"])
        value = module.from_rotvec(
            orient_vectors(rotation_vector, component_first)
        )
        actual_xyzw = _native_xyzw(
            module, value, is_jpl, is_numpy, component_first
        )
        native_xyzw = quaternion_xyzw(case["exp_result_quaternion_xyzw"])
        _same_rotation(
            actual_xyzw, native_xyzw, tolerance,
            f"{backend_name}: Godot {engine} exp {case['case_id']}",
        )
        if case["case_id"] not in {"z_180_degrees", "antipodal_z_90_branch"}:
            np.testing.assert_allclose(
                actual_xyzw, native_xyzw, rtol=0.0, atol=tolerance,
                err_msg=f"{backend_name}: Godot exp coefficient/sign {case['case_id']}",
            )
        assert case["exp_result_is_finite"]
        assert np.isclose(case["exp_result_norm"], 1.0, atol=tolerance, rtol=0.0)

    expected_logs = {
        "identity": (0.0, 0.0, 0.0),
        "z_90_degrees": (0.0, 0.0, np.pi / 2.0),
        "z_180_degrees": (0.0, 0.0, np.pi),
        "antipodal_z_90": (0.0, 0.0, -3.0 * np.pi / 2.0),
        "antipodal_z_180": (0.0, 0.0, -np.pi),
    }
    for case in cases["log"]:
        native_xyzw = quaternion_xyzw(case["input_quaternion_xyzw"])
        native_log = vector_xyz(case["rotation_vector_xyz"])
        np.testing.assert_allclose(
            native_log,
            expected_logs[case["case_id"]],
            rtol=0.0,
            atol=2e-5,
            err_msg=f"Godot Quaternion.log full rotation vector {case['case_id']}",
        )
        value = _backend_q(
            module, native_xyzw, is_jpl, is_numpy, component_first
        )
        actual_log = module.to_rotvec(value)
        if not case["case_id"].startswith("antipodal") and case["case_id"] != "z_180_degrees":
            np.testing.assert_allclose(
                np.asarray(tuple(actual_log), dtype=float).reshape(3),
                native_log,
                rtol=0.0,
                atol=tolerance,
                err_msg=f"{backend_name}: Godot {engine} log vs QuatOps to_rotvec {case['case_id']}",
            )

        reconstructed = module.from_rotvec(
            orient_vectors(native_log, component_first)
        )
        reconstructed_xyzw = _native_xyzw(
            module, reconstructed, is_jpl, is_numpy, component_first
        )
        if case["case_id"] not in {"z_180_degrees", "antipodal_z_90"}:
            np.testing.assert_allclose(
                reconstructed_xyzw,
                native_xyzw,
                rtol=0.0,
                atol=tolerance,
                err_msg=f"Godot log then QuatOps exp keeps q sign {case['case_id']}",
            )


@pytest.mark.parametrize("engine", GODOT_FIXTURE_IDS)
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_godot_shortest_arc_vector_constructor_matches_quatops(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    cases = FIXTURES[engine]["extended_conversions"]["from_to"]
    for case in cases:
        source = vector_xyz(case["input_from_xyz"])
        target = vector_xyz(case["input_to_xyz"])
        source_unit = vector_xyz(case["source_unit_xyz"])
        target_unit = vector_xyz(case["target_unit_xyz"])
        value = module.from_two_vectors(source, target)
        actual_xyzw = _native_xyzw(
            module, value, is_jpl, is_numpy, component_first
        )
        actual_target = np.asarray(module.rotate(value, source_unit), dtype=float)
        np.testing.assert_allclose(
            actual_target,
            target_unit,
            rtol=0.0,
            atol=tolerance,
            err_msg=f"{backend_name}: QuatOps from_two_vectors action {case['case_id']}",
        )
        assert case["quaternion_is_finite"]
        assert case["target_matches"] == (case["target_alignment_error"] <= 2e-5)
        if (
            case["shortest_rotation_orientation_is_unique"]
            and case["target_alignment_error"] <= tolerance
        ):
            _same_rotation(
                actual_xyzw,
                quaternion_xyzw(case["quaternion_xyzw"]),
                tolerance,
                f"{backend_name}: Godot {engine} Quaternion(from,to) {case['case_id']}",
            )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
@pytest.mark.parametrize("engine", GODOT_FIXTURE_IDS)
def test_godot_native_shortest_and_no_shortest_path_slerp(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    for case in FIXTURES[engine]["extended_conversions"]["slerp"]:
        lhs_xyzw, rhs_xyzw = quaternion_xyzw(case["lhs_xyzw"]), quaternion_xyzw(case["rhs_xyzw"])
        lhs = _backend_q(module, lhs_xyzw, is_jpl, is_numpy, component_first)
        rhs = _backend_q(module, rhs_xyzw, is_jpl, is_numpy, component_first)
        shortest = module.slerp(lhs, rhs, float(case["alpha"]))
        shortest_xyzw = _native_xyzw(module, shortest, is_jpl, is_numpy, component_first)
        _same_rotation(
            shortest_xyzw, quaternion_xyzw(case["slerp_xyzw"]), tolerance,
            f"{backend_name}: Godot slerp {case['case_id']} t={case['alpha']}",
        )
        if case["slerp_is_finite"]:
            np.testing.assert_allclose(
                shortest_xyzw,
                quaternion_xyzw(case["slerp_xyzw"]),
                rtol=0.0,
                atol=tolerance,
                err_msg=f"{backend_name}: Godot slerp coefficient sign {case['case_id']} t={case['alpha']}",
            )
        if case["case_id"] == "antipodal":
            with pytest.raises(ValueError, match="not unique"):
                module.slerp(lhs, rhs, float(case["alpha"]), shortest=False)
            continue
        full_path = module.slerp(lhs, rhs, float(case["alpha"]), shortest=False)
        full_path_xyzw = _native_xyzw(
            module, full_path, is_jpl, is_numpy, component_first
        )
        comparison_tolerance = (
            1e-3 if case["case_id"] == "nearly_identical" else tolerance
        )
        _same_rotation(
            full_path_xyzw, quaternion_xyzw(case["slerpni_xyzw"]), comparison_tolerance,
            f"{backend_name}: Godot slerpni {case['case_id']} t={case['alpha']}",
        )
        np.testing.assert_allclose(
            full_path_xyzw,
            quaternion_xyzw(case["slerpni_xyzw"]),
            rtol=0.0,
            atol=comparison_tolerance,
            err_msg=f"{backend_name}: Godot slerpni coefficient sign {case['case_id']} t={case['alpha']}",
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_shortest_slerp_clamping_and_unclamped_routes(
    backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES["unity"]
    for case in FIXTURES["unity"]["extended_conversions"]["slerp"]:
        lhs_xyzw, rhs_xyzw = quaternion_xyzw(case["lhs_xyzw"]), quaternion_xyzw(case["rhs_xyzw"])
        lhs = _backend_q(module, lhs_xyzw, is_jpl, is_numpy, component_first)
        rhs = _backend_q(module, rhs_xyzw, is_jpl, is_numpy, component_first)
        alpha = float(case["alpha"])
        unclamped = module.slerp(lhs, rhs, alpha)
        unclamped_xyzw = _native_xyzw(
            module, unclamped, is_jpl, is_numpy, component_first
        )
        engine_unclamped_xyzw = quaternion_xyzw(
            case["unity_engine_slerp_unclamped"]["result_xyzw"]
        )
        _same_rotation(
            unclamped_xyzw, engine_unclamped_xyzw, tolerance,
            f"{backend_name}: Unity SlerpUnclamped {case['case_id']} t={alpha}",
        )
        _same_rotation(
            quaternion_xyzw(case["unity_mathematics_slerp"]["result_xyzw"]),
            engine_unclamped_xyzw, tolerance,
            f"Unity math.slerp and SlerpUnclamped {case['case_id']} t={alpha}",
        )

        clamped_alpha = min(1.0, max(0.0, alpha))
        clamped = module.slerp(lhs, rhs, clamped_alpha)
        clamped_xyzw = _native_xyzw(
            module, clamped, is_jpl, is_numpy, component_first
        )
        engine_clamped_xyzw = quaternion_xyzw(case["unity_engine_slerp"]["result_xyzw"])
        _same_rotation(
            clamped_xyzw, engine_clamped_xyzw, tolerance,
            f"{backend_name}: Unity Quaternion.Slerp clamp {case['case_id']} t={alpha}",
        )


@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_unity_lerp_nlerp_clamping_and_unclamped_routes(
    backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES["unity"]
    for case in FIXTURES["unity"]["extended_conversions"]["lerp"]:
        lhs_xyzw = quaternion_xyzw(case["lhs_xyzw"])
        rhs_xyzw = quaternion_xyzw(case["rhs_xyzw"])
        lhs = _backend_q(module, lhs_xyzw, is_jpl, is_numpy, component_first)
        rhs = _backend_q(module, rhs_xyzw, is_jpl, is_numpy, component_first)
        alpha = float(case["alpha"])

        actual_unclamped = module.nlerp(lhs, rhs, alpha)
        actual_unclamped_xyzw = _native_xyzw(
            module, actual_unclamped, is_jpl, is_numpy, component_first
        )
        unity_unclamped = case["unity_engine_lerp_unclamped"]
        mathematics_nlerp = case["unity_mathematics_nlerp"]
        _same_rotation(
            actual_unclamped_xyzw,
            quaternion_xyzw(unity_unclamped["result_xyzw"]),
            tolerance,
            f"{backend_name}: Unity LerpUnclamped {case['case_id']} t={alpha}",
        )
        _same_rotation(
            actual_unclamped_xyzw,
            quaternion_xyzw(mathematics_nlerp["result_xyzw"]),
            tolerance,
            f"{backend_name}: Unity math.nlerp {case['case_id']} t={alpha}",
        )

        clamped_alpha = min(1.0, max(0.0, alpha))
        actual_clamped = module.nlerp(lhs, rhs, clamped_alpha)
        actual_clamped_xyzw = _native_xyzw(
            module, actual_clamped, is_jpl, is_numpy, component_first
        )
        _same_rotation(
            actual_clamped_xyzw,
            quaternion_xyzw(case["unity_engine_lerp"]["result_xyzw"]),
            tolerance,
            f"{backend_name}: Unity Quaternion.Lerp clamp {case['case_id']} t={alpha}",
        )

        for route in (unity_unclamped, mathematics_nlerp):
            assert route["is_finite"]
            assert np.isclose(route["norm"], 1.0, atol=tolerance, rtol=0.0)


@pytest.mark.parametrize("engine", (*GODOT_FIXTURE_IDS, "unity"))
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_engine_normalization_edges_are_recorded_and_compared_with_quatops(
    engine, backend_name, module, is_jpl, is_numpy, component_first
):
    tolerance = TOLERANCES[engine]
    cases = {
        case["case_id"]: case
        for case in FIXTURES[engine]["extended_conversions"]["normalization"]
    }
    for case_id in ("identity", "below_threshold", "above_threshold", "nonunit"):
        case = cases[case_id]
        native_xyzw = (
            quaternion_xyzw(case["normalized_xyzw"])
            if engine.startswith("godot")
            else quaternion_xyzw(case["unity_engine_normalized_xyzw"])
        )
        value = _backend_q(
            module, quaternion_xyzw(case["input_xyzw"]), is_jpl, is_numpy, component_first
        )
        normalized = module.normalize(value)
        actual_xyzw = _native_xyzw(
            module, normalized, is_jpl, is_numpy, component_first
        )
        _same_rotation(
            actual_xyzw, native_xyzw, tolerance,
            f"{backend_name}: {engine} {case_id} normalization",
        )
        assert np.isclose(np.linalg.norm(native_xyzw), 1.0, atol=tolerance, rtol=0.0)

    zero = cases["zero"]
    assert zero["input_norm_squared"] == 0.0
    if engine.startswith("godot"):
        assert zero["normalized_is_finite"] is False
        assert set(zero["normalized_xyzw"].values()) == {None}
    else:
        assert zero["unity_engine_normalized_is_finite"] is True
        assert quaternion_xyzw(zero["unity_engine_normalized_xyzw"]).tolist() == [0.0, 0.0, 0.0, 1.0]
        assert zero["unity_mathematics_normalized_is_finite"] is False
        assert set(zero["unity_mathematics_normalized_xyzw_components"].values()) == {"NaN"}
        assert zero["unity_mathematics_normalized_norm"] == "NaN"


def test_native_long_path_samples_confirm_which_public_route_is_available():
    godot_rows = [
        next(
            row for row in FIXTURES[engine]["extended_conversions"]["slerp"]
            if row["case_id"] == "long_path" and row["alpha"] == 0.5
        )
        for engine in GODOT_FIXTURE_IDS
    ]
    unity = next(
        row for row in FIXTURES["unity"]["extended_conversions"]["slerp"]
        if row["case_id"] == "long_path" and row["alpha"] == 0.5
    )
    for godot in godot_rows:
        assert np.linalg.norm(
            quaternion_xyzw(godot["slerp_xyzw"]) - quaternion_xyzw(godot["slerpni_xyzw"])
        ) > 0.5
    _same_rotation(
        quaternion_xyzw(unity["unity_engine_slerp_unclamped"]["result_xyzw"]),
        quaternion_xyzw(unity["unity_mathematics_slerp"]["result_xyzw"]),
        TOLERANCES["unity"],
        "Unity shortest-path routes agree for the long-path endpoint signs",
    )
    assert "not exposed" in FIXTURES["unity"]["proof_profile"]["slerp_full_path"]
