"""Three.js 0.187.0 native Node capture checked against SciPy and QuatOps."""

import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.support.native_engine_support import (
    from_engine_xyzw,
    quaternion_coefficient_error,
    to_hamilton_xyzw,
)
from tests.support.oracle_support import BACKENDS, orient_matrices, orient_vectors
from tests.support.paths import REPOSITORY_ROOT, TESTS_ROOT

FIXTURE_PATH = (
    TESTS_ROOT / "golden" / "oracles" / "threejs" / "0.187.0" / "native_rotation.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
OBS = FIXTURE["observations"]
CORPUS_SHA256 = "07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f"
TOLERANCE = 3e-12


def _rotation(xyzw):
    return Rotation.from_quat(np.asarray(xyzw, dtype=float))


def _orientation_error(lhs, rhs):
    return float(
        np.max(np.abs(_rotation(lhs).as_matrix() - _rotation(rhs).as_matrix()))
    )


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
        "euler_regressions": 1,
        "axis_angle": 5,
        "quaternion_normalization": 9,
        "multiplication": 3,
        "shortest_slerp": 7,
        "independent_matrices": 3,
        "two_vector_alignment": 2,
    }


def test_threejs_near_lock_readback_regression_is_retained():
    row = OBS["euler_regressions"][0]
    assert row["case_id"] == "boundary-offset-ZYX-+1-+1e-04"
    assert row["input_source"]["actual_bits_hex"]
    expected = Rotation.from_euler(
        row["order"], _ordered_angles(row["input_radians_xyz"], row["order"])
    ).as_matrix()
    np.testing.assert_allclose(
        _rotation(row["euler_quaternion_xyzw"]).as_matrix(),
        expected,
        rtol=0.0,
        atol=TOLERANCE,
    )
    residual = _orientation_error(
        row["quaternion_readback_roundtrip_xyzw"],
        row["euler_quaternion_xyzw"],
    )
    assert residual > 1e-4


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
        expected = Rotation.from_matrix(
            np.asarray(row["input_matrix_rows"])
        ).as_matrix()
        native = row["matrix_to_quaternion_xyzw"]
        np.testing.assert_allclose(
            _rotation(native).as_matrix(), expected, rtol=0.0, atol=TOLERANCE
        )
        np.testing.assert_allclose(
            row["quaternion_reconstructed_matrix_rows"],
            expected,
            rtol=0.0,
            atol=TOLERANCE,
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
            actual_vector,
            to_vector,
            rtol=0.0,
            atol=TOLERANCE,
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
                actual,
                expected,
                rtol=0.0,
                atol=2e-13,
                err_msg=f"{backend_name}: Three.js product {row['case_id']}",
            )

    for row in OBS["shortest_slerp"]:
        start_xyzw = np.asarray(row["a_input_wxyz"])[[1, 2, 3, 0]]
        end_xyzw = np.asarray(row["b_input_wxyz"])[[1, 2, 3, 0]]
        start = from_engine_xyzw(module, start_xyzw, is_jpl, is_numpy, component_first)
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
        source = np.asarray(row["input_wxyz"], dtype=float)[[1, 2, 3, 0]]
        if np.linalg.norm(source) == 0.0:
            # Three.js maps a zero quaternion to identity in normalize(); this
            # is recorded as native behavior, not asserted as QuatOps parity.
            assert row["normalized_xyzw"] == [0.0, 0.0, 0.0, 1.0]
            continue
        value = from_engine_xyzw(module, source, is_jpl, is_numpy, component_first)
        normalized = module.normalize(value)
        actual_xyzw = to_hamilton_xyzw(
            module, normalized, is_jpl, is_numpy, component_first
        )
        expected_xyzw = row["normalized_xyzw"]
        np.testing.assert_allclose(
            actual_xyzw,
            expected_xyzw,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{backend_name}: Three.js normalize {row['case_id']}",
        )


def test_threejs_fixture_hash_fields_match_checked_in_sources():
    script_path = (
        REPOSITORY_ROOT
        / "experiments"
        / "oracles"
        / "threejs"
        / "0.187.0"
        / "capture_three.mjs"
    )
    lock_path = script_path.with_name("package-lock.json")
    assert (
        hashlib.sha256(script_path.read_bytes()).hexdigest()
        == FIXTURE["provenance"]["exporter_sha256"]
    )
    assert (
        hashlib.sha256(lock_path.read_bytes()).hexdigest()
        == FIXTURE["provenance"]["package_lock_sha256"]
    )
