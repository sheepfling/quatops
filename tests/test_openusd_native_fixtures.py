"""Tests for the frozen OpenUSD 26.11 quaternion and transform-stack capture."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parent.parent
FIXTURE_PATH = ROOT / "tests" / "golden" / "transports" / "openusd" / "26.11" / "native_rotation.json"
STAGE_PATH = ROOT / "experiments" / "transports" / "openusd" / "26.11" / "assets" / "rotation_stack.usda"
LOCK_PATH = ROOT / "experiments" / "transports" / "openusd" / "26.11" / "uv.lock"
EXPORTER_PATH = ROOT / "experiments" / "transports" / "openusd" / "26.11" / "capture_openusd.py"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
OBS = FIXTURE["observations"]
CORPUS_SHA256 = "07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f"
TOLERANCE = 3e-12


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_nonstandard_json_constant(value):
    raise ValueError(value)


def _hamilton(lhs, rhs):
    aw, ax, ay, az = lhs
    bw, bx, by, bz = rhs
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=float,
    )


def _wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return q[[1, 2, 3, 0]]


def _rotation_error(lhs_wxyz, rhs_wxyz):
    lhs = Rotation.from_quat(_wxyz_to_xyzw(lhs_wxyz)).as_matrix()
    rhs = Rotation.from_quat(_wxyz_to_xyzw(rhs_wxyz)).as_matrix()
    return float(np.max(np.abs(lhs - rhs)))


def _slerp_shortest(a, b, amount):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        value = (1.0 - amount) * a + amount * b
        return value / np.linalg.norm(value)
    angle = math.acos(dot)
    return (
        math.sin((1.0 - amount) * angle) / math.sin(angle) * a
        + math.sin(amount * angle) / math.sin(angle) * b
    )


def _row_translation(vector):
    result = np.eye(4)
    result[3, :3] = vector
    return result


def _row_rotation(matrix3):
    result = np.eye(4)
    result[:3, :3] = matrix3
    return result


def test_openusd_capture_provenance_and_input_stage_are_frozen():
    assert FIXTURE["evidence_kind"] == "native_openusd_gf_and_usdgeom_runtime_capture"
    assert FIXTURE["runtime"]["distribution"] == "usd-core"
    assert FIXTURE["runtime"]["distribution_version"] == "26.11"
    assert FIXTURE["runtime"]["openusd_version_tuple"] == [0, 26, 11]
    assert FIXTURE["runtime"]["python_version"].startswith("3.14.6 ")
    assert FIXTURE["provenance"]["input_corpus_sha256"] == CORPUS_SHA256
    assert FIXTURE["provenance"]["input_stage_sha256"] == _sha256(STAGE_PATH)
    assert FIXTURE["provenance"]["lockfile_sha256"] == _sha256(LOCK_PATH)
    assert FIXTURE["provenance"]["exporter_sha256"] == _sha256(EXPORTER_PATH)
    assert FIXTURE["counts"] == {
        "quaternion_algebra": 9,
        "axis_angle_and_rotation": 5,
        "slerp": 7,
        "stage_prims": 7,
    }
    assert len(FIXTURE["runtime"]["native_binary_files"]) == 4
    assert all(item["sha256"] for item in FIXTURE["runtime"]["native_binary_files"])
    zero_row = next(row for row in OBS["quaternion_algebra"] if row["case_id"] == "invalid-rotation-zero")
    assert "NaN" in json.dumps(zero_row)
    json.loads(
        FIXTURE_PATH.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )


def test_openusd_quatf_quatd_algebra_storage_and_action_match_independent_formulas():
    fixed = np.asarray((0.5, 0.25, -0.75, 1.25), dtype=float)
    vectors = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, -3)), dtype=float)
    for row in OBS["quaternion_algebra"]:
        q = np.asarray(row["input_wxyz"], dtype=float)
        np.testing.assert_allclose(row["quatd_wxyz"], q, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            row["quatf_wxyz"], q.astype(np.float32).astype(float), rtol=0.0, atol=0.0
        )
        qf = q.astype(np.float32).astype(float)
        qf_norm = np.linalg.norm(qf)
        assert row["quatf_length"] == pytest.approx(qf_norm, abs=2e-6)
        expected_qf_normalized = (
            np.asarray((1.0, 0.0, 0.0, 0.0))
            if qf_norm < 1e-10
            else qf / qf_norm
        )
        np.testing.assert_allclose(
            row["quatf_normalized_wxyz"], expected_qf_normalized, rtol=0.0, atol=2e-7
        )
        np.testing.assert_allclose(
            row["quatf_normalize_in_place_wxyz"], expected_qf_normalized, rtol=0.0, atol=2e-7
        )
        np.testing.assert_allclose(
            row["quatf_conjugate_wxyz"], np.r_[qf[0], -qf[1:]], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            row["quatf_product_with_fixed_wxyz"],
            _hamilton(qf, (0.5, 0.25, -0.75, 1.25)),
            rtol=0.0,
            atol=2e-6,
        )
        if row["quatf_length"] == 0.0:
            conjugate_qf = np.r_[qf[0], -qf[1:]]
            if np.all(qf == 0.0):
                expected_qf_inverse = ["NaN"] * 4
            else:
                expected_qf_inverse = [
                    "NaN"
                    if value == 0.0
                    else "Infinity" if value > 0.0 else "-Infinity"
                    for value in conjugate_qf
                ]
            assert row["quatf_inverse_wxyz"] == expected_qf_inverse
        else:
            np.testing.assert_allclose(
                row["quatf_inverse_wxyz"],
                np.r_[qf[0], -qf[1:]] / (qf_norm * qf_norm),
                rtol=2e-6,
                atol=0.0,
            )
        norm = np.linalg.norm(q)
        assert row["quatd_length"] == pytest.approx(norm, abs=2e-14)
        expected_normalized = (
            np.asarray((1.0, 0.0, 0.0, 0.0))
            if norm < 1e-10
            else q / norm
        )
        np.testing.assert_allclose(
            row["quatd_normalized_wxyz"], expected_normalized, rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            row["quatd_normalize_in_place_wxyz"], expected_normalized, rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            row["quatd_conjugate_wxyz"], np.r_[q[0], -q[1:]], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            row["quatd_product_with_fixed_wxyz"], _hamilton(q, fixed), rtol=0.0, atol=2e-14
        )
        if norm == 0.0:
            assert row["quatd_inverse_wxyz"] == ["NaN"] * 4
            assert all(
                item["transformed_xyz"] == ["NaN"] * 3
                for item in row["quatd_transform_vectors_xyz"]
            )
            assert all(
                item["transformed_xyz"] == ["NaN"] * 3
                for item in row["quatf_transform_vectors_xyz"]
            )
            continue

        expected_inverse = np.r_[q[0], -q[1:]] / (norm * norm)
        np.testing.assert_allclose(
            row["quatd_inverse_wxyz"], expected_inverse, rtol=1e-15, atol=0.0
        )
        rotation = Rotation.from_quat(_wxyz_to_xyzw(q))
        inverse_rotation = rotation.inv()
        for probe, vector in zip(row["quatd_transform_vectors_xyz"], vectors):
            assert probe["input_xyz"] == vector.tolist()
            np.testing.assert_allclose(
                probe["transformed_xyz"], rotation.apply(vector), rtol=0.0, atol=TOLERANCE
            )
            np.testing.assert_allclose(
                probe["inverse_transformed_xyz"],
                inverse_rotation.apply(vector),
                rtol=0.0,
                atol=TOLERANCE,
            )
        if row["quatf_length"] == 0.0:
            assert all(
                probe["transformed_xyz"] == ["NaN"] * 3
                and probe["inverse_transformed_xyz"] == ["NaN"] * 3
                for probe in row["quatf_transform_vectors_xyz"]
            )
        else:
            qf_rotation = Rotation.from_quat(_wxyz_to_xyzw(qf))
            qf_inverse_rotation = qf_rotation.inv()
            for probe, vector in zip(row["quatf_transform_vectors_xyz"], vectors):
                np.testing.assert_allclose(
                    probe["transformed_xyz"],
                    qf_rotation.apply(vector.astype(np.float32)).astype(float),
                    rtol=0.0,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    probe["inverse_transformed_xyz"],
                    qf_inverse_rotation.apply(vector.astype(np.float32)).astype(float),
                    rtol=0.0,
                    atol=2e-6,
                )


def test_openusd_rotation_axis_angle_and_gf_slerp_are_separately_checked():
    vector_probe = np.asarray((0.31, -0.47, 0.83))
    for row in OBS["axis_angle_and_rotation"]:
        expected = Rotation.from_rotvec(
            np.asarray(row["input_axis_xyz"], dtype=float) * row["input_angle_radians"]
        )
        matrix = np.asarray(row["rotation_matrix_rows"], dtype=float)[:3, :3]
        np.testing.assert_allclose(
            matrix, expected.as_matrix().T, rtol=0.0, atol=2e-12
        )
        assert row["rotation_get_angle_degrees"] == pytest.approx(
            row["input_angle_degrees"], abs=2e-12
        )
        np.testing.assert_allclose(
            row["rotation_matrix_transform_dir_xyz"],
            vector_probe @ matrix,
            rtol=0.0,
            atol=TOLERANCE,
        )
        assert _rotation_error(
            row["rotation_get_quaternion_wxyz"],
            np.r_[expected.as_quat()[3], expected.as_quat()[:3]],
        ) <= TOLERANCE

    for row in OBS["slerp"]:
        expected = _slerp_shortest(
            row["input_a_wxyz"], row["input_b_wxyz"], row["t"]
        )
        actual = np.asarray(row["slerp_wxyz"], dtype=float)
        assert _rotation_error(actual, expected) <= 4e-12
        assert row["slerp_norm"] == pytest.approx(1.0, abs=4e-12)


def test_openusd_saved_stage_keeps_axis_metadata_separate_and_evaluates_row_vector_stacks():
    stage = OBS["stage"]
    assert stage["stage_up_axis_metadata"] == "Y"
    assert stage["stage_meters_per_unit"] == 1.0
    assert stage["axis_probe_world_translations"] == {
        "X": [1.0, 0.0, 0.0],
        "Y": [0.0, 1.0, 0.0],
        "Z": [0.0, 0.0, 1.0],
    }
    parent = stage["parent"]
    child = stage["child"]
    reset = stage["reset_child"]
    assert [op["name"] for op in parent["ordered_xform_ops"]] == [
        "xformOp:translate",
        "xformOp:orient",
    ]
    assert [op["name"] for op in child["ordered_xform_ops"]] == [
        "xformOp:rotateXYZ",
        "xformOp:translate",
    ]
    assert reset["resets_xform_stack"] is True

    parent_q = np.asarray((0.9238795325112867, 0.0, 0.0, 0.3826834323650898))
    parent_rotation = Rotation.from_quat(_wxyz_to_xyzw(parent_q)).as_matrix().T
    parent_expected = _row_rotation(parent_rotation) @ _row_translation((1.25, -2.0, 0.5))
    np.testing.assert_allclose(parent["local_matrix_rows"], parent_expected, rtol=0.0, atol=TOLERANCE)

    child_rotation = Rotation.from_euler("xyz", (25.0, -35.0, 70.0), degrees=True).as_matrix().T
    child_expected = _row_translation((-0.75, 1.5, 2.25)) @ _row_rotation(child_rotation)
    np.testing.assert_allclose(child["local_matrix_rows"], child_expected, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(
        child["world_matrix_rows"],
        np.asarray(child["local_matrix_rows"]) @ np.asarray(parent["world_matrix_rows"]),
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(reset["world_matrix_rows"], reset["local_matrix_rows"], rtol=0.0, atol=0.0)

    point = np.asarray((0.31, -0.47, 0.83, 1.0))
    direction = np.asarray((0.31, -0.47, 0.83, 0.0))
    for row in (parent, child, reset):
        matrix = np.asarray(row["world_matrix_rows"], dtype=float)
        np.testing.assert_allclose(
            row["world_transform_point_xyz"], (point @ matrix)[:3], rtol=0.0, atol=TOLERANCE
        )
        np.testing.assert_allclose(
            row["world_transform_direction_xyz"],
            (direction @ matrix)[:3],
            rtol=0.0,
            atol=TOLERANCE,
        )
