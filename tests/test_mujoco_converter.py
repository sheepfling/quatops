"""Replay selected MuJoCo direct-quaternion boundary witnesses through QuatOps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.python import hamilton

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/golden/oracles/mujoco/3.17.0/converter"
CONTRACT = json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads((FIXTURE / "inputs.json").read_text(encoding="utf-8"))
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
SHARED_PATH = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
SHARED = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
OBS = WITNESSES["observations"]
CANONICAL = {
    item["case_id"]: np.asarray(item["canonical_wxyz"], dtype=float)
    for item in SHARED["orientations"]
}
PROBE = np.asarray(SHARED["probe_vector_xyz"], dtype=float)
PROBES = np.vstack((np.eye(3), PROBE))
ATOL = 3e-12


def matrix(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def check(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ATOL, err_msg=message)


def test_mujoco_boundary_contract_and_capture_provenance_are_explicit():
    assert (
        CONTRACT["contract_id"] == "quatops-mujoco-3.17.0-direct-quaternion-boundary-v1"
    )
    assert CONTRACT["canonical"]["frame"] == {
        "positive_x": "right",
        "positive_y": "forward",
        "positive_z": "up",
    }
    assert CONTRACT["target_boundary"]["source_to_target_coordinate_map"] == [
        "x",
        "y",
        "z",
    ]
    assert CONTRACT["target_boundary"]["probe_frame_assignment"]["meaning"].startswith(
        "explicit test frame assignment"
    )
    assert CONTRACT["target_boundary"]["canonical_to_native"].startswith(
        "identity on WXYZ"
    )
    assert CONTRACT["target"]["entry_points"] == [
        "mju_quat2Mat",
        "mju_rotVecQuat",
        "mju_mat2Quat",
        "mju_mulQuat",
        "mju_axisAngle2Quat",
    ]
    assert "robot link" in CONTRACT["excludes"][1]
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["target"]["package_version"] == "3.17.0"
    assert MANIFEST["target"]["engine_version"] == "3.17.0"
    assert MANIFEST["target"]["native_scalar_dtype"] == "float64"
    assert MANIFEST["selection"] == {
        "retained_orientation_cases": 8,
        "native_created_orientations": 1,
        "noncommuting_product_orders": 2,
        "native_quaternion_actions_per_orientation": 4,
    }
    full = json.loads(
        (ROOT / "tests/golden/oracles/mujoco/3.17.0/native_rotation.json").read_text(
            encoding="utf-8"
        )
    )
    paths = {
        "capture_script_sha256": ROOT
        / "experiments/converters/mujoco/3.17.0/capture_native.py",
        "contract_sha256": FIXTURE / "contract.json",
        "converter_input_spec_sha256": ROOT
        / "experiments/converters/mujoco/3.17.0/converter_inputs.json",
        "selected_inputs_sha256": SHARED_PATH,
        "witnesses_sha256": FIXTURE / "witnesses.json",
        "full_oracle_sha256": ROOT
        / "tests/golden/oracles/mujoco/3.17.0/native_rotation.json",
        "pinned_lockfile_sha256": ROOT / "experiments/oracles/mujoco/3.17.0/uv.lock",
    }
    for field, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["provenance"][field]
        ), field
    capture_source = paths["capture_script_sha256"].read_text(encoding="utf-8")
    assert (
        "import quatops" not in capture_source and "from quatops" not in capture_source
    )
    assert sorted(
        (item["path"], item["sha256"]) for item in MANIFEST["native_binary_files"]
    ) == sorted(
        (item["path"], item["sha256"])
        for item in full["runtime"]["native_binary_files_under_package"]
    )
    assert (FIXTURE / "inputs.json").read_bytes() == (
        ROOT / "experiments/converters/mujoco/3.17.0/converter_inputs.json"
    ).read_bytes()


def test_mujoco_native_quat2mat_rotvec_and_mat2quat_preserve_canonical_action():
    selected_ids = INPUTS["retained_case_ids"]
    rows = {item["case_id"]: item for item in OBS["canonical_to_native"]}
    assert set(rows) == set(selected_ids)
    for case_id in selected_ids:
        canonical = CANONICAL[case_id]
        native = rows[case_id]
        q_native = np.asarray(native["native_input_wxyz"], dtype=float)
        np.testing.assert_array_equal(q_native, canonical)
        expected_matrix = matrix(canonical)
        check(
            native["native_mju_quat2Mat_rows"],
            expected_matrix,
            f"MuJoCo mju_quat2Mat {case_id}",
        )
        expected_actions = np.asarray([hamilton.rotate(canonical, p) for p in PROBES])
        check(
            native["native_mju_rotVecQuat_basis_and_probe_xyz"],
            expected_actions,
            f"MuJoCo mju_rotVecQuat {case_id}",
        )
        check(
            native["native_mju_rotVecQuat_basis_and_probe_xyz"],
            (expected_matrix @ PROBES.T).T,
            f"MuJoCo basis action {case_id}",
        )
        check(
            matrix(native["native_mju_mat2Quat_roundtrip_wxyz"]),
            expected_matrix,
            f"MuJoCo mju_mat2Quat roundtrip {case_id}",
        )

    check(
        rows["antipode_mixed_a"]["native_mju_rotVecQuat_basis_and_probe_xyz"],
        rows["mixed_a"]["native_mju_rotVecQuat_basis_and_probe_xyz"],
        "MuJoCo q versus -q",
    )


def test_mujoco_native_axisangle_constructor_converts_back_to_canonical():
    case = INPUTS["native_constructed_orientation"]
    native = OBS["native_created"][0]
    expected_matrix = Rotation.from_rotvec(
        np.asarray(case["axis_xyz"]) * case["angle_radians"]
    ).as_matrix()
    q_native = np.asarray(native["native_input_wxyz"], dtype=float)
    check(
        native["native_mju_quat2Mat_rows"],
        expected_matrix,
        "MuJoCo native-created quaternion matrix",
    )
    check(
        native["native_mju_rotVecQuat_basis_and_probe_xyz"],
        (expected_matrix @ PROBES.T).T,
        "MuJoCo native-created vector action",
    )
    check(
        [hamilton.rotate(q_native, p) for p in PROBES],
        native["native_mju_rotVecQuat_basis_and_probe_xyz"],
        "MuJoCo native-to-canonical identity map",
    )
    check(
        matrix(native["native_mju_mat2Quat_roundtrip_wxyz"]),
        expected_matrix,
        "MuJoCo native-created reverse matrix roundtrip",
    )


def test_mujoco_mulquat_matches_quatops_noncommuting_composition_both_orders():
    rows = {item["case_id"]: item for item in OBS["composition"]}
    for case_id, lhs_id, rhs_id in (
        ("mixed_a_times_mixed_b", "mixed_a", "mixed_b"),
        ("mixed_b_times_mixed_a", "mixed_b", "mixed_a"),
    ):
        row = rows[case_id]
        expected = np.asarray(
            hamilton.multiply(CANONICAL[lhs_id], CANONICAL[rhs_id]), dtype=float
        )
        np.testing.assert_allclose(
            row["native_lhs_wxyz"], CANONICAL[lhs_id], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            row["native_rhs_wxyz"], CANONICAL[rhs_id], rtol=0.0, atol=0.0
        )
        check(
            row["native_input_wxyz"],
            expected,
            f"MuJoCo mju_mulQuat coefficients {case_id}",
        )
        check(
            row["native_mju_quat2Mat_rows"],
            matrix(expected),
            f"MuJoCo product matrix {case_id}",
        )
        check(
            row["native_mju_rotVecQuat_basis_and_probe_xyz"],
            [hamilton.rotate(expected, p) for p in PROBES],
            f"MuJoCo product action {case_id}",
        )

    forward = np.asarray(
        rows["mixed_a_times_mixed_b"]["native_mju_rotVecQuat_basis_and_probe_xyz"][-1]
    )
    reverse = np.asarray(
        rows["mixed_b_times_mixed_a"]["native_mju_rotVecQuat_basis_and_probe_xyz"][-1]
    )
    assert np.max(np.abs(forward - reverse)) > 0.1
