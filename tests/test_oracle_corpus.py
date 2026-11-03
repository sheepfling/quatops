"""Integrity and input-domain checks for the shared native-capture corpus."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

CORPUS_PATH = (
    Path(__file__).parent.parent
    / "experiments"
    / "oracles"
    / "shared"
    / "orientation_inputs_v1.json"
)
CORPUS_SHA256 = "eaf7605b3298ec9dbdf7838f43f2da61d90af5351bb15e455d3eb37c6c6b5446"
EXPECTED_COUNTS = {
    "axis_angle_inputs": 5,
    "euler_boundaries": 7,
    "euler_inputs": 1,
    "integration_inputs": 4,
    "interpolation_pairs": 7,
    "matrix_inputs": 4,
    "multiplication_pairs": 3,
    "quaternion_inputs": 9,
    "trajectories": 3,
}
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_shared_corpus_is_the_frozen_input_only_release():
    assert hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest() == CORPUS_SHA256
    assert CORPUS["schema_version"] == "1.0"
    assert CORPUS["kind"] == "inputs_only_not_goldens"
    for group, count in EXPECTED_COUNTS.items():
        assert len(CORPUS[group]) == count


def test_corpus_case_ids_are_unique_within_each_operation_family():
    for group in EXPECTED_COUNTS:
        case_ids = [row["case_id"] for row in CORPUS[group]]
        assert len(case_ids) == len(set(case_ids)), group


def test_corpus_matrix_inputs_label_proper_and_invalid_rotation_domains():
    valid_count = 0
    invalid_count = 0
    for case in CORPUS["matrix_inputs"]:
        matrix = np.asarray(case["rows"], dtype=float)
        if case["domain"] == "proper_rotation":
            valid_count += 1
            np.testing.assert_allclose(
                matrix @ matrix.T, np.eye(3), rtol=0.0, atol=2e-12
            )
            assert np.linalg.det(matrix) == pytest.approx(1.0, abs=2e-12)
            np.testing.assert_allclose(
                Rotation.from_matrix(matrix).as_matrix(),
                matrix,
                rtol=0.0,
                atol=2e-12,
            )
        else:
            invalid_count += 1
            assert case["domain"] == "invalid_rotation"
    assert valid_count == 3
    assert invalid_count == 1


def test_shared_inputs_record_which_broad_discovery_corpus_was_reduced():
    assert (
        CORPUS["retention"]["source_discovery_corpus_sha256"]
        == "07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f"
    )
    assert CORPUS["retention"]["discovery_counts"] == {
        "axis_angle_inputs": 60,
        "euler_boundaries": 540,
        "euler_inputs": 89,
        "integration_inputs": 48,
        "interpolation_pairs": 7,
        "matrix_inputs": 11,
        "multiplication_pairs": 3,
        "quaternion_inputs": 140,
        "signed_axis_maps": 48,
        "trajectories": 3,
    }
    assert "signed_axis_maps" not in CORPUS


def test_trajectory_inputs_keep_strictly_increasing_times():
    for trajectory in CORPUS["trajectories"]:
        times = np.asarray(trajectory["times_seconds"], dtype=float)
        assert len(times) == len(trajectory["values_xyz_degrees"])
        assert np.all(np.diff(times) > 0.0)
