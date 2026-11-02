"""Frozen upstream Euler fixtures for game-library convention checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.oracle_support import (
    BACKENDS,
    as_hamilton_coefficients,
    orient_vectors,
    vectors_to_trailing,
)

FIXTURE_PATH = Path(__file__).parent / "golden" / "engine_euler_fixtures.json"
FIXTURES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def _engine_sequence_and_angles(case):
    by_axis = dict(zip("xyz", case["input_degrees_xyz"]))
    angles = tuple(by_axis[axis.lower()] for axis in case["engine_order"])
    sequence = case["engine_order"] if case["intrinsic"] else case["engine_order"].lower()
    return sequence, angles


@pytest.mark.parametrize("case", FIXTURES, ids=[case["case_id"] for case in FIXTURES])
@pytest.mark.parametrize(
    "backend_name,module,is_jpl,is_numpy,component_first",
    BACKENDS,
    ids=[backend[0] for backend in BACKENDS],
)
def test_quatops_euler_matches_frozen_engine_golden(
    case, backend_name, module, is_jpl, is_numpy, component_first
):
    sequence, angles = _engine_sequence_and_angles(case)
    angle_input = orient_vectors(np.asarray(angles, dtype=float), component_first)
    actual = module.from_euler(angle_input, sequence, degrees=True)
    actual_wxyz = np.asarray(
        module.to_wxyz(actual), dtype=float
    )
    actual_wxyz = vectors_to_trailing(actual_wxyz, component_first).reshape((-1, 4))

    # Compare orientation up to one global quaternion sign. The fixture's
    # published decimal precision is limited, so retain its declared tolerance.
    actual_hamilton_wxyz = as_hamilton_coefficients(actual_wxyz, is_jpl)
    actual_xyzw = actual_hamilton_wxyz[:, (1, 2, 3, 0)]
    np.testing.assert_allclose(
        np.linalg.norm(actual_xyzw[0]), 1.0, rtol=0.0, atol=2e-15,
        err_msg=f"{backend_name}: {case['case_id']} unit norm",
    )
    expected_xyzw = np.asarray(case["expected_quaternion_xyzw"], dtype=float)
    coefficient_error = min(
        np.linalg.norm(actual_xyzw[0] - expected_xyzw),
        np.linalg.norm(actual_xyzw[0] + expected_xyzw),
    )
    assert coefficient_error <= case["quaternion_atol"], backend_name

    actual_matrices = Rotation.from_quat(actual_xyzw).as_matrix()
    expected_matrix = np.asarray(case["expected_matrix_rows"], dtype=float)
    np.testing.assert_allclose(
        actual_matrices[0], expected_matrix,
        rtol=0.0,
        atol=case["matrix_atol"],
        err_msg=f"{backend_name}: {case['case_id']}",
    )
