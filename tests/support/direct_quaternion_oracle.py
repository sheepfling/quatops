"""Replay direct Hamilton WXYZ native witnesses against QuatOps and SciPy."""

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.python import hamilton
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from tests.support.evidence_oracles import matrix_from_wxyz

_BASIS = np.eye(3)
_ATOL = 3e-12


def _check(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_ATOL, err_msg=message)


def check_orientations(observations, canonical, probe, *, basis_field, probe_field):
    rows = {row["case_id"]: row for row in observations["canonical_to_native"]}
    assert set(rows) == set(NATIVE_RETAINED_CASE_IDS)
    for case_id in NATIVE_RETAINED_CASE_IDS:
        native = rows[case_id]
        quaternion = canonical[case_id]
        np.testing.assert_array_equal(native["native_input_wxyz"], quaternion)
        _check(
            native[basis_field],
            [hamilton.rotate(quaternion, p) for p in _BASIS],
            case_id,
        )
        if case_id == "mixed_a":
            _check(native[probe_field], hamilton.rotate(quaternion, probe), case_id)
    _check(
        rows["antipode_mixed_a"][basis_field], rows["mixed_a"][basis_field], "antipode"
    )


def check_native_constructor(observations, inputs, *, basis_field, quaternion_field):
    case = inputs["native_constructed_orientation"]
    native = observations["native_created"][0]
    expected = Rotation.from_rotvec(
        np.asarray(case["axis_xyz"]) * case["angle_radians"]
    ).as_matrix()
    quaternion = np.asarray(native[quaternion_field], dtype=float)
    _check(matrix_from_wxyz(quaternion), expected, "native-created quaternion")
    _check(native[basis_field], (expected @ _BASIS.T).T, "native-created vector action")
    _check(
        [hamilton.rotate(quaternion, p) for p in _BASIS],
        native[basis_field],
        "native-to-canonical vector action",
    )


def check_composition(observations, canonical, *, basis_field):
    rows = {row["case_id"]: row for row in observations["composition"]}
    for case_id, lhs_id, rhs_id in (
        ("mixed_a_times_mixed_b", "mixed_a", "mixed_b"),
        ("mixed_b_times_mixed_a", "mixed_b", "mixed_a"),
    ):
        row = rows[case_id]
        expected = np.asarray(hamilton.multiply(canonical[lhs_id], canonical[rhs_id]))
        np.testing.assert_allclose(
            row["native_lhs_wxyz"], canonical[lhs_id], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            row["native_rhs_wxyz"], canonical[rhs_id], rtol=0.0, atol=0.0
        )
        _check(row["native_input_wxyz"], expected, case_id)
        _check(
            row[basis_field], [hamilton.rotate(expected, p) for p in _BASIS], case_id
        )
    forward = np.asarray(rows["mixed_a_times_mixed_b"][basis_field])
    reverse = np.asarray(rows["mixed_b_times_mixed_a"][basis_field])
    assert np.max(np.abs(forward - reverse)) > 0.1
