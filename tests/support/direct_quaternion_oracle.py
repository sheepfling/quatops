"""Replay direct Hamilton WXYZ native witnesses against QuatOps and SciPy."""

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.python import hamilton
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from tests.support.evidence_oracles import matrix_from_wxyz
from tests.support.paths import REPOSITORY_ROOT

_BASIS = np.eye(3)
_ATOL = 3e-12


def check_contract_and_provenance(contract, manifest, *, source, fixture):
    """Check the shared direct-quaternion contract and pinned capture inputs."""
    assert contract["canonical"]["frame"] == {
        "positive_x": "right",
        "positive_y": "forward",
        "positive_z": "up",
    }
    boundary = contract["target_boundary"]
    assert boundary["source_to_target_coordinate_map"] == ["x", "y", "z"]
    assert boundary["probe_frame_assignment"]["meaning"].startswith(
        "explicit test frame assignment"
    )
    assert boundary["canonical_to_native"].startswith("identity on WXYZ")
    assert manifest["execution"]["was_run"] is True
    assert manifest["selection"] == {
        "retained_orientation_cases": 5,
        "native_created_orientations": 1,
        "noncommuting_product_orders": 2,
        "native_basis_images_per_orientation": 3,
        "native_mixed_probe_case_id": "mixed_a",
    }
    paths = {
        "contract_sha256": source / "semantic_contract.json",
        "converter_input_spec_sha256": source / "converter_inputs.json",
        "selected_inputs_sha256": REPOSITORY_ROOT
        / "experiments/interoperability/v1/canonical_inputs.json",
        "witnesses_sha256": fixture / "witnesses.json",
    }
    for field, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == manifest["provenance"][field]
        ), field
    replay = manifest["unified_lock_replay"]
    replay_paths = {
        "capture_script_sha256": source / "capture_native.py",
        "shared_capture_script_sha256": REPOSITORY_ROOT
        / "scripts/direct_quaternion_capture.py",
        "lockfile_sha256": REPOSITORY_ROOT / "experiments/converters/python/uv.lock",
    }
    for field, path in replay_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == replay[field], field
    assert replay["witnesses_sha256"] == manifest["provenance"]["witnesses_sha256"]
    for field in ("capture_script_sha256", "shared_capture_script_sha256"):
        capture_source = replay_paths[field].read_text(encoding="utf-8")
        assert "import quatops" not in capture_source
        assert "from quatops" not in capture_source


def canonical_cases_and_probe(shared):
    """Read the shared canonical inputs for direct-library replay."""
    cases = {
        item["case_id"]: np.asarray(item["canonical_wxyz"], dtype=float)
        for item in shared["orientations"]
    }
    return cases, np.asarray(shared["probe_vector_xyz"], dtype=float)


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
