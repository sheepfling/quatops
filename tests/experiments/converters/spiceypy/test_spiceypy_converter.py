"""Replay selected CSPICE direct-quaternion boundary witnesses through QuatOps."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.python import hamilton
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter"
CONTRACT_PATH = ROOT / "experiments/converters/spiceypy/8.3.0/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads(
    (ROOT / "experiments/converters/spiceypy/8.3.0/converter_inputs.json").read_text(
        encoding="utf-8"
    )
)
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
SHARED_PATH = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
SHARED = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
OBS = WITNESSES["observations"]
CANONICAL = {
    item["case_id"]: np.asarray(item["canonical_wxyz"], dtype=float)
    for item in SHARED["orientations"]
}
PROBE = np.asarray(SHARED["probe_vector_xyz"], dtype=float)
BASIS = np.eye(3)
ATOL = 3e-12


def matrix(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def check(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ATOL, err_msg=message)


def test_spiceypy_boundary_contract_and_capture_provenance_are_explicit():
    assert (
        CONTRACT["contract_id"]
        == "quatops-spiceypy-8.3.0-direct-quaternion-boundary-v1"
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
    assert CONTRACT["target"]["entry_points"] == ["q2m", "mxv", "m2q", "qxq", "axisar"]
    assert "eul2m" in CONTRACT["excludes"][0]
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["target"]["wrapper_version"] == "8.3.0"
    assert MANIFEST["target"]["toolkit_version"] == "CSPICE_N0067"
    assert MANIFEST["selection"] == {
        "retained_orientation_cases": 5,
        "native_created_orientations": 1,
        "noncommuting_product_orders": 2,
        "native_basis_images_per_orientation": 3,
        "native_mixed_probe_case_id": "mixed_a",
    }
    paths = {
        "reproduction_capture_script_sha256": ROOT
        / "experiments/converters/spiceypy/8.3.0/capture_native.py",
        "contract_sha256": CONTRACT_PATH,
        "converter_input_spec_sha256": ROOT
        / "experiments/converters/spiceypy/8.3.0/converter_inputs.json",
        "selected_inputs_sha256": SHARED_PATH,
        "witnesses_sha256": FIXTURE / "witnesses.json",
        "pinned_lockfile_sha256": ROOT
        / "experiments/converters/spiceypy/8.3.0/uv.lock",
    }
    for field, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == MANIFEST["provenance"][field]
        ), field
    assert (
        hashlib.sha256(
            read_provenance_source(
                ROOT, MANIFEST["provenance"]["capture_script_archive"]
            )
        ).hexdigest()
        == MANIFEST["provenance"]["capture_script_sha256"]
    )
    capture_source = paths["reproduction_capture_script_sha256"].read_text(
        encoding="utf-8"
    )
    assert (
        "import quatops" not in capture_source and "from quatops" not in capture_source
    )
    assert len(MANIFEST["provenance"]["loaded_cspice_binary_sha256"]) == 64
    assert (
        MANIFEST["provenance"]["loaded_cspice_binary_path"]
        .lower()
        .endswith("cspice.dll")
    )


def test_spiceypy_native_quaternion_action_preserves_canonical_physical_action():
    selected_ids = NATIVE_RETAINED_CASE_IDS
    rows = {item["case_id"]: item for item in OBS["canonical_to_native"]}
    assert set(rows) == set(selected_ids)
    for case_id in selected_ids:
        canonical = CANONICAL[case_id]
        native = rows[case_id]
        q_native = np.asarray(native["native_input_wxyz"], dtype=float)
        np.testing.assert_array_equal(q_native, canonical)
        expected_actions = np.asarray([hamilton.rotate(canonical, p) for p in BASIS])
        check(
            native["native_mxv_basis_xyz"],
            expected_actions,
            f"SPICE mxv {case_id}",
        )
        if case_id == "mixed_a":
            check(
                native["native_mxv_probe_xyz"],
                hamilton.rotate(canonical, PROBE),
                "SPICE mixed-vector action",
            )

    # The antipode is a distinct coefficient input but must have identical physical action.
    check(
        rows["antipode_mixed_a"]["native_mxv_basis_xyz"],
        rows["mixed_a"]["native_mxv_basis_xyz"],
        "SPICE q versus -q",
    )


def test_spiceypy_native_created_axisar_m2q_route_converts_back_to_canonical():
    case = INPUTS["native_constructed_orientation"]
    native = OBS["native_created"][0]
    expected_matrix = Rotation.from_rotvec(
        np.asarray(case["axis_xyz"]) * case["angle_radians"]
    ).as_matrix()
    q_native = np.asarray(native["native_m2q_wxyz"], dtype=float)
    check(matrix(q_native), expected_matrix, "SPICE axisar/m2q orientation")
    check(
        native["native_mxv_basis_xyz"],
        (expected_matrix @ BASIS.T).T,
        "SPICE native-created mxv",
    )
    check(
        [hamilton.rotate(q_native, p) for p in BASIS],
        native["native_mxv_basis_xyz"],
        "SPICE native-to-canonical identity map",
    )


def test_spiceypy_qxq_matches_quatops_noncommuting_composition_both_orders():
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
        check(row["native_input_wxyz"], expected, f"SPICE qxq coefficients {case_id}")
        check(
            row["native_mxv_basis_xyz"],
            [hamilton.rotate(expected, p) for p in BASIS],
            f"SPICE qxq action {case_id}",
        )

    forward = np.asarray(rows["mixed_a_times_mixed_b"]["native_mxv_basis_xyz"])
    reverse = np.asarray(rows["mixed_b_times_mixed_a"]["native_mxv_basis_xyz"])
    assert np.max(np.abs(forward - reverse)) > 0.1
