"""Replay OpenCV Rodrigues converter witnesses through QuatOps."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.python import hamilton
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/oracles/opencv/4.13.0/converter"
CONTRACT = json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
SPEC_PATH = ROOT / "experiments/converters/opencv/4.13.0/converter_inputs.json"
INPUTS = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
SHARED_PATH = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
SHARED = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
CANONICAL = {
    item["case_id"]: np.asarray(item["canonical_wxyz"], dtype=float)
    for item in SHARED["orientations"]
}
PROBE = np.asarray(SHARED["probe_vector_xyz"], dtype=float)
BASIS = np.eye(3)
ATOL = {"float32": 2.5e-6, "float64": 4e-12}


def matrix(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def check(actual, expected, dtype, message):
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=ATOL[dtype], err_msg=message
    )


def test_opencv_contract_and_pinned_capture_provenance():
    assert CONTRACT["contract_id"] == "quatops-opencv-4.13.0-rodrigues-boundary-v1"
    assert CONTRACT["target_boundary"]["probe_frame_assignment"]["meaning"].startswith(
        "explicit numeric test frame"
    )
    assert CONTRACT["target"]["scalar_profiles"] == ["float32", "float64"]
    assert CONTRACT["target"]["interface"] == "cv2.Rodrigues and cv2.gemm"
    assert "camera/world extrinsics" in CONTRACT["excludes"][1]
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["target"]["version"] == "4.13.0"
    assert MANIFEST["target"]["distribution_version"] == "4.13.0.92"
    assert MANIFEST["target"]["native_scalar_profiles"] == ["float32", "float64"]
    assert MANIFEST["selection"] == {
        "orientation_cases_per_precision": 8,
        "native_created_orientations_per_precision": 1,
        "noncommuting_product_orders_per_precision": 2,
        "native_basis_images_per_orientation": 3,
        "native_mixed_probe_case_id": "mixed_a",
    }

    provenance = MANIFEST["provenance"]
    paths = {
        "preparation_spec_sha256": ROOT
        / "experiments/converters/opencv/4.13.0/converter_inputs.json",
        "witnesses_sha256": FIXTURE / "witnesses.json",
    }
    for field, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field], field

    reproduction = MANIFEST["reproduction"]
    reproduction_paths = {
        "capture_script_sha256": ROOT
        / "experiments/converters/opencv/4.13.0/capture_native.py",
        "preparation_script_sha256": ROOT
        / "experiments/converters/opencv/4.13.0/prepare_inputs.py",
        "lockfile_sha256": ROOT / "experiments/converters/opencv/4.13.0/uv.lock",
    }
    for field, path in reproduction_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reproduction[field], (
            field
        )
    assert (
        provenance["shared_canonical_inputs_sha256"]
        == hashlib.sha256(SHARED_PATH.read_bytes()).hexdigest()
    )
    assert INPUTS["shared_canonical_inputs"] == SHARED_PATH.relative_to(ROOT).as_posix()
    native_capture = reproduction_paths["capture_script_sha256"].read_text(
        encoding="utf-8"
    )
    assert (
        "import quatops" not in native_capture and "from quatops" not in native_capture
    )
    assert MANIFEST["native_binaries"] and all(
        item["sha256"] for item in MANIFEST["native_binaries"]
    )


def test_opencv_rodrigues_forward_reverse_and_vector_actions_match_quatops():
    profile_rows = WITNESSES["observations"]["canonical_to_native"]
    input_case_ids = INPUTS["retained_case_ids"]
    for dtype in INPUTS["precision_profiles"]:
        rows = {
            item["case_id"]: item for item in profile_rows if item["dtype"] == dtype
        }
        assert set(rows) == set(input_case_ids)
        for case_id in input_case_ids:
            row = rows[case_id]
            q = CANONICAL[case_id]
            rotvec = np.asarray(hamilton.to_rotation_vector(q), dtype=dtype)
            expected = matrix(q)
            check(
                row["native_input_rotation_vector_xyz_radians"],
                rotvec,
                dtype,
                f"OpenCV input rotation vector {case_id}/{dtype}",
            )
            check(
                row["native_gemm_basis_actions_xyz"],
                [hamilton.rotate(q, p) for p in BASIS],
                dtype,
                f"OpenCV native gemm basis action {case_id}/{dtype}",
            )
            check(
                row["native_gemm_basis_actions_xyz"],
                (expected @ BASIS.T).T,
                dtype,
                f"OpenCV basis action {case_id}/{dtype}",
            )
            if case_id == "mixed_a":
                check(
                    row["native_gemm_probe_action_xyz"],
                    hamilton.rotate(q, PROBE),
                    dtype,
                    f"OpenCV mixed probe action {case_id}/{dtype}",
                )
            recovered_q = hamilton.from_rotation_vector(
                row["native_matrix_to_rotation_vector_xyz_radians"]
            )
            check(
                matrix(recovered_q),
                expected,
                dtype,
                f"OpenCV native-to-canonical route {case_id}/{dtype}",
            )

        check(
            rows["antipode_mixed_a"]["native_gemm_basis_actions_xyz"],
            rows["mixed_a"]["native_gemm_basis_actions_xyz"],
            dtype,
            f"OpenCV q versus -q/{dtype}",
        )


def test_opencv_native_axis_angle_and_noncommuting_matrix_composition():
    ctor = INPUTS["native_constructed_orientation"]
    expected_native_q = np.asarray(
        hamilton.from_rotation_vector(
            np.asarray(ctor["axis_xyz"]) * ctor["angle_radians"]
        )
    )
    expected_native_matrix = matrix(expected_native_q)
    for dtype in INPUTS["precision_profiles"]:
        native = next(
            item
            for item in WITNESSES["observations"]["native_created"]
            if item["dtype"] == dtype
        )
        check(
            native["native_gemm_basis_actions_xyz"],
            (expected_native_matrix @ BASIS.T).T,
            dtype,
            f"OpenCV native axis-angle action/{dtype}",
        )
        recovered_q = hamilton.from_rotation_vector(
            native["native_matrix_to_rotation_vector_xyz_radians"]
        )
        check(
            matrix(recovered_q),
            expected_native_matrix,
            dtype,
            f"OpenCV axis-angle reverse/{dtype}",
        )

        products = {
            row["case_id"]: row
            for row in WITNESSES["observations"]["composition"]
            if row["dtype"] == dtype
        }
        for case_id, lhs_id, rhs_id in (
            ("mixed_a_times_mixed_b", "mixed_a", "mixed_b"),
            ("mixed_b_times_mixed_a", "mixed_b", "mixed_a"),
        ):
            row = products[case_id]
            expected_q = np.asarray(
                hamilton.multiply(CANONICAL[lhs_id], CANONICAL[rhs_id])
            )
            expected_matrix = matrix(expected_q)
            check(
                row["native_gemm_basis_actions_xyz"],
                [hamilton.rotate(expected_q, p) for p in BASIS],
                dtype,
                f"OpenCV gemm composition action {case_id}/{dtype}",
            )
            check(
                matrix(
                    hamilton.from_rotation_vector(
                        row["native_matrix_to_rotation_vector_xyz_radians"]
                    )
                ),
                expected_matrix,
                dtype,
                f"OpenCV composition to canonical {case_id}/{dtype}",
            )

        forward = np.asarray(
            products["mixed_a_times_mixed_b"]["native_gemm_basis_actions_xyz"]
        )
        reverse = np.asarray(
            products["mixed_b_times_mixed_a"]["native_gemm_basis_actions_xyz"]
        )
        assert np.max(np.abs(forward - reverse)) > 0.1
