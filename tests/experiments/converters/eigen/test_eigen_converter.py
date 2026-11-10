"""Replay native Eigen quaternion, storage, and matrix-construction witnesses."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

import numpy as np

from quatops.python import hamilton
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from tests.support.evidence_oracles import matrix_from_wxyz as matrix
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/oracles/eigen/5.0.1/converter"
CONTRACT_PATH = ROOT / "experiments/converters/eigen/5.0.1/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
SPEC_PATH = ROOT / "experiments/converters/eigen/5.0.1/converter_inputs.json"
INPUTS = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
SOURCE = json.loads(
    (ROOT / "experiments/oracles/eigen/5.0.1/source.json").read_text(encoding="utf-8")
)
SHARED_PATH = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
SHARED = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
CANONICAL = {
    item["case_id"]: np.asarray(item["canonical_wxyz"], dtype=float)
    for item in SHARED["orientations"]
}
BASIS = np.eye(3)
ATOL = {"float32": 2e-6, "float64": 4e-12}


def check(actual, expected, dtype, message):
    np.testing.assert_allclose(
        actual, expected, rtol=0.0, atol=ATOL[dtype], err_msg=message
    )


def coeffs_xyzw(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return q[[1, 2, 3, 0]]


def test_eigen_contract_source_pin_and_native_build_provenance():
    assert CONTRACT["contract_id"] == "quatops-eigen-5.0.1-quaternion-boundary-v1"
    assert (
        CONTRACT["target_boundary"]["constructor_order"]
        == "Eigen::Quaternion(w, x, y, z)"
    )
    assert CONTRACT["target_boundary"]["coeffs_access_order"] == "XYZW"
    assert CONTRACT["target_boundary"]["probe_frame_assignment"]["meaning"].startswith(
        "explicit numeric test frame"
    )
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["target"]["version"] == "5.0.1"
    assert MANIFEST["target"]["scalar_profiles"] == ["float32", "float64"]
    assert MANIFEST["native_build"]["cxx_standard"] == "C++17"
    assert (
        "GCC" in MANIFEST["native_build"]["compiler_version"]
        and "15.2.0" in MANIFEST["native_build"]["compiler_version"]
    )
    assert MANIFEST["native_build"]["version_macros"] == [3, 5, 0, 1]
    assert MANIFEST["selection"] == {
        "orientation_cases_per_scalar_profile": 5,
        "native_created_orientations_per_scalar_profile": 1,
        "noncommuting_product_orders_per_scalar_profile": 2,
        "native_basis_images_per_orientation": 3,
        "native_mixed_probe_case_id": "mixed_a",
        "matrix_to_quaternion_case_ids": ["mixed_a", "nonprincipal_halfturn"],
        "matrix_layout_case_id": "mixed_a",
    }
    provenance = MANIFEST["provenance"]
    paths = {
        "reproduction_capture_runner_sha256": ROOT
        / "experiments/converters/eigen/5.0.1/capture_native.py",
        "reproduction_native_cpp_source_sha256": ROOT
        / "experiments/converters/eigen/5.0.1/capture_native.cpp",
        "contract_sha256": CONTRACT_PATH,
        "converter_input_spec_sha256": SPEC_PATH,
        "selected_inputs_sha256": SHARED_PATH,
        "witnesses_sha256": FIXTURE / "witnesses.json",
        "eigen_source_metadata_sha256": ROOT
        / "experiments/oracles/eigen/5.0.1/source.json",
        "reproduction_lockfile_sha256": ROOT
        / "experiments/converters/python313/uv.lock",
    }
    for field, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field], field
    replay = MANIFEST["compact_lock_replay"]
    assert replay["lockfile_sha256"] == provenance["pre_shared_lockfile_sha256"]
    assert replay["witnesses_sha256"] == provenance["witnesses_sha256"]
    assert (
        MANIFEST["shared_lock_replay"]["lockfile_sha256"]
        == provenance["reproduction_lockfile_sha256"]
    )
    assert (
        MANIFEST["shared_lock_replay"]["witnesses_sha256"]
        == provenance["witnesses_sha256"]
    )
    archived_source = read_provenance_source(
        ROOT, provenance["native_cpp_source_archive"]
    )
    assert (
        hashlib.sha256(archived_source).hexdigest()
        == provenance["native_cpp_source_sha256"]
    )
    assert provenance["reproduction_witness_sha256"] == provenance["witnesses_sha256"]
    assert provenance["eigen_source_archive_sha256"] == SOURCE["archive_sha256"]
    assert provenance["eigen_version_header_sha256"] == SOURCE["version_header_sha256"]
    archive_url = urlsplit(SOURCE["official_archive_url"])
    assert archive_url.scheme == "https"
    assert archive_url.netloc == "gitlab.com"
    assert archive_url.path == "/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz"
    cpp = paths["reproduction_native_cpp_source_sha256"].read_text(encoding="utf-8")
    runner = paths["reproduction_capture_runner_sha256"].read_text(encoding="utf-8")
    assert "#include <Eigen/Geometry>" in cpp
    assert (
        "#include <quatops" not in cpp.lower()
        and '#include "quatops' not in cpp.lower()
    )
    assert "import quatops" not in runner and "from quatops" not in runner


def test_eigen_wxyz_constructor_xyzw_storage_and_native_actions():
    profiles = {item["dtype"]: item for item in WITNESSES["observations"]["profiles"]}
    for dtype, profile in profiles.items():
        assert profile["scalar_bytes"] == (4 if dtype == "float32" else 8)
        rows = {item["case_id"]: item for item in profile["orientations"]}
        assert set(rows) == set(NATIVE_RETAINED_CASE_IDS)
        for case_id, row in rows.items():
            canonical = CANONICAL[case_id]
            expected_matrix = matrix(canonical)
            check(
                row["quaternion"]["wxyz"],
                canonical,
                dtype,
                f"Eigen WXYZ accessors {case_id}/{dtype}",
            )
            if case_id == "mixed_a":
                check(
                    row["quaternion"]["coeffs_xyzw"],
                    coeffs_xyzw(canonical),
                    dtype,
                    f"Eigen coeffs() order {case_id}/{dtype}",
                )
            else:
                assert "coeffs_xyzw" not in row["quaternion"]
            check(
                row["native_quaternion_basis_actions_xyz"],
                [hamilton.rotate(canonical, p) for p in BASIS],
                dtype,
                f"Eigen quaternion vector action {case_id}/{dtype}",
            )
            if case_id == "mixed_a":
                check(
                    row["rotation_matrix"]["rows"],
                    expected_matrix,
                    dtype,
                    f"Eigen selected matrix rows/{dtype}",
                )
                check(
                    row["rotation_matrix"]["column_major_storage"],
                    expected_matrix.flatten(order="F"),
                    dtype,
                    f"Eigen selected matrix storage/{dtype}",
                )
            if case_id in {"mixed_a", "nonprincipal_halfturn"}:
                check(
                    matrix(row["matrix_to_quaternion"]["wxyz"]),
                    expected_matrix,
                    dtype,
                    f"Eigen matrix-to-quaternion {case_id}/{dtype}",
                )

        check(
            rows["antipode_mixed_a"]["native_quaternion_basis_actions_xyz"],
            rows["mixed_a"]["native_quaternion_basis_actions_xyz"],
            dtype,
            f"Eigen q versus -q/{dtype}",
        )


def test_eigen_angleaxis_reverse_conversion_and_noncommuting_products():
    ctor = INPUTS["native_constructed_orientation"]
    expected_q = np.asarray(
        hamilton.from_rotation_vector(
            np.asarray(ctor["axis_xyz"]) * ctor["angle_radians"]
        )
    )
    expected_matrix = matrix(expected_q)
    profiles = {item["dtype"]: item for item in WITNESSES["observations"]["profiles"]}
    for dtype, profile in profiles.items():
        native = profile["native_created"][0]
        check(
            native["quaternion"]["wxyz"],
            expected_q,
            dtype,
            f"Eigen AngleAxis coefficients/{dtype}",
        )
        check(
            native["native_quaternion_basis_actions_xyz"],
            (expected_matrix @ BASIS.T).T,
            dtype,
            f"Eigen AngleAxis action/{dtype}",
        )
        check(
            matrix(native["matrix_to_quaternion"]["wxyz"]),
            expected_matrix,
            dtype,
            f"Eigen AngleAxis matrix reverse/{dtype}",
        )

        products = {item["case_id"]: item for item in profile["composition"]}
        for case_id, lhs_id, rhs_id in (
            ("mixed_a_times_mixed_b", "mixed_a", "mixed_b"),
            ("mixed_b_times_mixed_a", "mixed_b", "mixed_a"),
        ):
            row = products[case_id]
            expected_product = np.asarray(
                hamilton.multiply(CANONICAL[lhs_id], CANONICAL[rhs_id])
            )
            check(
                row["quaternion"]["wxyz"],
                expected_product,
                dtype,
                f"Eigen quaternion product {case_id}/{dtype}",
            )
            check(
                row["native_quaternion_basis_actions_xyz"],
                [hamilton.rotate(expected_product, p) for p in BASIS],
                dtype,
                f"Eigen product vector action {case_id}/{dtype}",
            )

        forward = np.asarray(
            products["mixed_a_times_mixed_b"]["native_quaternion_basis_actions_xyz"]
        )
        reverse = np.asarray(
            products["mixed_b_times_mixed_a"]["native_quaternion_basis_actions_xyz"]
        )
        assert np.max(np.abs(forward - reverse)) > 0.1
