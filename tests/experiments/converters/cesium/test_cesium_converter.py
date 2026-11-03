"""End-to-end QuatOps local-ENU to Cesium WGS84 ECEF converter proof."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from quatops.python import hamilton
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/transports/cesium/1.124.0"
CONTRACT = json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS = json.loads((FIXTURE / "inputs.json").read_text(encoding="utf-8"))
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
OBSERVATIONS = {row["case_id"]: row for row in CAPTURE["observations"]}
ATOL = 2e-11


def values(item):
    if isinstance(item, dict) and "values" in item and "dtype" in item:
        return item["values"]
    if isinstance(item, list):
        return [values(child) for child in item]
    if isinstance(item, dict):
        return {key: values(child) for key, child in item.items()}
    return item


def rotation_matrix(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def check(actual, expected, message, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol, err_msg=message)


def test_cesium_capture_provenance_and_explicit_frame_contract():
    assert (
        MANIFEST["review_state"]
        == "native capture retained; QuatOps replay covered by tests/experiments/converters/cesium/test_cesium_converter.py"
    )
    assert MANIFEST["target"]["name"] == "CesiumJS"
    assert MANIFEST["target"]["package_version"] == "1.124.0"
    assert MANIFEST["target"]["api_version"] == "1.124"
    assert MANIFEST["execution"]["was_run"] is True
    assert MANIFEST["execution"]["node"] == "v24.21.0"
    assert MANIFEST["selection"]["retained_observations"] == 5
    assert MANIFEST["review_state"].startswith("native capture retained")
    assert CONTRACT["forward"].startswith("R_ECEF_from_model = R_ECEF_from_ENU")
    assert INPUTS["frame_contract"]["map_canonical_to_cesium_enu"] == ["x", "y", "z"]
    assert INPUTS["frame_contract"]["canonical_local"]["positive_z"] == "up"
    assert INPUTS["frame_contract"]["cesium_local"]["positive_z"] == "ellipsoid normal"
    provenance = MANIFEST["provenance"]
    hash_paths = {
        "converter_input_sha256": FIXTURE / "inputs.json",
        "canonical_inputs_sha256": ROOT
        / "experiments/interoperability/v1/canonical_inputs.json",
        "witnesses_sha256": FIXTURE / "witnesses.json",
    }
    for field, path in hash_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field]

    reproduction = MANIFEST["reproduction"]
    reproduction_paths = {
        "geographic_inputs_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/geographic_inputs.json",
        "input_generator_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/create_inputs.py",
        "capture_script_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/capture_native.mjs",
        "package_lock_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/package-lock.json",
        "numeric_helper_sha256": ROOT
        / "experiments/converters/shared/numeric_evidence.mjs",
        "retention_script_sha256": ROOT
        / "scripts/retain_cesium_converter_witnesses.py",
        "retention_helper_sha256": ROOT / "scripts/evidence_json.py",
    }
    for field, path in reproduction_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reproduction[field]


def test_quatops_generated_local_quaternions_match_cesium_native_actions_and_roundtrips():
    probes = np.asarray(INPUTS["probe_vectors_xyz"], dtype=float)
    by_id = {row["case_id"]: row for row in INPUTS["cases"]}
    for case_id, item in by_id.items():
        observation = OBSERVATIONS[case_id]
        output = observation["outputs"]
        q = np.asarray(item["canonical_wxyz"], dtype=float)
        q_xyzw = np.asarray(item["quatops_cesium_local_xyzw"], dtype=float)
        q_expected_xyzw = q[[1, 2, 3, 0]]
        check(q_xyzw, q_expected_xyzw, f"QuatOps coefficient map for {case_id}")

        local_rotation = rotation_matrix(q)
        local_translation = np.asarray(item["local_translation_xyz_m"], dtype=float)
        check(
            values(output["local_rotation_matrix_rows"]),
            local_rotation,
            f"Cesium local matrix for {case_id}",
        )
        expected_local_images = (local_rotation @ probes.T).T
        check(
            values(output["local_basis_and_probe_images"]),
            expected_local_images,
            f"Cesium local vector action for {case_id}",
        )

        enu_to_ecef = np.asarray(values(output["enu_to_ecef_matrix_rows"]), dtype=float)
        local_to_model = np.eye(4)
        local_to_model[:3, :3] = local_rotation
        local_to_model[:3, 3] = local_translation
        expected_world = enu_to_ecef @ local_to_model
        model_to_ecef = np.asarray(
            values(output["model_to_ecef_matrix_rows"]), dtype=float
        )
        check(
            model_to_ecef,
            expected_world,
            f"Cesium ENU-to-ECEF composition for {case_id}",
        )

        local_points = (local_rotation @ probes.T).T + local_translation
        expected_world_points = (enu_to_ecef[:3, :3] @ local_points.T).T + enu_to_ecef[
            :3, 3
        ]
        check(
            values(output["world_points_xyz_m"]),
            expected_world_points,
            f"Cesium ECEF point action for {case_id}",
            atol=2e-9,
        )
        expected_world_directions = (enu_to_ecef[:3, :3] @ local_rotation @ probes.T).T
        check(
            values(output["world_directions_xyz"]),
            expected_world_directions,
            f"Cesium ECEF direction action for {case_id}",
        )
        check(
            values(output["point_roundtrip_local_xyz_m"]),
            probes,
            f"Cesium ECEF/local inverse point for {case_id}",
            atol=2e-9,
        )
        check(
            values(output["ecef_origin_xyz_m"]),
            enu_to_ecef[:3, 3],
            f"Cesium ENU origin for {case_id}",
            atol=2e-9,
        )

        world_q = np.asarray(
            values(output["model_to_ecef_quaternion_xyzw"]), dtype=float
        )
        recovered_local_q = np.asarray(
            values(output["local_quaternion_recovered_from_ecef_xyzw"]), dtype=float
        )
        check(
            Rotation.from_quat(world_q).as_matrix(),
            expected_world[:3, :3],
            f"Cesium ECEF quaternion for {case_id}",
        )
        check(
            Rotation.from_quat(recovered_local_q).as_matrix(),
            local_rotation,
            f"Cesium recovered local quaternion for {case_id}",
        )


def test_cesium_composition_antipode_and_geographic_branch_witnesses_are_discriminating():
    for case_id in OBSERVATIONS:
        assert "geo-" in case_id
    assert "antipode_mixed_a@geo-09" in OBSERVATIONS
    assert "mixed_a_times_mixed_b@geo-03" in OBSERVATIONS
    antipode = {row["case_id"]: row for row in INPUTS["cases"]}[
        "antipode_mixed_a@geo-09"
    ]
    positive = {row["case_id"]: row for row in INPUTS["cases"]}["mixed_a@geo-03"]
    check(
        rotation_matrix(antipode["canonical_wxyz"]),
        rotation_matrix(positive["canonical_wxyz"]),
        "q and -q encode the same physical orientation",
    )

    a = {row["case_id"]: row for row in INPUTS["cases"]}["mixed_a@geo-03"][
        "canonical_wxyz"
    ]
    b = {row["case_id"]: row for row in INPUTS["cases"]}["mixed_b@geo-04"][
        "canonical_wxyz"
    ]
    expected_product = np.asarray(hamilton.multiply(a, b))
    composed = {row["case_id"]: row for row in INPUTS["cases"]}[
        "mixed_a_times_mixed_b@geo-03"
    ]
    check(
        composed["canonical_wxyz"],
        expected_product,
        "QuatOps noncommuting product is what was serialized",
    )
    wrong_product = np.asarray(hamilton.multiply(b, a))
    assert (
        np.max(
            np.abs(rotation_matrix(wrong_product) - rotation_matrix(expected_product))
        )
        > 0.1
    )
