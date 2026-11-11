"""End-to-end QuatOps local-ENU to Cesium WGS84 ECEF converter proof."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.converters.cesium.inputs import build_inputs, serialize_inputs
from quatops.experimental.semantics import (
    Ellipsoid,
    GeodeticOrigin,
    bind_cesium_enu_frame,
)
from quatops.python import hamilton
from tests.support.evidence_oracles import ecef_origin
from tests.support.evidence_oracles import matrix_from_wxyz as rotation_matrix
from tests.support.evidence_oracles import native_values as values
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/transports/cesium/1.124.0"
CONTRACT_PATH = ROOT / "experiments/converters/cesium/1.124.0/semantic_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
INPUTS_BYTES = serialize_inputs(build_inputs(ROOT))
INPUTS = json.loads(INPUTS_BYTES)
CAPTURE = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
OBSERVATIONS = {row["case_id"]: row for row in CAPTURE["observations"]}
ATOL = 2e-11


def enu_to_ecef(origin_xyz):
    x, y, z = origin_xyz
    semi_major = 6_378_137.0
    flattening = 1.0 / 298.257223563
    semi_minor = semi_major * (1.0 - flattening)
    up = np.array([x / semi_major**2, y / semi_major**2, z / semi_minor**2])
    up /= np.linalg.norm(up)
    east = np.array([-y, x, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.column_stack((east, north, up))


def check(actual, expected, message, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol, err_msg=message)


def test_cesium_capture_provenance_and_explicit_frame_contract():
    assert MANIFEST["review_state"] == (
        "native capture retained; QuatOps replay covered by "
        "tests/experiments/converters/cesium/test_cesium_converter.py"
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
    assert provenance["input_serialization_note"].startswith(
        "JSON formatting changed only"
    )
    assert (
        provenance["captured_converter_input_sha256"]
        == provenance["converter_input_sha256"]
    )
    hash_paths = {
        "canonical_inputs_sha256": ROOT
        / "experiments/interoperability/v1/canonical_inputs.json",
        "witnesses_sha256": FIXTURE / "witnesses.json",
    }
    for field, path in hash_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[field]
    assert (
        hashlib.sha256(INPUTS_BYTES).hexdigest() == provenance["converter_input_sha256"]
    )

    reproduction = MANIFEST["reproduction"]
    reproduction_paths = {
        "geographic_inputs_sha256": ROOT
        / "experiments/oracles/shared/geographic_inputs_v1.json",
        "input_generator_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/create_inputs.py",
        "input_builder_sha256": ROOT / "experiments/converters/cesium/inputs.py",
        "capture_script_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/capture_native.mjs",
        "distribution_sha256": ROOT
        / "experiments/converters/cesium/1.124.0/distribution.json",
        "distribution_installer_sha256": ROOT
        / "experiments/converters/cesium/distribution.py",
        "numeric_helper_sha256": ROOT
        / "experiments/converters/shared/numeric_evidence.mjs",
        "retention_formatter_sha256": ROOT / "scripts/evidence_format.py",
    }
    for field, path in reproduction_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reproduction[field]
    assert (
        MANIFEST["bundled_reproduction"]["native_script_sha256"]
        == reproduction["capture_script_sha256"]
    )
    assert (
        MANIFEST["bundled_reproduction"]["reproduction_runner_sha256"]
        == hashlib.sha256(
            (ROOT / "experiments/run_converter_capture.py").read_bytes()
        ).hexdigest()
    )
    for item in ("lock", "manifest"):
        assert (
            hashlib.sha256(
                read_provenance_source(
                    ROOT, provenance[f"native_package_{item}_archive"]
                )
            ).hexdigest()
            == provenance[f"native_package_{item}_sha256"]
        )
    assert (
        hashlib.sha256(
            read_provenance_source(ROOT, reproduction["input_generator_archive"])
        ).hexdigest()
        == MANIFEST["provenance"]["captured_input_generator_sha256"]
    )


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
        expected_local_images = (local_rotation @ probes.T).T
        if case_id == "mixed_a@geo-03":
            check(
                values(output["local_basis_and_probe_images"]),
                expected_local_images,
                f"Cesium local vector action for {case_id}",
            )
        else:
            assert "local_basis_and_probe_images" not in output

        origin = ecef_origin(
            item["longitude_degrees"],
            item["latitude_degrees"],
            item["height_m"],
        )
        enu_rotation = enu_to_ecef(origin)
        world_rotation = enu_rotation @ local_rotation
        expected_world_directions = (world_rotation @ probes.T).T
        actual_world_directions = values(output["world_directions_xyz"])
        check(
            actual_world_directions,
            expected_world_directions[: len(actual_world_directions)],
            f"Cesium ECEF direction action for {case_id}",
        )
        expected_world_probe = origin + enu_rotation @ (
            local_rotation @ probes[-1] + local_translation
        )
        check(
            values(output["point_roundtrip_local_xyz_m"]),
            [probes[-1]],
            f"Cesium ECEF/local inverse point for {case_id}",
            atol=2e-9,
        )
        check(
            values(output["ecef_origin_xyz_m"]),
            origin,
            f"Cesium ENU origin for {case_id}",
            atol=2e-9,
        )
        check(
            values(output["world_points_xyz_m"]),
            [expected_world_probe],
            f"Cesium ECEF probe point for {case_id}",
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
            world_rotation,
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


def test_enu_semantic_binding_matches_cesium_native_world_basis_at_site():
    case_id = "mixed_a@geo-03"
    item = next(row for row in INPUTS["cases"] if row["case_id"] == case_id)
    output = OBSERVATIONS[case_id]["outputs"]
    geodetic_origin = GeodeticOrigin(
        item["longitude_degrees"], item["latitude_degrees"], item["height_m"]
    )
    binding = bind_cesium_enu_frame(
        name=f"enu@{case_id}",
        origin=geodetic_origin,
        ellipsoid=Ellipsoid("WGS84", 6_378_137.0, 6_356_752.314245179),
    )

    # Recover Cesium's local ENU-to-ECEF basis from its captured world basis
    # action and the independently recorded local orientation input.
    native_world_basis = np.asarray(
        values(output["world_directions_xyz"])[:3], dtype=float
    ).T
    local_rotation = rotation_matrix(item["canonical_wxyz"])
    native_enu_to_ecef = native_world_basis @ local_rotation.T
    check(
        binding.coordinate_frame.basis_to_reference.matrix,
        native_enu_to_ecef,
        "semantic ENU frame binding matches Cesium's native basis action",
    )
    assert binding.origin == geodetic_origin
    assert binding.ellipsoid is not None and binding.ellipsoid.name == "WGS84"
