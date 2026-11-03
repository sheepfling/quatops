"""Replay the local Cesium/Three.js captures against separate references."""
from __future__ import annotations

import json
from pathlib import Path

from shared.capture_io import sha256
from tools.verify_local_native import CESIUM, PROJECT, THREE, VALIDATOR, verify_cesium, verify_three


def _manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def _check_run_hashes(folder: Path) -> None:
    manifest = _manifest(folder)
    assert sha256(Path(manifest["exporter"]["path"])) == manifest["exporter"]["sha256"]
    assert sha256(Path(manifest["corpus"]["path"])) == manifest["corpus"]["sha256"]
    for artifact in manifest["artifacts"]:
        path = folder / artifact["path"] if not Path(artifact["path"]).is_absolute() else Path(artifact["path"])
        assert sha256(path) == artifact["sha256"]
    assert manifest["execution"]["was_run"] is True
    assert manifest["coverage"]["quatops_replay_count"] == 0


def test_cesium_run_identity_and_raw_hashes() -> None:
    manifest = _manifest(CESIUM)
    assert manifest["product"]["name"] == "cesium"
    assert manifest["product"]["version"] == "1.124"
    assert manifest["execution"]["runtime"] == "v24.21.0"
    assert manifest["execution"]["platform"] == "win32"
    assert manifest["coverage"]["native_observation_count"] == 171
    _check_run_hashes(CESIUM)
    lock = PROJECT / "experiments/quatops_capture_wave2/js/package-lock.json"
    lock_sha = sha256(lock)
    assert {item["sha256"] for item in manifest["product"]["dependencies"] if item["name"] == "npm lock"} == {lock_sha}
    packages = json.loads(lock.read_text(encoding="utf-8"))["packages"]
    assert packages["node_modules/cesium"]["version"] == "1.124.0"
    for package, version in (
        ("@cesium/engine", "13.0.0"),
        ("@cesium/widgets", "10.0.0"),
        ("@zip.js/zip.js", "2.7.34"),
    ):
        matches = [
            entry["version"]
            for name, entry in packages.items()
            if name.endswith("/" + package)
        ]
        assert matches and set(matches) == {version}


def test_cesium_native_conversions_match_scipy_and_preserve_geographic_edge() -> None:
    result = verify_cesium()
    errors = result["max_abs_errors"]
    assert result["native_observation_count"] == 171
    assert result["counts"]["local_hpr_matrix"] == 89
    assert result["counts"]["slerp"] == 35
    assert result["counts"]["geographic_composition"] == 33
    for metric in (
        "local_hpr_matrix",
        "local_hpr_quaternion_matrix",
        "readback_hpr_matrix",
        "basis_action",
        "matrix_to_quaternion",
        "slerp",
        "geographic_composition",
        "geographic_quaternion_matrix",
        "geographic_enu_vs_surface_normal",
        "geographic_up_vs_radii_normal",
    ):
        assert errors[metric] < 2e-12, (metric, errors[metric])
    assert errors["hamilton_product"] == 0.0
    assert errors["geographic_origin"] < 1e-8
    assert errors["geographic_point"] < 1e-12
    assert errors["geographic_direction"] < 1e-12
    assert errors["geographic_point_roundtrip"] < 2e-9
    # Preserve the observed difference from a frame derived only from the
    # input geodetic latitude, largest at the 400 km altitude case.
    assert 1e-4 < errors["geographic_enu_vs_geodetic_latitude"] < 2e-4


def test_threejs_gltf_run_identity_and_raw_hashes() -> None:
    manifest = _manifest(THREE)
    assert manifest["product"]["name"] == "three_and_gltfloader"
    assert manifest["product"]["version"] == "187"
    assert manifest["execution"]["runtime"] == "v24.21.0"
    assert manifest["execution"]["platform"] == "win32"
    assert manifest["coverage"]["native_observation_count"] == 649
    _check_run_hashes(THREE)


def test_threejs_euler_and_gltf_loader_outputs_match_independent_references() -> None:
    result = verify_three()
    assert result["three_euler_case_count"] == 534
    assert result["three_euler_matrix_and_orientation_max_abs_error"] < 2e-12
    assert result["gltf_loader_sample_count"] == 112
    assert result["gltf_loader_max_abs_matrix_error"] < 1e-6
    assert result["global_cubic_sign_max_abs_matrix_error"] < 1e-12
    assert result["single_key_sign_change_max_abs_matrix_difference"] > 1e-3


def test_khronos_validator_accepts_all_wave_two_assets() -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(VALIDATOR.glob("*.json"))]
    assert len(reports) == 12
    for result in reports:
        issues = result["report"]["issues"]
        assert result["validator_version"] == "2.0.0-dev.3.10"
        assert issues["numErrors"] == 0
        assert issues["numWarnings"] == 0
        assert issues["numInfos"] == 0
        assert issues["numHints"] == 0
