"""Integrity checks for the auxiliary Wave 2 engine/transport evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WAVE = ROOT / "experiments/quatops_capture_wave2"

IMPORTED_RUNS = (
    (
        "opencv-4.13.0-first-pass",
        ROOT / "tests/golden/oracles/opencv/4.13.0",
        "OpenCV",
        "4.13.0",
        264,
    ),
    (
        "proj-geographic",
        ROOT / "tests/golden/oracles/proj/9.5.1",
        "PROJ via pyproj",
        "9.5.1",
        11,
    ),
    (
        "vtk-9.6.2-gltf",
        ROOT / "tests/golden/transports/gltf/2.0.0/vtk/9.6.2",
        "VTK vtkGLTFImporter",
        "9.6.2",
        112,
    ),
    (
        "trimesh-4.11.1-static",
        ROOT / "tests/golden/transports/gltf/2.0.0/trimesh/4.11.1",
        "Trimesh glTF loader",
        "4.11.1",
        12,
    ),
)

LOCAL_RUNS = (
    (ROOT / "tests/golden/oracles/cesium/1.124.0", "cesium", "1.124", 171),
    (
        ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/wave2",
        "three_and_gltfloader",
        "187",
        649,
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observations(folder: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_imported_native_runs_are_preserved_in_versioned_fixtures() -> None:
    for run_id, fixture, product, version, count in IMPORTED_RUNS:
        source = WAVE / "runs" / run_id
        for item in source.iterdir():
            assert item.is_file()
            assert (fixture / item.name).read_bytes() == item.read_bytes()

        manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["evidence_kind"] == "native_runtime"
        assert manifest["product"]["name"] == product
        assert manifest["product"]["version"] == version
        assert manifest["execution"]["was_run"] is True
        assert manifest["coverage"]["native_observation_count"] == count
        assert manifest["coverage"]["quatops_replay_count"] == 0
        records = _observations(fixture)
        assert sum(record["status"] == "ok" for record in records) == count
        for artifact in manifest["artifacts"]:
            path = artifact["path"]
            if path.startswith("/"):
                continue  # Original packet input paths remain provenance strings.
            assert _sha256(fixture / path) == artifact["sha256"]


def test_local_cesium_and_threejs_run_manifests_are_pinned() -> None:
    for folder, product, version, count in LOCAL_RUNS:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["evidence_kind"] == "native_runtime"
        assert manifest["product"]["name"] == product
        assert manifest["product"]["version"] == version
        assert manifest["execution"]["runtime"] == "v24.21.0"
        assert manifest["execution"]["platform"] == "win32"
        assert manifest["execution"]["was_run"] is True
        assert manifest["coverage"]["native_observation_count"] == count
        assert manifest["coverage"]["quatops_replay_count"] == 0
        for artifact in manifest["artifacts"]:
            path = artifact["path"]
            if Path(path).is_absolute():
                continue  # Input corpus assets are shared by the Wave 2 package.
            assert _sha256(folder / path) == artifact["sha256"]


def test_scipy_comparison_report_retains_quaternion_and_geographic_results() -> None:
    report = json.loads(
        (WAVE / "verification/local_native_comparison.json").read_text(encoding="utf-8")
    )
    assert report["evidence_kind"] == "derived_reference"
    assert report["references"]["scipy"] == "1.17.0"
    assert report["quatops_executed"] is False
    cesium = report["cesium"]["max_abs_errors"]
    assert cesium["local_hpr_matrix"] < 1e-12
    assert cesium["slerp"] < 1e-12
    assert cesium["hamilton_product"] == 0.0
    assert cesium["geographic_origin"] < 1e-8
    assert cesium["geographic_up_vs_radii_normal"] < 1e-12
    assert 1e-4 < cesium["geographic_enu_vs_geodetic_latitude"] < 2e-4
    three = report["threejs_and_gltf"]
    assert three["three_euler_case_count"] == 534
    assert three["three_euler_matrix_and_orientation_max_abs_error"] < 1e-12
    assert three["gltf_loader_sample_count"] == 112
    assert three["gltf_loader_max_abs_matrix_error"] < 1e-6
    assert three["global_cubic_sign_max_abs_matrix_error"] < 1e-12
    assert three["single_key_sign_change_max_abs_matrix_difference"] > 1e-3


def test_khronos_reports_cover_twelve_assets_without_issues() -> None:
    reports = sorted(
        (ROOT / "tests/golden/transports/gltf/2.0.0/khronos-validator/2.0.0-dev.3.10/wave2").glob("*.json")
    )
    assert len(reports) == 12
    for path in reports:
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["validator_version"] == "2.0.0-dev.3.10"
        issues = result["report"]["issues"]
        assert [issues[key] for key in ("numErrors", "numWarnings", "numInfos", "numHints")] == [0, 0, 0, 0]
