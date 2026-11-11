"""Strict PROJ native capture schema and reproduction provenance checks."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.proj_capture import ProjCapture, load_proj_capture
from scripts.evidence_schema.shared_inputs import GeographicInputCorpus
from tests.support.evidence_oracles import ecef_origin
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

FIXTURE = REPOSITORY_ROOT / "tests/golden/oracles/proj/9.5.1"


def test_all_proj_version_manifests_validate_their_witnesses() -> None:
    evidence = FIXTURE.parent
    versions = sorted(
        path.parent
        for path in evidence.glob("*/manifest.json")
        if path.parent.name != "shared"
    )
    assert versions
    for version in versions:
        assert load_proj_capture(REPOSITORY_ROOT, version).observations, version


def witness_reference() -> tuple[str, str]:
    manifest = read_evidence_document(FIXTURE / "manifest.json")
    artifact = manifest["artifacts"][0]
    return artifact["path"], artifact["sha256"]


def test_proj_capture_observations_use_a_target_specific_strict_schema() -> None:
    reference, _ = witness_reference()
    capture = load_proj_capture(REPOSITORY_ROOT, FIXTURE)
    document = (FIXTURE / reference).read_text(encoding="utf-8")

    assert len(capture.observations) == 11
    assert capture.schema_version == "2.0"
    assert capture.dtype == "binary64"

    lines = document.splitlines()

    def raw_capture():
        return {
            **json.loads(lines[0]),
            "observations": [json.loads(x) for x in lines[1:]],
        }

    malformed = raw_capture()
    malformed["observations"][0]["origin_ecef_m"].append(1.0)
    with pytest.raises(ValidationError):
        ProjCapture.model_validate_json(json.dumps(malformed), strict=True)

    malformed = raw_capture()
    malformed["observations"][0]["enu_endpoints_ecef_m"][0].pop()
    with pytest.raises(ValidationError):
        ProjCapture.model_validate_json(json.dumps(malformed), strict=True)

    malformed = raw_capture()
    malformed["observations"][1]["case_id"] = malformed["observations"][0]["case_id"]
    with pytest.raises(ValidationError, match="duplicate case IDs"):
        ProjCapture.model_validate_json(json.dumps(malformed), strict=True)

    malformed = raw_capture()
    malformed["dtype"] = "binary32"
    with pytest.raises(ValidationError):
        ProjCapture.model_validate_json(json.dumps(malformed), strict=True)

    header = json.loads(lines[0])
    header["unexpected"] = True
    with pytest.raises(ValidationError):
        ProjCapture.from_jsonl("\n".join([json.dumps(header), *lines[1:]]))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_proj_capture_rejects_nonfinite_native_values(nonfinite: float) -> None:
    reference, _ = witness_reference()
    path = FIXTURE / reference
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["origin_ecef_m"][0] = nonfinite
    lines[1] = json.dumps(row)
    with pytest.raises(ValidationError):
        ProjCapture.from_jsonl("\n".join(lines))


def test_proj_capture_preserves_integer_form_negative_zero() -> None:
    reference, _ = witness_reference()
    lines = (FIXTURE / reference).read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["origin_ecef_m"][1] = -0.0
    lines[1] = json.dumps(row).replace("-0.0", "-0", 1)
    capture = ProjCapture.from_jsonl("\n".join(lines))
    assert math.copysign(1, capture.observations[0].origin_ecef_m[1]) == -1


def test_proj_native_outputs_match_independent_wgs84_enu_transform() -> None:
    root = REPOSITORY_ROOT
    capture = load_proj_capture(root, FIXTURE)
    inputs = GeographicInputCorpus.model_validate_json(
        (root / "experiments/oracles/shared/geographic_inputs_v1.json").read_bytes(),
        strict=True,
    )
    locations = {row.case_id: row for row in inputs.locations}
    assert {row.case_id for row in capture.observations} == set(locations)
    probe = np.asarray(inputs.local_point_xyz_m, dtype=float)
    for row in capture.observations:
        location = locations[row.case_id]
        lon = math.radians(location.longitude_degrees)
        lat = math.radians(location.latitude_degrees)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        origin = ecef_origin(
            location.longitude_degrees, location.latitude_degrees, location.height_m
        )
        basis = np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ]
        )
        np.testing.assert_allclose(row.origin_ecef_m, origin, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(row.enu_endpoints_ecef_m) - row.origin_ecef_m,
            basis.T,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            row.probe_ecef_m, origin + basis @ probe, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(row.readback_enu_m, probe, rtol=0, atol=1e-8)


@pytest.fixture
def isolated_proj_evidence(tmp_path):
    root = tmp_path / "repository"
    version = root / "tests/golden/oracles/proj/9.5.1"
    version.mkdir(parents=True)
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest["artifacts"][0]["path"]
    target = (version / reference).resolve()
    target.parent.mkdir()
    target.write_bytes((FIXTURE / reference).read_bytes())
    for key in ("input_path", "exporter_path"):
        path = manifest["reproduction"][key]
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / path).read_bytes())
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, version, manifest


@pytest.mark.parametrize(
    ("reference", "error", "message"),
    [
        ("../missing/{name}", FileNotFoundError, None),
        ("../../outside/{name}", ValueError, "escapes"),
        ("absolute", ValueError, "relative"),
        ("../shared/misnamed.jsonl", ValueError, "filename"),
    ],
)
def test_shared_proj_witness_rejects_bad_paths(
    isolated_proj_evidence, reference, error, message
) -> None:
    root, version, manifest = isolated_proj_evidence
    target = (version / manifest["artifacts"][0]["path"]).resolve()
    manifest["artifacts"][0]["path"] = (
        target.as_posix()
        if reference == "absolute"
        else reference.format(name=target.name)
    )
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(error, match=message):
        load_proj_capture(root, version)


def test_shared_proj_witness_rejects_corrupt_data(isolated_proj_evidence) -> None:
    root, version, manifest = isolated_proj_evidence
    assert len(load_proj_capture(root, version).observations) == 11
    target = (version / manifest["artifacts"][0]["path"]).resolve()
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_proj_capture(root, version)


def test_proj_shared_witness_cannot_imply_an_unrun_version(
    isolated_proj_evidence,
) -> None:
    root, version, manifest = isolated_proj_evidence
    other_version = version.with_name("9.6.0")
    version.rename(other_version)
    with pytest.raises(ValueError, match="version"):
        load_proj_capture(root, other_version)
    manifest["product"]["version"] = "9.6.0"
    (other_version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_proj_capture(root, other_version)


@pytest.mark.parametrize(
    ("field", "value"),
    [("was_run", False), ("exit_code", 1), ("was_run", 1), ("exit_code", False)],
)
def test_proj_manifest_requires_successful_native_runs(
    isolated_proj_evidence, field, value
) -> None:
    root, version, manifest = isolated_proj_evidence
    manifest["compact_reproduction"][field] = value
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_proj_capture(root, version)


def test_proj_manifest_rejects_missing_native_run(isolated_proj_evidence) -> None:
    root, version, manifest = isolated_proj_evidence
    del manifest["compact_reproduction"]
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_proj_capture(root, version)


def test_proj_manifest_rejects_mismatched_native_run_inputs(
    isolated_proj_evidence,
) -> None:
    root, version, manifest = isolated_proj_evidence
    manifest["compact_reproduction"]["input_sha256"] = "0" * 64
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="digests"):
        load_proj_capture(root, version)


def test_proj_manifest_rejects_corrupt_shared_input_corpus(
    isolated_proj_evidence,
) -> None:
    root, version, manifest = isolated_proj_evidence
    inputs = root / manifest["reproduction"]["input_path"]
    inputs.write_bytes(inputs.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="input corpus digest"):
        load_proj_capture(root, version)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pyproj_version", "0.0", "pyproj version"),
        ("capture_operations_completed", False, "must complete"),
        ("compared_witness_exact", False, "exact witness"),
        ("observations_sha256", "0" * 64, "digests"),
        ("exporter_sha256", "0" * 64, "digests"),
        ("native_binary_sha256", "invalid", "native_binary_sha256"),
        ("observation_count", 1, "corpus or run"),
    ],
)
def test_proj_manifest_rejects_inconsistent_run_metadata(
    isolated_proj_evidence, field, value, message
) -> None:
    root, version, manifest = isolated_proj_evidence
    manifest["compact_reproduction"][field] = value
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_proj_capture(root, version)


def test_proj_manifest_rejects_wrong_pyproj_dependency(isolated_proj_evidence) -> None:
    root, version, manifest = isolated_proj_evidence
    manifest["product"]["dependencies"][0]["version"] = "0.0"
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pyproj version"):
        load_proj_capture(root, version)


def test_proj_reproduction_hashes_preserve_prior_and_current_source_identity() -> None:
    root = REPOSITORY_ROOT
    fixture = root / "tests/golden/oracles/proj/9.5.1"
    manifest = read_evidence_document(fixture / "manifest.json")
    baseline = read_evidence_document(fixture / "audit.json")["native_baseline"]
    assert (
        hashlib.sha256((fixture / baseline["observations"]).read_bytes()).hexdigest()
        == baseline["observations_sha256"]
    )
    reproduction = manifest["reproduction"]
    paths = {
        "input_sha256": root / reproduction["input_path"],
        "exporter_sha256": root / reproduction["exporter_path"],
        "digest_helper_sha256": root / "scripts/evidence_schema/utilities.py",
        "lockfile_sha256": root / "experiments/converters/python/uv.lock",
    }
    for field, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reproduction[field], (
            field
        )
    archives = {
        "previous_checkout_exporter_sha256": "experiments/provenance/reproduction-v1.zip#proj-9.5.1/capture_proj.py",
        "previous_checkout_capture_helper_sha256": reproduction[
            "capture_helper_archive"
        ],
    }
    for field, reference in archives.items():
        content = read_provenance_source(root, reference)
        assert hashlib.sha256(content).hexdigest() == reproduction[field], field
    assert (
        manifest["direct_reproduction"]["reproduction_exporter_sha256"]
        == reproduction["previous_envelope_exporter_sha256"]
    )
    assert manifest["direct_reproduction"]["observation_count"] == 11
    assert (
        manifest["unified_lock_replay"]["observations_sha256"]
        == manifest["direct_reproduction"]["observations_sha256"]
    )
    assert (
        manifest["unified_lock_replay"]["lockfile_sha256"]
        == reproduction["lockfile_sha256"]
    )
    compact = manifest["compact_reproduction"]
    assert compact["observations_sha256"] == baseline["observations_sha256"]
    assert compact["exporter_sha256"] == reproduction["exporter_sha256"]
    assert compact["native_binary_sha256"] == manifest["product"]["binary_sha256"]
    assert (
        baseline["previous_selected_observations_sha256"]
        == manifest["retention"]["previous_selected_observations_sha256"]
    )
    assert baseline["audit_time_reproduction_exporter_sha256"] == (
        "9295111b0a0d2ddf36c207ee50c7078a28202b60911a5589301299622a665a9a"
    )
