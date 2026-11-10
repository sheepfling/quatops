"""Strict PROJ native capture schema and reproduction provenance checks."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.proj_capture import ProjObservation
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source


def test_proj_capture_observations_use_a_target_specific_strict_schema() -> None:
    observations_path = (
        REPOSITORY_ROOT / "tests/golden/oracles/proj/9.5.1/observations.jsonl"
    )
    lines = observations_path.read_text(encoding="utf-8").splitlines()
    observations = [
        ProjObservation.model_validate_json(line, strict=True) for line in lines
    ]

    assert len(observations) == 11
    assert len(observations[0].native_outputs.origin_ecef_m.values) == 3
    assert len(observations[0].native_outputs.enu_basis_endpoints_ecef_m.values) == 3

    malformed = json.loads(lines[0])
    malformed["native_outputs"]["origin_ecef_m"]["values"].append(1.0)
    with pytest.raises(ValidationError):
        ProjObservation.model_validate_json(json.dumps(malformed), strict=True)

    malformed = json.loads(lines[0])
    malformed["native_outputs"]["enu_basis_endpoints_ecef_m"]["values"][0].pop()
    with pytest.raises(ValidationError):
        ProjObservation.model_validate_json(json.dumps(malformed), strict=True)


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
    assert (
        hashlib.sha256((root / reproduction["input_path"]).read_bytes()).hexdigest()
        == reproduction["input_sha256"]
    )
    archived = read_provenance_source(
        root, "experiments/provenance/reproduction-v1.zip#proj-9.5.1/capture_proj.py"
    )
    current_exporter = root / reproduction["exporter_path"]
    requirements = root / "experiments/oracles/proj/9.5.1/requirements.txt"

    assert (
        hashlib.sha256(archived).hexdigest()
        == reproduction["previous_checkout_exporter_sha256"]
    )
    assert (
        hashlib.sha256(current_exporter.read_bytes()).hexdigest()
        == reproduction["exporter_sha256"]
    )
    assert (
        hashlib.sha256(requirements.read_bytes()).hexdigest()
        == reproduction["requirements_sha256"]
    )
    archived_helper = read_provenance_source(
        root, reproduction["capture_helper_archive"]
    )
    current_helper = root / reproduction["capture_helper_path"]
    assert (
        hashlib.sha256(archived_helper).hexdigest()
        == reproduction["previous_checkout_capture_helper_sha256"]
    )
    assert (
        hashlib.sha256(current_helper.read_bytes()).hexdigest()
        == reproduction["capture_helper_sha256"]
    )
    assert baseline["audit_time_reproduction_exporter_sha256"] == (
        "9295111b0a0d2ddf36c207ee50c7078a28202b60911a5589301299622a665a9a"
    )
