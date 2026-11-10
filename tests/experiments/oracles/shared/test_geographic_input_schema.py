"""Strict schema checks for the shared geographic input corpus."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.proj_capture import ProjObservation
from scripts.evidence_schema.shared_inputs import GeographicInputCorpus
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

CORPUS = REPOSITORY_ROOT / "experiments/oracles/shared/geographic_inputs_v1.json"


def test_geographic_corpus_is_semantically_typed() -> None:
    corpus = GeographicInputCorpus.model_validate_json(CORPUS.read_bytes(), strict=True)
    assert corpus.ellipsoid == "WGS84"
    assert len(corpus.locations) == 11
    assert corpus.locations[3].case_id == "geo-03"


@pytest.mark.parametrize("invalid", ["0", True])
def test_geographic_coordinates_reject_numeric_strings_and_booleans(
    invalid: object,
) -> None:
    corpus = read_evidence_document(CORPUS)
    corpus["locations"][0]["longitude_degrees"] = invalid

    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate(corpus, strict=True)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_geographic_json_rejects_nonfinite_coordinate_tokens(token: str) -> None:
    corpus = read_evidence_document(CORPUS)
    corpus["locations"][0]["longitude_degrees"] = 246813579
    payload = json.dumps(corpus, separators=(",", ":"))
    payload = payload.replace(
        '"longitude_degrees":246813579', f'"longitude_degrees":{token}', 1
    )

    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate_json(payload, strict=True)


def test_geographic_json_accepts_integer_and_float_numbers_and_round_trips() -> None:
    corpus = read_evidence_document(CORPUS)
    corpus["locations"][0]["longitude_degrees"] = -122
    corpus["locations"][0]["latitude_degrees"] = 47.5
    corpus["locations"][0]["height_m"] = 12
    capture = GeographicInputCorpus.model_validate(corpus, strict=True)

    assert capture.locations[0].longitude_degrees == -122.0
    assert capture.locations[0].latitude_degrees == 47.5
    assert capture.locations[0].height_m == 12.0
    adapter = TypeAdapter(GeographicInputCorpus)
    restored = adapter.validate_json(adapter.dump_json(capture), strict=True)
    assert restored == capture


def test_geographic_input_rejects_integer_overflow_but_preserves_binary64_rounding():
    corpus = read_evidence_document(CORPUS)
    corpus["locations"][0]["longitude_degrees"] = 10**1000
    with pytest.raises(ValidationError, match="overflows the target binary64"):
        GeographicInputCorpus.model_validate(corpus, strict=True)

    corpus["locations"][0]["longitude_degrees"] = 9007199254740993
    parsed = GeographicInputCorpus.model_validate(corpus, strict=True)
    exact_integer = parsed.locations[0].longitude_degrees
    assert exact_integer == 9007199254740993
    assert float(exact_integer) == 9007199254740992.0


def test_geographic_schema_rejects_extra_fields_and_wrong_versions() -> None:
    corpus = read_evidence_document(CORPUS)
    corpus["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate(corpus, strict=True)

    corpus["schema_version"] = "1.0"
    corpus["unexpected"] = True
    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate(corpus, strict=True)


def test_proj_capture_observations_use_a_target_specific_strict_schema() -> None:
    observations_path = (
        REPOSITORY_ROOT / "tests/golden/oracles/proj/9.5.1/observations.jsonl"
    )
    lines = observations_path.read_text(encoding="utf-8").splitlines()
    observations = [
        ProjObservation.model_validate_json(line, strict=True) for line in lines
    ]

    assert len(observations) == 11
    assert observations[0].native_outputs.origin_ecef_m.shape == [3]
    assert observations[0].native_outputs.enu_basis_endpoints_ecef_m.shape == [3, 3]

    malformed = json.loads(lines[0])
    malformed["native_outputs"]["origin_ecef_m"]["shape"] = [4]
    with pytest.raises(ValidationError):
        ProjObservation.model_validate_json(json.dumps(malformed), strict=True)


def test_proj_reproduction_hashes_preserve_prior_and_current_source_identity() -> None:
    root = REPOSITORY_ROOT
    manifest = read_evidence_document(
        root / "tests/golden/oracles/proj/9.5.1/manifest.json"
    )
    reproduction = manifest["reproduction"]
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
    assert reproduction["previous_manifest_exporter_sha256"] == (
        "9295111b0a0d2ddf36c207ee50c7078a28202b60911a5589301299622a665a9a"
    )
