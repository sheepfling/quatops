"""Strict schema checks for the shared geographic input corpus."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.shared_inputs import GeographicInputCorpus
from tests.support.paths import REPOSITORY_ROOT

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


def test_geographic_schema_rejects_extra_fields_and_wrong_versions() -> None:
    corpus = read_evidence_document(CORPUS)
    corpus["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate(corpus, strict=True)

    corpus["schema_version"] = "1.0"
    corpus["unexpected"] = True
    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate(corpus, strict=True)


def test_proj_reproduction_hashes_preserve_prior_and_current_source_identity() -> None:
    root = REPOSITORY_ROOT
    manifest = read_evidence_document(
        root / "tests/golden/oracles/proj/9.5.1/manifest.json"
    )
    reproduction = manifest["reproduction"]
    archived = (
        root / "experiments/provenance/reproduction-v1/proj-9.5.1/capture_proj.py.txt"
    )
    current_exporter = root / reproduction["exporter_path"]
    requirements = root / "experiments/oracles/proj/9.5.1/requirements.txt"

    assert (
        hashlib.sha256(archived.read_bytes()).hexdigest()
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
    assert reproduction["previous_manifest_exporter_sha256"] == (
        "9295111b0a0d2ddf36c207ee50c7078a28202b60911a5589301299622a665a9a"
    )
