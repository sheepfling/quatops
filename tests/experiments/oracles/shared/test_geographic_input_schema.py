"""Strict schema checks for the shared geographic input corpus."""

from __future__ import annotations

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
