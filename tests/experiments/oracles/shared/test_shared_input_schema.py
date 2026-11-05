"""Strict semantic validation for the maintained shared input corpora."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.shared_inputs import (
    GeographicInputCorpus,
    SharedCanonicalInputCapture,
    resolve_orientation_selection,
)
from tests.support.paths import REPOSITORY_ROOT


def _canonical() -> dict[str, object]:
    return read_evidence_document(
        REPOSITORY_ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    )


def _geographic() -> dict[str, object]:
    return read_evidence_document(
        REPOSITORY_ROOT / "experiments/oracles/shared/geographic_inputs_v1.json"
    )


def test_shared_inputs_validate_contract_uniqueness_references_and_finite_values():
    canonical = SharedCanonicalInputCapture.model_validate(_canonical(), strict=True)
    geographic = GeographicInputCorpus.model_validate(_geographic(), strict=True)
    assert len(canonical.orientations) == 8
    assert len(geographic.locations) == 11
    assert geographic.locations[-1].latitude_degrees == -90


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["contract"].update({"algebra": "JPL"}),
        lambda data: data["orientations"].append(data["orientations"][0].copy()),
        lambda data: data["composition"].update({"left_case_id": "missing"}),
        lambda data: data["source_frame"].update({"positive_x": "left"}),
        lambda data: data["source_frame"].update({"positive_y": "north"}),
        lambda data: data["source_frame"].update({"positive_z": "forward"}),
        lambda data: data["contract"].update({"frame_mapping": "implicit frames"}),
        lambda data: data["orientations"][0].update(
            {"canonical_wxyz": [float("nan"), 0, 0, 0]}
        ),
        lambda data: data["orientations"][0].update({"canonical_wxyz": [1, 1, 0, 0]}),
        lambda data: data.update({"probe_vector_xyz": [0, 1]}),
    ],
)
def test_invalid_canonical_input_is_rejected(mutate):
    document = _canonical()
    mutate(document)
    with pytest.raises(ValidationError):
        SharedCanonicalInputCapture.model_validate(document, strict=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["locations"].append(data["locations"][0].copy()),
        lambda data: data["locations"][0].update({"height_m": float("inf")}),
        lambda data: data["heading_pitch_roll_radians"].append([0, 1]),
        lambda data: data.update({"local_point_xyz_m": [0, 1, float("nan")]}),
    ],
)
def test_invalid_geographic_input_is_rejected(mutate):
    document = _geographic()
    mutate(document)
    with pytest.raises(ValidationError):
        GeographicInputCorpus.model_validate(document, strict=True)


def test_finite_out_of_range_longitude_and_exact_pole_remain_valid():
    document = _geographic()
    document["locations"][0]["longitude_degrees"] = 540
    document["locations"][0]["latitude_degrees"] = 90
    parsed = GeographicInputCorpus.model_validate(document, strict=True)
    assert parsed.locations[0].longitude_degrees == 540
    assert parsed.locations[0].latitude_degrees == 90


def test_finite_latitude_outside_physical_range_is_retained_as_a_probe():
    document = _geographic()
    document["locations"][0]["latitude_degrees"] = 91.0
    parsed = GeographicInputCorpus.model_validate(document, strict=True)
    assert parsed.locations[0].latitude_degrees == 91.0


def test_input_corpus_source_bytes_keep_their_recorded_identity():
    canonical_path = (
        REPOSITORY_ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    )
    geographic_path = (
        REPOSITORY_ROOT / "experiments/oracles/shared/geographic_inputs_v1.json"
    )
    assert hashlib.sha256(canonical_path.read_bytes()).hexdigest() == (
        "a6f0630806b69d3b3b29a036aa10f63bfe214668329c661598951f84f8f41eec"
    )
    assert hashlib.sha256(geographic_path.read_bytes()).hexdigest() == (
        "619099b732593c6c45f4f7eb64714e4c7d79b2c96dcd070777522e3736e4f085"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["locations"][0].update(
            {"longitude_degrees": 9007199254740993}
        ),
        lambda data: data["locations"][0].update(
            {"latitude_degrees": 9007199254740993}
        ),
        lambda data: data["locations"][0].update({"height_m": 9007199254740993}),
    ],
)
def test_geographic_input_retains_integers_until_native_binary64_conversion(mutate):
    document = _geographic()
    mutate(document)
    parsed = GeographicInputCorpus.model_validate(document, strict=True)
    values = (
        parsed.locations[0].longitude_degrees,
        parsed.locations[0].latitude_degrees,
        parsed.locations[0].height_m,
    )
    assert 9007199254740993 in values
    assert float(9007199254740993) == 9007199254740992.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update({"probe_vector_xyz": [9007199254740993, 0, 0]}),
    ],
)
def test_canonical_input_retains_integers_until_native_binary64_conversion(mutate):
    document = _canonical()
    mutate(document)
    parsed = SharedCanonicalInputCapture.model_validate(document, strict=True)
    assert parsed.probe_vector_xyz[0] == 9007199254740993
    assert float(9007199254740993) == 9007199254740992.0


@pytest.mark.parametrize(
    "selected,required",
    [
        (["mixed_a", "mixed_a"], set()),
        (["not-present"], set()),
        (["mixed_a"], {"mixed_b"}),
    ],
)
def test_orientation_selection_rejects_duplicates_unresolved_and_missing_required(
    selected, required
):
    capture = SharedCanonicalInputCapture.model_validate(_canonical(), strict=True)
    with pytest.raises(ValueError):
        resolve_orientation_selection(capture, selected, required_case_ids=required)
