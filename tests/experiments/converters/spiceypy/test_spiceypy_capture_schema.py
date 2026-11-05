"""Strict finite-value validation for SpiceyPy native capture products."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.shared_inputs import SharedCanonicalInputCapture
from scripts.evidence_schema.spiceypy_capture import (
    validate_spiceypy_capture_matches_inputs,
)
from scripts.evidence_schema.targets import (
    SpiceyPyCapture,
    SpiceyPyInputs,
    SpiceyPyOrientation,
)
from tests.support.paths import REPOSITORY_ROOT


def _orientation() -> dict[str, object]:
    return {
        "case_id": "probe",
        "native_input_wxyz": [1, -0.0, 0, 0],
        "native_mxv_basis_xyz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_input_wxyz", [float("nan"), 0, 0, 0]),
        ("native_mxv_basis_xyz", [[float("inf"), 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("native_mxv_probe_xyz", [0, float("-inf"), 0]),
        ("native_lhs_wxyz", [1, 2, float("nan"), 4]),
        ("native_rhs_wxyz", [1, 2, 3, float("inf")]),
    ],
)
def test_spice_observations_reject_nonfinite_values(field, value):
    row = _orientation()
    row[field] = value
    with pytest.raises(ValidationError):
        SpiceyPyOrientation.model_validate(row, strict=True)


def test_spice_observation_accepts_signed_zero_and_nonunit_finite_values():
    row = _orientation()
    row["native_input_wxyz"] = [123, -0.0, 0, 0]
    parsed = SpiceyPyOrientation.model_validate(row, strict=True)
    assert parsed.native_input_wxyz[0] == 123
    assert str(parsed.native_input_wxyz[1]).startswith("-")


@pytest.mark.parametrize("axis", [[0, 0, 0], [2, -3, 0]])
def test_spice_constructor_requires_identity_and_allows_zero_or_nonunit_axis(axis):
    spec = read_evidence_document(
        REPOSITORY_ROOT / "experiments/converters/spiceypy/8.3.0/converter_inputs.json"
    )
    spec["native_constructed_orientation"]["axis_xyz"] = axis
    assert (
        SpiceyPyInputs.model_validate(
            spec, strict=True
        ).native_constructed_orientation.axis_xyz
        == axis
    )
    spec["native_constructed_orientation"].pop("case_id")
    with pytest.raises(ValidationError):
        SpiceyPyInputs.model_validate(spec, strict=True)


@pytest.mark.parametrize(
    "axis,angle",
    [
        ([float("nan"), 0, 0], 1.0),
        ([0, float("inf"), 0], 1.0),
        ([0, 0, 1], float("inf")),
    ],
)
def test_spice_constructor_rejects_nonfinite_axis_or_angle(axis, angle):
    spec = read_evidence_document(
        REPOSITORY_ROOT / "experiments/converters/spiceypy/8.3.0/converter_inputs.json"
    )
    spec["native_constructed_orientation"]["axis_xyz"] = axis
    spec["native_constructed_orientation"]["angle_radians"] = angle
    with pytest.raises(ValidationError):
        SpiceyPyInputs.model_validate(spec, strict=True)


def _load_valid_spice_capture():
    return SpiceyPyCapture.model_validate(
        read_evidence_document(
            REPOSITORY_ROOT
            / "tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json"
        ),
        strict=True,
    )


def _load_spice_inputs():
    return SpiceyPyInputs.model_validate(
        read_evidence_document(
            REPOSITORY_ROOT
            / "experiments/converters/spiceypy/8.3.0/converter_inputs.json"
        ),
        strict=True,
    )


def _load_canonical_inputs():
    return SharedCanonicalInputCapture.model_validate(
        read_evidence_document(
            REPOSITORY_ROOT / "experiments/interoperability/v1/canonical_inputs.json"
        ),
        strict=True,
    )


def test_spice_capture_requires_complete_selected_recipe():
    capture = _load_valid_spice_capture()
    assert len(capture.observations.canonical_to_native) == 8
    assert len(capture.observations.native_created) == 1
    assert len(capture.observations.composition) == 2
    root = read_evidence_document(
        REPOSITORY_ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json"
    )
    root["observations"]["canonical_to_native"].pop()
    with pytest.raises(ValidationError):
        SpiceyPyCapture.model_validate(root, strict=True)
    root = read_evidence_document(
        REPOSITORY_ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json"
    )
    root["observations"]["native_created"] = []
    with pytest.raises(ValidationError):
        SpiceyPyCapture.model_validate(root, strict=True)
    root = read_evidence_document(
        REPOSITORY_ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json"
    )
    root["observations"]["composition"].pop()
    with pytest.raises(ValidationError):
        SpiceyPyCapture.model_validate(root, strict=True)


def test_spice_capture_inputs_and_product_operands_link_to_selected_corpus():
    capture = _load_valid_spice_capture()
    inputs = _load_spice_inputs()
    shared = _load_canonical_inputs()
    validate_spiceypy_capture_matches_inputs(capture, inputs, shared)

    root = read_evidence_document(
        REPOSITORY_ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json"
    )
    root["observations"]["composition"][0]["native_lhs_wxyz"] = [2, 3, 4, 5]
    changed = SpiceyPyCapture.model_validate(root, strict=True)
    with pytest.raises(ValueError, match="operands differ"):
        validate_spiceypy_capture_matches_inputs(changed, inputs, shared)

    root = read_evidence_document(
        REPOSITORY_ROOT / "tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json"
    )
    root["observations"]["canonical_to_native"][0]["native_input_wxyz"] = [0, 1, 0, 0]
    changed = SpiceyPyCapture.model_validate(root, strict=True)
    with pytest.raises(ValueError, match="input differs from selected input"):
        validate_spiceypy_capture_matches_inputs(changed, inputs, shared)


@pytest.mark.parametrize("invalid_ids", [[], ["identity"] * 8, ["identity"]])
def test_spice_input_requires_the_complete_unique_fixed_selection(invalid_ids):
    spec = read_evidence_document(
        REPOSITORY_ROOT / "experiments/converters/spiceypy/8.3.0/converter_inputs.json"
    )
    spec["retained_case_ids"] = invalid_ids
    with pytest.raises(ValidationError):
        SpiceyPyInputs.model_validate(spec, strict=True)
