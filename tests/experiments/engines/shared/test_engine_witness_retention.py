"""Engine witness selection keeps the raw native rows intact."""

from __future__ import annotations

import json
from copy import deepcopy
from math import copysign

import pytest

import scripts.retain_engine_witnesses as retention
from scripts.evidence_schema.engine_captures import ENGINE_RETAINED_CASE_IDS
from scripts.retain_engine_witnesses import (
    compact_native_vectors,
    retain_canonical_interop,
)
from tests.support.paths import REPOSITORY_ROOT


def test_godot_retention_rejects_a_different_native_version(tmp_path, monkeypatch):
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "canonical_interop": {"reverse_composition": {}},
                "engine": {"name": "Godot Engine", "version": "4.8-dev1 (official)"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        retention, "retain_canonical_interop", lambda capture, _: capture
    )
    with pytest.raises(ValueError, match="native godot version differs"):
        retention.retain("godot", "4.7-stable", capture, tmp_path, REPOSITORY_ROOT)


def test_direct_capture_rejects_unselected_cases():
    inputs = {case_id: [1.0, 0.0, 0.0, 0.0] for case_id in ENGINE_RETAINED_CASE_IDS}
    capture = {
        "orientation_cases": [
            {"case_id": "quarter_x", "canonical_wxyz": [1.0, 0.0, 0.0, 0.0]}
        ]
    }
    with pytest.raises(ValueError, match="unexpected"):
        retain_canonical_interop(capture, inputs)


def test_native_basis_vectors_are_compacted_without_changing_components():
    original = {
        "native_rotated_positive_basis_vectors_xyz": [
            {"x": -0.0, "y": 0.25, "z": -0.75},
            [0.5, 0.0, 1.0],
            {"x": 1.0, "y": 2.0, "z": 3.0},
        ],
        "native_rotated_probe_vector_xyz": {"x": 0.5, "y": -1.0, "z": 2.0},
    }
    retained = compact_native_vectors(original)
    assert retained["native_rotated_positive_basis_vectors_xyz"] == [
        [-0.0, 0.25, -0.75],
        [0.5, 0.0, 1.0],
        [1.0, 2.0, 3.0],
    ]
    assert (
        copysign(1.0, retained["native_rotated_positive_basis_vectors_xyz"][0][0])
        == -1.0
    )
    assert retained["native_rotated_probe_vector_xyz"] == [0.5, -1.0, 2.0]
    assert isinstance(original["native_rotated_positive_basis_vectors_xyz"][0], dict)
    with pytest.raises(ValueError, match="only x, y and z"):
        compact_native_vectors(
            {"native_rotated_probe_vector_xyz": {"x": 1, "y": 2, "z": 3, "w": 4}}
        )


def test_precision_profiles_are_rebuilt_without_mutating_capture_rows():
    canonical_input = [0.5, -0.5, 0.5, -0.5]
    capture = {
        "precision_captures": {
            "float32": {
                "orientation_cases": [
                    {
                        "case_id": "mixed_a",
                        "canonical_wxyz": [0.5, -0.5000001, 0.5, -0.5],
                        "native_quaternion": [0.5, -0.5, 0.5, -0.5],
                        "native_rotated_positive_basis_vectors_xyz": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "native_rotated_probe_vector_xyz": [1.0, 2.0, 3.0],
                        "native_result": {"rotated_probe_xyz": [1.0, 2.0, 3.0]},
                    }
                ],
                "reverse_composition": {"case_id": "product-reverse"},
            }
        }
    }
    original = deepcopy(capture)

    retained = retain_canonical_interop(capture, {"mixed_a": canonical_input})

    assert capture == original
    assert (
        capture["precision_captures"]["float32"]["orientation_cases"][0]
        == (original["precision_captures"]["float32"]["orientation_cases"][0])
    )
    retained_case = retained["precision_captures"]["float32"]["orientation_cases"][0]
    assert "canonical_wxyz" not in retained_case
    assert "native_input_wxyz" not in retained_case
    assert retained_case["case_id"] == "mixed_a"
    assert retained_case["native_rotated_probe_vector_xyz"] == [1.0, 2.0, 3.0]
    assert retained_case["native_result"] == {"rotated_probe_xyz": [1.0, 2.0, 3.0]}
    retained_case["native_result"]["rotated_probe_xyz"][0] = 99.0
    assert capture == original
    assert canonical_input == [0.5, -0.5, 0.5, -0.5]


def test_unity_mixed_probe_is_attached_without_changing_its_values():
    capture = {
        "native_mixed_probe_xyz": {"x": -0.0, "y": 1.0, "z": 2.0},
        "orientation_cases": [
            {"case_id": "mixed_a", "canonical_wxyz": [0.8, 0.2, -0.4, 0.4]}
        ],
    }
    retained = retain_canonical_interop(capture, {"mixed_a": [0.8, 0.2, -0.4, 0.4]})
    assert retained["orientation_cases"][0]["native_rotated_probe_vector_xyz"] == [
        -0.0,
        1.0,
        2.0,
    ]
    assert (
        copysign(
            1.0, retained["orientation_cases"][0]["native_rotated_probe_vector_xyz"][0]
        )
        == -1.0
    )
    capture["orientation_cases"][0]["native_rotated_probe_vector_xyz"] = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="only once"):
        retain_canonical_interop(capture, {"mixed_a": [0.8, 0.2, -0.4, 0.4]})


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (
            [{"case_id": "unexpected", "canonical_wxyz": [1.0, 0.0, 0.0, 0.0]}],
            "unexpected",
        ),
        (
            [
                {
                    "case_id": "mixed",
                    "canonical_wxyz": [0.5, -0.5, 0.5, -0.5],
                },
                {
                    "case_id": "mixed",
                    "canonical_wxyz": [0.5, -0.5, 0.5, -0.5],
                },
            ],
            "duplicate",
        ),
        ([], "missing"),
    ],
)
def test_retention_rejects_unexpected_duplicate_and_missing_cases(cases, message):
    with pytest.raises(ValueError, match=message):
        retain_canonical_interop(
            {"orientation_cases": cases},
            {"mixed": [0.5, -0.5, 0.5, -0.5]},
        )


def test_retention_rejects_non_quaternion_native_inputs():
    with pytest.raises(ValueError, match="at least 4"):
        retain_canonical_interop(
            {
                "orientation_cases": [
                    {"case_id": "mixed", "canonical_wxyz": [1.0, 0.0, 0.0]}
                ]
            },
            {"mixed": [0.5, -0.5, 0.5, -0.5]},
        )
