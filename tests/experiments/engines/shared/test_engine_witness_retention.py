"""Engine witness selection keeps the raw native rows intact."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.retain_engine_witnesses import retain_canonical_interop


def test_precision_profiles_are_rebuilt_without_mutating_capture_rows():
    native_input = [0.5, 0.5, 0.5, 0.5]
    canonical_input = [0.5, -0.5, 0.5, -0.5]
    capture = {
        "precision_captures": {
            "float32": {
                "orientation_cases": [
                    {
                        "case_id": "mixed",
                        "canonical_wxyz": native_input,
                        "native_result": {"rotated_probe_xyz": [1.0, 2.0, 3.0]},
                    }
                ],
                "reverse_composition": {"case_id": "product-reverse"},
            }
        }
    }
    original = deepcopy(capture)

    retained = retain_canonical_interop(capture, {"mixed": canonical_input})

    assert capture == original
    assert (
        capture["precision_captures"]["float32"]["orientation_cases"][0]
        == (original["precision_captures"]["float32"]["orientation_cases"][0])
    )
    retained_case = retained["precision_captures"]["float32"]["orientation_cases"][0]
    assert retained_case["native_input_wxyz"] == native_input
    assert retained_case["canonical_wxyz"] == canonical_input
    assert retained_case["native_result"] == {"rotated_probe_xyz": [1.0, 2.0, 3.0]}
    retained_case["native_result"]["rotated_probe_xyz"][0] = 99.0
    retained_case["canonical_wxyz"][0] = 99.0
    assert capture == original
    assert canonical_input == [0.5, -0.5, 0.5, -0.5]


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (
            [{"case_id": "unexpected", "canonical_wxyz": [1.0, 0.0, 0.0, 0.0]}],
            "unexpected",
        ),
        (
            [
                {"case_id": "mixed", "canonical_wxyz": [1.0, 0.0, 0.0, 0.0]},
                {"case_id": "mixed", "canonical_wxyz": [1.0, 0.0, 0.0, 0.0]},
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
    with pytest.raises(TypeError, match="four numeric coefficients"):
        retain_canonical_interop(
            {
                "orientation_cases": [
                    {"case_id": "mixed", "canonical_wxyz": [1.0, 0.0, 0.0]}
                ]
            },
            {"mixed": [0.5, -0.5, 0.5, -0.5]},
        )
