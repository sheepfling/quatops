"""Input-to-output identity checks for the fixed SpiceyPy capture recipe."""

from __future__ import annotations

from scripts.evidence_schema.converter_captures import SpiceyPyCapture, SpiceyPyInputs
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    resolve_orientation_selection,
)


def validate_spiceypy_capture_matches_inputs(
    capture: SpiceyPyCapture,
    inputs: SpiceyPyInputs,
    shared: SharedCanonicalInputCapture,
) -> None:
    """Verify selected IDs and recorded native product operands before writing."""
    selected = resolve_orientation_selection(
        shared,
        inputs.retained_case_ids,
        required_case_ids={"mixed_a", "mixed_b"},
    )
    selected_by_id = {row.case_id: row for row in selected}
    observations = capture.observations
    if tuple(row.case_id for row in observations.canonical_to_native) != tuple(
        inputs.retained_case_ids
    ):
        raise ValueError(
            "SpiceyPy captured orientations do not match selected input IDs"
        )
    for observed in observations.canonical_to_native:
        expected = [
            float(value) for value in selected_by_id[observed.case_id].canonical_wxyz
        ]
        if observed.native_input_wxyz != expected:
            raise ValueError(
                f"SpiceyPy orientation {observed.case_id!r} input differs from selected input"
            )
    if tuple(row.case_id for row in observations.native_created) != (
        inputs.native_constructed_orientation.case_id,
    ):
        raise ValueError(
            "SpiceyPy constructed output does not match the selected constructor ID"
        )
    expected_products = (
        ("mixed_a_times_mixed_b", "mixed_a", "mixed_b"),
        ("mixed_b_times_mixed_a", "mixed_b", "mixed_a"),
    )
    if tuple(row.case_id for row in observations.composition) != tuple(
        row[0] for row in expected_products
    ):
        raise ValueError(
            "SpiceyPy composition outputs do not match both selected product IDs"
        )
    for observed, (case_id, lhs_id, rhs_id) in zip(
        observations.composition, expected_products
    ):
        lhs = selected_by_id[lhs_id].canonical_wxyz
        rhs = selected_by_id[rhs_id].canonical_wxyz
        expected_lhs = [float(value) for value in lhs]
        expected_rhs = [float(value) for value in rhs]
        if (
            observed.native_lhs_wxyz != expected_lhs
            or observed.native_rhs_wxyz != expected_rhs
        ):
            raise ValueError(
                f"SpiceyPy product {case_id!r} operands differ from selected inputs"
            )
