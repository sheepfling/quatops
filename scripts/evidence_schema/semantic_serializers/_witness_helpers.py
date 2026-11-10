"""Small shared operations used by target-specific witness serializers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from scripts.evidence_schema.crystallized import CanonicalInputs
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


class _CaseRow(Protocol):
    case_id: str


class _DirectInputDocument(Protocol):
    @property
    def shared_canonical_inputs(self) -> str: ...

    @property
    def retained_case_ids(self) -> Sequence[str]: ...


_Row = TypeVar("_Row", bound=_CaseRow)

CANONICAL_RETAINED_ORIENTATION_CASE_IDS = NATIVE_RETAINED_CASE_IDS


def canonical_orientation_selection(
    available_case_ids: Sequence[str], *, context: str
) -> tuple[str, ...]:
    """Resolve the common compact witness selection against target inputs."""
    if len(set(available_case_ids)) != len(available_case_ids):
        raise ValueError(f"{context} inputs contain duplicate retained case IDs")
    missing = [
        case_id
        for case_id in CANONICAL_RETAINED_ORIENTATION_CASE_IDS
        if case_id not in available_case_ids
    ]
    if missing:
        raise ValueError(
            f"{context} inputs omit common witness cases: {', '.join(missing)}"
        )
    return CANONICAL_RETAINED_ORIENTATION_CASE_IDS


def canonical_orientations_by_case(
    canonical_inputs: CanonicalInputs,
) -> dict[str, list[float]]:
    """Index the shared quaternion inputs without changing their values."""
    return {row.case_id: row.coefficients_wxyz for row in canonical_inputs.orientations}


def validate_direct_canonical_inputs(
    inputs: _DirectInputDocument,
    context: SemanticSerializationContext,
    *,
    target: str,
) -> None:
    """Check a direct library experiment's canonical file and case selection."""
    referenced_path = (
        context.repository_root / inputs.shared_canonical_inputs
    ).resolve()
    if referenced_path != context.canonical_input_path.resolve():
        raise ValueError(f"{target} references a different canonical input file")
    expected_case_ids = [row.case_id for row in context.canonical_inputs.orientations]
    if list(inputs.retained_case_ids) != expected_case_ids:
        raise ValueError(f"{target} retained case IDs differ from shared inputs")


def validate_canonical_input_digest(
    actual_digest: str,
    context: SemanticSerializationContext,
    *,
    target: str,
) -> None:
    """Require a target input document to cite this exact canonical corpus."""
    if actual_digest != context.canonical_input_sha256:
        raise ValueError(f"{target} canonical input digest differs")


def validate_canonical_quaternion(
    target: str,
    case_id: str,
    coefficients: list[float],
    canonical_by_case: dict[str, list[float]],
) -> None:
    """Require a target input quaternion to preserve the named shared case."""
    expected = canonical_by_case.get(case_id)
    if expected is None or coefficients != expected:
        raise ValueError(
            f"{target} input {case_id!r} differs from shared canonical inputs"
        )


def select_retained_rows(
    rows: Sequence[_Row], retained_case_ids: Sequence[str], *, context: str
) -> list[_Row]:
    """Select inputs in declared order and reject missing or duplicate rows."""
    by_case_id = {row.case_id: row for row in rows}
    if len(by_case_id) != len(rows):
        raise ValueError(f"{context} capture contains duplicate case IDs")
    if len(set(retained_case_ids)) != len(retained_case_ids):
        raise ValueError(f"{context} inputs contain duplicate retained case IDs")
    missing = [case_id for case_id in retained_case_ids if case_id not in by_case_id]
    if missing:
        raise ValueError(
            f"{context} capture is missing retained cases: {', '.join(missing)}"
        )
    return [by_case_id[case_id] for case_id in retained_case_ids]


def native_product_case_ids(case_id: str) -> tuple[str, str]:
    """Read the native capture's stable ``left_times_right`` case label."""
    try:
        left_case_id, right_case_id = case_id.split("_times_", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"unsupported native product case ID: {case_id}") from exc
    return left_case_id, right_case_id
