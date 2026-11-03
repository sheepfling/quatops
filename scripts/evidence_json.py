"""Strict Pydantic validation for repository-owned evidence JSON objects."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn, cast

from pydantic import JsonValue, RootModel


class EvidenceDocument(RootModel[dict[str, JsonValue]]):
    """A JSON object with recursively valid JSON values."""


class EvidencePayload(RootModel[JsonValue]):
    """Any standards-compliant JSON value with recursively valid contents."""


class EvidenceRows(RootModel[list[dict[str, JsonValue]]]):
    """A JSON array whose rows contain only JSON values."""


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is not allowed")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)


def _decode_json(payload: str | bytes) -> JsonValue:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    value = json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    _reject_non_finite(value)
    return cast(JsonValue, value)


def parse_evidence_payload(payload: str | bytes) -> JsonValue:
    """Parse strict UTF-8 JSON and validate its recursive value tree with Pydantic."""
    decoded = _decode_json(payload)
    return EvidencePayload.model_validate(decoded, strict=True).root


def parse_evidence_document(payload: str | bytes) -> dict[str, Any]:
    """Parse an object document, rejecting non-object roots and invalid values."""
    parsed = EvidenceDocument.model_validate(_decode_json(payload), strict=True)
    return cast(dict[str, Any], parsed.root)


def read_evidence_document(path: Path) -> dict[str, Any]:
    """Read and validate a repository evidence JSON object from disk."""
    return parse_evidence_document(path.read_bytes())


def validate_evidence_payload(path: Path) -> None:
    """Validate any evidence JSON root without rewriting captured bytes."""
    parse_evidence_payload(path.read_bytes())


def read_evidence_rows(path: Path) -> list[dict[str, Any]]:
    """Read and validate a JSON array of native observation objects."""
    parsed = EvidenceRows.model_validate(_decode_json(path.read_bytes()), strict=True)
    return cast(list[dict[str, Any]], parsed.root)
