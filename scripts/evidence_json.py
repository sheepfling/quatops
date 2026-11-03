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


def dump_evidence_json(value: Any, *, max_inline_length: int = 120) -> str:
    """Render readable JSON with short numeric/vector arrays kept on one line."""

    def render(item: Any, depth: int) -> str:
        prefix = "  " * depth
        if isinstance(item, dict):
            if not item:
                return "{}"
            rows = ["{"]
            entries = list(item.items())
            for index, (key, child) in enumerate(entries):
                rendered = render(child, depth + 1).splitlines()
                rows.append(
                    "  " * (depth + 1)
                    + json.dumps(key, ensure_ascii=False)
                    + ": "
                    + rendered[0]
                )
                rows.extend(rendered[1:])
                if index < len(entries) - 1:
                    rows[-1] += ","
            rows.append(prefix + "}")
            return "\n".join(rows)
        if isinstance(item, list):
            if not item:
                return "[]"
            is_scalar_array = all(not isinstance(child, (dict, list)) for child in item)
            is_matrix = all(
                isinstance(child, list)
                and all(not isinstance(element, (dict, list)) for element in child)
                for child in item
            )
            if is_scalar_array or is_matrix:
                compact = json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(", ", ": "),
                )
                if len(compact) <= max_inline_length:
                    return compact
            rows = ["["]
            for index, child in enumerate(item):
                rendered = render(child, depth + 1).splitlines()
                rows.append("  " * (depth + 1) + rendered[0])
                rows.extend(rendered[1:])
                if index < len(item) - 1:
                    rows[-1] += ","
            rows.append(prefix + "]")
            return "\n".join(rows)
        return json.dumps(item, ensure_ascii=False, allow_nan=False)

    return render(value, 0) + "\n"


def _contains_nonfinite_marker(value: Any) -> bool:
    if isinstance(value, dict):
        return "nonfinite" in value or any(
            _contains_nonfinite_marker(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_nonfinite_marker(item) for item in value)
    return False


def compact_numeric_evidence(value: Any) -> Any:
    """Drop inferable shape and duplicate bits, keeping signed-zero/nonfinite bits."""
    if isinstance(value, dict):
        if "values" in value and "dtype" in value:
            result = {
                "values": compact_numeric_evidence(value["values"]),
                "dtype": value["dtype"],
            }
            bits = value.get("ieee754_hex_be", [])
            if any(
                bit in {"80000000", "8000000000000000"} for bit in bits
            ) or _contains_nonfinite_marker(value["values"]):
                result["ieee754_hex_be"] = bits
            return result
        return {key: compact_numeric_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_numeric_evidence(item) for item in value]
    return value
