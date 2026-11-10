"""Strict Pydantic validation for repository-owned evidence JSON objects."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn, cast

from pydantic import JsonValue, RootModel

from scripts.evidence_format import dump_evidence_json

__all__ = ["dump_evidence_json"]


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


def parse_evidence_document(payload: str | bytes) -> dict[str, JsonValue]:
    """Parse an object document, rejecting non-object roots and invalid values."""
    parsed = EvidenceDocument.model_validate(_decode_json(payload), strict=True)
    return parsed.root


def read_evidence_document(path: Path) -> dict[str, JsonValue]:
    """Read and validate a repository evidence JSON object from disk."""
    return parse_evidence_document(path.read_bytes())


def validate_evidence_payload(path: Path) -> None:
    """Validate any evidence JSON root without rewriting captured bytes."""
    parse_evidence_payload(path.read_bytes())


def read_evidence_rows(path: Path) -> list[dict[str, JsonValue]]:
    """Read and validate a JSON array of native observation objects."""
    parsed = EvidenceRows.model_validate(_decode_json(path.read_bytes()), strict=True)
    return parsed.root


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
            if "ieee754_hex_be" in value and (
                any(bit in {"80000000", "8000000000000000"} for bit in bits)
                or _contains_nonfinite_marker(value["values"])
            ):
                result["ieee754_hex_be"] = bits
            return result
        return {key: compact_numeric_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_numeric_evidence(item) for item in value]
    return value


def _numeric_shape(value: object) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        shape = _numeric_shape(value[0])
        if any(_numeric_shape(child) != shape for child in value[1:]):
            raise ValueError("numeric values must have a rectangular shape")
        return [len(value), *shape]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return []
    if (
        isinstance(value, dict)
        and set(value) == {"nonfinite"}
        and value["nonfinite"] in ("nan", "+inf", "-inf")
    ):
        return []
    raise TypeError("expected a numeric scalar, nonfinite tag, or list")


def select_numeric_evidence(item: object, selection: int | slice) -> dict[str, object]:
    """Select outer values and their flat scalar bit sidecar without inventing bits."""
    if not isinstance(item, dict) or "values" not in item or "dtype" not in item:
        raise TypeError("expected a typed native numeric value")
    values = item["values"]
    if not isinstance(values, list):
        raise TypeError("numeric selection requires an outer list")
    shape = _numeric_shape(values)
    if "shape" in item:
        declared = item["shape"]
        if (
            not isinstance(declared, list)
            or any(type(size) is not int or size < 0 for size in declared)
            or declared[: len(shape)] != shape
            or (0 not in shape and len(declared) != len(shape))
        ):
            raise ValueError("numeric shape does not match values")
        # Empty axes hide trailing dimensions; retain their declared rank.
        shape = declared
    width = math.prod(shape[1:])
    if isinstance(selection, bool) or not isinstance(selection, (int, slice)):
        raise TypeError("numeric selection must be an integer or slice")
    selected = values[selection]
    if isinstance(selection, int):
        index = selection if selection >= 0 else len(values) + selection
        indices = range(index, index + 1)
        selected_shape = shape[1:]
    else:
        indices = range(*selection.indices(len(values)))
        selected_shape = [len(indices), *shape[1:]]
    result: dict[str, object] = {"values": selected, "dtype": item["dtype"]}
    if "shape" in item:
        result["shape"] = selected_shape
    if "ieee754_hex_be" in item:
        bits = item["ieee754_hex_be"]
        if not isinstance(bits, list) or len(bits) != len(values) * width:
            raise ValueError("numeric bit sidecar length does not match values")
        result["ieee754_hex_be"] = [
            bit
            for index in indices
            for bit in bits[index * width : (index + 1) * width]
        ]
    return result
