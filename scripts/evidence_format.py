"""Standard-library formatting for readable, compact numeric evidence JSON."""

from __future__ import annotations

import json
from typing import Any


def dump_evidence_json(value: Any, *, max_inline_length: int = 120) -> str:
    """Render JSON with short numeric/vector arrays kept on one line."""

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
