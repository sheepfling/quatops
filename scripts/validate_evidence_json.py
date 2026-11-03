"""Validate retained and maintained quaternion evidence JSON as JSON values."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_EVIDENCE_ROOTS = (
    ROOT / "tests" / "golden",
    ROOT / "experiments" / "oracles",
    ROOT / "experiments" / "converters",
    ROOT / "experiments" / "transports",
)
_EXCLUDED_PARTS = {".venv", "__pycache__", "node_modules", "build", "install"}


def evidence_json_files() -> list[Path]:
    """Collect maintained evidence JSON and omit local runtimes/build outputs."""
    files = {
        path
        for evidence_root in _EVIDENCE_ROOTS
        if evidence_root.is_dir()
        for path in evidence_root.rglob("*.json")
        if not (_EXCLUDED_PARTS.intersection(path.parts))
    }
    return sorted(files)


def _validate_file(path: Path) -> None:
    from scripts.evidence_json import validate_evidence_payload

    try:
        validate_evidence_payload(path)
    except Exception as error:
        raise ValueError(f"invalid evidence JSON: {path.relative_to(ROOT)}") from error


def main() -> None:
    files = evidence_json_files()
    if not files:
        raise RuntimeError("no retained evidence JSON files were found")

    for path in files:
        _validate_file(path)
    print(f"Validated {len(files)} retained evidence JSON files")


if __name__ == "__main__":
    main()
