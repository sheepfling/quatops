"""Validate retained and maintained quaternion evidence JSON as JSON values."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evidence_json import validate_evidence_payload

_EXCLUDED_PARTS = frozenset(
    {".venv", "__pycache__", "node_modules", "build", "install"}
)


def evidence_json_files(repository_root: Path) -> list[Path]:
    """Collect maintained evidence JSON and omit local runtimes/build outputs."""
    evidence_roots = (
        repository_root / "tests" / "golden",
        repository_root / "experiments" / "oracles",
        repository_root / "experiments" / "converters",
        repository_root / "experiments" / "transports",
    )
    files = {
        path
        for evidence_root in evidence_roots
        if evidence_root.is_dir()
        for path in evidence_root.rglob("*.json")
        if not (_EXCLUDED_PARTS.intersection(path.parts))
    }
    return sorted(files)


def _validate_file(path: Path, repository_root: Path) -> None:
    try:
        validate_evidence_payload(path)
    except Exception as error:
        raise ValueError(
            f"invalid evidence JSON: {path.relative_to(repository_root)}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    repository_root = parser.parse_args().repository_root.resolve(strict=True)
    files = evidence_json_files(repository_root)
    if not files:
        raise RuntimeError("no retained evidence JSON files were found")

    for path in files:
        _validate_file(path, repository_root)
    print(f"Validated {len(files)} retained evidence JSON files")


if __name__ == "__main__":
    main()
