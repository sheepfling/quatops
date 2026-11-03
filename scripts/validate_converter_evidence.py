"""Validate the shared envelope and cross-file identity of converter fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from scripts.converter_evidence import (
        converter_fixture_directories,
        load_converter_evidence,
    )

    directories = converter_fixture_directories(ROOT)
    if not directories:
        raise RuntimeError("no converter evidence packages were found")

    for directory in directories:
        package = load_converter_evidence(directory, repository_root=ROOT)
        print(
            f"PASS {package.target.name} {package.target.version}: "
            f"{len(package.observations)} observation groups "
            f"({package.evidence_kind})"
        )
    print(f"Validated {len(directories)} converter evidence packages")


if __name__ == "__main__":
    main()
