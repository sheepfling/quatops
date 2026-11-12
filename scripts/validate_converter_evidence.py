"""Validate the shared envelope and cross-file identity of converter fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.converter_evidence import converter_fixture_directories
from scripts.evidence_schema.engine_evidence import ENGINE_VERSIONS
from scripts.evidence_schema.validation import validate_evidence_packages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    repository_root = parser.parse_args().repository_root.resolve(strict=True)
    directories = converter_fixture_directories(repository_root)
    if not directories:
        raise RuntimeError("no converter evidence packages were found")

    captures = validate_evidence_packages(repository_root)
    for capture in captures:
        print(f"PASS {capture.target} {capture.version}: normalized native evidence")
    unique_witnesses = {
        capture.witness_path: capture.serialization for capture in captures
    }
    print(
        f"Validated {len(directories)} converter packages and "
        f"{len(ENGINE_VERSIONS)} engine version packages; "
        f"{sum(len(item.witnesses) for item in unique_witnesses.values())} native witnesses"
    )


if __name__ == "__main__":
    main()
