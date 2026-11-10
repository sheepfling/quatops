"""Validate the shared envelope and cross-file identity of converter fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.converter_evidence import (
    converter_fixture_directories,
    load_converter_evidence,
)
from scripts.evidence_schema.engine_evidence import ENGINE_VERSIONS, load_engine_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    repository_root = parser.parse_args().repository_root.resolve(strict=True)
    directories = converter_fixture_directories(repository_root)
    if not directories:
        raise RuntimeError("no converter evidence packages were found")

    for directory in directories:
        package = load_converter_evidence(directory, repository_root=repository_root)
        print(
            f"PASS {package.target.name} {package.target.version}: "
            f"{package.observation_group_count} typed observation groups "
            f"({package.evidence_kind})"
        )
    for engine, version in ENGINE_VERSIONS:
        package = load_engine_package(repository_root, engine, version)
        print(
            f"PASS {package.capture.target.engine} {version}: typed native witness package"
        )
    print(
        f"Validated {len(directories)} converter packages and "
        f"{len(ENGINE_VERSIONS)} engine version packages"
    )


if __name__ == "__main__":
    main()
