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
    from scripts.evidence_schema.engine_product import (
        ENGINE_VERSIONS,
        load_engine_package,
    )

    directories = converter_fixture_directories(ROOT)
    if not directories:
        raise RuntimeError("no converter evidence packages were found")

    for directory in directories:
        package = load_converter_evidence(directory, repository_root=ROOT)
        print(
            f"PASS {package.target.name} {package.target.version}: "
            f"{package.observation_group_count} typed observation groups "
            f"({package.evidence_kind})"
        )
    for engine, version in ENGINE_VERSIONS:
        package = load_engine_package(ROOT, engine, version)
        print(
            f"PASS {package.capture.target.engine} {version}: typed native witness package"
        )
    print(
        f"Validated {len(directories)} converter packages and "
        f"{len(ENGINE_VERSIONS)} engine version packages"
    )


if __name__ == "__main__":
    main()
