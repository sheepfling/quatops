"""Build and integrity-check a reviewable QuatOps evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE_PATHS = (
    "AGENTS.md",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "pyproject.toml",
    "MANIFEST.in",
    "quality-gates.toml",
    "docs",
    "src/quatops",
    "experiments",
    "tests",
    "scripts",
)
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "intermediate",
    "binaries",
    "saved",
    "library",
    "logs",
    "usersettings",
    ".godot",
    "__pycache__",
    ".pytest_cache",
    ".pytest-semantic-review-tmp",
    ".ruff_cache",
    ".mypy_cache",
    ".rumdl_cache",
    ".benchmarks",
    "build",
    "dist",
    ".tox",
    ".nox",
    ".cache",
}
GENERATED_INPUTS = {
    "experiments/converters/cesium/1.124.0/converter_inputs.json",
    "experiments/converters/gltf/2.0.0/converter_inputs.json",
    "experiments/converters/gltf/2.0.0/quatops_hierarchy.gltf",
    "experiments/converters/openusd/26.11/converter_inputs.json",
    "experiments/converters/openusd/26.11/quatops_hierarchy.usda",
    "experiments/transports/gltf/2.0.0/assets/rotation_transport.gltf",
    "experiments/transports/gltf/2.0.0/assets/rotation_transport.bin",
}


def included_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in INCLUDE_PATHS:
        path = root / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if not candidate.is_file():
                    continue
                relative_candidate = candidate.relative_to(root)
                if relative_candidate.as_posix() in GENERATED_INPUTS:
                    continue
                if any(
                    part.casefold() in EXCLUDED_PARTS
                    or part.casefold().endswith(".egg-info")
                    for part in relative_candidate.parts
                ):
                    continue
                if candidate.suffix.casefold() in {".pyc", ".pyo", ".log"}:
                    continue
                parts = relative_candidate.parts
                if (
                    parts[:2] == ("experiments", "unity")
                    and "ProjectSettings" in parts
                    and candidate.name != "ProjectVersion.txt"
                ):
                    continue
                if parts[:2] == ("experiments", "unreal") and "Config" in parts:
                    continue
                files.add(candidate)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="repository root whose files should be included",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="archive destination (default: repository-root/quatops-evidence-bundle.zip)",
    )
    args = parser.parse_args()

    root = args.repository_root.resolve(strict=True)
    output = (args.output or root / "quatops-evidence-bundle.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = included_files(root)
    if not files:
        parser.error("no evidence files found")

    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with ZipFile(
            temporary, "w", compression=ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        with ZipFile(temporary, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP integrity check failed at {bad_member}")
            if len(archive.namelist()) != len(files):
                raise RuntimeError(
                    "ZIP member count does not match included file count"
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Archive: {output}")
    print(f"Files: {len(files)}")
    print(f"Size: {output.stat().st_size:,} bytes")
    print("Integrity: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
