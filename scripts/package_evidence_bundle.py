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
    "uv.lock",
    "MANIFEST.in",
    "quality-gates.toml",
    "quatops_frame_semantics_audit_138b730.md",
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
                if any(
                    part.casefold() in EXCLUDED_PARTS
                    for part in relative_candidate.parts
                ):
                    continue
                if candidate.suffix in {".pyc", ".pyo"}:
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
