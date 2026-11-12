"""Locate the repository for auxiliary scripts and test support."""

from __future__ import annotations

from pathlib import Path


def find_repository_root(start: Path) -> Path:
    """Find the nearest repository containing the project, source, and scripts."""
    path = start.resolve(strict=True)
    directory = path.parent if path.is_file() else path
    for candidate in (directory, *directory.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src/quatops").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate
    raise ValueError(f"Cannot find the QuatOps repository from {start}")
