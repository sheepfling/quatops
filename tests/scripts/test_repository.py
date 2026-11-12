"""Repository discovery works from nested paths without Git metadata."""

from pathlib import Path

import pytest

from scripts.repository import find_repository_root


def test_discovers_repository_from_nested_file(tmp_path: Path):
    root = tmp_path / "checkout"
    (root / "src/quatops").mkdir(parents=True)
    (root / "scripts/nested").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    source = root / "scripts/nested/check.py"
    source.write_text("", encoding="utf-8")
    assert find_repository_root(source) == root
    assert find_repository_root(source.parent) == root


def test_rejects_paths_outside_a_repository(tmp_path: Path):
    with pytest.raises(ValueError, match="Cannot find the QuatOps repository"):
        find_repository_root(tmp_path)
