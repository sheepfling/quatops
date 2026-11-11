"""Stable paths to the repository and test-data roots."""

from pathlib import Path

from scripts.repository import find_repository_root

REPOSITORY_ROOT = find_repository_root(Path(__file__))
TESTS_ROOT = REPOSITORY_ROOT / "tests"
