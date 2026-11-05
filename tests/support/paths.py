"""Stable paths to the repository and test-data roots."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPOSITORY_ROOT / "tests"
