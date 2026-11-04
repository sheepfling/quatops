"""Shared identifiers and file digests for evidence assembly."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug_identifier(value: str) -> str:
    """Normalize a display name for a stable evidence identifier."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
