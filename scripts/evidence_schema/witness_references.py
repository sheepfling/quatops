"""Resolve and verify immutable evidence files within an allowed directory."""

from __future__ import annotations

import hashlib
from pathlib import Path


def read_evidence_file(
    base: Path,
    reference: str,
    allowed_root: Path,
    expected_sha256: str,
    *,
    context: str,
) -> bytes:
    """Return exact checked bytes after rejecting paths outside the allowed root."""
    if not reference or "\\" in reference or Path(reference).is_absolute():
        raise ValueError(f"{context} path must be a relative POSIX path")
    root = allowed_root.resolve(strict=True)
    path = (base / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{context} path escapes its allowed directory")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{context} digest differs from its manifest")
    return payload
