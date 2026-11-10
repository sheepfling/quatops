"""Shared identifiers and file digests for evidence assembly."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retention_paths(description: str | None) -> tuple[Path, Path]:
    """Parse the common native-capture and repository paths for selectors."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    return args.capture_dir.resolve(), args.repository_root.resolve(strict=True)


def retention_provenance(root: Path, script: Path) -> dict[str, str]:
    """Identify the selector and the shared code used to write its evidence."""
    return {
        "retention_script_sha256": sha256_file(script),
        "retention_digest_helper_sha256": sha256_file(Path(__file__)),
        "retention_helper_sha256": sha256_file(root / "scripts/evidence_json.py"),
        "retention_formatter_sha256": sha256_file(root / "scripts/evidence_format.py"),
    }
