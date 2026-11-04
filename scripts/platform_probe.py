"""Shared runtime platform probes for portable evidence tooling."""

from __future__ import annotations

import sys


def is_windows() -> bool:
    """Return whether the current Python process runs on Windows."""
    return sys.platform == "win32"


def executable_suffix() -> str:
    """Return the executable filename suffix for the current platform."""
    return ".exe" if is_windows() else ""
