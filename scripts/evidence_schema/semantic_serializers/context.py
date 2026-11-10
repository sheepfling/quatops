"""Shared input context supplied to every converter semantic serializer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.evidence_schema.crystallized import CanonicalInputs


@dataclass(frozen=True)
class SemanticSerializationContext:
    """Canonical corpus identity and resolved paths for one serialization run."""

    canonical_inputs: CanonicalInputs
    canonical_input_path: Path
    canonical_input_sha256: str
    repository_root: Path
