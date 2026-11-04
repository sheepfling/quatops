"""Strict numeric primitives shared by target capture schemas."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Scalar = float
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vector3 = Annotated[list[Scalar], Field(min_length=3, max_length=3)]
Quaternion4 = Annotated[list[Scalar], Field(min_length=4, max_length=4)]
Matrix3 = Annotated[list[Vector3], Field(min_length=3, max_length=3)]
Matrix4 = Annotated[
    list[Annotated[list[Scalar], Field(min_length=4, max_length=4)]],
    Field(min_length=4, max_length=4),
]
BasisImages = Annotated[list[Vector3], Field(min_length=3, max_length=3)]


def finite_max_abs(values: Iterable[float], label: str) -> float:
    """Return a finite maximum absolute value; reject empty/non-finite inputs."""
    finite_values = [float(value) for value in values]
    if not finite_values:
        raise ValueError(f"{label} is empty")
    for value in finite_values:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite value: {value}")
    return max(abs(value) for value in finite_values)


class StrictEvidenceModel(BaseModel):
    """Immutable, strict model for the normalized evidence product."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NativeCaptureModel(BaseModel):
    """Strict typed model for target-specific capture packages."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
