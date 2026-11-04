"""Strict numeric primitives shared by target capture schemas."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
)

Scalar = float


def _ensure_binary64_convertible(value: object) -> object:
    """Reject numeric source values that overflow binary64 without rounding ints."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("input number must be an integer or float")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(
            "input number overflows the target binary64 representation"
        ) from error
    if not math.isfinite(converted):
        raise ValueError("input number overflows the target binary64 representation")
    return value


# Keep parsed integers intact; adapters explicitly convert to binary64 at the
# native boundary, where ordinary binary64 rounding is part of the contract.
FiniteInputNumber = Annotated[
    Union[StrictInt, FiniteFloat], AfterValidator(_ensure_binary64_convertible)
]
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
