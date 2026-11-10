"""Strict numeric primitives shared by target capture schemas."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
)

Scalar = float

NATIVE_RETAINED_CASE_IDS = (
    "identity",
    "mixed_a",
    "mixed_b",
    "antipode_mixed_a",
    "nonprincipal_halfturn",
)


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


def validate_signed_axis_map(values: list[str], *, context: str) -> None:
    """Validate a declared signed permutation of three coordinate axes."""
    if len(values) != 3 or any(
        value not in {"x", "-x", "y", "-y", "z", "-z"} for value in values
    ):
        raise ValueError(f"{context} is not a signed-axis triple")
    if len({value.lstrip("-") for value in values}) != 3:
        raise ValueError(f"{context} is not a signed-axis permutation")


def mapped_handedness(
    source: str, values: list[str], *, context: str
) -> Literal["right-handed", "left-handed"]:
    """Compute the handedness induced by a declared signed-axis map."""
    validate_signed_axis_map(values, context=context)
    axes = [value.lstrip("-") for value in values]
    inversions = sum(
        axes[left] > axes[right] for left in range(3) for right in range(left + 1, 3)
    )
    negative_axes = sum(value.startswith("-") for value in values)
    determinant = -1 if (inversions + negative_axes) % 2 else 1
    source_sign = 1 if source == "right-handed" else -1
    return "right-handed" if determinant * source_sign > 0 else "left-handed"


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


class SemanticContractReference(NativeCaptureModel):
    """Manifest link to an experiment-owned semantic contract file."""

    file: str
    sha256: SHA256
