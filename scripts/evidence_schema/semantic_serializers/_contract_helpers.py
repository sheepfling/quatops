"""Shared helpers for translating declared native contracts into common shapes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal, Union

from scripts.evidence_schema.converter_captures import (
    EigenBoundaryContract,
    EigenContract,
    MuJoCoBoundaryContract,
    MuJoCoContract,
    OpenCVContract,
    SpiceyPyBoundaryContract,
    SpiceyPyContract,
)
from scripts.evidence_schema.crystallized import (
    CoordinateFrame,
    FrameSemantics,
    canonical_source_frame_semantics,
)

_QuaternionBoundaryTarget = Union[
    EigenContract, MuJoCoContract, OpenCVContract, SpiceyPyContract
]
_NativeQuaternionBoundary = Union[
    EigenBoundaryContract, MuJoCoBoundaryContract, SpiceyPyBoundaryContract
]


def coefficient_order(value: str) -> Literal["wxyz", "xyzw"]:
    normalized = value.lower().replace(" ", "")
    if normalized == "wxyz":
        return "wxyz"
    if normalized == "xyzw":
        return "xyzw"
    raise ValueError(f"unsupported declared quaternion order: {value!r}")


def handedness(value: str) -> Literal["right-handed", "left-handed"]:
    normalized = value.lower()
    if normalized in {"right", "right-handed"}:
        return "right-handed"
    if normalized in {"left", "left-handed"}:
        return "left-handed"
    raise ValueError(f"unsupported declared handedness: {value!r}")


def coordinate_frame(name: str, handedness_value: str, axes: object) -> CoordinateFrame:
    if not isinstance(axes, Mapping):
        raise TypeError(f"frame {name!r} must declare its positive axes")
    return CoordinateFrame(
        name=name,
        handedness=handedness(handedness_value),
        positive_x=str(axes["positive_x"]),
        positive_y=str(axes["positive_y"]),
        positive_z=str(axes["positive_z"]),
    )


def signed_axis_map(values: list[str]) -> list[str]:
    if len(values) != 3 or any(
        value not in {"x", "-x", "y", "-y", "z", "-z"} for value in values
    ):
        raise ValueError(f"invalid declared signed-axis map: {values!r}")
    if len({value.lstrip("-") for value in values}) != 3:
        raise ValueError(f"axis map must be a signed permutation: {values!r}")
    return values


def canonical_frame(contract: _QuaternionBoundaryTarget) -> CoordinateFrame:
    canonical = contract.canonical
    return coordinate_frame(
        "QuatOps canonical frame",
        canonical.handedness,
        canonical.frame.model_dump(),
    )


def probe_frame(contract: _QuaternionBoundaryTarget) -> CoordinateFrame:
    canonical = contract.canonical
    probe = contract.target_boundary.probe_frame_assignment
    canonical_axes = canonical.frame.model_dump()
    probe_axes = probe.model_dump(exclude={"meaning"})
    if any(canonical_axes[key] != probe_axes[key] for key in canonical_axes):
        raise ValueError("target probe frame differs from its declared canonical frame")
    return coordinate_frame(
        f"{contract.target.name} explicit numeric probe frame",
        canonical.handedness,
        probe_axes,
    )


def canonical_frame_semantics() -> FrameSemantics:
    return canonical_source_frame_semantics()


def numeric_probe_semantics(statement: str) -> FrameSemantics:
    return FrameSemantics(
        role="numeric_probe",
        authority="fixture_only",
        statement=statement,
    )


def canonical_order(
    contract: _QuaternionBoundaryTarget,
) -> Literal["wxyz", "xyzw"]:
    order = coefficient_order(contract.canonical.coefficient_order)
    if contract.canonical.algebra.lower() != "hamilton":
        raise ValueError("converter contract does not declare Hamilton algebra")
    if not contract.canonical.rotation_action.lower().startswith("active"):
        raise ValueError("converter contract does not declare active rotation action")
    if handedness(contract.canonical.handedness) != "right-handed":
        raise ValueError("this converter product requires an explicit RH source frame")
    return order


def native_order(boundary: _NativeQuaternionBoundary) -> Literal["wxyz", "xyzw"]:
    declared_order = getattr(boundary, "coefficient_order", None)
    if isinstance(declared_order, str):
        return coefficient_order(declared_order)

    constructor = getattr(boundary, "constructor_order", None)
    if isinstance(constructor, str):
        components = "".join(re.findall(r"\b([wxyz])\b", constructor, re.IGNORECASE))
        if components:
            return coefficient_order(components)
    raise ValueError(
        "target contract does not explicitly declare native coefficient order"
    )
