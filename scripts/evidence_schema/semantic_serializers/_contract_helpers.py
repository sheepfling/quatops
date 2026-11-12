"""Shared helpers for translating declared native contracts into common shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, Union

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
)
from scripts.evidence_schema.primitives import (
    AxisMeaning,
    FrameHandednessLabel,
    QuaternionCoefficientOrder,
    SignedAxis,
    mapped_handedness,
    validate_signed_axis_map,
)


class _FrameAxes(Protocol):
    handedness: FrameHandednessLabel
    positive_x: AxisMeaning
    positive_y: AxisMeaning
    positive_z: AxisMeaning


_QuaternionBoundaryTarget = Union[
    EigenContract, MuJoCoContract, OpenCVContract, SpiceyPyContract
]
_NativeQuaternionBoundary = Union[
    EigenBoundaryContract, MuJoCoBoundaryContract, SpiceyPyBoundaryContract
]


def coefficient_order(
    value: QuaternionCoefficientOrder,
) -> Literal["wxyz", "xyzw"]:
    normalized = value.lower().replace(" ", "")
    if normalized == "wxyz":
        return "wxyz"
    if normalized == "xyzw":
        return "xyzw"
    raise ValueError(f"unsupported declared quaternion order: {value!r}")


def handedness(
    value: FrameHandednessLabel,
) -> Literal["right-handed", "left-handed"]:
    normalized = value.lower()
    if normalized in {"right", "right-handed"}:
        return "right-handed"
    if normalized in {"left", "left-handed"}:
        return "left-handed"
    raise ValueError(f"unsupported declared handedness: {value!r}")


def coordinate_frame(
    name: str,
    handedness_value: FrameHandednessLabel,
    axes: Mapping[str, AxisMeaning],
) -> CoordinateFrame:
    if not isinstance(axes, Mapping):
        raise TypeError(f"frame {name!r} must declare its positive axes")
    return CoordinateFrame(
        name=name,
        handedness=handedness(handedness_value),
        positive_x=axes["positive_x"],
        positive_y=axes["positive_y"],
        positive_z=axes["positive_z"],
    )


def signed_axis_map(values: list[SignedAxis]) -> list[SignedAxis]:
    validate_signed_axis_map(values, context="declared axis map")
    return values


def frame_axes(frame: _FrameAxes) -> tuple[AxisMeaning, AxisMeaning, AxisMeaning]:
    """Return the three positive-axis meanings from a validated frame model."""
    return frame.positive_x, frame.positive_y, frame.positive_z


def frame_signature(frame: _FrameAxes) -> tuple[object, ...]:
    """Return the semantic frame identity shared by capture and contract models."""
    return (
        frame.handedness,
        *frame_axes(frame),
        getattr(frame, "forward", None),
        getattr(frame, "upAxis", None),
    )


_AXIS_OPPOSITES: dict[AxisMeaning, AxisMeaning] = {
    "right": "left",
    "left": "right",
    "east": "west",
    "west": "east",
    "forward": "backward",
    "back": "forward",
    "backward": "forward",
    "north": "south",
    "south": "north",
    "up": "down",
    "down": "up",
    "ellipsoid normal": "ellipsoid inward normal",
    "ellipsoid inward normal": "ellipsoid normal",
}


def opposite_axis_meaning(value: AxisMeaning) -> AxisMeaning:
    """Return the declared positive meaning opposite to one frame-axis meaning."""
    return _AXIS_OPPOSITES[value]


def derive_signed_axis_map(
    source: Sequence[AxisMeaning], destination: Sequence[AxisMeaning]
) -> list[SignedAxis]:
    """Derive a basis map only from explicit, corresponding axis meanings."""
    if len(source) != 3 or len(destination) != 3:
        raise ValueError("source and destination frames must each name three axes")
    source_tokens = ("x", "y", "z")
    source_meanings: list[AxisMeaning] = [
        "backward" if meaning == "back" else meaning for meaning in source
    ]
    mapping: list[SignedAxis] = []
    for destination_meaning in destination:
        meaning: AxisMeaning = (
            "backward" if destination_meaning == "back" else destination_meaning
        )
        if meaning in source_meanings:
            index = source_meanings.index(meaning)
            mapping.append(source_tokens[index])
            continue
        inverse = opposite_axis_meaning(meaning)
        if inverse not in source_meanings:
            raise ValueError(
                "cannot derive an axis map for destination meaning "
                f"{destination_meaning!r}"
            )
        index = source_meanings.index(inverse)
        mapping.append(f"-{source_tokens[index]}")
    validate_signed_axis_map(mapping, context="derived frame axis map")
    return mapping


def map_vector_components(
    values: Sequence[float], axes: Sequence[SignedAxis]
) -> list[float]:
    """Map a vector into destination coordinates under a signed-axis triple."""
    if len(values) != 3:
        raise ValueError("a mapped vector must have three components")
    validate_signed_axis_map(axes, context="declared axis map")
    source_index = {"x": 0, "y": 1, "z": 2}
    return [
        (-1.0 if axis.startswith("-") else 1.0) * values[source_index[axis.lstrip("-")]]
        for axis in axes
    ]


def map_hamilton_quaternion_wxyz(
    values: Sequence[float], axes: Sequence[SignedAxis]
) -> list[float]:
    """Map active Hamilton WXYZ coefficients through one orthogonal basis map."""
    if len(values) != 4:
        raise ValueError("a mapped quaternion must have four components")
    determinant = (
        1.0
        if mapped_handedness(
            "right-handed", axes, context="declared quaternion axis map"
        )
        == "right-handed"
        else -1.0
    )
    mapped_vector = map_vector_components(values[1:], axes)
    return [values[0], *(determinant * value for value in mapped_vector)]


def validate_mapped_vector(
    actual: Sequence[float],
    source: Sequence[float],
    axes: Sequence[SignedAxis],
    *,
    context: str,
) -> None:
    """Reject target input vectors that disagree with their declared axis map."""
    expected = map_vector_components(source, axes)
    if list(actual) != expected:
        raise ValueError(f"{context} does not match its declared signed-axis map")


def validate_mapped_vectors(
    actual: Sequence[Sequence[float]],
    source: Sequence[Sequence[float]],
    axes: Sequence[SignedAxis],
    *,
    context: str,
) -> None:
    """Validate a target vector corpus against one signed-axis map."""
    if len(actual) != len(source):
        raise ValueError(f"{context} has a different vector count from its source")
    for index, (actual_vector, source_vector) in enumerate(zip(actual, source)):
        validate_mapped_vector(
            actual_vector,
            source_vector,
            axes,
            context=f"{context} vector {index}",
        )


def validate_mapped_quaternion_wxyz(
    actual: Sequence[float],
    source: Sequence[float],
    axes: Sequence[SignedAxis],
    *,
    context: str,
) -> None:
    """Reject target quaternion inputs that disagree with the declared basis map."""
    expected = map_hamilton_quaternion_wxyz(source, axes)
    if list(actual) != expected:
        raise ValueError(f"{context} does not match its declared signed-axis map")


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
    boundary = contract.target_boundary
    axis_map = (
        boundary.coordinate_map
        if isinstance(contract, OpenCVContract)
        else boundary.source_to_target_coordinate_map
    )
    if axis_map != ["x", "y", "z"]:
        raise ValueError(
            "target probe frame matches canonical axes but its map is not identity"
        )
    return coordinate_frame(
        f"{contract.target.name} explicit numeric probe frame",
        canonical.handedness,
        probe_axes,
    )


def numeric_probe_semantics(statement: str) -> FrameSemantics:
    return FrameSemantics(
        role="numeric_probe",
        authority="fixture_only",
        statement=statement,
    )


def canonical_order(
    contract: _QuaternionBoundaryTarget,
) -> Literal["wxyz", "xyzw"]:
    return coefficient_order(contract.canonical.coefficient_order)


def native_order(boundary: _NativeQuaternionBoundary) -> Literal["wxyz", "xyzw"]:
    if not isinstance(boundary, EigenBoundaryContract):
        return coefficient_order(boundary.coefficient_order)
    return "wxyz"
