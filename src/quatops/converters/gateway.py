"""The common semantic-orientation to target-converter gateway."""

from __future__ import annotations

from typing import Union

from ..semantics import (
    CoordinateFrame,
    FrameCorrespondence,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    SemanticQuaternionMap,
)
from .targets import (
    ConverterProfile,
    TargetOrientation,
    TargetOrientationBatch,
    get_converter,
)

Converter = Union[str, ConverterProfile]


def _profile(converter: Converter) -> ConverterProfile:
    if isinstance(converter, str):
        return get_converter(converter)
    if isinstance(converter, ConverterProfile):
        return converter
    raise TypeError("converter must be a converter ID or ConverterProfile")


def _frame_map(
    source_parent: CoordinateFrame,
    source_child: CoordinateFrame,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
) -> SemanticQuaternionMap:
    return SemanticQuaternionMap(
        world=FrameCorrespondence(source_parent, target_parent),
        body=FrameCorrespondence(source_child, target_child),
    )


def convert_orientation_to_target(
    orientation: PhysicalOrientation,
    converter: Converter,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
) -> TargetOrientation:
    """Convert one physical orientation through the shared target gateway."""
    if not isinstance(orientation, PhysicalOrientation):
        raise TypeError("orientation must be a PhysicalOrientation")
    frame_map = _frame_map(
        orientation.parent_frame,
        orientation.child_frame,
        target_parent,
        target_child,
    )
    return _profile(converter).to_target(orientation.to_framed_hamilton(), frame_map)


def convert_orientation_from_target(
    orientation: TargetOrientation,
    converter: Converter,
    source_parent: CoordinateFrame,
    source_child: CoordinateFrame,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
) -> PhysicalOrientation:
    """Convert one target value to a frame-complete physical orientation."""
    frame_map = _frame_map(source_parent, source_child, target_parent, target_child)
    canonical = _profile(converter).from_target(orientation, frame_map)
    return PhysicalOrientation.from_framed_hamilton(
        canonical,
        parent_frame=source_parent,
        child_frame=source_child,
    )


def convert_orientation_batch_to_target(
    orientations: PhysicalOrientationBatch,
    converter: Converter,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
    component_axis: int = -1,
) -> TargetOrientationBatch:
    """Convert a physical orientation batch through the same gateway."""
    if not isinstance(orientations, PhysicalOrientationBatch):
        raise TypeError("orientations must be a PhysicalOrientationBatch")
    from .numpy import _require_unit_quaternion_batch

    _require_unit_quaternion_batch(orientations.values, component_axis)
    frame_map = _frame_map(
        orientations.parent_frame,
        orientations.child_frame,
        target_parent,
        target_child,
    )
    return _profile(converter).to_target_array(
        orientations.to_framed_batch(), frame_map, component_axis
    )


def convert_orientation_batch_from_target(
    orientations: TargetOrientationBatch,
    converter: Converter,
    source_parent: CoordinateFrame,
    source_child: CoordinateFrame,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
    component_axis: int = -1,
) -> PhysicalOrientationBatch:
    """Convert a target batch to a frame-complete physical orientation batch."""
    frame_map = _frame_map(source_parent, source_child, target_parent, target_child)
    canonical = _profile(converter).from_target_array(
        orientations, frame_map, component_axis
    )
    from .numpy import _require_unit_quaternion_batch

    _require_unit_quaternion_batch(canonical.values, component_axis)
    return PhysicalOrientationBatch(
        values=canonical.values,
        parent_frame=source_parent,
        child_frame=source_child,
    )
