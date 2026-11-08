"""The common semantic-orientation to target-converter gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    EvidenceStatus,
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


@dataclass(frozen=True)
class OrientationConverter:
    """Reusable target conversion with explicit source and target frame pairs.

    Construct one instance per application frame mapping, then reuse it for
    scalar and batch values. It resolves the target profile and compiles the
    parent/child coordinate maps once. No axes are inferred from the converter
    ID.
    """

    converter: Converter
    source_parent: CoordinateFrame
    source_child: CoordinateFrame
    target_parent: CoordinateFrame
    target_child: CoordinateFrame
    _profile: ConverterProfile = field(init=False, repr=False, compare=False)
    _mapping: SemanticQuaternionMap = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        profile = _profile(self.converter)
        mapping = _frame_map(
            self.source_parent,
            self.source_child,
            self.target_parent,
            self.target_child,
        )
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_mapping", mapping)

    @property
    def converter_id(self) -> str:
        """Return the resolved target profile ID."""
        return self._profile.converter_id

    @property
    def evidence_status(self) -> EvidenceStatus:
        """Return the selected profile's acceptance state without upgrading it."""
        return self._profile.evidence_status

    def to_target(self, orientation: PhysicalOrientation) -> TargetOrientation:
        """Convert one orientation after checking its configured source frames."""
        if not isinstance(orientation, PhysicalOrientation):
            raise TypeError("orientation must be a PhysicalOrientation")
        self._check_source_frames(orientation.parent_frame, orientation.child_frame)
        return self._profile.to_target(orientation.to_framed_hamilton(), self._mapping)

    def from_target(self, orientation: TargetOrientation) -> PhysicalOrientation:
        """Convert one native value into the configured physical source frames."""
        canonical = self._profile.from_target(orientation, self._mapping)
        return PhysicalOrientation.from_framed_hamilton(
            canonical,
            parent_frame=self.source_parent,
            child_frame=self.source_child,
        )

    def to_target_batch(
        self,
        orientations: PhysicalOrientationBatch,
        component_axis: int = -1,
    ) -> TargetOrientationBatch:
        """Convert a batch through the precompiled target and frame mapping."""
        if not isinstance(orientations, PhysicalOrientationBatch):
            raise TypeError("orientations must be a PhysicalOrientationBatch")
        self._check_source_frames(orientations.parent_frame, orientations.child_frame)
        from .numpy import _require_unit_quaternion_batch

        _require_unit_quaternion_batch(orientations.values, component_axis)
        return self._profile.to_target_array(
            orientations.to_framed_batch(), self._mapping, component_axis
        )

    def from_target_batch(
        self,
        orientations: TargetOrientationBatch,
        component_axis: int | None = None,
    ) -> PhysicalOrientationBatch:
        """Convert a target batch into the configured physical source frames."""
        canonical = self._profile.from_target_array(
            orientations, self._mapping, component_axis
        )
        from .numpy import _require_unit_quaternion_batch

        _require_unit_quaternion_batch(
            canonical.values,
            orientations.component_axis if component_axis is None else component_axis,
        )
        return PhysicalOrientationBatch(
            values=canonical.values,
            parent_frame=self.source_parent,
            child_frame=self.source_child,
        )

    def _check_source_frames(
        self, parent_frame: CoordinateFrame, child_frame: CoordinateFrame
    ) -> None:
        if not self.source_parent.has_same_definition(parent_frame):
            raise ValueError("orientation parent frame does not match this converter")
        if not self.source_child.has_same_definition(child_frame):
            raise ValueError("orientation child frame does not match this converter")


def convert_orientation_to_target(
    orientation: PhysicalOrientation,
    converter: Converter,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
) -> TargetOrientation:
    """Convert one physical orientation through the shared target gateway."""
    if not isinstance(orientation, PhysicalOrientation):
        raise TypeError("orientation must be a PhysicalOrientation")
    converter_context = OrientationConverter(
        converter,
        orientation.parent_frame,
        orientation.child_frame,
        target_parent,
        target_child,
    )
    return converter_context.to_target(orientation)


def convert_orientation_from_target(
    orientation: TargetOrientation,
    converter: Converter,
    source_parent: CoordinateFrame,
    source_child: CoordinateFrame,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
) -> PhysicalOrientation:
    """Convert one target value to a frame-complete physical orientation."""
    return OrientationConverter(
        converter, source_parent, source_child, target_parent, target_child
    ).from_target(orientation)


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
    return OrientationConverter(
        converter,
        orientations.parent_frame,
        orientations.child_frame,
        target_parent,
        target_child,
    ).to_target_batch(orientations, component_axis)


def convert_orientation_batch_from_target(
    orientations: TargetOrientationBatch,
    converter: Converter,
    source_parent: CoordinateFrame,
    source_child: CoordinateFrame,
    target_parent: CoordinateFrame,
    target_child: CoordinateFrame,
    component_axis: int | None = None,
) -> PhysicalOrientationBatch:
    """Convert a target batch to a frame-complete physical orientation batch."""
    return OrientationConverter(
        converter, source_parent, source_child, target_parent, target_child
    ).from_target_batch(orientations, component_axis)
