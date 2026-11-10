"""The common semantic-orientation to target-converter gateway."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Union

from ..semantics import (
    CoordinateFrame,
    FrameCorrespondence,
    PhysicalOrientation,
    PhysicalOrientationBatch,
    SemanticFrameBinding,
    SemanticQuaternionMap,
)
from ..semantics.roles import SEMANTIC_CONTRACTS
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


def _resolve_target_frame(
    frame: CoordinateFrame | SemanticFrameBinding,
    *,
    profile: ConverterProfile,
) -> tuple[CoordinateFrame, SemanticFrameBinding | None]:
    if isinstance(frame, CoordinateFrame):
        return frame, None
    if not isinstance(frame, SemanticFrameBinding):
        raise TypeError("target frames must be CoordinateFrame or SemanticFrameBinding")
    if frame.target_family is None:
        raise ValueError("semantic frame binding does not declare a target family")
    if frame.target_family.casefold() != profile.target.casefold():
        raise ValueError(
            f"semantic role {frame.contract_id!r} belongs to "
            f"{frame.target_family}, not {profile.target}"
        )
    contract = SEMANTIC_CONTRACTS[frame.contract_id]
    if profile.converter_id not in contract.converter_ids:
        raise ValueError(
            f"semantic role {frame.contract_id!r} is not registered for "
            f"converter {profile.converter_id!r}"
        )
    return frame.coordinate_frame, frame


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
    target_parent: CoordinateFrame | SemanticFrameBinding
    target_child: CoordinateFrame | SemanticFrameBinding
    _profile: ConverterProfile = field(init=False, repr=False, compare=False)
    _mapping: SemanticQuaternionMap = field(init=False, repr=False, compare=False)
    _target_role_bindings: tuple[
        SemanticFrameBinding | None, SemanticFrameBinding | None
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        profile = _profile(self.converter)
        target_parent, parent_binding = _resolve_target_frame(
            self.target_parent, profile=profile
        )
        target_child, child_binding = _resolve_target_frame(
            self.target_child, profile=profile
        )
        mapping = _frame_map(
            self.source_parent,
            self.source_child,
            target_parent,
            target_child,
        )
        object.__setattr__(self, "target_parent", target_parent)
        object.__setattr__(self, "target_child", target_child)
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_mapping", mapping)
        object.__setattr__(
            self, "_target_role_bindings", (parent_binding, child_binding)
        )

    @property
    def converter_id(self) -> str:
        """Return the resolved target profile ID."""
        return self._profile.converter_id

    @property
    def evidence_status(self) -> EvidenceStatus:
        """Return the selected profile's acceptance state without upgrading it."""
        return self._profile.evidence_status

    @property
    def target_semantic_bindings(
        self,
    ) -> tuple[SemanticFrameBinding | None, SemanticFrameBinding | None]:
        """Return the parent/child semantic bindings, preserving their context."""
        return self._target_role_bindings

    @property
    def target_semantic_binding_ids(self) -> tuple[str, ...]:
        """Return the IDs of authoritative roles used for target frames."""
        return tuple(
            binding.contract_id
            for binding in self._target_role_bindings
            if binding is not None
        )

    def to_target(self, orientation: PhysicalOrientation) -> TargetOrientation:
        """Convert one orientation after checking its configured source frames."""
        if not isinstance(orientation, PhysicalOrientation):
            raise TypeError("orientation must be a PhysicalOrientation")
        self._check_source_frames(orientation.parent_frame, orientation.child_frame)
        target = self._profile.to_target(
            orientation.to_framed_hamilton(), self._mapping
        )
        return replace(
            target,
            parent_semantic_binding=self._target_role_bindings[0],
            child_semantic_binding=self._target_role_bindings[1],
        )

    def from_target(self, orientation: TargetOrientation) -> PhysicalOrientation:
        """Convert one native value into the configured physical source frames."""
        if not isinstance(orientation, TargetOrientation):
            raise TypeError("orientation must be a TargetOrientation")
        self._check_target_bindings(
            orientation.parent_semantic_binding,
            orientation.child_semantic_binding,
        )
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
        target = self._profile.to_target_array(
            orientations.to_framed_batch(), self._mapping, component_axis
        )
        return replace(
            target,
            parent_semantic_binding=self._target_role_bindings[0],
            child_semantic_binding=self._target_role_bindings[1],
        )

    def from_target_batch(
        self,
        orientations: TargetOrientationBatch,
        component_axis: int | None = None,
    ) -> PhysicalOrientationBatch:
        """Convert a target batch into the configured physical source frames."""
        if not isinstance(orientations, TargetOrientationBatch):
            raise TypeError("orientations must be a TargetOrientationBatch")
        self._check_target_bindings(
            orientations.parent_semantic_binding,
            orientations.child_semantic_binding,
        )
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

    def _check_target_bindings(
        self,
        parent_binding: SemanticFrameBinding | None,
        child_binding: SemanticFrameBinding | None,
    ) -> None:
        expected_parent, expected_child = self._target_role_bindings
        if not _same_semantic_binding(parent_binding, expected_parent):
            raise ValueError(
                "target parent semantic binding does not match this converter"
            )
        if not _same_semantic_binding(child_binding, expected_child):
            raise ValueError(
                "target child semantic binding does not match this converter"
            )


def _same_semantic_binding(
    actual: SemanticFrameBinding | None,
    expected: SemanticFrameBinding | None,
) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if not (
        actual.contract_id == expected.contract_id
        and actual.authority == expected.authority
        and actual.evidence_reference == expected.evidence_reference
        and actual.target_family == expected.target_family
        and actual.coordinate_frame.has_identical_definition(expected.coordinate_frame)
        and actual.origin == expected.origin
        and actual.ellipsoid == expected.ellipsoid
    ):
        return False
    if actual.reference_frame is None or expected.reference_frame is None:
        if actual.reference_frame is not expected.reference_frame:
            return False
    elif not actual.reference_frame.has_identical_definition(expected.reference_frame):
        return False
    actual_alignment, expected_alignment = actual.alignment, expected.alignment
    if actual_alignment is None or expected_alignment is None:
        return actual_alignment is expected_alignment
    return (
        actual_alignment.reference_frame.has_identical_definition(
            expected_alignment.reference_frame
        )
        and actual_alignment.basis_to_reference.matrix
        == expected_alignment.basis_to_reference.matrix
        and actual_alignment.purpose == expected_alignment.purpose
        and actual_alignment.provider == expected_alignment.provider
        and actual_alignment.alignment_source == expected_alignment.alignment_source
        and actual_alignment.alignment_evidence_reference
        == expected_alignment.alignment_evidence_reference
        and actual_alignment.space_context == expected_alignment.space_context
    )


def convert_orientation_to_target(
    orientation: PhysicalOrientation,
    converter: Converter,
    target_parent: CoordinateFrame | SemanticFrameBinding,
    target_child: CoordinateFrame | SemanticFrameBinding,
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
    target_parent: CoordinateFrame | SemanticFrameBinding,
    target_child: CoordinateFrame | SemanticFrameBinding,
) -> PhysicalOrientation:
    """Convert one target value to a frame-complete physical orientation."""
    return OrientationConverter(
        converter, source_parent, source_child, target_parent, target_child
    ).from_target(orientation)


def convert_orientation_batch_to_target(
    orientations: PhysicalOrientationBatch,
    converter: Converter,
    target_parent: CoordinateFrame | SemanticFrameBinding,
    target_child: CoordinateFrame | SemanticFrameBinding,
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
    target_parent: CoordinateFrame | SemanticFrameBinding,
    target_child: CoordinateFrame | SemanticFrameBinding,
    component_axis: int | None = None,
) -> PhysicalOrientationBatch:
    """Convert a target batch to a frame-complete physical orientation batch."""
    return OrientationConverter(
        converter, source_parent, source_child, target_parent, target_child
    ).from_target_batch(orientations, component_axis)
