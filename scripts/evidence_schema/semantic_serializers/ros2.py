"""Semantic serialization for the ROS 2 tf2 frame-chain experiment."""

from __future__ import annotations

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import Ros2Capture, Ros2Contract
from scripts.evidence_schema.crystallized import (
    FrameAxisMapping,
    FrameChainContract,
    FrameChainSerialization,
    FrameChainWitness,
    FrameMessageWitness,
    FrameSemantics,
    FrameTransform,
    PointAction,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    coefficient_order,
    coordinate_frame,
    signed_axis_map,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


def serialize(
    package: ConverterEvidencePackage, _context: SemanticSerializationContext
) -> FrameChainSerialization:
    """Map tf2 lookups, stamped message fields, and point action semantics."""
    if not isinstance(package.contract, Ros2Contract):
        raise TypeError("ROS 2 serializer received a non-ROS contract")
    if not isinstance(package.target_capture, Ros2Capture):
        raise TypeError("ROS 2 serializer received a non-ROS capture")
    return FrameChainSerialization(
        layer="frame-transport",
        contract=serialize_ros2_contract(package.contract, package.target_capture),
        witnesses=serialize_ros2_witnesses(package.target_capture),
    )


def serialize_ros2_contract(
    contract: Ros2Contract, capture: Ros2Capture
) -> FrameChainContract:
    """Map tf2's declared frame chain to the shared frame contract."""
    native_inputs = capture.native_inputs
    return FrameChainContract(
        kind="frame_chain",
        canonical_frame=coordinate_frame(
            "QuatOps canonical frame",
            contract.frames.canonical_axes.handedness,
            contract.frames.canonical_axes.model_dump(),
        ),
        canonical_frame_semantics=canonical_source_frame_semantics(),
        quaternion_order=coefficient_order(contract.rotation.native_order),
        vector_action=contract.rotation.action,
        lookup_direction=contract.frame_direction,
        frame_mappings=[
            FrameAxisMapping(
                frame=native_inputs.target_frame,
                axes_from_canonical=signed_axis_map(
                    contract.frames.ros_world_from_canonical_world
                ),
                semantics=FrameSemantics(
                    role="world_frame_id",
                    authority="fixture_only",
                    statement=(
                        "The frame ID and signed-axis mapping are assigned by "
                        "this capture fixture; the ID alone does not establish "
                        "physical world axes."
                    ),
                ),
            ),
            FrameAxisMapping(
                frame="base",
                axes_from_canonical=signed_axis_map(
                    contract.frames.ros_base_from_canonical_base
                ),
                semantics=FrameSemantics(
                    role="base_frame_id",
                    authority="fixture_only",
                    statement=(
                        "The base frame and its signed-axis mapping are assigned "
                        "by this capture fixture."
                    ),
                ),
            ),
            FrameAxisMapping(
                frame=native_inputs.source_frame,
                axes_from_canonical=signed_axis_map(
                    contract.frames.ros_sensor_optical_from_canonical_sensor
                ),
                semantics=FrameSemantics(
                    role="sensor_frame_id",
                    authority="fixture_only",
                    statement=(
                        "The sensor frame ID and its signed-axis mapping are "
                        "assigned by this capture fixture; no physical mounting "
                        "is inferred."
                    ),
                ),
            ),
        ],
        scope=[
            "named-frame transform direction",
            "two-edge noncommuting frame chain",
            "forward and reverse lookup",
            "message-field round trip",
            "native point action",
        ],
        excludes=contract.exclusions,
    )


def serialize_ros2_witnesses(capture: Ros2Capture) -> list[FrameChainWitness]:
    by_operation = {row.operation_id: row for row in capture.observations}
    forward = by_operation["lookup-world-from-sensor"]
    reverse = by_operation["lookup-sensor-from-world"]
    inputs = capture.native_inputs

    def input_transform(name: str, parent: str, child: str) -> FrameTransform:
        transform = getattr(inputs, name)
        return FrameTransform(
            parent_frame=parent,
            child_frame=child,
            quaternion=transform.quaternion_xyzw.values,
            coefficient_order="xyzw",
            translation_xyz=transform.translation_xyz.values,
        )

    def lookup_transform(row) -> FrameTransform:
        transform = row.native_outputs.lookup
        parent, child = (
            (inputs.target_frame, inputs.source_frame)
            if row.operation_id == "lookup-world-from-sensor"
            else (inputs.source_frame, inputs.target_frame)
        )
        return FrameTransform(
            parent_frame=parent,
            child_frame=child,
            quaternion=transform.quaternion_xyzw.values,
            coefficient_order="xyzw",
            translation_xyz=transform.translation_xyz.values,
        )

    def message_roundtrip(row) -> FrameTransform:
        transform = row.native_outputs.message_roundtrip
        if transform is None:
            raise ValueError("forward ROS lookup lacks its message round trip")
        return FrameTransform(
            parent_frame=inputs.target_frame,
            child_frame=inputs.source_frame,
            quaternion=transform.quaternion_xyzw.values,
            coefficient_order="xyzw",
            translation_xyz=transform.translation_xyz.values,
        )

    def message_fields(row) -> FrameMessageWitness:
        message = row.native_outputs.lookup_message
        if message is None:
            raise ValueError("forward ROS lookup lacks TransformStamped fields")
        return FrameMessageWitness(
            parent_frame=message.parent_frame,
            child_frame=message.child_frame,
            stamp_sec=message.stamp_sec,
            stamp_nanosec=message.stamp_nanosec,
            quaternion=message.rotation_xyzw.values,
            coefficient_order="xyzw",
            translation_xyz=message.translation_xyz.values,
        )

    probe_output = forward.native_outputs.probe_in_target_xyz
    if probe_output is None:
        raise ValueError("forward ROS lookup lacks its transformed point")
    point_actions = [
        PointAction(
            source_frame=inputs.source_frame,
            target_frame=inputs.target_frame,
            input_xyz=inputs.probe_in_source_xyz.values,
            output_xyz=probe_output.values,
        )
    ]
    return [
        FrameChainWitness(
            kind="frame_chain",
            case_id="world-base-sensor",
            intermediate_transforms=[
                input_transform("world_from_base", inputs.target_frame, "base"),
                input_transform("base_from_sensor", "base", inputs.source_frame),
            ],
            lookups=[lookup_transform(forward), lookup_transform(reverse)],
            messages=[message_fields(forward)],
            point_actions=point_actions,
            message_roundtrips=[message_roundtrip(forward)],
        )
    ]
