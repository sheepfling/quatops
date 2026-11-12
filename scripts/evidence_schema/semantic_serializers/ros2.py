"""Semantic serialization for the ROS 2 tf2 frame-chain experiment."""

from __future__ import annotations

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    Ros2Capture,
    Ros2Contract,
    RosTransform,
)
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
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> FrameChainSerialization:
    """Map tf2 lookups, stamped message fields, and point action semantics."""
    if not isinstance(package.contract, Ros2Contract):
        raise TypeError("ROS 2 serializer received a non-ROS contract")
    if not isinstance(package.target_capture, Ros2Capture):
        raise TypeError("ROS 2 serializer received a non-ROS capture")
    if (
        package.target.name != package.contract.target.name
        or package.target.version != package.contract.target.tf2_version
    ):
        raise ValueError("ROS 2 manifest target differs from semantic contract")
    _validate_ros_inputs(package.contract, package.target_capture, context)
    return FrameChainSerialization(
        layer="frame-transport",
        contract=serialize_ros2_contract(package.contract, package.target_capture),
        witnesses=serialize_ros2_witnesses(package.target_capture),
    )


def _validate_ros_inputs(
    contract: Ros2Contract,
    capture: Ros2Capture,
    context: SemanticSerializationContext,
) -> None:
    """Bind frame declarations and operation IDs to the shared input context."""
    numeric_fields = [
        capture.native_inputs.base_from_sensor.quaternion_xyzw,
        capture.native_inputs.base_from_sensor.translation_xyz,
        capture.native_inputs.world_from_base.quaternion_xyzw,
        capture.native_inputs.world_from_base.translation_xyz,
        capture.native_inputs.probe_in_source_xyz,
    ]
    for observation in capture.observations:
        outputs = observation.native_outputs
        numeric_fields.extend(
            [outputs.lookup.quaternion_xyzw, outputs.lookup.translation_xyz]
        )
        if outputs.lookup_message is not None:
            numeric_fields.extend(
                [
                    outputs.lookup_message.rotation_xyzw,
                    outputs.lookup_message.translation_xyz,
                ]
            )
        if outputs.message_roundtrip is not None:
            numeric_fields.extend(
                [
                    outputs.message_roundtrip.quaternion_xyzw,
                    outputs.message_roundtrip.translation_xyz,
                ]
            )
        if outputs.probe_in_target_xyz is not None:
            numeric_fields.append(outputs.probe_in_target_xyz)
    if any(field.dtype != "binary64" for field in numeric_fields):
        raise ValueError("ROS 2 capture numeric fields must be tagged binary64")

    canonical_axes = contract.frames.canonical_axes
    shared_frame = context.canonical_inputs.source_frame
    if (
        canonical_axes.handedness != "right"
        or shared_frame.handedness != "right-handed"
        or (
            canonical_axes.positive_x,
            canonical_axes.positive_y,
            canonical_axes.positive_z,
        )
        != (
            shared_frame.positive_x,
            shared_frame.positive_y,
            shared_frame.positive_z,
        )
    ):
        raise ValueError("ROS canonical frame differs from shared canonical inputs")
    if contract.rotation.algebra != context.canonical_inputs.contract.algebra:
        raise ValueError(
            "ROS quaternion algebra differs from shared canonical contract"
        )
    expected_apis = {
        "tf2::BufferCore::setTransform",
        "tf2::BufferCore::lookupTransform",
        "tf2::Transform",
        "geometry_msgs/msg/TransformStamped",
        "tf2::fromMsg",
    }
    if (
        len(set(contract.native_api)) != len(contract.native_api)
        or set(contract.native_api) != expected_apis
    ):
        raise ValueError("ROS native API list differs from the pinned capture scope")
    observations = {row.operation_id: row for row in capture.observations}
    if len(observations) != len(capture.observations) or set(observations) != {
        "lookup-world-from-sensor",
        "lookup-sensor-from-world",
    }:
        raise ValueError("ROS capture must contain the forward and reverse lookups")
    native = capture.native_inputs
    forward = observations["lookup-world-from-sensor"].native_outputs
    message = forward.lookup_message
    roundtrip = forward.message_roundtrip
    if message is None or roundtrip is None:
        raise ValueError("ROS forward lookup lacks its message roundtrip evidence")
    if (
        message.parent_frame != native.target_frame
        or message.child_frame != native.source_frame
        or (message.stamp_sec, message.stamp_nanosec) != (native.stamp_sec, 0)
    ):
        raise ValueError("ROS frame IDs or timestamp differ from the captured lookup")
    if (
        roundtrip.quaternion_xyzw.values != forward.lookup.quaternion_xyzw.values
        or roundtrip.translation_xyz.values != forward.lookup.translation_xyz.values
    ):
        raise ValueError("ROS message roundtrip differs from the forward lookup")
    for mapping in (
        contract.frames.ros_world_from_canonical_world,
        contract.frames.ros_base_from_canonical_base,
        contract.frames.ros_sensor_optical_from_canonical_sensor,
    ):
        signed_axis_map(mapping)


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
    if len(by_operation) != len(capture.observations) or set(by_operation) != {
        "lookup-world-from-sensor",
        "lookup-sensor-from-world",
    }:
        raise ValueError("ROS capture has duplicate or unexpected operation IDs")
    forward = by_operation["lookup-world-from-sensor"]
    reverse = by_operation["lookup-sensor-from-world"]
    inputs = capture.native_inputs

    def frame_transform(
        transform: RosTransform, parent: str, child: str
    ) -> FrameTransform:
        return FrameTransform(
            parent_frame=parent,
            child_frame=child,
            quaternion=transform.quaternion_xyzw.values,
            coefficient_order="xyzw",
            translation_xyz=transform.translation_xyz.values,
        )

    probe_output = forward.native_outputs.probe_in_target_xyz
    if probe_output is None:
        raise ValueError("forward ROS lookup lacks its transformed point")
    message = forward.native_outputs.lookup_message
    if message is None:
        raise ValueError("forward ROS lookup lacks TransformStamped fields")
    roundtrip = forward.native_outputs.message_roundtrip
    if roundtrip is None:
        raise ValueError("forward ROS lookup lacks its message round trip")
    return [
        FrameChainWitness(
            kind="frame_chain",
            case_id="world-base-sensor",
            intermediate_transforms=[
                frame_transform(inputs.world_from_base, inputs.target_frame, "base"),
                frame_transform(inputs.base_from_sensor, "base", inputs.source_frame),
            ],
            lookups=[
                frame_transform(
                    forward.native_outputs.lookup,
                    inputs.target_frame,
                    inputs.source_frame,
                ),
                frame_transform(
                    reverse.native_outputs.lookup,
                    inputs.source_frame,
                    inputs.target_frame,
                ),
            ],
            messages=[
                FrameMessageWitness(
                    parent_frame=message.parent_frame,
                    child_frame=message.child_frame,
                    stamp_sec=message.stamp_sec,
                    stamp_nanosec=message.stamp_nanosec,
                    quaternion=message.rotation_xyzw.values,
                    coefficient_order="xyzw",
                    translation_xyz=message.translation_xyz.values,
                )
            ],
            point_actions=[
                PointAction(
                    source_frame=inputs.source_frame,
                    target_frame=inputs.target_frame,
                    input_xyz=inputs.probe_in_source_xyz.values,
                    output_xyz=probe_output.values,
                )
            ],
            message_roundtrips=[
                frame_transform(roundtrip, inputs.target_frame, inputs.source_frame)
            ],
        )
    ]
