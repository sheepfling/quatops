"""Strict schemas for the complete stdout emitted by the ROS 2 native exporter."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, model_validator

from scripts.evidence_schema.primitives import (
    FiniteInputNumber,
    NativeCaptureModel,
)

FiniteVector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
FiniteQuaternion4 = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
FiniteMatrix3 = Annotated[list[FiniteVector3], Field(min_length=3, max_length=3)]
InputVector3 = Annotated[list[FiniteInputNumber], Field(min_length=3, max_length=3)]


def reject_lost_nonfinite_values(value: object) -> None:
    """Reject nulls emitted by nlohmann for non-finite native numbers."""
    if value is None:
        raise ValueError(
            "Native ROS stdout contains null; nlohmann JSON maps NaN/Infinity to null, "
            "so the original non-finite value is lost and cannot be captured safely"
        )
    if isinstance(value, dict):
        for item in value.values():
            reject_lost_nonfinite_values(item)
    elif isinstance(value, list):
        for item in value:
            reject_lost_nonfinite_values(item)


class RosFullTransform(NativeCaptureModel):
    quaternion_xyzw: FiniteQuaternion4
    translation_xyz: FiniteVector3
    matrix_rows: FiniteMatrix3


class RosChainEdge(NativeCaptureModel):
    rpy_radians: InputVector3
    translation_xyz: InputVector3


class RosChainInputs(NativeCaptureModel):
    case_id: Literal["world-base-sensor"]
    world_from_base: RosChainEdge
    base_from_sensor: RosChainEdge
    probe_in_source_xyz: InputVector3
    stamp_sec: int


class RosLookupInputs(NativeCaptureModel):
    target_frame: str
    source_frame: str
    stamp_sec: int
    world_from_base: RosFullTransform
    base_from_sensor: RosFullTransform
    probe_in_source_xyz: FiniteVector3
    world_from_base_rpy: InputVector3
    base_from_sensor_rpy: InputVector3


class RosLookupMessage(NativeCaptureModel):
    parent_frame: str
    child_frame: str
    stamp_sec: int
    stamp_nanosec: int
    translation_xyz: FiniteVector3
    rotation_xyzw: FiniteQuaternion4


class RosLookupOutputs(NativeCaptureModel):
    lookup_message: RosLookupMessage
    lookup: RosFullTransform
    message_roundtrip: RosFullTransform
    probe_in_target_xyz: FiniteVector3
    direct_world_from_sensor: RosFullTransform
    direct_sensor_from_world: RosFullTransform
    world_probe_sequential: FiniteVector3


class RosLookupObservation(NativeCaptureModel):
    case_id: str
    operation_id: Literal["lookup-world-from-sensor", "lookup-sensor-from-world"]
    native_entry_point: str
    native_inputs: RosLookupInputs
    native_outputs: RosLookupOutputs


class RosFullCapture(NativeCaptureModel):
    native_scalar_bytes: int
    compiler: str
    cpp_standard: int
    flags: str
    observations: list[RosLookupObservation]

    @model_validator(mode="after")
    def validate_operation_identity(self) -> RosFullCapture:
        keys = [(row.case_id, row.operation_id) for row in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("ROS stdout contains duplicate (case_id, operation_id)")
        for row in self.observations:
            expected = (
                ("world", "sensor")
                if row.operation_id == "lookup-world-from-sensor"
                else ("sensor", "world")
            )
            if (
                row.native_inputs.target_frame,
                row.native_inputs.source_frame,
            ) != expected:
                raise ValueError(
                    f"{row.operation_id} frame direction does not match its operation"
                )
            message = row.native_outputs.lookup_message
            if (message.parent_frame, message.child_frame) != expected:
                raise ValueError(
                    f"{row.operation_id} message frames do not match its lookup"
                )
            if (
                message.stamp_sec != row.native_inputs.stamp_sec
                or message.stamp_nanosec != 0
            ):
                raise ValueError(
                    f"{row.operation_id} message timestamp does not match its input"
                )
            lookup = row.native_outputs.lookup
            if message.translation_xyz != lookup.translation_xyz or not any(
                all(
                    math.isclose(left, sign * right, rel_tol=0.0, abs_tol=2e-12)
                    for left, right in zip(
                        message.rotation_xyzw, lookup.quaternion_xyzw
                    )
                )
                for sign in (-1.0, 1.0)
            ):
                raise ValueError(
                    f"{row.operation_id} message fields differ from the lookup"
                )
        lookups = self.observations
        lookup_ops = {row.operation_id for row in lookups}
        required = {"lookup-world-from-sensor", "lookup-sensor-from-world"}
        if lookup_ops != required:
            raise ValueError(
                f"ROS stdout must contain both lookup directions: {sorted(required)}"
            )
        if len({row.case_id for row in lookups}) != 1:
            raise ValueError("ROS lookup directions must belong to one case ID")
        chain_signatures = {
            (
                row.native_inputs.stamp_sec,
                row.native_inputs.world_from_base.model_dump_json(),
                row.native_inputs.base_from_sensor.model_dump_json(),
                tuple(row.native_inputs.probe_in_source_xyz),
            )
            for row in lookups
        }
        if len(chain_signatures) != 1:
            raise ValueError(
                "ROS lookup directions must share the same chain, probe and timestamp"
            )
        return self


def validate_ros_capture_inputs(
    capture: RosFullCapture, inputs: RosChainInputs
) -> None:
    """Link both native frame-chain observations to the exact declared inputs."""
    for row in capture.observations:
        native = row.native_inputs
        if (
            row.case_id != inputs.case_id
            or native.stamp_sec != inputs.stamp_sec
            or native.world_from_base_rpy != inputs.world_from_base.rpy_radians
            or native.base_from_sensor_rpy != inputs.base_from_sensor.rpy_radians
            or native.world_from_base.translation_xyz
            != inputs.world_from_base.translation_xyz
            or native.base_from_sensor.translation_xyz
            != inputs.base_from_sensor.translation_xyz
            or native.probe_in_source_xyz != inputs.probe_in_source_xyz
        ):
            raise ValueError("ROS native chain inputs differ from the declared recipe")
