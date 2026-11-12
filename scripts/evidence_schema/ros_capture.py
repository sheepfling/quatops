"""Typed validation for the selected ROS 2 native frame-chain capture."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Optional

from pydantic import Field, FiniteFloat, model_validator

from scripts.evidence_schema.primitives import FiniteInputNumber, NativeCaptureModel

Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
InputVector3 = Annotated[list[FiniteInputNumber], Field(min_length=3, max_length=3)]


class RosChainEdge(NativeCaptureModel):
    rpy_radians: InputVector3
    translation_xyz: InputVector3


class RosChainInputs(NativeCaptureModel):
    case_id: Literal["world-base-sensor"]
    world_from_base: RosChainEdge
    base_from_sensor: RosChainEdge
    probe_in_source_xyz: InputVector3
    stamp_sec: int


class RosTransform(NativeCaptureModel):
    quaternion_xyzw: Quaternion
    translation_xyz: Vector3


class RosNativeInputs(NativeCaptureModel):
    world_from_base: RosTransform
    base_from_sensor: RosTransform
    probe_in_source_xyz: Vector3
    source_frame: Literal["sensor"]
    target_frame: Literal["world"]
    stamp_sec: int
    world_from_base_rpy: InputVector3
    base_from_sensor_rpy: InputVector3


class RosMessage(NativeCaptureModel):
    parent_frame: Literal["world"]
    child_frame: Literal["sensor"]
    stamp_sec: int
    stamp_nanosec: Literal[0]
    translation_xyz: Vector3
    rotation_xyzw: Quaternion


class RosOutputs(NativeCaptureModel):
    lookup: RosTransform
    lookup_message: Optional[RosMessage] = None  # noqa: UP045 -- Python 3.9
    message_roundtrip: Optional[RosTransform] = None  # noqa: UP045 -- Python 3.9
    probe_in_target_xyz: Optional[Vector3] = None  # noqa: UP045 -- Python 3.9


class RosObservation(NativeCaptureModel):
    operation_id: Literal["lookup-world-from-sensor", "lookup-sensor-from-world"]
    native_outputs: RosOutputs


class RosDirectCapture(NativeCaptureModel):
    native_scalar_bytes: Literal[8]
    compiler: str
    cpp_standard: int
    flags: str
    case_id: Literal["world-base-sensor"]
    native_inputs: RosNativeInputs
    observations: Annotated[list[RosObservation], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def validate_lookup(self) -> RosDirectCapture:
        if [item.operation_id for item in self.observations] != [
            "lookup-world-from-sensor",
            "lookup-sensor-from-world",
        ]:
            raise ValueError("ROS capture must contain both lookup directions in order")
        forward, reverse = (item.native_outputs for item in self.observations)
        message = forward.lookup_message
        if (
            message is None
            or forward.message_roundtrip is None
            or forward.probe_in_target_xyz is None
        ):
            raise ValueError(
                "forward lookup requires message, roundtrip and point action"
            )
        if any(
            value is not None
            for value in (
                reverse.lookup_message,
                reverse.message_roundtrip,
                reverse.probe_in_target_xyz,
            )
        ):
            raise ValueError("reverse lookup contains unexpected extra observations")
        if message.stamp_sec != self.native_inputs.stamp_sec:
            raise ValueError("lookup message timestamp differs from the input")
        if message.translation_xyz != forward.lookup.translation_xyz or not any(
            all(
                math.isclose(a, sign * b, rel_tol=0.0, abs_tol=2e-12)
                for a, b in zip(message.rotation_xyzw, forward.lookup.quaternion_xyzw)
            )
            for sign in (-1.0, 1.0)
        ):
            raise ValueError("lookup message fields differ from the lookup")
        return self

    def validate_inputs(self, inputs: RosChainInputs) -> None:
        native = self.native_inputs
        if (
            self.case_id != inputs.case_id
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
