"""Strict schemas for the complete stdout emitted by the ROS 2 native exporter."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, FiniteFloat, model_validator

from scripts.evidence_schema.primitives import (
    FiniteInputNumber,
    NativeCaptureModel,
)

FiniteVector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
FiniteQuaternion4 = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
FiniteMatrix3 = Annotated[list[FiniteVector3], Field(min_length=3, max_length=3)]
InputVector3 = Annotated[list[FiniteInputNumber], Field(min_length=3, max_length=3)]
InputQuaternion4 = Annotated[list[FiniteInputNumber], Field(min_length=4, max_length=4)]
InputMatrix3 = Annotated[list[InputVector3], Field(min_length=3, max_length=3)]


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


class RosEulerInputs(NativeCaptureModel):
    roll_pitch_yaw_radians: InputVector3
    probe_vectors_xyz: list[FiniteVector3]


class RosEulerOutputs(NativeCaptureModel):
    quaternion_xyzw: FiniteQuaternion4
    matrix_rows: FiniteMatrix3
    readback_rpy: FiniteVector3
    message_xyzw: FiniteQuaternion4
    message_roundtrip_xyzw: FiniteQuaternion4
    basis_and_probe_images: list[FiniteVector3]


class RosEulerObservation(NativeCaptureModel):
    case_id: str
    operation_id: Literal["setRPY-message-action"]
    native_entry_point: str
    native_inputs: RosEulerInputs
    native_outputs: RosEulerOutputs


class RosProductsInputs(NativeCaptureModel):
    a_xyzw: FiniteQuaternion4
    b_xyzw: FiniteQuaternion4


class RosProductsOutputs(NativeCaptureModel):
    ab_xyzw: FiniteQuaternion4
    ba_xyzw: FiniteQuaternion4


class RosProductsObservation(NativeCaptureModel):
    case_id: str
    operation_id: Literal["products"]
    native_entry_point: str
    native_inputs: RosProductsInputs
    native_outputs: RosProductsOutputs


class RosMatrixInputs(NativeCaptureModel):
    matrix_rows: FiniteMatrix3


class RosMatrixOutputs(NativeCaptureModel):
    quaternion_xyzw: FiniteQuaternion4


class RosMatrixObservation(NativeCaptureModel):
    case_id: str
    operation_id: Literal["matrix-to-quaternion"]
    native_entry_point: str
    native_inputs: RosMatrixInputs
    native_outputs: RosMatrixOutputs


class RosLookupInputs(NativeCaptureModel):
    target_frame: str
    source_frame: str
    stamp_sec: int
    world_from_base: RosFullTransform
    base_from_sensor: RosFullTransform
    probe_in_source_xyz: FiniteVector3


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


class RosWorldLookupObservation(NativeCaptureModel):
    case_id: str
    operation_id: Literal["lookup-world-from-sensor"]
    native_entry_point: str
    native_inputs: RosLookupInputs
    native_outputs: RosLookupOutputs


class RosSensorLookupObservation(NativeCaptureModel):
    case_id: str
    operation_id: Literal["lookup-sensor-from-world"]
    native_entry_point: str
    native_inputs: RosLookupInputs
    native_outputs: RosLookupOutputs


RosFullObservation = Annotated[
    Union[
        RosEulerObservation,
        RosProductsObservation,
        RosMatrixObservation,
        RosWorldLookupObservation,
        RosSensorLookupObservation,
    ],
    Field(discriminator="operation_id"),
]


class RosFullCapture(NativeCaptureModel):
    native_scalar_bytes: int
    compiler: str
    cpp_standard: int
    flags: str
    observations: list[RosFullObservation]

    @model_validator(mode="after")
    def validate_operation_identity(self) -> RosFullCapture:
        keys = [(row.case_id, row.operation_id) for row in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("ROS stdout contains duplicate (case_id, operation_id)")
        for row in self.observations:
            if isinstance(row, (RosWorldLookupObservation, RosSensorLookupObservation)):
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
                if (
                    message.translation_xyz != lookup.translation_xyz
                    or message.rotation_xyzw != lookup.quaternion_xyzw
                ):
                    raise ValueError(
                        f"{row.operation_id} message fields differ from the lookup"
                    )
        lookups = [
            row
            for row in self.observations
            if isinstance(row, (RosWorldLookupObservation, RosSensorLookupObservation))
        ]
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


class RosCorpusEuler(NativeCaptureModel):
    case_id: str
    xyz: InputVector3
    actual_bits_hex: list[str]
    family: str
    numeric_format: str
    order_label: Optional[str]  # noqa: UP045 -- Pydantic evaluates this on the Python 3.9 floor.
    units: str


class RosCorpusProduct(NativeCaptureModel):
    case_id: str
    a_wxyz: InputQuaternion4
    b_wxyz: InputQuaternion4
    family: str


class RosCorpusMatrix(NativeCaptureModel):
    case_id: str
    domain: str
    rows: InputMatrix3


class RosSelectedCorpus(NativeCaptureModel):
    euler_inputs: list[RosCorpusEuler]
    multiplication_pairs: list[RosCorpusProduct]
    matrix_inputs: list[RosCorpusMatrix]
    vectors_xyz: list[InputVector3]

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> RosSelectedCorpus:
        for label, rows in (
            ("Euler", self.euler_inputs),
            ("product", self.multiplication_pairs),
            ("matrix", self.matrix_inputs),
        ):
            ids = [row.case_id for row in rows]
            if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
                raise ValueError(
                    f"ROS selected {label} case IDs must be nonempty and unique"
                )
        return self


def validate_ros_capture_corpus_link(
    capture: RosFullCapture, corpus: RosSelectedCorpus
) -> None:
    """Link echoed native inputs and consumed probes to the selected source corpus."""
    by_key = {(row.case_id, row.operation_id): row for row in capture.observations}
    expected_keys = {
        *((row.case_id, "setRPY-message-action") for row in corpus.euler_inputs),
        *((row.case_id, "products") for row in corpus.multiplication_pairs),
        *(
            (row.case_id, "matrix-to-quaternion")
            for row in corpus.matrix_inputs
            if row.domain == "proper_rotation"
        ),
        ("world-base-sensor", "lookup-world-from-sensor"),
        ("world-base-sensor", "lookup-sensor-from-world"),
    }
    if set(by_key) != expected_keys:
        raise ValueError("ROS stdout operation IDs do not match the selected corpus")

    for source in corpus.euler_inputs:
        observation = by_key[(source.case_id, "setRPY-message-action")]
        if not isinstance(observation, RosEulerObservation):
            raise TypeError(
                f"ROS Euler case {source.case_id!r} has the wrong operation"
            )
        if observation.native_inputs.roll_pitch_yaw_radians != source.xyz:
            raise ValueError(
                f"ROS Euler input {source.case_id!r} differs from its corpus row"
            )
        expected_probes = [
            [float(value) for value in probe] for probe in corpus.vectors_xyz
        ]
        if observation.native_inputs.probe_vectors_xyz != expected_probes:
            raise ValueError(
                f"ROS Euler probe inputs {source.case_id!r} differ from the selected corpus"
            )
        if len(observation.native_outputs.basis_and_probe_images) != len(
            expected_probes
        ):
            raise ValueError(
                f"ROS Euler output {source.case_id!r} has the wrong probe count"
            )

    for source in corpus.multiplication_pairs:
        observation = by_key[(source.case_id, "products")]
        if not isinstance(observation, RosProductsObservation):
            raise TypeError(
                f"ROS product case {source.case_id!r} has the wrong operation"
            )
        expected_a = [float(value) for value in source.a_wxyz[1:]] + [
            float(source.a_wxyz[0])
        ]
        expected_b = [float(value) for value in source.b_wxyz[1:]] + [
            float(source.b_wxyz[0])
        ]
        if (
            observation.native_inputs.a_xyzw != expected_a
            or observation.native_inputs.b_xyzw != expected_b
        ):
            raise ValueError(
                f"ROS product inputs {source.case_id!r} differ from its corpus row"
            )

    for source in corpus.matrix_inputs:
        if source.domain != "proper_rotation":
            continue
        observation = by_key[(source.case_id, "matrix-to-quaternion")]
        if not isinstance(observation, RosMatrixObservation):
            raise TypeError(
                f"ROS matrix case {source.case_id!r} has the wrong operation"
            )
        expected_rows = [[float(value) for value in row] for row in source.rows]
        if observation.native_inputs.matrix_rows != expected_rows:
            raise ValueError(
                f"ROS matrix input {source.case_id!r} differs from its corpus row"
            )
