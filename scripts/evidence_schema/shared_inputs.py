"""Typed source model and reducer for the shared canonical input corpus."""

from __future__ import annotations

from typing import Literal

from pydantic import FiniteFloat

from scripts.evidence_schema.crystallized import (
    CanonicalInputContract,
    CanonicalInputs,
    CanonicalOrientationInput,
    CoordinateFrame,
)
from scripts.evidence_schema.primitives import (
    NativeCaptureModel,
    Quaternion4,
    Vector3,
)


class SharedFrameInput(NativeCaptureModel):
    handedness: Literal["right-handed"]
    positive_x: str
    positive_y: str
    positive_z: str


class SharedQuaternionInput(NativeCaptureModel):
    case_id: str
    canonical_wxyz: Quaternion4


class SharedCompositionInput(NativeCaptureModel):
    left_case_id: str
    right_case_id: str
    product_order: str
    reverse_product_order: str


class SharedCanonicalContract(NativeCaptureModel):
    algebra: Literal["Hamilton", "JPL"]
    coefficient_order: Literal["wxyz", "xyzw"]
    rotation_action: Literal["active", "passive"]
    vector_action: str
    matrix_action: str
    coordinate_system: Literal["right-handed", "left-handed"]
    construction_angle_units: Literal["radians"]
    normalization: str
    frame_mapping: str


class SharedCanonicalInputCapture(NativeCaptureModel):
    schema_version: Literal[1]
    contract_id: str
    contract: SharedCanonicalContract
    source_frame: SharedFrameInput
    probe_vector_xyz: Vector3
    orientations: list[SharedQuaternionInput]
    composition: SharedCompositionInput


class GeographicLocationInput(NativeCaptureModel):
    case_id: str
    longitude_degrees: FiniteFloat
    latitude_degrees: FiniteFloat
    height_m: FiniteFloat


class GeographicInputCorpus(NativeCaptureModel):
    """Typed WGS84 geographic inputs shared by native frame evidence."""

    schema_version: Literal["1.0"]
    ellipsoid: Literal["WGS84"]
    kind: Literal["inputs_not_expected_outputs"]
    locations: list[GeographicLocationInput]
    heading_pitch_roll_radians: list[Vector3]
    local_point_xyz_m: Vector3
    local_direction_xyz: Vector3
    exact_pole_note: str


def crystallize_shared_inputs(capture: SharedCanonicalInputCapture) -> CanonicalInputs:
    """Remove repeated labels while retaining one canonical case table."""
    return CanonicalInputs(
        input_set_id="canonical-rotation-inputs-v1",
        contract_id=capture.contract_id,
        contract=CanonicalInputContract(
            algebra=capture.contract.algebra,
            coefficient_order=capture.contract.coefficient_order,
            rotation_action=capture.contract.rotation_action,
            vector_action=capture.contract.vector_action,
            matrix_action=capture.contract.matrix_action,
            coordinate_system=capture.contract.coordinate_system,
            construction_angle_units=capture.contract.construction_angle_units,
            normalization=capture.contract.normalization,
            frame_mapping=capture.contract.frame_mapping,
        ),
        source_frame=CoordinateFrame(
            name="QuatOps canonical source frame",
            handedness=capture.source_frame.handedness,
            positive_x=capture.source_frame.positive_x,
            positive_y=capture.source_frame.positive_y,
            positive_z=capture.source_frame.positive_z,
        ),
        orientations=[
            CanonicalOrientationInput(
                case_id=row.case_id,
                coefficients_wxyz=row.canonical_wxyz,
            )
            for row in capture.orientations
        ],
        probe_vector_xyz=capture.probe_vector_xyz,
        composition_left_case_id=capture.composition.left_case_id,
        composition_right_case_id=capture.composition.right_case_id,
        product_order=capture.composition.product_order,
    )
