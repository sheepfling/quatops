"""Typed source model and reducer for the shared canonical input corpus."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Optional

from pydantic import Field, model_validator

from scripts.evidence_schema.crystallized import (
    CanonicalInputContract,
    CanonicalInputs,
    CanonicalOrientationInput,
    CoordinateFrame,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.primitives import (
    FiniteInputNumber,
    NativeCaptureModel,
)

FiniteVector3 = Annotated[list[FiniteInputNumber], Field(min_length=3, max_length=3)]
FiniteQuaternion4 = Annotated[
    list[FiniteInputNumber], Field(min_length=4, max_length=4)
]


class SharedFrameInput(NativeCaptureModel):
    handedness: Literal["right-handed"]
    positive_x: Literal["right"]
    positive_y: Literal["forward"]
    positive_z: Literal["up"]


class SharedQuaternionInput(NativeCaptureModel):
    case_id: str
    canonical_wxyz: FiniteQuaternion4


class SharedCompositionInput(NativeCaptureModel):
    left_case_id: str
    right_case_id: str
    product_order: str
    reverse_product_order: str


class SharedCanonicalContract(NativeCaptureModel):
    algebra: Literal["Hamilton"]
    coefficient_order: Literal["wxyz"]
    rotation_action: Literal["active"]
    vector_action: Literal["q * (0, v) * conjugate(q)"]
    matrix_action: Literal["column vectors"]
    coordinate_system: Literal["right-handed"]
    construction_angle_units: Literal["radians"]
    normalization: Literal["unit quaternion"]
    frame_mapping: Literal["source and destination frames are explicit"]


class SharedCanonicalInputCapture(NativeCaptureModel):
    schema_version: Literal[1]
    contract_id: str
    contract: SharedCanonicalContract
    source_frame: SharedFrameInput
    probe_vector_xyz: FiniteVector3
    orientations: list[SharedQuaternionInput]
    composition: SharedCompositionInput

    @model_validator(mode="after")
    def validate_canonical_contract(self) -> SharedCanonicalInputCapture:
        ids = [row.case_id for row in self.orientations]
        if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("canonical orientation IDs must be nonempty and unique")
        by_id = set(ids)
        for row in self.orientations:
            binary64 = [float(value) for value in row.canonical_wxyz]
            norm = math.sqrt(sum(value * value for value in binary64))
            if not math.isfinite(norm) or not math.isclose(
                norm, 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"canonical quaternion {row.case_id!r} is not unit length"
                )
        if (
            self.composition.left_case_id not in by_id
            or self.composition.right_case_id not in by_id
        ):
            raise ValueError(
                "canonical composition references an unresolved orientation ID"
            )
        return self


class GeographicLocationInput(NativeCaptureModel):
    case_id: str
    longitude_degrees: FiniteInputNumber
    latitude_degrees: FiniteInputNumber
    height_m: FiniteInputNumber


class GeographicInputCorpus(NativeCaptureModel):
    """Typed finite probes, not assertions that every coordinate is physical WGS84.

    Finite wrapped longitudes and out-of-range latitudes remain useful cross-
    target probes; individual target APIs may reject them differently.
    """

    schema_version: Literal["1.0"]
    ellipsoid: Literal["WGS84"]
    kind: Literal["inputs_not_expected_outputs"]
    locations: list[GeographicLocationInput]
    heading_pitch_roll_radians: list[FiniteVector3]
    local_point_xyz_m: FiniteVector3
    local_direction_xyz: FiniteVector3
    exact_pole_note: str

    @model_validator(mode="after")
    def validate_geographic_inputs(self) -> GeographicInputCorpus:
        ids = [row.case_id for row in self.locations]
        if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("geographic location IDs must be nonempty and unique")
        return self


def resolve_orientation_selection(
    capture: SharedCanonicalInputCapture,
    selected_case_ids: list[str],
    *,
    required_case_ids: Optional[set[str]] = None,  # noqa: UP045 -- supports Python 3.9.
) -> list[SharedQuaternionInput]:
    """Resolve a declared selection without collapsing duplicates in a dict."""
    if any(not case_id for case_id in selected_case_ids):
        raise ValueError("selected orientation IDs must be nonempty")
    if len(selected_case_ids) != len(set(selected_case_ids)):
        raise ValueError("selected orientation IDs must be unique")
    by_id = {row.case_id: row for row in capture.orientations}
    missing = set(selected_case_ids).difference(by_id)
    if missing:
        raise ValueError(f"selected orientation IDs are unresolved: {sorted(missing)}")
    required = required_case_ids or set()
    absent = required.difference(selected_case_ids)
    if absent:
        raise ValueError(
            f"required orientation IDs are missing from selection: {sorted(absent)}"
        )
    return [by_id[case_id] for case_id in selected_case_ids]


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
        source_frame_semantics=canonical_source_frame_semantics(),
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
