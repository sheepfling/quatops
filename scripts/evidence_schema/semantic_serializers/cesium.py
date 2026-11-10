"""Semantic serialization for CesiumJS geographic frame experiments."""

from __future__ import annotations

import re

from quatops.scalar import multiply as multiply_wxyz
from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    CesiumCapture,
    CesiumContract,
    CesiumInputs,
)
from scripts.evidence_schema.crystallized import (
    FrameSemantics,
    GeographicContract,
    GeographicOrientationWitness,
    GeographicSerialization,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    canonical_frame_semantics,
    coefficient_order,
    coordinate_frame,
    signed_axis_map,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientations_by_case,
    validate_canonical_input_digest,
    validate_canonical_quaternion,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> GeographicSerialization:
    """Map Cesium ENU/ECEF observations with the declared local alignment."""
    if not isinstance(package.contract, CesiumContract):
        raise TypeError("Cesium serializer received a non-Cesium contract")
    if not isinstance(package.target_capture, CesiumCapture):
        raise TypeError("Cesium serializer received a non-Cesium capture")
    if not isinstance(package.inputs, CesiumInputs):
        raise TypeError("Cesium serializer requires typed Cesium inputs")
    _validate_canonical_inputs(package.inputs, context)
    return GeographicSerialization(
        layer="geographic-transport",
        contract=serialize_cesium_contract(package.contract, package.inputs),
        witnesses=serialize_cesium_witnesses(package.target_capture, package.inputs),
    )


def serialize_cesium_contract(
    contract: CesiumContract, inputs: CesiumInputs
) -> GeographicContract:
    """Map Cesium's documented local ENU frame and declared alignment."""
    storage = contract.storage.upper()
    native_order_match = re.search(r"CESIUM\s+(WXYZ|XYZW)", storage)
    if native_order_match is None:
        raise ValueError("Cesium storage declaration lacks a native component order")
    canonical = inputs.frame_contract.canonical_local
    local = contract.local_frame
    if "Transforms.eastNorthUpToFixedFrame" not in contract.native_api:
        raise ValueError("Cesium frame claim lacks its ENU placement API capture")
    if (local.positive_x, local.positive_y, local.positive_z) != (
        "east",
        "north",
        "ellipsoid normal",
    ):
        raise ValueError(
            "Cesium target frame must use its documented east/north/up axes; "
            "caller alignment belongs in the input frame contract"
        )
    return GeographicContract(
        kind="geographic",
        canonical_frame=coordinate_frame(
            "QuatOps canonical local frame",
            canonical.handedness,
            canonical.model_dump(),
        ),
        canonical_frame_semantics=canonical_frame_semantics(),
        local_frame=coordinate_frame(
            "Cesium local ENU", local.handedness, local.model_dump()
        ),
        local_frame_semantics=FrameSemantics(
            role="local_geographic_enu",
            authority="target_documented",
            statement=(
                "The captured API creates the east-north-up local frame at the "
                "specified geodetic origin and ellipsoid."
            ),
            evidence_reference=(
                "https://cesium.com/downloads/cesiumjs/releases/1.124/Build/"
                "Documentation/Transforms.html#eastNorthUpToFixedFrame"
            ),
        ),
        local_axes_from_canonical=signed_axis_map(contract.source_map),
        local_alignment_authority="application_declared",
        global_frame_name="ECEF",
        geodetic_datum=contract.geodesy,
        forward_expression=contract.forward,
        inverse_expression=contract.inverse,
        quaternion_order=coefficient_order(native_order_match.group(1)),
        scope=[
            "local quaternion placement at multiple geodetic origins",
            "native basis and mixed-vector actions",
            "ECEF point and direction action",
            "inverse point and orientation recovery",
            "antipodal quaternion input",
        ],
        excludes=contract.excludes,
    )


def _validate_canonical_inputs(
    inputs: CesiumInputs, context: SemanticSerializationContext
) -> None:
    validate_canonical_input_digest(
        inputs.canonical_input_sha256, context, target="Cesium"
    )
    canonical_by_case = canonical_orientations_by_case(context.canonical_inputs)
    composition_case = (
        f"{context.canonical_inputs.composition_left_case_id} * "
        f"{context.canonical_inputs.composition_right_case_id}"
    )
    for row in inputs.cases:
        case_id = row.canonical_orientation_case
        if case_id in canonical_by_case:
            validate_canonical_quaternion(
                "Cesium", case_id, row.canonical_wxyz, canonical_by_case
            )
            continue
        if case_id != composition_case:
            raise ValueError(f"Cesium input references unknown case {case_id!r}")
        expected = multiply_wxyz(
            canonical_by_case[context.canonical_inputs.composition_left_case_id],
            canonical_by_case[context.canonical_inputs.composition_right_case_id],
        )
        if any(
            abs(actual - wanted) > 1e-15
            for actual, wanted in zip(row.canonical_wxyz, expected)
        ):
            raise ValueError("Cesium composed input differs from shared product")


def serialize_cesium_witnesses(
    capture: CesiumCapture, inputs: CesiumInputs
) -> list[GeographicOrientationWitness]:
    cases = {row.case_id: row for row in inputs.cases}
    witnesses: list[GeographicOrientationWitness] = []
    for observation in capture.observations:
        item = cases[observation.case_id]
        output = observation.outputs
        local_actions = (
            output.local_basis_and_probe_images.values
            if output.local_basis_and_probe_images is not None
            else None
        )
        world_directions = output.world_directions_xyz.values
        witnesses.append(
            GeographicOrientationWitness(
                kind="geographic_orientation",
                case_id=observation.case_id,
                location_case=item.location_case,
                longitude_degrees=item.longitude_degrees,
                latitude_degrees=item.latitude_degrees,
                height_m=item.height_m,
                coefficient_order="xyzw",
                local_coefficients=item.quatops_cesium_local_xyzw,
                local_translation_xyz_m=item.local_translation_xyz_m,
                ecef_coefficients=output.model_to_ecef_quaternion_xyzw.values,
                recovered_local_coefficients=(
                    output.local_quaternion_recovered_from_ecef_xyzw.values
                ),
                local_basis_images_xyz=local_actions[:3] if local_actions else None,
                local_probe_image_xyz=(
                    local_actions[3]
                    if local_actions is not None and len(local_actions) > 3
                    else None
                ),
                world_directions_xyz=world_directions[:3],
                world_probe_direction_xyz=(
                    world_directions[3] if len(world_directions) > 3 else None
                ),
                ecef_origin_xyz_m=output.ecef_origin_xyz_m.values,
                world_probe_point_ecef_xyz_m=output.world_points_xyz_m.values[0],
                recovered_probe_point_local_xyz_m=(
                    output.point_roundtrip_local_xyz_m.values[0]
                ),
            )
        )
    return witnesses
