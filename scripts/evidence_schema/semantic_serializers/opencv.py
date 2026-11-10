"""Semantic serialization for OpenCV's Rodrigues rotation-vector route."""

from __future__ import annotations

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    OpenCVCapture,
    OpenCVContract,
    OpenCVInputs,
)
from scripts.evidence_schema.crystallized import (
    RotationVectorBoundaryContract,
    RotationVectorBoundaryWitness,
    RotationVectorCompositionWitness,
    RotationVectorSerialization,
    RotationVectorWitness,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    canonical_frame,
    canonical_order,
    numeric_probe_semantics,
    probe_frame,
    signed_axis_map,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientation_selection,
    select_retained_rows,
    validate_direct_canonical_inputs,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)
from scripts.evidence_schema.tolerances import tolerances_for_target


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> RotationVectorSerialization:
    """Map OpenCV matrices and Rodrigues vectors to rotation witnesses."""
    if not isinstance(package.contract, OpenCVContract):
        raise TypeError("OpenCV serializer received a non-OpenCV contract")
    if not isinstance(package.target_capture, OpenCVCapture):
        raise TypeError("OpenCV serializer received a non-OpenCV capture")
    if not isinstance(package.inputs, OpenCVInputs):
        raise TypeError("OpenCV serializer requires typed OpenCV inputs")
    validate_direct_canonical_inputs(package.inputs, context, target="OpenCV")
    return RotationVectorSerialization(
        layer="native-quaternion-api",
        contract=serialize_opencv_contract(package.contract),
        witnesses=serialize_opencv_witnesses(package.target_capture, package.inputs),
    )


def serialize_opencv_contract(
    contract: OpenCVContract,
) -> RotationVectorBoundaryContract:
    """Map OpenCV's declared Rodrigues route to the common boundary contract."""
    canonical_coefficient_order = canonical_order(contract)
    boundary = contract.target_boundary
    if not boundary.rotation_vector_from_canonical or not boundary.native_to_canonical:
        raise ValueError("rotation-vector contract must declare both conversion paths")
    return RotationVectorBoundaryContract(
        kind="rotation_vector_boundary",
        canonical_algebra="Hamilton",
        canonical_order=canonical_coefficient_order,
        rotation_action="active",
        source_frame=canonical_frame(contract),
        source_frame_semantics=canonical_source_frame_semantics(),
        destination_frame=probe_frame(contract),
        destination_frame_semantics=numeric_probe_semantics(
            boundary.probe_frame_assignment.meaning
        ),
        destination_axes_from_source=signed_axis_map(boundary.coordinate_map),
        angle_units="radians",
        native_representation=boundary.representation,
        native_matrix_action=boundary.native_matrix_action,
        conversion_rule=boundary.rotation_vector_from_canonical,
        inverse_rule=boundary.native_to_canonical,
        tolerances=tolerances_for_target(contract.target.name),
        scope=[
            "quaternion-to-rotation-vector boundary",
            "native basis and selected mixed-vector actions",
            "native matrix-to-rotation-vector round trip on a mixed orientation",
            "both orders of a noncommuting matrix product",
        ],
        excludes=contract.excludes,
    )


def serialize_opencv_witnesses(
    capture: OpenCVCapture, inputs: OpenCVInputs
) -> list[RotationVectorBoundaryWitness]:
    observations = capture.observations
    retained_case_ids = canonical_orientation_selection(
        inputs.retained_case_ids, context="OpenCV"
    )
    witnesses: list[RotationVectorBoundaryWitness] = []
    for profile in inputs.precision_profiles:
        profile_rows = [
            row for row in observations.canonical_to_native if row.dtype == profile
        ]
        retained_rows = select_retained_rows(
            profile_rows, retained_case_ids, context=f"OpenCV {profile}"
        )
        witnesses.extend(
            RotationVectorWitness(
                kind="rotation_vector",
                case_id=f"{row.case_id}@{row.dtype}",
                precision=row.dtype,
                input_rotation_vector_xyz=row.native_input_rotation_vector_xyz_radians,
                output_rotation_vector_xyz=row.native_matrix_to_rotation_vector_xyz_radians,
                basis_images_xyz=row.native_gemm_basis_actions_xyz,
                probe_image_xyz=row.native_gemm_probe_action_xyz,
            )
            for row in retained_rows
        )
    witnesses.extend(
        RotationVectorWitness(
            kind="rotation_vector",
            case_id=f"{row.case_id}@{row.dtype}",
            precision=row.dtype,
            input_rotation_vector_xyz=row.native_input_rotation_vector_xyz_radians,
            output_rotation_vector_xyz=row.native_matrix_to_rotation_vector_xyz_radians,
            basis_images_xyz=row.native_gemm_basis_actions_xyz,
        )
        for row in observations.native_created
    )
    witnesses.extend(
        RotationVectorCompositionWitness(
            kind="rotation_vector_composition",
            case_id=f"{row.case_id}@{row.dtype}",
            left_case_id=row.native_lhs_case_id,
            right_case_id=row.native_rhs_case_id,
            product_order=row.product_order,
            precision=row.dtype,
            output_rotation_vector_xyz=row.native_matrix_to_rotation_vector_xyz_radians,
            basis_images_xyz=row.native_gemm_basis_actions_xyz,
        )
        for row in observations.composition
    )
    return witnesses
