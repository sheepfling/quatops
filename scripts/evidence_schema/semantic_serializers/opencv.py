"""Semantic serialization for OpenCV's Rodrigues rotation-vector route."""

from __future__ import annotations

from math import cos, sin

from quatops.scalar import from_rotation_vector, multiply as multiply_wxyz
from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    OPENCV_NATIVE_CASE_IDS,
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
    map_hamilton_quaternion_wxyz,
    map_vector_components,
    numeric_probe_semantics,
    probe_frame,
    signed_axis_map,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientation_selection,
    select_retained_rows,
    validate_direct_canonical_inputs,
    validate_quaternion_actions,
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
    _validate_precision_profiles(
        package.contract, package.target_capture, package.inputs
    )
    _validate_capture_actions(
        package.contract, package.target_capture, package.inputs, context
    )
    return RotationVectorSerialization(
        layer="native-quaternion-api",
        contract=serialize_opencv_contract(package.contract),
        witnesses=serialize_opencv_witnesses(package.target_capture, package.inputs),
    )


def _validate_precision_profiles(
    contract: OpenCVContract,
    capture: OpenCVCapture,
    inputs: OpenCVInputs,
) -> None:
    """Require all declared OpenCV scalar profiles in inputs and observations."""
    expected = contract.target.scalar_profiles
    supplied = inputs.precision_profiles
    observed = {
        row.dtype
        for rows in (
            capture.observations.canonical_to_native,
            capture.observations.native_created,
            capture.observations.composition,
        )
        for row in rows
    }
    if (
        len(set(expected)) != len(expected)
        or len(set(supplied)) != len(supplied)
        or set(expected) != set(supplied)
        or set(expected) != observed
    ):
        raise ValueError("OpenCV target, input, and capture scalar profiles differ")


def _validate_capture_actions(
    contract: OpenCVContract,
    capture: OpenCVCapture,
    inputs: OpenCVInputs,
    context: SemanticSerializationContext,
) -> None:
    """Tie native Rodrigues vectors and matrix actions to shared rotations."""
    canonical_by_case = {
        row.case_id: row.coefficients_wxyz
        for row in context.canonical_inputs.orientations
    }
    axis_map = contract.target_boundary.coordinate_map
    probe = map_vector_components(context.canonical_inputs.probe_vector_xyz, axis_map)
    tolerance_by_profile = {
        item.profile: item.absolute for item in tolerances_for_target("OpenCV")
    }
    left_case = context.canonical_inputs.composition_left_case_id
    right_case = context.canonical_inputs.composition_right_case_id
    constructor = inputs.native_constructed_orientation
    half_angle = constructor.angle_radians / 2.0
    constructed = map_hamilton_quaternion_wxyz(
        [
            cos(half_angle),
            *(sin(half_angle) * component for component in constructor.axis_xyz),
        ],
        axis_map,
    )

    for profile in inputs.precision_profiles:
        tolerance = tolerance_by_profile[profile]
        canonical_rows = [
            row
            for row in capture.observations.canonical_to_native
            if row.dtype == profile
        ]
        canonical_rows = select_retained_rows(
            canonical_rows,
            OPENCV_NATIVE_CASE_IDS,
            context=f"OpenCV {profile}",
        )
        for row in canonical_rows:
            expected = map_hamilton_quaternion_wxyz(
                canonical_by_case[row.case_id], axis_map
            )
            validate_quaternion_actions(
                expected,
                row.native_gemm_basis_actions_xyz,
                tolerance=tolerance,
                context=f"OpenCV {profile} {row.case_id}",
                probe_vector_xyz=probe if row.case_id == "mixed_a" else None,
                probe_image_xyz=row.native_gemm_probe_action_xyz,
            )
            for label, rotation_vector in (
                ("input", row.native_input_rotation_vector_xyz_radians),
                ("roundtrip", row.native_matrix_to_rotation_vector_xyz_radians),
            ):
                validate_quaternion_actions(
                    from_rotation_vector(rotation_vector),
                    row.native_gemm_basis_actions_xyz,
                    tolerance=tolerance,
                    context=f"OpenCV {profile} {row.case_id} {label} Rodrigues value",
                )

        created_rows = [
            row for row in capture.observations.native_created if row.dtype == profile
        ]
        if len(created_rows) != 1 or created_rows[0].case_id != constructor.case_id:
            raise ValueError(f"OpenCV {profile} constructor capture differs from input")
        created = created_rows[0]
        validate_quaternion_actions(
            constructed,
            created.native_gemm_basis_actions_xyz,
            tolerance=tolerance,
            context=f"OpenCV {profile} native-created orientation",
        )
        for label, rotation_vector in (
            ("input", created.native_input_rotation_vector_xyz_radians),
            ("roundtrip", created.native_matrix_to_rotation_vector_xyz_radians),
        ):
            validate_quaternion_actions(
                from_rotation_vector(rotation_vector),
                created.native_gemm_basis_actions_xyz,
                tolerance=tolerance,
                context=f"OpenCV {profile} native-created {label} Rodrigues value",
            )

        product_rows = [
            row for row in capture.observations.composition if row.dtype == profile
        ]
        expected_pairs = {
            (left_case, right_case),
            (right_case, left_case),
        }
        actual_pairs = {
            (row.native_lhs_case_id, row.native_rhs_case_id) for row in product_rows
        }
        if (
            len(product_rows) != 2
            or len(actual_pairs) != 2
            or actual_pairs != expected_pairs
            or any(
                row.product_order != "lhs matrix * rhs matrix; rhs acts first"
                or row.case_id
                != f"{row.native_lhs_case_id}_times_{row.native_rhs_case_id}"
                for row in product_rows
            )
        ):
            raise ValueError(f"OpenCV {profile} composition cases differ from inputs")
        for row in product_rows:
            expected_product = map_hamilton_quaternion_wxyz(
                multiply_wxyz(
                    canonical_by_case[row.native_lhs_case_id],
                    canonical_by_case[row.native_rhs_case_id],
                ),
                axis_map,
            )
            validate_quaternion_actions(
                expected_product,
                row.native_gemm_basis_actions_xyz,
                tolerance=tolerance,
                context=f"OpenCV {profile} composition {row.case_id}",
            )
            validate_quaternion_actions(
                from_rotation_vector(row.native_matrix_to_rotation_vector_xyz_radians),
                row.native_gemm_basis_actions_xyz,
                tolerance=tolerance,
                context=f"OpenCV {profile} composition {row.case_id} roundtrip",
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
    canonical_orientation_selection(inputs.retained_case_ids, context="OpenCV")
    witnesses: list[RotationVectorBoundaryWitness] = []
    for profile in inputs.precision_profiles:
        profile_rows = [
            row for row in observations.canonical_to_native if row.dtype == profile
        ]
        if len(profile_rows) != len(OPENCV_NATIVE_CASE_IDS):
            raise ValueError(f"Unexpected OpenCV {profile} witness count")
        retained_rows = select_retained_rows(
            profile_rows, OPENCV_NATIVE_CASE_IDS, context=f"OpenCV {profile}"
        )
        witnesses.extend(
            RotationVectorWitness(
                kind="rotation_vector",
                case_id=f"{row.case_id}@{row.dtype}",
                precision=row.dtype,
                input_rotation_vector_xyz=row.native_input_rotation_vector_xyz_radians,
                output_rotation_vector_xyz=(
                    row.native_matrix_to_rotation_vector_xyz_radians
                ),
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
