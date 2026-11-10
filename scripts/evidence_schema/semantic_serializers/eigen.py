"""Semantic serialization for the Eigen quaternion API experiment."""

from __future__ import annotations

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.contracts import serialize_direct_quaternion_contract
from scripts.evidence_schema.converter_captures import (
    EigenCapture,
    EigenContract,
    EigenInputs,
)
from scripts.evidence_schema.crystallized import (
    CompositionWitness,
    NativeQuaternionApiSerialization,
    OrientationWitness,
    QuaternionBoundaryWitness,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientation_selection,
    select_retained_rows,
    validate_direct_canonical_inputs,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> NativeQuaternionApiSerialization:
    """Map Eigen's constructor, coefficient, action, and product evidence."""
    if not isinstance(package.contract, EigenContract):
        raise TypeError("Eigen serializer received a non-Eigen contract")
    if not isinstance(package.target_capture, EigenCapture):
        raise TypeError("Eigen serializer received a non-Eigen capture")
    if not isinstance(package.inputs, EigenInputs):
        raise TypeError("Eigen serializer requires typed Eigen inputs")
    validate_direct_canonical_inputs(package.inputs, context, target="Eigen")
    composition = package.inputs.composition
    if (
        composition.left_case_id != context.canonical_inputs.composition_left_case_id
        or composition.right_case_id
        != context.canonical_inputs.composition_right_case_id
        or (
            composition.product_order is not None
            and composition.product_order != context.canonical_inputs.product_order
        )
    ):
        raise ValueError("Eigen composition input differs from shared inputs")
    return NativeQuaternionApiSerialization(
        layer="native-quaternion-api",
        contract=serialize_direct_quaternion_contract(package.contract),
        witnesses=serialize_eigen_witnesses(package.target_capture, package.inputs),
    )


def serialize_eigen_witnesses(
    capture: EigenCapture, inputs: EigenInputs
) -> list[QuaternionBoundaryWitness]:
    retained_case_ids = canonical_orientation_selection(
        inputs.retained_case_ids, context="Eigen"
    )
    witnesses: list[QuaternionBoundaryWitness] = []
    for profile in capture.observations.profiles:
        retained_rows = select_retained_rows(
            profile.orientations, retained_case_ids, context=f"Eigen {profile.dtype}"
        )
        witnesses.extend(
            OrientationWitness(
                kind="orientation",
                case_id=row.case_id,
                precision=profile.dtype,
                coefficient_order="wxyz",
                coefficients=row.quaternion.wxyz,
                storage_coefficients_xyzw=(
                    row.quaternion.coeffs_xyzw if row.case_id == "mixed_a" else None
                ),
                basis_images_xyz=row.native_quaternion_basis_actions_xyz,
                probe_image_xyz=row.native_quaternion_probe_action_xyz,
            )
            for row in retained_rows
        )
        witnesses.extend(
            OrientationWitness(
                kind="orientation",
                case_id=row.case_id,
                precision=profile.dtype,
                coefficient_order="wxyz",
                coefficients=row.quaternion.wxyz,
                basis_images_xyz=row.native_quaternion_basis_actions_xyz,
                constructor_axis_xyz=inputs.native_constructed_orientation.axis_xyz,
                constructor_angle_radians=inputs.native_constructed_orientation.angle_radians,
            )
            for row in profile.native_created
        )
        witnesses.extend(
            CompositionWitness(
                kind="composition",
                case_id=row.case_id,
                left_case_id=row.lhs_case_id,
                right_case_id=row.rhs_case_id,
                product_order=row.product_order,
                precision=profile.dtype,
                coefficient_order="wxyz",
                coefficients=row.quaternion.wxyz,
                basis_images_xyz=row.native_quaternion_basis_actions_xyz,
            )
            for row in profile.composition
        )
    return witnesses
