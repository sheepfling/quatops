"""Semantic serialization for the Eigen quaternion API experiment."""

from __future__ import annotations

import math

from quatops.scalar import multiply as multiply_wxyz
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
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    map_hamilton_quaternion_wxyz,
    map_vector_components,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientation_selection,
    canonical_orientations_by_case,
    native_product_case_ids,
    select_retained_rows,
    validate_direct_canonical_inputs,
    validate_quaternion_actions,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)
from scripts.evidence_schema.tolerances import tolerances_for_target


def _require_coefficients_close(
    actual: list[float],
    expected: list[float],
    *,
    tolerance: float,
    context: str,
) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(value, wanted, rel_tol=0.0, abs_tol=tolerance)
        for value, wanted in zip(actual, expected)
    ):
        raise ValueError(f"{context} differs from its declared frame conversion")


def _validate_eigen_capture(
    contract: EigenContract,
    capture: EigenCapture,
    inputs: EigenInputs,
    context: SemanticSerializationContext,
) -> None:
    """Bind captured Eigen profile values to their typed conversion contract."""
    target_profiles = contract.target.scalar_profiles
    input_profiles = inputs.scalar_profiles
    capture_profiles = [profile.dtype for profile in capture.observations.profiles]
    if (
        len(set(target_profiles)) != len(target_profiles)
        or len(set(input_profiles)) != len(input_profiles)
        or len(set(capture_profiles)) != len(capture_profiles)
        or set(target_profiles) != set(input_profiles)
        or set(target_profiles) != set(capture_profiles)
    ):
        raise ValueError("Eigen target, input, and capture scalar profiles differ")

    canonical_by_case = canonical_orientations_by_case(context.canonical_inputs)
    axis_map = contract.target_boundary.source_to_target_coordinate_map
    tolerances = {
        item.profile: item.absolute for item in tolerances_for_target("Eigen")
    }
    constructor = inputs.native_constructed_orientation
    half_angle = constructor.angle_radians / 2.0
    constructed_wxyz = map_hamilton_quaternion_wxyz(
        [
            math.cos(half_angle),
            *(math.sin(half_angle) * component for component in constructor.axis_xyz),
        ],
        axis_map,
    )
    left_case = context.canonical_inputs.composition_left_case_id
    right_case = context.canonical_inputs.composition_right_case_id
    expected_products = {(left_case, right_case), (right_case, left_case)}

    for profile in capture.observations.profiles:
        precision = profile.dtype
        tolerance = tolerances[precision]
        if profile.scalar_bytes != (4 if precision == "float32" else 8):
            raise ValueError(f"Eigen {precision} scalar width differs from its profile")
        retained = select_retained_rows(
            profile.orientations,
            canonical_orientation_selection(inputs.retained_case_ids, context="Eigen"),
            context=f"Eigen {precision}",
        )
        for row in retained:
            _require_coefficients_close(
                row.quaternion.wxyz,
                map_hamilton_quaternion_wxyz(canonical_by_case[row.case_id], axis_map),
                tolerance=tolerance,
                context=f"Eigen {precision} {row.case_id}",
            )
            probe_image = row.native_quaternion_probe_action_xyz
            if row.case_id == "mixed_a" and probe_image is None:
                raise ValueError(f"Eigen {precision} mixed_a lacks its probe action")
            validate_quaternion_actions(
                row.quaternion.wxyz,
                row.native_quaternion_basis_actions_xyz,
                tolerance=tolerance,
                context=f"Eigen {precision} {row.case_id}",
                probe_vector_xyz=(
                    map_vector_components(
                        context.canonical_inputs.probe_vector_xyz, axis_map
                    )
                    if probe_image is not None
                    else None
                ),
                probe_image_xyz=probe_image,
            )

        if (
            len(profile.native_created) != 1
            or profile.native_created[0].case_id != constructor.case_id
            or profile.native_created[0].constructor
            != "Eigen::Quaternion(Eigen::AngleAxis)"
        ):
            raise ValueError(
                f"Eigen {precision} constructor evidence differs from input"
            )
        _require_coefficients_close(
            profile.native_created[0].quaternion.wxyz,
            constructed_wxyz,
            tolerance=tolerance,
            context=f"Eigen {precision} native-created quaternion",
        )
        validate_quaternion_actions(
            profile.native_created[0].quaternion.wxyz,
            profile.native_created[0].native_quaternion_basis_actions_xyz,
            tolerance=tolerance,
            context=f"Eigen {precision} native-created quaternion",
        )

        product_cases = [
            native_product_case_ids(row.case_id) for row in profile.composition
        ]
        if (
            len(set(product_cases)) != len(product_cases)
            or set(product_cases) != expected_products
        ):
            raise ValueError(f"Eigen {precision} composition cases differ from inputs")
        for row in profile.composition:
            if (
                row.lhs_case_id,
                row.rhs_case_id,
            ) != native_product_case_ids(row.case_id):
                raise ValueError(
                    f"Eigen {precision} composition case label is inconsistent"
                )
            if row.product_order != context.canonical_inputs.product_order:
                raise ValueError(f"Eigen {precision} product order differs from inputs")
            expected_product = map_hamilton_quaternion_wxyz(
                multiply_wxyz(
                    canonical_by_case[row.lhs_case_id],
                    canonical_by_case[row.rhs_case_id],
                ),
                axis_map,
            )
            _require_coefficients_close(
                row.quaternion.wxyz,
                expected_product,
                tolerance=tolerance,
                context=f"Eigen {precision} {row.case_id}",
            )
            validate_quaternion_actions(
                row.quaternion.wxyz,
                row.native_quaternion_basis_actions_xyz,
                tolerance=tolerance,
                context=f"Eigen {precision} composition {row.case_id}",
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
    _validate_eigen_capture(
        package.contract, package.target_capture, package.inputs, context
    )
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
        mixed = next(row for row in retained_rows if row.case_id == "mixed_a")
        storage = mixed.quaternion.coeffs_xyzw
        if storage is None:
            raise ValueError(f"Eigen {profile.dtype} lacks native coeffs() evidence")
        wxyz = mixed.quaternion.wxyz
        if storage != [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]:
            raise ValueError(
                f"Eigen {profile.dtype} coeffs() differs from WXYZ accessors"
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
                constructor_angle_radians=(
                    inputs.native_constructed_orientation.angle_radians
                ),
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
