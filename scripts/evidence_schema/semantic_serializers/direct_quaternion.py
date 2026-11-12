"""Serialize MuJoCo and CSPICE native quaternion actions into shared witnesses."""

from __future__ import annotations

import math
from typing import Union

from quatops.scalar import multiply as multiply_wxyz
from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.contracts import serialize_direct_quaternion_contract
from scripts.evidence_schema.converter_captures import (
    MuJoCoCapture,
    MuJoCoContract,
    MuJoCoInputs,
    MuJoCoOrientation,
    SpiceyPyCapture,
    SpiceyPyContract,
    SpiceyPyInputs,
    SpiceyPyOrientation,
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

NativeOrientation = Union[MuJoCoOrientation, SpiceyPyOrientation]


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


def _validate_native_coefficients(
    contract: MuJoCoContract | SpiceyPyContract,
    capture: MuJoCoCapture | SpiceyPyCapture,
    inputs: MuJoCoInputs | SpiceyPyInputs,
    context: SemanticSerializationContext,
    *,
    target: str,
) -> None:
    """Check captured inputs and products against the shared quaternion corpus."""
    canonical_by_case = canonical_orientations_by_case(context.canonical_inputs)
    axis_map = contract.target_boundary.source_to_target_coordinate_map
    tolerance_by_profile = {
        item.profile: item.absolute
        for item in tolerances_for_target(contract.target.name)
    }
    tolerance = tolerance_by_profile["float64"]
    retained_ids = canonical_orientation_selection(
        inputs.retained_case_ids, context=target
    )
    retained = select_retained_rows(
        capture.observations.canonical_to_native,
        retained_ids,
        context=target,
    )
    for row in retained:
        expected = map_hamilton_quaternion_wxyz(
            canonical_by_case[row.case_id], axis_map
        )
        _require_coefficients_close(
            row.native_input_wxyz,
            expected,
            tolerance=tolerance,
            context=f"{target} {row.case_id} native input",
        )
        basis, probe = _action(row)
        if row.case_id == "mixed_a" and probe is None:
            raise ValueError(f"{target} mixed_a lacks its probe action")
        validate_quaternion_actions(
            row.native_input_wxyz,
            basis,
            tolerance=tolerance,
            context=f"{target} {row.case_id}",
            probe_vector_xyz=(
                map_vector_components(
                    context.canonical_inputs.probe_vector_xyz, axis_map
                )
                if probe is not None
                else None
            ),
            probe_image_xyz=probe,
        )

    left_case = context.canonical_inputs.composition_left_case_id
    right_case = context.canonical_inputs.composition_right_case_id
    expected_products = {(left_case, right_case), (right_case, left_case)}
    products = capture.observations.composition
    product_cases = [native_product_case_ids(row.case_id) for row in products]
    if len(set(product_cases)) != len(product_cases) or set(product_cases) != (
        expected_products
    ):
        raise ValueError(f"{target} composition cases differ from shared inputs")
    for row in products:
        left_case_id, right_case_id = native_product_case_ids(row.case_id)
        if row.product_order != context.canonical_inputs.product_order:
            raise ValueError(f"{target} composition order differs from shared inputs")
        if row.native_lhs_wxyz is None or row.native_rhs_wxyz is None:
            raise ValueError(f"{target} composition lacks native operands")
        expected_left = map_hamilton_quaternion_wxyz(
            canonical_by_case[left_case_id], axis_map
        )
        expected_right = map_hamilton_quaternion_wxyz(
            canonical_by_case[right_case_id], axis_map
        )
        _require_coefficients_close(
            row.native_lhs_wxyz,
            expected_left,
            tolerance=tolerance,
            context=f"{target} {row.case_id} lhs",
        )
        _require_coefficients_close(
            row.native_rhs_wxyz,
            expected_right,
            tolerance=tolerance,
            context=f"{target} {row.case_id} rhs",
        )
        expected_product = map_hamilton_quaternion_wxyz(
            multiply_wxyz(
                canonical_by_case[left_case_id], canonical_by_case[right_case_id]
            ),
            axis_map,
        )
        _require_coefficients_close(
            row.native_input_wxyz,
            expected_product,
            tolerance=tolerance,
            context=f"{target} {row.case_id} product",
        )
        basis, probe = _action(row)
        validate_quaternion_actions(
            row.native_input_wxyz,
            basis,
            tolerance=tolerance,
            context=f"{target} {row.case_id} composition",
            probe_vector_xyz=(
                map_vector_components(
                    context.canonical_inputs.probe_vector_xyz, axis_map
                )
                if probe is not None
                else None
            ),
            probe_image_xyz=probe,
        )

    constructor = inputs.native_constructed_orientation
    half_angle = constructor.angle_radians / 2.0
    expected_constructed = map_hamilton_quaternion_wxyz(
        [
            math.cos(half_angle),
            *(math.sin(half_angle) * component for component in constructor.axis_xyz),
        ],
        axis_map,
    )
    native_created = capture.observations.native_created
    if len(native_created) != 1 or native_created[0].case_id != constructor.case_id:
        raise ValueError(f"{target} native constructor case differs from its input")
    created = native_created[0]
    if isinstance(created, MuJoCoOrientation):
        if created.native_constructor != "mju_axisAngle2Quat(axis, angle)":
            raise ValueError("MuJoCo constructor API differs from its contract")
        created_coefficients = created.native_input_wxyz
        created_basis = created.native_mju_rotVecQuat_basis_xyz
    else:
        if created.native_constructor != "axisar(axis, angle) -> m2q(matrix)":
            raise ValueError("SpiceyPy constructor API differs from its contract")
        created_coefficients = created.native_m2q_wxyz
        created_basis = created.native_mxv_basis_xyz
    _require_coefficients_close(
        created_coefficients,
        expected_constructed,
        tolerance=tolerance,
        context=f"{target} native-created quaternion",
    )
    validate_quaternion_actions(
        created_coefficients,
        created_basis,
        tolerance=tolerance,
        context=f"{target} native-created quaternion",
    )


def _action(row: NativeOrientation) -> tuple[list[list[float]], list[float] | None]:
    """Select the target API's captured vector actions without recomputing them."""
    if isinstance(row, MuJoCoOrientation):
        return row.native_mju_rotVecQuat_basis_xyz, row.native_mju_rotVecQuat_probe_xyz
    return row.native_mxv_basis_xyz, row.native_mxv_probe_xyz


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> NativeQuaternionApiSerialization:
    """Normalize the selected native rows while retaining target-specific parsing."""
    contract, capture, inputs = package.contract, package.target_capture, package.inputs
    if isinstance(contract, MuJoCoContract):
        target = "MuJoCo"
        if not isinstance(capture, MuJoCoCapture) or not isinstance(
            inputs, MuJoCoInputs
        ):
            raise TypeError(
                "MuJoCo serializer requires typed MuJoCo capture and inputs"
            )
    elif isinstance(contract, SpiceyPyContract):
        target = "SpiceyPy"
        if not isinstance(capture, SpiceyPyCapture) or not isinstance(
            inputs, SpiceyPyInputs
        ):
            raise TypeError(
                "SpiceyPy serializer requires typed SpiceyPy capture and inputs"
            )
    else:
        raise TypeError("direct quaternion serializer received an unsupported contract")

    validate_direct_canonical_inputs(inputs, context, target=target)
    _validate_native_coefficients(
        contract,
        capture,
        inputs,
        context,
        target=target,
    )
    observations = capture.observations
    retained = select_retained_rows(
        observations.canonical_to_native,
        canonical_orientation_selection(inputs.retained_case_ids, context=target),
        context=target,
    )
    witnesses: list[QuaternionBoundaryWitness] = []
    for row in retained:
        basis, probe = _action(row)
        witnesses.append(
            OrientationWitness(
                kind="orientation",
                case_id=row.case_id,
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=basis,
                probe_image_xyz=probe,
            )
        )

    constructor = inputs.native_constructed_orientation
    if isinstance(capture, MuJoCoCapture):
        for row in capture.observations.native_created:
            basis, _ = _action(row)
            witnesses.append(
                OrientationWitness(
                    kind="orientation",
                    case_id=row.case_id,
                    precision="float64",
                    coefficient_order="wxyz",
                    coefficients=row.native_input_wxyz,
                    basis_images_xyz=basis,
                    constructor_axis_xyz=constructor.axis_xyz,
                    constructor_angle_radians=constructor.angle_radians,
                )
            )
    else:
        witnesses.extend(
            OrientationWitness(
                kind="orientation",
                case_id=row.case_id,
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_m2q_wxyz,
                basis_images_xyz=row.native_mxv_basis_xyz,
                constructor_axis_xyz=constructor.axis_xyz,
                constructor_angle_radians=constructor.angle_radians,
            )
            for row in capture.observations.native_created
        )

    for row in observations.composition:
        if row.product_order is None:
            raise ValueError(
                f"{target} product {row.case_id!r} lacks its observed product order"
            )
        left, right = native_product_case_ids(row.case_id)
        basis, _ = _action(row)
        witnesses.append(
            CompositionWitness(
                kind="composition",
                case_id=row.case_id,
                left_case_id=left,
                right_case_id=right,
                product_order=row.product_order,
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=basis,
            )
        )
    return NativeQuaternionApiSerialization(
        layer="native-quaternion-api",
        contract=serialize_direct_quaternion_contract(contract),
        witnesses=witnesses,
    )
