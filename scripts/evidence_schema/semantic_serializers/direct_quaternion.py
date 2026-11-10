"""Serialize MuJoCo and CSPICE native quaternion actions into shared witnesses."""

from __future__ import annotations

from typing import Union

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
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientation_selection,
    native_product_case_ids,
    select_retained_rows,
    validate_direct_canonical_inputs,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)

NativeOrientation = Union[MuJoCoOrientation, SpiceyPyOrientation]


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
