"""Semantic serialization for the SpiceyPy/CSPICE quaternion experiment."""

from __future__ import annotations

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.contracts import serialize_direct_quaternion_contract
from scripts.evidence_schema.converter_captures import (
    SpiceyPyCapture,
    SpiceyPyContract,
    SpiceyPyInputs,
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
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    native_product_case_ids as _native_product_case_ids,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> NativeQuaternionApiSerialization:
    """Map CSPICE's q2m/mxv, m2q, and product observations."""
    if not isinstance(package.contract, SpiceyPyContract):
        raise TypeError("SpiceyPy serializer received a non-SpiceyPy contract")
    if not isinstance(package.target_capture, SpiceyPyCapture):
        raise TypeError("SpiceyPy serializer received a non-SpiceyPy capture")
    if not isinstance(package.inputs, SpiceyPyInputs):
        raise TypeError("SpiceyPy serializer requires typed SpiceyPy inputs")
    validate_direct_canonical_inputs(package.inputs, context, target="SpiceyPy")
    return NativeQuaternionApiSerialization(
        layer="native-quaternion-api",
        contract=serialize_direct_quaternion_contract(package.contract),
        witnesses=serialize_spiceypy_witnesses(package.target_capture, package.inputs),
    )


def serialize_spiceypy_witnesses(
    capture: SpiceyPyCapture, inputs: SpiceyPyInputs
) -> list[QuaternionBoundaryWitness]:
    observations = capture.observations
    retained_case_ids = canonical_orientation_selection(
        inputs.retained_case_ids, context="SpiceyPy"
    )
    witnesses: list[QuaternionBoundaryWitness] = []
    retained_rows = select_retained_rows(
        observations.canonical_to_native,
        retained_case_ids,
        context="SpiceyPy",
    )
    witnesses.extend(
        OrientationWitness(
            kind="orientation",
            case_id=row.case_id,
            precision="float64",
            coefficient_order="wxyz",
            coefficients=row.native_input_wxyz,
            basis_images_xyz=row.native_mxv_basis_xyz,
            probe_image_xyz=row.native_mxv_probe_xyz,
        )
        for row in retained_rows
    )
    witnesses.extend(
        OrientationWitness(
            kind="orientation",
            case_id=row.case_id,
            precision="float64",
            coefficient_order="wxyz",
            coefficients=row.native_m2q_wxyz,
            basis_images_xyz=row.native_mxv_basis_xyz,
            constructor_axis_xyz=inputs.native_constructed_orientation.axis_xyz,
            constructor_angle_radians=inputs.native_constructed_orientation.angle_radians,
        )
        for row in observations.native_created
    )
    for row in observations.composition:
        if row.product_order is None:
            raise ValueError(
                f"SpiceyPy product {row.case_id!r} lacks its observed product order"
            )
        left_case_id, right_case_id = _native_product_case_ids(row.case_id)
        witnesses.append(
            CompositionWitness(
                kind="composition",
                case_id=row.case_id,
                left_case_id=left_case_id,
                right_case_id=right_case_id,
                product_order=row.product_order,
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=row.native_mxv_basis_xyz,
            )
        )
    return witnesses
