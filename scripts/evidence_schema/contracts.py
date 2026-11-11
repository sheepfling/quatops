"""Shared contract constructors for common native rotation API families."""

from __future__ import annotations

from scripts.evidence_schema.converter_captures import (
    EigenBoundaryContract,
    EigenContract,
    MuJoCoBoundaryContract,
    MuJoCoContract,
    SpiceyPyContract,
)
from scripts.evidence_schema.crystallized import (
    QuaternionBoundaryContract,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    canonical_frame,
    canonical_order,
    coefficient_order,
    native_order,
    numeric_probe_semantics,
    probe_frame,
    signed_axis_map,
)
from scripts.evidence_schema.tolerances import tolerances_for_target


def serialize_direct_quaternion_contract(
    contract: EigenContract | MuJoCoContract | SpiceyPyContract,
) -> QuaternionBoundaryContract:
    """Serialize the shared direct-quaternion boundary declared by one library."""
    boundary = contract.target_boundary
    canonical_coefficient_order = canonical_order(contract)
    native_coefficient_order = native_order(boundary)
    conversion = boundary.canonical_to_native
    inverse = boundary.native_to_canonical
    if not conversion or not inverse:
        raise ValueError("direct quaternion contract must declare both directions")
    if isinstance(boundary, EigenBoundaryContract):
        native_storage = (
            coefficient_order(boundary.coeffs_access_order)
            if boundary.coeffs_access_order
            else None
        )
        native_action = boundary.vector_action
    elif isinstance(boundary, MuJoCoBoundaryContract):
        native_storage = None
        native_action = boundary.observed_quat2mat_vector_action
    else:
        native_storage = None
        native_action = boundary.observed_q2m_vector_action
    return QuaternionBoundaryContract(
        kind="quaternion_boundary",
        canonical_algebra="Hamilton",
        canonical_order=canonical_coefficient_order,
        rotation_action="active",
        source_frame=canonical_frame(contract),
        source_frame_semantics=canonical_source_frame_semantics(),
        destination_frame=probe_frame(contract),
        destination_frame_semantics=numeric_probe_semantics(
            boundary.probe_frame_assignment.meaning
        ),
        destination_axes_from_source=signed_axis_map(
            boundary.source_to_target_coordinate_map
        ),
        native_algebra="Hamilton",
        native_order=native_coefficient_order,
        native_storage_order=native_storage,
        native_vector_action=(
            native_action
            or "native quaternion action captured on the declared coordinate basis"
        ),
        conversion_rule=conversion,
        inverse_rule=inverse,
        input_domain=contract.canonical.normalization,
        tolerances=tolerances_for_target(contract.target.name),
        scope=[
            "canonical-to-native coefficient boundary",
            "native quaternion action on three basis vectors",
            "native-created orientation for reverse conversion",
            "both orders of a noncommuting rotation product",
        ],
        excludes=contract.excludes,
    )
