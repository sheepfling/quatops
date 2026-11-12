"""Semantic serializer for explicitly contracted engine rotation captures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from quatops.scalar import multiply as multiply_wxyz
from scripts.evidence_schema.crystallized import (
    CanonicalInputs,
    CompositionWitness,
    CoordinateFrame,
    EngineQuaternionSerialization,
    FrameSemantics,
    OrientationWitness,
    PrecisionTolerance,
    QuaternionBoundaryContract,
    QuaternionBoundaryWitness,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.engine_captures import (
    ENGINE_RETAINED_CASE_IDS,
    EngineCaptureDocument,
    EngineComposition,
    EngineContractDocument,
    EngineOrientation,
)
from scripts.evidence_schema.primitives import (
    AxisMeaning,
    NativeScalarProfile,
    mapped_handedness,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    derive_signed_axis_map,
    map_hamilton_quaternion_wxyz,
    map_vector_components,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    select_retained_rows,
    validate_quaternion_actions,
)


class EngineSemanticPackage(Protocol):
    """Loaded engine contract and native observation pair."""

    contract_path: Path
    witness_path: Path
    contract: EngineContractDocument
    capture: EngineCaptureDocument


def _native_coefficients(
    canonical_wxyz: list[float],
    package: EngineSemanticPackage,
) -> list[float]:
    """Apply the declared frame basis and native coefficient order."""
    mapped_wxyz = map_hamilton_quaternion_wxyz(
        canonical_wxyz,
        package.contract.conversion.destination_axes_from_source,
    )
    if package.contract.engine.native_quaternion_order == "wxyz":
        return mapped_wxyz
    return mapped_wxyz[1:] + mapped_wxyz[:1]


def _native_wxyz(coefficients: list[float], order: str) -> list[float]:
    if order == "wxyz":
        return coefficients
    return [coefficients[3], *coefficients[:3]]


def _require_coefficients_close(
    actual: list[float],
    expected: list[float],
    *,
    tolerance: float,
    context: str,
) -> None:
    """Check captured target coefficients against the declared conversion."""
    if len(actual) != len(expected) or any(
        not math.isclose(value, wanted, rel_tol=0.0, abs_tol=tolerance)
        for value, wanted in zip(actual, expected)
    ):
        raise ValueError(f"{context} differs from its declared frame conversion")


def _validate_frame_map(
    package: EngineSemanticPackage,
) -> tuple[AxisMeaning, AxisMeaning, AxisMeaning]:
    conversion = package.contract.conversion
    axis_map = conversion.destination_axes_from_source
    derived_handedness = mapped_handedness(
        "right-handed", axis_map, context="native frame map"
    )
    canonical = conversion.canonical_frame
    declared_labels = (
        conversion.native_frame.positive_x,
        conversion.native_frame.positive_y,
        conversion.native_frame.positive_z,
    )
    derived_map = derive_signed_axis_map(
        (canonical.positive_x, canonical.positive_y, canonical.positive_z),
        declared_labels,
    )
    if axis_map != derived_map:
        raise ValueError("declared native axis meanings disagree with signed map")
    declared_handedness = package.contract.conversion.native_frame.handedness
    if declared_handedness is not None:
        expected_hand = (
            "right-handed"
            if declared_handedness.lower() in {"right", "right-handed"}
            else "left-handed"
            if declared_handedness.lower() in {"left", "left-handed"}
            else None
        )
        if expected_hand != derived_handedness:
            raise ValueError(
                "declared native handedness disagrees with signed-axis map"
            )
    return declared_labels


def _validate_capture_contract(package: EngineSemanticPackage) -> None:
    """Require native identity and interoperability labels to match the contract."""
    engine = package.contract.engine
    capture_target = package.capture.target
    for field in (
        "engine",
        "api",
        "version",
        "native_scalar_profiles",
        "native_quaternion_order",
        "native_algebra",
        "native_vector_action",
    ):
        if getattr(capture_target, field) != getattr(engine, field):
            raise ValueError(f"engine capture {field} differs from its contract")
    interop = package.capture.canonical_interop
    conversion = package.contract.conversion
    if (
        interop.contract_id != conversion.contract_id
        or interop.native_quaternion_order != engine.native_quaternion_order
        or interop.frame_map_destination_axes_from_source
        != conversion.destination_axes_from_source
    ):
        raise ValueError("engine interoperability labels differ from its contract")


def _profile_cases(
    package: EngineSemanticPackage,
) -> list[
    tuple[
        NativeScalarProfile,
        list[EngineOrientation],
        EngineComposition,
        EngineComposition,
    ]
]:
    interop = package.capture.canonical_interop
    declared_profiles = package.contract.engine.native_scalar_profiles
    tolerances = package.contract.conversion.native_precision_tolerance
    if len(set(declared_profiles)) != len(declared_profiles):
        raise ValueError("engine contract contains duplicate scalar profiles")
    if set(tolerances) != set(declared_profiles):
        raise ValueError("engine tolerances differ from the declared scalar profiles")
    if interop.precision_captures is not None:
        if set(interop.precision_captures) != set(declared_profiles):
            raise ValueError(
                "engine native captures differ from the declared scalar profiles"
            )
        return [
            (
                profile,
                capture.orientation_cases,
                capture.composition,
                capture.reverse_composition,
            )
            for profile, capture in sorted(interop.precision_captures.items())
        ]
    if (
        interop.orientation_cases is None
        or interop.composition is None
        or interop.reverse_composition is None
    ):
        raise ValueError("engine capture lacks either profiles or flat witnesses")
    if len(declared_profiles) != 1:
        raise ValueError("flat engine capture must declare one precision profile")
    return [
        (
            declared_profiles[0],
            interop.orientation_cases,
            interop.composition,
            interop.reverse_composition,
        )
    ]


def _orientation_witnesses(
    package: EngineSemanticPackage, canonical_inputs: CanonicalInputs
) -> list[QuaternionBoundaryWitness]:
    identity = package.contract.engine
    shared = {
        row.case_id: row.coefficients_wxyz for row in canonical_inputs.orientations
    }
    retained_case_ids = ENGINE_RETAINED_CASE_IDS
    output: list[QuaternionBoundaryWitness] = []
    for precision, cases, composition, reverse in _profile_cases(package):
        tolerance = package.contract.conversion.native_precision_tolerance[precision]
        retained_rows = select_retained_rows(
            cases, retained_case_ids, context=f"{identity.engine} {precision}"
        )
        for row in retained_rows:
            if row.case_id not in shared:
                raise ValueError(
                    f"{identity.engine} {row.case_id} is absent from shared inputs"
                )
            _require_coefficients_close(
                row.native_quaternion,
                _native_coefficients(shared[row.case_id], package),
                tolerance=tolerance,
                context=f"{identity.engine} {precision} {row.case_id}",
            )
            probe_image = row.native_rotated_probe_vector_xyz
            if row.case_id == "mixed_a" and probe_image is None:
                raise ValueError(f"{identity.engine} mixed_a lacks its probe action")
            validate_quaternion_actions(
                _native_wxyz(row.native_quaternion, identity.native_quaternion_order),
                row.native_rotated_positive_basis_vectors_xyz,
                tolerance=tolerance,
                context=f"{identity.engine} {precision} {row.case_id}",
                probe_vector_xyz=(
                    map_vector_components(
                        canonical_inputs.probe_vector_xyz,
                        package.contract.conversion.destination_axes_from_source,
                    )
                    if probe_image is not None
                    else None
                ),
                probe_image_xyz=probe_image,
            )
            output.append(
                OrientationWitness(
                    kind="orientation",
                    case_id=row.case_id,
                    precision=precision,
                    coefficient_order=identity.native_quaternion_order,
                    coefficients=row.native_quaternion,
                    basis_images_xyz=row.native_rotated_positive_basis_vectors_xyz,
                    probe_image_xyz=(
                        probe_image
                        if row.case_id == "mixed_a" and probe_image is not None
                        else None
                    ),
                )
            )
        expected_orders = (
            (
                canonical_inputs.composition_left_case_id,
                canonical_inputs.composition_right_case_id,
                canonical_inputs.product_order,
            ),
            (
                canonical_inputs.composition_right_case_id,
                canonical_inputs.composition_left_case_id,
                canonical_inputs.product_order,
            ),
        )
        for row, expected in zip((composition, reverse), expected_orders):
            if (row.left_case_id, row.right_case_id, row.product_order) != expected:
                raise ValueError(
                    "native composition cases differ from shared product orders"
                )
            if row.left_case_id not in shared or row.right_case_id not in shared:
                raise ValueError(
                    "native composition references an unknown canonical case"
                )
            expected_product = multiply_wxyz(
                shared[row.left_case_id], shared[row.right_case_id]
            )
            _require_coefficients_close(
                row.native_product_quaternion,
                _native_coefficients(expected_product, package),
                tolerance=tolerance,
                context=(
                    f"{identity.engine} {precision} composition "
                    f"{row.left_case_id} * {row.right_case_id}"
                ),
            )
            validate_quaternion_actions(
                _native_wxyz(
                    row.native_product_quaternion, identity.native_quaternion_order
                ),
                row.native_rotated_positive_basis_vectors_xyz,
                tolerance=tolerance,
                context=(
                    f"{identity.engine} {precision} composition "
                    f"{row.left_case_id} * {row.right_case_id}"
                ),
            )
            output.append(
                CompositionWitness(
                    kind="composition",
                    case_id=f"{row.left_case_id}_times_{row.right_case_id}",
                    left_case_id=row.left_case_id,
                    right_case_id=row.right_case_id,
                    product_order=row.product_order,
                    precision=precision,
                    coefficient_order=identity.native_quaternion_order,
                    coefficients=row.native_product_quaternion,
                    basis_images_xyz=row.native_rotated_positive_basis_vectors_xyz,
                )
            )
    return output


def serialize_engine_semantics(
    package: EngineSemanticPackage,
    canonical_inputs: CanonicalInputs,
) -> EngineQuaternionSerialization:
    """Interpret a native engine package under its declared frame contract."""
    contract = package.contract
    engine = contract.engine
    conversion = contract.conversion
    _validate_capture_contract(package)
    declared_labels = _validate_frame_map(package)
    handedness = mapped_handedness(
        "right-handed",
        conversion.destination_axes_from_source,
        context="native frame map",
    )
    source = conversion.canonical_frame
    if source.handedness.lower() not in {"right", "right-handed"}:
        raise ValueError("engine interoperability source frame must be right-handed")
    if conversion.canonical_action != "active":
        raise ValueError("engine native witness contract must declare active rotations")
    if (
        conversion.canonical_algebra != canonical_inputs.contract.algebra
        or conversion.canonical_order != canonical_inputs.contract.coefficient_order
        or conversion.canonical_action != canonical_inputs.contract.rotation_action
    ):
        raise ValueError(
            "engine canonical quaternion contract differs from shared inputs"
        )
    if (
        source.positive_x != canonical_inputs.source_frame.positive_x
        or source.positive_y != canonical_inputs.source_frame.positive_y
        or source.positive_z != canonical_inputs.source_frame.positive_z
    ):
        raise ValueError("engine source frame differs from shared canonical inputs")
    native_frame = CoordinateFrame(
        name=f"{engine.engine} declared native coordinate basis",
        handedness=handedness,
        positive_x=declared_labels[0],
        positive_y=declared_labels[1],
        positive_z=declared_labels[2],
    )
    common_contract = QuaternionBoundaryContract(
        kind="quaternion_boundary",
        canonical_algebra=conversion.canonical_algebra,
        canonical_order=conversion.canonical_order,
        rotation_action=conversion.canonical_action,
        source_frame=CoordinateFrame(
            name="QuatOps canonical frame",
            handedness=(
                "right-handed"
                if source.handedness.lower() in {"right", "right-handed"}
                else "left-handed"
            ),
            positive_x=source.positive_x,
            positive_y=source.positive_y,
            positive_z=source.positive_z,
        ),
        source_frame_semantics=canonical_source_frame_semantics(),
        destination_frame=native_frame,
        destination_frame_semantics=FrameSemantics(
            role=(
                "numeric_probe"
                if conversion.native_frame.declared_probe_semantics is not None
                else "native_coordinate_basis"
            ),
            authority=(
                "fixture_only"
                if conversion.native_frame.declared_probe_semantics is not None
                else "application_declared"
            ),
            statement=(
                conversion.native_frame.declared_probe_semantics
                or (
                    "The versioned converter contract explicitly assigns this "
                    "target axis basis for the captured API. The capture does "
                    "not establish a universal asset or model-forward axis."
                )
            ),
        ),
        destination_axes_from_source=conversion.destination_axes_from_source,
        native_algebra=engine.native_algebra,
        native_order=engine.native_quaternion_order,
        native_vector_action=engine.native_vector_action,
        conversion_rule=conversion.conversion_rule,
        inverse_rule=conversion.inverse_rule,
        input_domain=conversion.input_domain,
        tolerances=[
            PrecisionTolerance(profile=profile, absolute=tolerance, relative=0.0)
            for profile, tolerance in sorted(
                conversion.native_precision_tolerance.items()
            )
        ],
        scope=[
            conversion.scope,
            "native basis actions for selected orientations",
            "both orders of a noncommuting native product",
        ],
        excludes=conversion.excludes,
    )
    return EngineQuaternionSerialization(
        layer="engine-quaternion",
        contract=common_contract,
        witnesses=_orientation_witnesses(package, canonical_inputs),
    )
