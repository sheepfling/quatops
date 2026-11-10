"""Semantic serializer for explicitly contracted engine rotation captures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Protocol

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
    EngineCaptureDocument,
    EngineComposition,
    EngineContractDocument,
    EngineOrientation,
    NativeVector3,
    Vector3Object,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientation_selection,
    select_retained_rows,
)


class EngineSemanticPackage(Protocol):
    """Loaded engine contract and native observation pair."""

    contract_path: Path
    witness_path: Path
    contract: EngineContractDocument
    capture: EngineCaptureDocument


def _vector(value: NativeVector3) -> list[float]:
    if isinstance(value, Vector3Object):
        return [value.x, value.y, value.z]
    return value


def _handedness_from_map(
    axis_map: list[str],
) -> Literal["right-handed", "left-handed"]:
    axes = [value.lstrip("-") for value in axis_map]
    inversions = sum(
        axes[left] > axes[right] for left in range(3) for right in range(left + 1, 3)
    )
    sign = -1 if inversions % 2 else 1
    if sum(value.startswith("-") for value in axis_map) % 2:
        sign *= -1
    return "right-handed" if sign > 0 else "left-handed"


def _opposite_axis_meaning(value: str) -> str:
    opposites = {
        "right": "left",
        "left": "right",
        "east": "west",
        "west": "east",
        "forward": "backward",
        "backward": "forward",
        "north": "south",
        "south": "north",
        "up": "down",
        "down": "up",
    }
    if value not in opposites:
        raise ValueError(f"cannot invert declared axis meaning {value!r}")
    return opposites[value]


def _native_axis_labels(package: EngineSemanticPackage) -> tuple[str, str, str]:
    frame = package.contract.conversion.native_frame
    labels = (frame.positive_x, frame.positive_y, frame.positive_z)
    if all(label is not None for label in labels):
        return labels  # type: ignore[return-value]
    declaration = frame.declared_probe_semantics
    if declaration is None:
        raise ValueError("native frame must declare positive axis meanings")
    match = re.search(r"\+X\s+([^,]+),\s*\+Y\s+([^,]+),\s*\+Z\s+(.+)", declaration)
    if match is None:
        raise ValueError("native probe frame declaration is not parseable")
    return (
        match.group(1).strip(),
        match.group(2).strip(),
        re.split(r"\s+for\s+", match.group(3).strip(), maxsplit=1)[0],
    )


def _validate_frame_map(package: EngineSemanticPackage) -> tuple[str, str, str]:
    conversion = package.contract.conversion
    axis_map = conversion.destination_axes_from_source
    normalized_axes = [axis.lstrip("-") for axis in axis_map]
    if (
        len(axis_map) != 3
        or any(axis not in {"x", "y", "z", "-x", "-y", "-z"} for axis in axis_map)
        or sorted(normalized_axes) != ["x", "y", "z"]
    ):
        raise ValueError("native frame map must be a signed-axis permutation")
    canonical = conversion.canonical_frame
    declared_labels = _native_axis_labels(package)
    canonical_labels = (
        canonical.positive_x,
        canonical.positive_y,
        canonical.positive_z,
    )
    for destination_index, axis in enumerate(axis_map):
        source_index = "xyz".index(axis.lstrip("-"))
        expected = canonical_labels[source_index]
        if axis.startswith("-"):
            expected = _opposite_axis_meaning(expected)
        if declared_labels[destination_index] != expected:
            raise ValueError(
                f"declared native axis {destination_index} disagrees with signed map"
            )
    derived_handedness = _handedness_from_map(axis_map)
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


def _validate_probe_vectors(
    package: EngineSemanticPackage, canonical_inputs: CanonicalInputs
) -> None:
    conversion = package.contract.conversion
    if len(canonical_inputs.probe_vector_xyz) != 3:
        raise ValueError("shared canonical input must include a three-vector probe")
    for profile, cases, _composition, _reverse in _profile_cases(package):
        tolerance = conversion.native_precision_tolerance.get(profile)
        if tolerance is None:
            raise ValueError(f"engine contract lacks tolerance for {profile}")
        probe_cases = [row for row in cases if row.case_id == "mixed_a"]
        if (
            len(probe_cases) != 1
            or probe_cases[0].native_rotated_probe_vector_xyz is None
        ):
            raise ValueError(
                f"engine {profile} witnesses must retain one mixed-vector action"
            )


def _profile_cases(
    package: EngineSemanticPackage,
) -> list[tuple[str, list[EngineOrientation], EngineComposition, EngineComposition]]:
    interop = package.capture.canonical_interop
    if interop.precision_captures is not None:
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
    if len(package.contract.engine.native_scalar_profiles) != 1:
        raise ValueError("flat engine capture must declare one precision profile")
    return [
        (
            package.contract.engine.native_scalar_profiles[0],
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
    retained_case_ids = canonical_orientation_selection(
        list(shared), context=identity.engine
    )
    output: list[QuaternionBoundaryWitness] = []
    for precision, cases, composition, reverse in _profile_cases(package):
        retained_rows = select_retained_rows(
            cases, retained_case_ids, context=f"{identity.engine} {precision}"
        )
        for row in retained_rows:
            if row.case_id not in shared:
                raise ValueError(
                    f"{identity.engine} {row.case_id} is absent from shared inputs"
                )
            probe_image = row.native_rotated_probe_vector_xyz
            if row.case_id == "mixed_a" and probe_image is None:
                raise ValueError(f"{identity.engine} mixed_a lacks its probe action")
            output.append(
                OrientationWitness(
                    kind="orientation",
                    case_id=row.case_id,
                    precision=precision,
                    coefficient_order=identity.native_quaternion_order,
                    coefficients=row.native_quaternion,
                    basis_images_xyz=[
                        _vector(value)
                        for value in row.native_rotated_positive_basis_vectors_xyz
                    ],
                    probe_image_xyz=(
                        _vector(probe_image)
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
                    basis_images_xyz=[
                        _vector(value)
                        for value in row.native_rotated_positive_basis_vectors_xyz
                    ],
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
    declared_labels = _validate_frame_map(package)
    _validate_probe_vectors(package, canonical_inputs)
    handedness = _handedness_from_map(conversion.destination_axes_from_source)
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
