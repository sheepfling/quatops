"""Normalize target semantic serializations into the shared evidence product."""

from __future__ import annotations

from collections.abc import Sequence

from scripts.evidence_schema.crystallized import (
    CanonicalInputs,
    CoordinateFrame,
    EngineQuaternionSerialization,
    EvidenceContract,
    FrameChainContract,
    FrameChainSerialization,
    FrameSemantics,
    GeographicContract,
    GeographicSerialization,
    NativeQuaternionApiSerialization,
    QuaternionBoundaryContract,
    RotationVectorBoundaryContract,
    RotationVectorSerialization,
    SceneTransportContract,
    SceneTransportSerialization,
    SemanticSerialization,
)


def _validate_documented_frame(
    reference: str | None,
    *,
    context: str,
) -> None:
    if reference is None or not reference.strip():
        raise ValueError(
            f"{context} is marked target_documented without an evidence reference"
        )


def _validate_frame_semantics(semantics: FrameSemantics, *, context: str) -> None:
    if not semantics.role.strip():
        raise ValueError(f"{context} has an empty semantic role")
    if not semantics.statement.strip():
        raise ValueError(f"{context} has an empty semantic statement")
    if semantics.authority == "target_documented":
        _validate_documented_frame(semantics.evidence_reference, context=context)


def _validate_coordinate_frame(frame: CoordinateFrame, *, context: str) -> None:
    if not frame.name.strip():
        raise ValueError(f"{context} has an empty frame name")
    if any(
        not axis.strip()
        for axis in (frame.positive_x, frame.positive_y, frame.positive_z)
    ):
        raise ValueError(f"{context} must declare all three positive axes")


def _validate_signed_axis_map(values: list[str], *, context: str) -> None:
    if len(values) != 3 or any(
        value not in {"x", "-x", "y", "-y", "z", "-z"} for value in values
    ):
        raise ValueError(f"{context} is not a signed-axis triple")
    if len({value.lstrip("-") for value in values}) != 3:
        raise ValueError(f"{context} is not a signed-axis permutation")


def _mapped_handedness(source: str, values: list[str], *, context: str) -> str:
    """Return the destination handedness implied by a signed basis map."""
    _validate_signed_axis_map(values, context=context)
    axes = [value.lstrip("-") for value in values]
    inversions = sum(
        axes[left] > axes[right] for left in range(3) for right in range(left + 1, 3)
    )
    negative_axes = sum(value.startswith("-") for value in values)
    determinant = -1 if (inversions + negative_axes) % 2 else 1
    source_sign = 1 if source == "right-handed" else -1
    return "right-handed" if determinant * source_sign > 0 else "left-handed"


def _validate_mapped_frame_handedness(
    source: CoordinateFrame,
    destination: CoordinateFrame,
    axis_map: list[str],
    *,
    context: str,
) -> None:
    expected = _mapped_handedness(
        source.handedness, axis_map, context=f"{context} axis map"
    )
    if destination.handedness != expected:
        raise ValueError(f"{context} handedness disagrees with its signed-axis map")


def _validate_contract_semantics(contract: EvidenceContract) -> None:
    if isinstance(
        contract, (QuaternionBoundaryContract, RotationVectorBoundaryContract)
    ):
        _validate_coordinate_frame(contract.source_frame, context="source frame")
        _validate_coordinate_frame(
            contract.destination_frame, context="destination frame"
        )
        for label, semantics in (
            ("source frame", contract.source_frame_semantics),
            ("destination frame", contract.destination_frame_semantics),
        ):
            _validate_frame_semantics(semantics, context=label)
        _validate_mapped_frame_handedness(
            contract.source_frame,
            contract.destination_frame,
            contract.destination_axes_from_source,
            context=(
                "quaternion destination frame"
                if isinstance(contract, QuaternionBoundaryContract)
                else "rotation-vector destination frame"
            ),
        )
        return
    if isinstance(contract, FrameChainContract):
        _validate_coordinate_frame(contract.canonical_frame, context="canonical frame")
        _validate_frame_semantics(
            contract.canonical_frame_semantics, context="canonical frame"
        )
        frame_ids = [mapping.frame for mapping in contract.frame_mappings]
        if any(not frame_id.strip() for frame_id in frame_ids):
            raise ValueError("frame-chain contract contains an empty frame ID")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame-chain contract declares duplicate frame IDs")
        for mapping in contract.frame_mappings:
            _validate_frame_semantics(
                mapping.semantics, context=f"frame {mapping.frame!r}"
            )
            _validate_signed_axis_map(
                mapping.axes_from_canonical,
                context=f"frame {mapping.frame!r} axis map",
            )
        return
    if isinstance(contract, GeographicContract):
        _validate_coordinate_frame(contract.canonical_frame, context="canonical frame")
        _validate_coordinate_frame(contract.local_frame, context="local frame")
        for label, semantics in (
            ("canonical frame", contract.canonical_frame_semantics),
            ("local frame", contract.local_frame_semantics),
        ):
            _validate_frame_semantics(semantics, context=label)
        _validate_mapped_frame_handedness(
            contract.canonical_frame,
            contract.local_frame,
            contract.local_axes_from_canonical,
            context="geographic local frame",
        )
        return
    if isinstance(contract, SceneTransportContract):
        _validate_coordinate_frame(contract.canonical_frame, context="canonical frame")
        _validate_coordinate_frame(contract.target_frame, context="target frame")
        for label, semantics in (
            ("canonical frame", contract.canonical_frame_semantics),
            ("target frame", contract.target_frame_semantics),
        ):
            _validate_frame_semantics(semantics, context=label)
        expected_role = {
            "caller_assigned_probe": "numeric_probe",
            "declared_stage": "stage",
        }[contract.target_frame_role]
        if contract.target_frame_semantics.role != expected_role:
            raise ValueError(
                "scene target frame role disagrees with its semantic declaration"
            )
        expected_authority = {
            "caller_assigned_probe": "fixture_only",
            "declared_stage": "application_declared",
        }[contract.target_frame_role]
        if contract.target_frame_semantics.authority != expected_authority:
            raise ValueError(
                "scene target frame authority disagrees with its declared role"
            )
        _validate_mapped_frame_handedness(
            contract.canonical_frame,
            contract.target_frame,
            contract.destination_axes_from_source,
            context="scene target frame",
        )
        return
    raise TypeError(f"unsupported evidence contract {type(contract).__name__}")


def _validate_source_frame(
    contract: EvidenceContract, canonical_inputs: CanonicalInputs
) -> None:
    if isinstance(
        contract, (QuaternionBoundaryContract, RotationVectorBoundaryContract)
    ):
        frame = contract.source_frame
        semantics = contract.source_frame_semantics
    elif isinstance(
        contract, (FrameChainContract, GeographicContract, SceneTransportContract)
    ):
        frame = contract.canonical_frame
        semantics = contract.canonical_frame_semantics
    else:
        raise TypeError(f"unsupported evidence contract {type(contract).__name__}")

    expected = canonical_inputs.source_frame
    if (
        frame.handedness != expected.handedness
        or frame.positive_x != expected.positive_x
        or frame.positive_y != expected.positive_y
        or frame.positive_z != expected.positive_z
    ):
        raise ValueError("target serializer source frame differs from shared inputs")
    expected_semantics = canonical_inputs.source_frame_semantics
    if semantics != expected_semantics:
        raise ValueError(
            "target serializer source-frame semantics differ from shared inputs"
        )


def _validate_contract_basis(
    contract: EvidenceContract, canonical_inputs: CanonicalInputs
) -> None:
    if isinstance(
        contract, (QuaternionBoundaryContract, RotationVectorBoundaryContract)
    ):
        if (
            contract.canonical_algebra != canonical_inputs.contract.algebra
            or contract.canonical_order != canonical_inputs.contract.coefficient_order
            or contract.rotation_action != canonical_inputs.contract.rotation_action
        ):
            raise ValueError(
                "target serializer rotation basis differs from shared inputs"
            )
        if (
            contract.source_frame.handedness
            != canonical_inputs.contract.coordinate_system
        ):
            raise ValueError(
                "target serializer source handedness differs from shared inputs"
            )


def _validate_witnesses(
    serialization: SemanticSerialization, native_precision: Sequence[str]
) -> None:
    if not serialization.witnesses:
        raise ValueError("target semantic serialization contains no witnesses")
    if isinstance(
        serialization,
        (EngineQuaternionSerialization, NativeQuaternionApiSerialization),
    ):
        contract = serialization.contract
        witnesses = serialization.witnesses
        for witness in witnesses:
            if witness.coefficient_order != contract.native_order:
                raise ValueError(
                    f"{witness.case_id!r} coefficient order disagrees with "
                    "the normalized quaternion contract"
                )
            if witness.precision not in native_precision:
                raise ValueError(
                    f"{witness.case_id!r} precision is absent from target metadata"
                )
        return

    if isinstance(serialization, RotationVectorSerialization):
        witnesses = serialization.witnesses
        if any(witness.precision not in native_precision for witness in witnesses):
            raise ValueError(
                "rotation-vector witness precision is absent from target metadata"
            )
        return

    if isinstance(serialization, FrameChainSerialization):
        contract = serialization.contract
        witnesses = serialization.witnesses
        mappings = {row.frame for row in contract.frame_mappings}
        for witness in witnesses:
            transforms = (
                *witness.intermediate_transforms,
                *witness.lookups,
                *witness.message_roundtrips,
            )
            if any(
                transform.coefficient_order != contract.quaternion_order
                for transform in transforms
            ) or any(
                message.coefficient_order != contract.quaternion_order
                for message in witness.messages
            ):
                raise ValueError(
                    "frame-chain witness coefficient order disagrees with "
                    "the normalized contract"
                )
            observed = {
                frame
                for transform in transforms
                for frame in (transform.parent_frame, transform.child_frame)
            }
            observed.update(
                frame
                for message in witness.messages
                for frame in (message.parent_frame, message.child_frame)
            )
            observed.update(
                frame
                for action in witness.point_actions
                for frame in (action.source_frame, action.target_frame)
            )
            if not observed <= mappings:
                raise ValueError(
                    f"frame-chain witnesses use undeclared frames: "
                    f"{sorted(observed - mappings)}"
                )
        return

    if isinstance(serialization, GeographicSerialization):
        contract = serialization.contract
        witnesses = serialization.witnesses
        for witness in witnesses:
            if witness.coefficient_order != contract.quaternion_order:
                raise ValueError(
                    f"geographic orientation {witness.case_id!r} coefficient order "
                    "disagrees with the normalized contract"
                )
        return

    if isinstance(serialization, SceneTransportSerialization):
        contract = serialization.contract
        witnesses = serialization.witnesses
        for witness in witnesses:
            if witness.coefficient_order != contract.quaternion_order:
                raise ValueError(
                    f"scene node {witness.node_id!r} coefficient order disagrees "
                    "with the normalized scene contract"
                )
        return

    raise TypeError(
        f"unsupported semantic serialization {type(serialization).__name__}"
    )


def normalize_capture(
    serialization: SemanticSerialization,
    canonical_inputs: CanonicalInputs,
    *,
    native_precision: Sequence[str],
) -> None:
    """Validate a target's typed witnesses against its explicit frame contract."""
    contract = serialization.contract
    _validate_contract_semantics(contract)
    _validate_source_frame(contract, canonical_inputs)
    _validate_contract_basis(contract, canonical_inputs)
    _validate_witnesses(serialization, native_precision)
