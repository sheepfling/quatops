"""Semantic serialization for OpenUSD stage and transform-stack experiments."""

from __future__ import annotations

import re

from scripts.converter_evidence import ConverterEvidencePackage
from scripts.evidence_schema.converter_captures import (
    OpenUsdCapture,
    OpenUsdContract,
    OpenUsdInputs,
)
from scripts.evidence_schema.crystallized import (
    FrameSemantics,
    SceneNodeWitness,
    SceneTransportContract,
    SceneTransportSerialization,
    canonical_source_frame_semantics,
)
from scripts.evidence_schema.semantic_serializers._contract_helpers import (
    coefficient_order,
    coordinate_frame,
    signed_axis_map,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_orientations_by_case,
    validate_canonical_input_digest,
    validate_canonical_quaternion,
)
from scripts.evidence_schema.semantic_serializers.context import (
    SemanticSerializationContext,
)


def serialize(
    package: ConverterEvidencePackage, context: SemanticSerializationContext
) -> SceneTransportSerialization:
    """Map OpenUSD serialized ops and stage hierarchy to scene witnesses."""
    if not isinstance(package.contract, OpenUsdContract):
        raise TypeError("OpenUSD serializer received a non-OpenUSD contract")
    if not isinstance(package.target_capture, OpenUsdCapture):
        raise TypeError("OpenUSD serializer received a non-OpenUSD capture")
    if not isinstance(package.inputs, OpenUsdInputs):
        raise TypeError("OpenUSD serializer requires typed OpenUSD inputs")
    _validate_canonical_inputs(package.inputs, context)
    return SceneTransportSerialization(
        layer="scene-transport",
        contract=serialize_openusd_contract(package),
        witnesses=serialize_openusd_witnesses(package.target_capture, package.inputs),
    )


def serialize_openusd_contract(
    package: ConverterEvidencePackage,
) -> SceneTransportContract:
    """Map OpenUSD stage declarations and fixture axis semantics."""
    if not isinstance(package.contract, OpenUsdContract):
        raise TypeError("expected a typed OpenUSD contract")
    contract = package.contract
    order_match = re.search(
        r"(WXYZ|XYZW)", contract.serialized_orientation, re.IGNORECASE
    )
    if order_match is None:
        raise ValueError("OpenUSD contract lacks a declared serialized order")
    frame_match = re.fullmatch(
        r"(right-handed|left-handed);\s*\+X\s+([^,]+),\s*\+Y\s+([^,]+),\s*\+Z\s+(.+)",
        contract.source.frame,
        re.IGNORECASE,
    )
    if frame_match is None:
        raise ValueError("OpenUSD source frame is not in the declared axis format")
    stage = contract.stage
    return SceneTransportContract(
        kind="scene_transport",
        format_name="OpenUSD USDA",
        format_version=package.target.version,
        loader_name=package.target.name,
        loader_version=package.target.version,
        canonical_frame=coordinate_frame(
            "QuatOps canonical frame",
            frame_match.group(1),
            {
                "positive_x": frame_match.group(2),
                "positive_y": frame_match.group(3),
                "positive_z": frame_match.group(4),
            },
        ),
        canonical_frame_semantics=canonical_source_frame_semantics(),
        target_frame=coordinate_frame(
            "OpenUSD declared stage frame", stage.handedness, stage.model_dump()
        ),
        target_frame_semantics=FrameSemantics(
            role="stage",
            authority="application_declared",
            statement=(
                "The stage declares its up axis and this fixture declares "
                "the remaining signed-axis meanings; this is not a model "
                "front or camera frame."
            ),
        ),
        target_frame_role="declared_stage",
        destination_axes_from_source=signed_axis_map(
            contract.destination_axes_from_source
        ),
        quaternion_order=coefficient_order(order_match.group(1)),
        matrix_action=contract.matrix_action,
        hierarchy_semantics=contract.matrix_action,
        scope=[
            "serialized orient operation readback",
            "local and world transform-stack matrices",
            "parent-child composition and resetXformStack",
            "world point action and inverse point round trip",
        ],
        excludes=contract.excludes,
    )


def _validate_canonical_inputs(
    inputs: OpenUsdInputs, context: SemanticSerializationContext
) -> None:
    validate_canonical_input_digest(
        inputs.canonical_input_sha256, context, target="OpenUSD"
    )
    canonical_by_case = canonical_orientations_by_case(context.canonical_inputs)
    for row in inputs.nodes:
        validate_canonical_quaternion(
            "OpenUSD", row.canonical_case_id, row.canonical_wxyz, canonical_by_case
        )


def serialize_openusd_witnesses(
    capture: OpenUsdCapture, inputs: OpenUsdInputs
) -> list[SceneNodeWitness]:
    input_by_name = {row.node: row for row in inputs.nodes}
    witnesses: list[SceneNodeWitness] = []
    for observation in capture.observations:
        item = input_by_name[observation.node]
        output = observation.outputs
        parent_id = None
        if observation.node == "child":
            parent_id = "parent"
        elif not output.resets_xform_stack:
            parent_id = observation.prim_path.rsplit("/", maxsplit=1)[0]
        witnesses.append(
            SceneNodeWitness(
                kind="scene_node",
                node_id=observation.node,
                source_case_id=item.canonical_case_id,
                source_coefficients_wxyz=item.canonical_wxyz,
                serialized_coefficients=item.usd_wxyz,
                parent_id=parent_id,
                prim_path=observation.prim_path,
                local_coefficients=output.orient_op_wxyz,
                coefficient_order="wxyz",
                local_translation_xyz=item.usd_translation_xyz,
                probe_point_local_xyz=inputs.usd_probe_vectors_xyz[-1],
                world_basis_images_xyz=output.world_basis_images_xyz,
                world_probe_point_xyz=output.world_probe_point_xyz,
                roundtrip_probe_point_xyz=output.roundtrip_probe_point_xyz,
                ordered_ops=capture.ordered_xform_ops,
                resets_parent_stack=output.resets_xform_stack,
            )
        )
    return witnesses
