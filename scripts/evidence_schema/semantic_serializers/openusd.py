"""Semantic serialization for OpenUSD stage and transform-stack experiments."""

from __future__ import annotations

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
    derive_signed_axis_map,
    frame_axes,
    frame_signature,
    map_vector_components,
    signed_axis_map,
    validate_mapped_quaternion_wxyz,
    validate_mapped_vectors,
)
from scripts.evidence_schema.semantic_serializers._witness_helpers import (
    canonical_probe_vectors,
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
    _validate_canonical_inputs(package.contract, package.inputs, context)
    _validate_observations(package.contract, package.target_capture, package.inputs)
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
    stage = contract.stage
    if (package.target.name, package.target.version) != (
        contract.target.name,
        contract.target.version,
    ):
        raise ValueError("OpenUSD manifest target differs from semantic contract")
    stage_metadata = package.manifest.get("stage_metadata")
    if (
        not isinstance(stage_metadata, dict)
        or stage_metadata.get("upAxis") != stage.upAxis
        or stage_metadata.get("metersPerUnit") != stage.metersPerUnit
    ):
        raise ValueError("OpenUSD stage metadata differs from its semantic contract")
    return SceneTransportContract(
        kind="scene_transport",
        format_name=f"{contract.target.name} {contract.serialization_format}",
        format_version=contract.target.version,
        loader_name=contract.target.name,
        loader_version=contract.target.version,
        orientation_field=contract.orientation_op,
        meters_per_unit=stage.metersPerUnit,
        canonical_frame=coordinate_frame(
            "QuatOps canonical frame",
            contract.source.frame.handedness,
            contract.source.frame.model_dump(),
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
        quaternion_order=coefficient_order(contract.source.storage),
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
    contract: OpenUsdContract,
    inputs: OpenUsdInputs,
    context: SemanticSerializationContext,
) -> None:
    validate_canonical_input_digest(
        inputs.canonical_input_sha256, context, target="OpenUSD"
    )
    frame = inputs.frame_contract
    expected_source = context.canonical_inputs.source_frame
    if (
        frame_signature(frame.canonical) != frame_signature(contract.source.frame)
        or frame_axes(frame.canonical) != frame_axes(expected_source)
        or expected_source.handedness != "right-handed"
    ):
        raise ValueError("OpenUSD source frame differs from shared or target contract")
    if (
        contract.source.algebra != context.canonical_inputs.contract.algebra
        or contract.source.rotation_action
        != context.canonical_inputs.contract.rotation_action
        or contract.source.storage.upper()
        != context.canonical_inputs.contract.coefficient_order.upper()
    ):
        raise ValueError(
            "OpenUSD quaternion contract differs from shared canonical contract"
        )
    if frame_signature(frame.usd_stage) != (
        contract.stage.handedness,
        contract.stage.positive_x,
        contract.stage.positive_y,
        contract.stage.positive_z,
        contract.stage.forward,
        contract.stage.upAxis,
    ):
        raise ValueError("OpenUSD input stage frame differs from target contract")
    if frame.destination_axes_from_source != contract.destination_axes_from_source:
        raise ValueError("OpenUSD input axis map differs from target contract")
    derived_map = derive_signed_axis_map(
        frame_axes(contract.source.frame), frame_axes(contract.stage)
    )
    if contract.destination_axes_from_source != derived_map:
        raise ValueError("OpenUSD frame meanings disagree with the declared axis map")
    expected_quaternion_semantics = (
        f"{contract.source.algebra} {contract.source.rotation_action}; "
        f"USDA quatd real-first {contract.source.storage}"
    )
    if frame.quaternion != expected_quaternion_semantics:
        raise ValueError("OpenUSD input quaternion semantics differ from contract")
    source_probes = canonical_probe_vectors(context)
    if inputs.probe_vectors_xyz != source_probes:
        raise ValueError("OpenUSD source probes differ from shared inputs")
    validate_mapped_vectors(
        inputs.usd_probe_vectors_xyz,
        source_probes,
        contract.destination_axes_from_source,
        context="OpenUSD probe vectors",
    )
    expected_nodes = {"parent", "child", "reset_child"}
    input_node_names = [row.node for row in inputs.nodes]
    if (
        len(set(input_node_names)) != len(input_node_names)
        or set(input_node_names) != expected_nodes
    ):
        raise ValueError(
            "OpenUSD inputs must contain the parent, child, and reset child"
        )
    canonical_by_case = canonical_orientations_by_case(context.canonical_inputs)
    for row in inputs.nodes:
        validate_canonical_quaternion(
            "OpenUSD", row.canonical_case_id, row.canonical_wxyz, canonical_by_case
        )
        validate_mapped_quaternion_wxyz(
            row.usd_wxyz,
            row.canonical_wxyz,
            contract.destination_axes_from_source,
            context=f"OpenUSD {row.node} quaternion",
        )
        expected_translation = map_vector_components(
            row.canonical_translation_xyz, contract.destination_axes_from_source
        )
        if row.usd_translation_xyz != expected_translation:
            raise ValueError(
                f"OpenUSD {row.node} translation differs from its axis map"
            )


def _validate_observations(
    contract: OpenUsdContract,
    capture: OpenUsdCapture,
    inputs: OpenUsdInputs,
) -> None:
    expected_entry_points = {
        "Usd.Stage.Open",
        "UsdGeom.Xformable.GetOrderedXformOps",
        "GetLocalTransformation",
        "UsdGeom.XformCache.GetLocalToWorldTransform",
        "Gf.Matrix4d.Transform",
        "Gf.Matrix4d.TransformDir",
    }
    if (
        len(set(capture.native_entry_points)) != len(capture.native_entry_points)
        or set(capture.native_entry_points) != expected_entry_points
        or contract.reopen_api not in capture.native_entry_points
    ):
        raise ValueError(
            "OpenUSD native entry points differ from the pinned capture scope"
        )
    expected_ops = ["xformOp:translate", contract.orientation_op]
    if (
        inputs.stack_contract.parent_ops != expected_ops
        or inputs.stack_contract.child_ops != expected_ops
        or inputs.stack_contract.reset_ops != ["!resetXformStack!", *expected_ops]
        or capture.ordered_xform_ops != expected_ops
    ):
        raise ValueError("OpenUSD xform-op stacks differ from the declared contract")
    observations = {row.node: row for row in capture.observations}
    if len(observations) != len(capture.observations) or set(observations) != {
        "parent",
        "child",
        "reset_child",
    }:
        raise ValueError(
            "OpenUSD capture must contain one observation per fixture node"
        )
    expected_paths = {
        "parent": "/Scene/Parent",
        "child": "/Scene/Parent/Child",
        "reset_child": "/Scene/Parent/ResetChild",
    }
    expected_resets = {"parent": False, "child": False, "reset_child": True}
    for node, observation in observations.items():
        if observation.prim_path != expected_paths[node]:
            raise ValueError(f"OpenUSD {node} observation path differs from fixture")
        if observation.outputs.resets_xform_stack != expected_resets[node]:
            raise ValueError(
                f"OpenUSD {node} reset-stack observation differs from fixture"
            )


def serialize_openusd_witnesses(
    capture: OpenUsdCapture, inputs: OpenUsdInputs
) -> list[SceneNodeWitness]:
    input_by_name = {row.node: row for row in inputs.nodes}
    parent_ids = {"parent": None, "child": "parent", "reset_child": None}
    witnesses: list[SceneNodeWitness] = []
    for observation in capture.observations:
        item = input_by_name[observation.node]
        output = observation.outputs
        witnesses.append(
            SceneNodeWitness(
                kind="scene_node",
                node_id=observation.node,
                source_case_id=item.canonical_case_id,
                source_coefficients_wxyz=item.canonical_wxyz,
                serialized_coefficients=item.usd_wxyz,
                parent_id=parent_ids[observation.node],
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
