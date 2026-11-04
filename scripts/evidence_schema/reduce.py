"""Reduce target-native capture models to the compact common evidence schema."""

from __future__ import annotations

from typing import Optional, Union

from scripts.evidence_schema.crystallized import (
    CompositionWitness,
    CrystallizationHeader,
    CrystallizedCapture,
    CrystallizedEvidence,
    CrystallizedWitnessSet,
    EvidenceWitness,
    FrameChainWitness,
    FrameMessageWitness,
    FrameTransform,
    GeographicOrientationWitness,
    OrientationWitness,
    PointAction,
    RotationVectorCompositionWitness,
    RotationVectorWitness,
    SceneNodeWitness,
)
from scripts.evidence_schema.targets import (
    CesiumCapture,
    CesiumInputs,
    EigenCapture,
    EigenInputs,
    MuJoCoCapture,
    MuJoCoInputs,
    OpenCVCapture,
    OpenCVInputs,
    OpenUsdCapture,
    OpenUsdInputs,
    Ros2Capture,
    SpiceyPyCapture,
    SpiceyPyInputs,
    TargetCapture,
    ThreeGltfCapture,
    ThreeGltfInputs,
)

_RETAINED_ORIENTATION_CASES = frozenset(
    {
        "identity",
        "quarter_x",
        "quarter_y",
        "quarter_z",
        "mixed_a",
        "antipode_mixed_a",
        "nonprincipal_halfturn",
    }
)
_PROBE_CASES = frozenset({"mixed_a"})
_NATIVE_INPUTS = Union[
    EigenInputs,
    MuJoCoInputs,
    OpenCVInputs,
    SpiceyPyInputs,
    CesiumInputs,
    OpenUsdInputs,
    ThreeGltfInputs,
]


def _basis(images: list[list[float]]) -> list[list[float]]:
    """Keep the three basis actions; the fourth discovery probe is optional."""
    return images[:3]


def _native_product_case_ids(case_id: str) -> tuple[str, str]:
    try:
        left_case_id, right_case_id = case_id.split("_times_", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"unsupported native product case ID: {case_id}") from exc
    return left_case_id, right_case_id


def _eigen(capture: EigenCapture, inputs: EigenInputs) -> list[EvidenceWitness]:
    witnesses: list[EvidenceWitness] = []
    for profile in capture.observations.profiles:
        for row in profile.orientations:
            if row.case_id not in _RETAINED_ORIENTATION_CASES:
                continue
            witnesses.append(
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
                    probe_image_xyz=(
                        row.native_quaternion_probe_action_xyz
                        if row.case_id in _PROBE_CASES
                        else None
                    ),
                )
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
                constructor_angle_radians=inputs.native_constructed_orientation.angle_radians,
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


def _mujoco(capture: MuJoCoCapture, inputs: MuJoCoInputs) -> list[EvidenceWitness]:
    observations = capture.observations
    witnesses: list[EvidenceWitness] = []
    for row in observations.canonical_to_native:
        if row.case_id not in _RETAINED_ORIENTATION_CASES:
            continue
        witnesses.append(
            OrientationWitness(
                kind="orientation",
                case_id=row.case_id,
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=row.native_mju_rotVecQuat_basis_xyz,
                probe_image_xyz=(
                    row.native_mju_rotVecQuat_probe_xyz
                    if row.case_id in _PROBE_CASES
                    else None
                ),
            )
        )
    witnesses.extend(
        OrientationWitness(
            kind="orientation",
            case_id=row.case_id,
            precision="float64",
            coefficient_order="wxyz",
            coefficients=row.native_input_wxyz,
            basis_images_xyz=row.native_mju_rotVecQuat_basis_xyz,
            constructor_axis_xyz=inputs.native_constructed_orientation.axis_xyz,
            constructor_angle_radians=inputs.native_constructed_orientation.angle_radians,
        )
        for row in observations.native_created
    )
    for row in observations.composition:
        left_case_id, right_case_id = _native_product_case_ids(row.case_id)
        witnesses.append(
            CompositionWitness(
                kind="composition",
                case_id=row.case_id,
                left_case_id=left_case_id,
                right_case_id=right_case_id,
                product_order=row.product_order or "lhs * rhs; rhs acts first",
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=row.native_mju_rotVecQuat_basis_xyz,
            )
        )
    return witnesses


def _opencv(capture: OpenCVCapture) -> list[EvidenceWitness]:
    observations = capture.observations
    witnesses: list[EvidenceWitness] = []
    for row in observations.canonical_to_native:
        if row.case_id not in _RETAINED_ORIENTATION_CASES:
            continue
        witnesses.append(
            RotationVectorWitness(
                kind="rotation_vector",
                case_id=f"{row.case_id}@{row.dtype}",
                precision=row.dtype,
                input_rotation_vector_xyz=row.native_input_rotation_vector_xyz_radians,
                output_rotation_vector_xyz=row.native_matrix_to_rotation_vector_xyz_radians,
                basis_images_xyz=row.native_gemm_basis_actions_xyz,
                probe_image_xyz=(
                    row.native_gemm_probe_action_xyz
                    if row.case_id == "mixed_a"
                    else None
                ),
            )
        )
    witnesses.extend(
        RotationVectorWitness(
            kind="rotation_vector",
            case_id=f"{row.case_id}@{row.dtype}",
            precision=row.dtype,
            input_rotation_vector_xyz=row.native_input_rotation_vector_xyz_radians,
            output_rotation_vector_xyz=row.native_matrix_to_rotation_vector_xyz_radians,
            basis_images_xyz=row.native_gemm_basis_actions_xyz,
        )
        for row in observations.native_created
    )
    witnesses.extend(
        RotationVectorCompositionWitness(
            kind="rotation_vector_composition",
            case_id=f"{row.case_id}@{row.dtype}",
            left_case_id=row.native_lhs_case_id,
            right_case_id=row.native_rhs_case_id,
            product_order=row.product_order,
            precision=row.dtype,
            output_rotation_vector_xyz=row.native_matrix_to_rotation_vector_xyz_radians,
            basis_images_xyz=row.native_gemm_basis_actions_xyz,
        )
        for row in observations.composition
    )
    return witnesses


def _spiceypy(
    capture: SpiceyPyCapture, inputs: SpiceyPyInputs
) -> list[EvidenceWitness]:
    observations = capture.observations
    witnesses: list[EvidenceWitness] = []
    for row in observations.canonical_to_native:
        if row.case_id not in _RETAINED_ORIENTATION_CASES:
            continue
        witnesses.append(
            OrientationWitness(
                kind="orientation",
                case_id=row.case_id,
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=row.native_mxv_basis_xyz,
                probe_image_xyz=(
                    row.native_mxv_probe_xyz if row.case_id in _PROBE_CASES else None
                ),
            )
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
        left_case_id, right_case_id = _native_product_case_ids(row.case_id)
        witnesses.append(
            CompositionWitness(
                kind="composition",
                case_id=row.case_id,
                left_case_id=left_case_id,
                right_case_id=right_case_id,
                product_order=row.product_order or "lhs * rhs; rhs acts first",
                precision="float64",
                coefficient_order="wxyz",
                coefficients=row.native_input_wxyz,
                basis_images_xyz=row.native_mxv_basis_xyz,
            )
        )
    return witnesses


def _ros2(capture: Ros2Capture) -> list[EvidenceWitness]:
    by_operation = {row.operation_id: row for row in capture.observations}
    forward = by_operation["lookup-world-from-sensor"]
    reverse = by_operation["lookup-sensor-from-world"]
    inputs = capture.native_inputs

    def input_transform(name: str, parent: str, child: str) -> FrameTransform:
        transform = getattr(inputs, name)
        return FrameTransform(
            parent_frame=parent,
            child_frame=child,
            quaternion_xyzw=transform.quaternion_xyzw.values,
            translation_xyz=transform.translation_xyz.values,
        )

    def lookup_transform(row) -> FrameTransform:
        transform = row.native_outputs.lookup
        parent, child = (
            ("world", "sensor")
            if row.operation_id == "lookup-world-from-sensor"
            else ("sensor", "world")
        )
        return FrameTransform(
            parent_frame=parent,
            child_frame=child,
            quaternion_xyzw=transform.quaternion_xyzw.values,
            translation_xyz=transform.translation_xyz.values,
        )

    def message_roundtrip(row) -> FrameTransform:
        transform = row.native_outputs.message_roundtrip
        if transform is None:
            raise ValueError("forward ROS lookup lacks its message round trip")
        return FrameTransform(
            parent_frame="world",
            child_frame="sensor",
            quaternion_xyzw=transform.quaternion_xyzw.values,
            translation_xyz=transform.translation_xyz.values,
        )

    def message_fields(row) -> FrameMessageWitness:
        message = row.native_outputs.lookup_message
        if message is None:
            raise ValueError("forward ROS lookup lacks TransformStamped fields")
        return FrameMessageWitness(
            parent_frame=message.parent_frame,
            child_frame=message.child_frame,
            stamp_sec=message.stamp_sec,
            stamp_nanosec=message.stamp_nanosec,
            quaternion_xyzw=message.rotation_xyzw.values,
            translation_xyz=message.translation_xyz.values,
        )

    probe_output = forward.native_outputs.probe_in_target_xyz
    if probe_output is None:
        raise ValueError("forward ROS lookup lacks its transformed point")
    point_actions = [
        PointAction(
            source_frame=inputs.source_frame,
            target_frame=inputs.target_frame,
            input_xyz=inputs.probe_in_source_xyz.values,
            output_xyz=probe_output.values,
        )
    ]
    return [
        FrameChainWitness(
            kind="frame_chain",
            case_id="world-base-sensor",
            intermediate_transforms=[
                input_transform("world_from_base", "world", "base"),
                input_transform("base_from_sensor", "base", "sensor"),
            ],
            lookups=[lookup_transform(forward), lookup_transform(reverse)],
            messages=[message_fields(forward)],
            point_actions=point_actions,
            message_roundtrips=[message_roundtrip(forward)],
        )
    ]


def _cesium(capture: CesiumCapture, inputs: CesiumInputs) -> list[EvidenceWitness]:
    cases = {row.case_id: row for row in inputs.cases}
    witnesses: list[EvidenceWitness] = []
    for observation in capture.observations:
        item = cases[observation.case_id]
        output = observation.outputs
        witnesses.append(
            GeographicOrientationWitness(
                kind="geographic_orientation",
                case_id=observation.case_id,
                location_case=item.location_case,
                longitude_degrees=item.longitude_degrees,
                latitude_degrees=item.latitude_degrees,
                height_m=item.height_m,
                local_coefficients_xyzw=item.quatops_cesium_local_xyzw,
                local_translation_xyz_m=item.local_translation_xyz_m,
                ecef_coefficients_xyzw=output.model_to_ecef_quaternion_xyzw.values,
                recovered_local_coefficients_xyzw=(
                    output.local_quaternion_recovered_from_ecef_xyzw.values
                ),
                local_basis_images_xyz=(
                    _basis(output.local_basis_and_probe_images.values)
                    if output.local_basis_and_probe_images is not None
                    else None
                ),
                local_probe_image_xyz=(
                    output.local_basis_and_probe_images.values[3]
                    if observation.case_id == "mixed_a@geo-03"
                    and output.local_basis_and_probe_images is not None
                    else None
                ),
                world_directions_xyz=_basis(output.world_directions_xyz.values),
                world_probe_direction_xyz=(
                    output.world_directions_xyz.values[3]
                    if observation.case_id == "mixed_a@geo-03"
                    else None
                ),
                ecef_origin_xyz_m=output.ecef_origin_xyz_m.values,
                world_probe_point_ecef_xyz_m=output.world_points_xyz_m.values[0],
                recovered_probe_point_local_xyz_m=(
                    output.point_roundtrip_local_xyz_m.values[0]
                ),
            )
        )
    return witnesses


def _gltf(capture: ThreeGltfCapture, inputs: ThreeGltfInputs) -> list[EvidenceWitness]:
    input_by_name = {row.node: row for row in inputs.nodes}
    witnesses: list[EvidenceWitness] = []
    for observation in capture.observations:
        input_name = observation.node.removeprefix("QuatOps").lower()
        item = input_by_name[input_name]
        output = observation.outputs
        witnesses.append(
            SceneNodeWitness(
                kind="scene_node",
                node_id=observation.node,
                source_coefficients_wxyz=item.canonical_wxyz,
                serialized_coefficients=item.gltf_xyzw,
                parent_id="QuatOpsParent" if input_name == "child" else None,
                local_coefficients=output.local_quaternion_xyzw.values,
                coefficient_order="xyzw",
                local_translation_xyz=item.gltf_translation_xyz,
                probe_point_local_xyz=(
                    inputs.gltf_probe_vectors_xyz[-1]
                    if output.world_probe_point_xyz is not None
                    else None
                ),
                local_basis_images_xyz=_basis(output.local_basis_images_xyz.values),
                world_basis_images_xyz=(
                    _basis(output.world_basis_images_xyz.values)
                    if output.world_basis_images_xyz is not None
                    else None
                ),
                world_probe_point_xyz=(
                    output.world_probe_point_xyz.values
                    if output.world_probe_point_xyz is not None
                    else None
                ),
                roundtrip_probe_point_xyz=(
                    output.roundtrip_probe_point_xyz.values
                    if output.roundtrip_probe_point_xyz is not None
                    else None
                ),
            )
        )
    return witnesses


def _openusd(capture: OpenUsdCapture, inputs: OpenUsdInputs) -> list[EvidenceWitness]:
    input_by_name = {row.node: row for row in inputs.nodes}
    witnesses: list[EvidenceWitness] = []
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
                world_basis_images_xyz=_basis(output.world_basis_images_xyz),
                world_probe_point_xyz=output.world_probe_point_xyz,
                roundtrip_probe_point_xyz=output.roundtrip_probe_point_xyz,
                ordered_ops=capture.ordered_xform_ops,
                resets_parent_stack=output.resets_xform_stack,
            )
        )
    return witnesses


def crystallize_capture(
    capture: TargetCapture,
    header: CrystallizationHeader,
    *,
    target_inputs: Optional[_NATIVE_INPUTS] = None,  # noqa: UP045
) -> CrystallizedCapture:
    """Reduce one target-specific capture to its typed common evidence record."""
    if isinstance(capture, EigenCapture) and isinstance(target_inputs, EigenInputs):
        witnesses = _eigen(capture, target_inputs)
    elif isinstance(capture, MuJoCoCapture) and isinstance(target_inputs, MuJoCoInputs):
        witnesses = _mujoco(capture, target_inputs)
    elif isinstance(capture, OpenCVCapture):
        witnesses = _opencv(capture)
    elif isinstance(capture, SpiceyPyCapture) and isinstance(
        target_inputs, SpiceyPyInputs
    ):
        witnesses = _spiceypy(capture, target_inputs)
    elif isinstance(capture, Ros2Capture):
        witnesses = _ros2(capture)
    elif isinstance(capture, CesiumCapture) and isinstance(target_inputs, CesiumInputs):
        witnesses = _cesium(capture, target_inputs)
    elif isinstance(capture, ThreeGltfCapture) and isinstance(
        target_inputs, ThreeGltfInputs
    ):
        witnesses = _gltf(capture, target_inputs)
    elif isinstance(capture, OpenUsdCapture) and isinstance(
        target_inputs, OpenUsdInputs
    ):
        witnesses = _openusd(capture, target_inputs)
    else:
        raise TypeError(
            f"{type(capture).__name__} requires its matching typed input model"
        )

    return CrystallizedCapture(
        record=CrystallizedEvidence(
            schema_version=1,
            evidence_id=header.evidence_id,
            layer=header.layer,
            target=header.target,
            contract_id=header.contract_id,
            contract=header.contract,
            witness_set_id=header.witness_set_id,
            provenance=header.provenance,
        ),
        witness_set=CrystallizedWitnessSet(
            schema_version=1,
            witness_set_id=header.witness_set_id,
            witnesses=witnesses,
        ),
    )
