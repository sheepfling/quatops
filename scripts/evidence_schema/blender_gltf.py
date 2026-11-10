"""Resolved schema for Blender's native glTF importer capture."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, StrictInt

from scripts.evidence_schema.primitives import (
    SHA256,
    NativeCaptureModel,
)

Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Quaternion4 = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
Vector4 = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
Matrix4 = Annotated[list[Vector4], Field(min_length=4, max_length=4)]
ComponentKeyValues = Annotated[
    list[Annotated[list[str], Field(min_length=3, max_length=3)]],
    Field(min_length=4, max_length=4),
]


class GltfTransportIdentity(NativeCaptureModel):
    name: Literal["glTF"]
    version: Literal["2.0"]
    source_coordinates: str


class BlenderGltfLoader(NativeCaptureModel):
    name: Literal["Blender glTF importer"]
    version: Literal["5.1.2"]
    version_tuple: Annotated[list[StrictInt], Field(min_length=3, max_length=3)]
    build_hash: str
    build_date: str
    build_platform: str
    python_version: str
    executable: str
    executable_sha256: SHA256
    importer_source_root: str
    importer_source_sha256: dict[str, SHA256]
    rotation_component_order: Literal[
        "WXYZ in mathutils Quaternion; outputs labelled explicitly"
    ]


class BlenderGltfOperatorArguments(NativeCaptureModel):
    filepath: str


class BlenderGltfProvenance(NativeCaptureModel):
    gltf_sha256: SHA256
    binary_sha256: SHA256
    exporter_sha256: SHA256
    exact_invocation: str
    operator_arguments: BlenderGltfOperatorArguments
    other_operator_properties: str
    captured_at_utc: str


class BlenderGltfSceneUnits(NativeCaptureModel):
    system: str
    scale_length: FiniteFloat
    system_rotation: str


class BlenderGltfScene(NativeCaptureModel):
    fps: StrictInt
    fps_base: FiniteFloat
    frame_start: StrictInt
    frame_end: StrictInt
    unit_settings: BlenderGltfSceneUnits


class BlenderGltfStaticTransform(NativeCaptureModel):
    matrix_basis_rows: Matrix4
    matrix_world_rows: Matrix4


class BlenderGltfStaticTransforms(NativeCaptureModel):
    parent_frame: BlenderGltfStaticTransform
    static_trs: BlenderGltfStaticTransform
    static_matrix: BlenderGltfStaticTransform


class BlenderGltfAxisImages(NativeCaptureModel):
    x: Vector3
    y: Vector3
    z: Vector3


class BlenderGltfFcurveSummary(NativeCaptureModel):
    data_path: Literal["rotation_quaternion"]
    array_indices: Annotated[list[StrictInt], Field(min_length=4, max_length=4)]
    interpolations_by_component: ComponentKeyValues
    left_handle_types_by_component: ComponentKeyValues
    right_handle_types_by_component: ComponentKeyValues


class BlenderGltfAnimationSample(NativeCaptureModel):
    sample: str
    time_seconds: FiniteFloat
    quaternion_wxyz: Quaternion4


class BlenderGltfAnimationFields(NativeCaptureModel):
    action_name: str
    target_object: str
    key_times_seconds_from_imported_fcurves: Annotated[
        list[FiniteFloat], Field(min_length=3, max_length=3)
    ]
    fcurves: BlenderGltfFcurveSummary
    samples: Annotated[list[BlenderGltfAnimationSample], Field(min_length=4)]


class BlenderGltfStepAnimation(BlenderGltfAnimationFields):
    interpolation: Literal["STEP"]


class BlenderGltfLinearAnimation(BlenderGltfAnimationFields):
    interpolation: Literal["LINEAR"]


class BlenderGltfCubicSplineAnimation(BlenderGltfAnimationFields):
    interpolation: Literal["CUBICSPLINE"]


class BlenderGltfObservations(NativeCaptureModel):
    static_transforms: BlenderGltfStaticTransforms
    source_translation_axis_images_in_blender_xyz: BlenderGltfAxisImages
    animations: tuple[
        BlenderGltfStepAnimation,
        BlenderGltfLinearAnimation,
        BlenderGltfCubicSplineAnimation,
    ]


class BlenderGltfCapture(NativeCaptureModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["native_blender_gltf_importer_capture"]
    transport: GltfTransportIdentity
    loader: BlenderGltfLoader
    provenance: BlenderGltfProvenance
    scene: BlenderGltfScene
    observations: BlenderGltfObservations
