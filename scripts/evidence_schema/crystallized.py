"""Common evidence records emitted after target-specific capture validation."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from scripts.evidence_schema.primitives import (
    AxisMeaning,
    BasisImages,
    Matrix4,
    NativeScalarProfile,
    NativeQuaternionVectorAction,
    NonNegativeFiniteFloat,
    ObservedProductOrder,
    PositiveFiniteFloat,
    PrecisionProfile,
    Quaternion4,
    SceneTransformSemantics,
    SignedAxis,
    StrictEvidenceModel,
    Vector3,
)


class CoordinateFrame(StrictEvidenceModel):
    """Named, explicit coordinate-frame axis meanings."""

    name: str
    handedness: Literal["right-handed", "left-handed"]
    positive_x: AxisMeaning
    positive_y: AxisMeaning
    positive_z: AxisMeaning


class FrameSemantics(StrictEvidenceModel):
    """Role and authority of a frame assignment, separate from its numbers."""

    role: str
    authority: Literal[
        "target_documented",
        "application_declared",
        "fixture_only",
        "unspecified",
    ]
    statement: str
    evidence_reference: Optional[str] = None  # noqa: UP045


def canonical_source_frame_semantics() -> FrameSemantics:
    """Describe the shared corpus frame without assigning target-world meaning."""
    return FrameSemantics(
        role="canonical_probe",
        authority="application_declared",
        statement=(
            "The shared corpus explicitly assigns this right-handed frame "
            "for interoperability comparisons; it is not an implicit world "
            "or asset frame."
        ),
    )


class PrecisionTolerance(StrictEvidenceModel):
    """Comparison rule for one native scalar profile."""

    profile: PrecisionProfile
    absolute: NonNegativeFiniteFloat
    relative: NonNegativeFiniteFloat


class _EvidenceContract(StrictEvidenceModel):
    """Scope and exclusions common to every normalized evidence contract."""

    scope: list[str]
    excludes: list[str]


class _RotationBoundaryContract(_EvidenceContract):
    """Frame and conversion fields shared by rotation boundary contracts."""

    canonical_algebra: Literal["Hamilton", "JPL"]
    canonical_order: Literal["wxyz", "xyzw"]
    rotation_action: Literal["active", "passive"]
    source_frame: CoordinateFrame
    source_frame_semantics: FrameSemantics
    destination_frame: CoordinateFrame
    destination_frame_semantics: FrameSemantics
    destination_axes_from_source: Annotated[
        list[SignedAxis], Field(min_length=3, max_length=3)
    ]
    conversion_rule: str
    inverse_rule: str
    tolerances: list[PrecisionTolerance]


class QuaternionBoundaryContract(_RotationBoundaryContract):
    """Quaternion coefficient and shared-basis conversion contract."""

    kind: Literal["quaternion_boundary"]
    native_algebra: Literal["Hamilton", "JPL"]
    native_order: Literal["wxyz", "xyzw"]
    native_storage_order: Optional[Literal["wxyz", "xyzw"]] = None  # noqa: UP045
    native_vector_action: NativeQuaternionVectorAction
    input_domain: str


class RotationVectorBoundaryContract(_RotationBoundaryContract):
    """Conversion contract for a native axis-angle/rotation-vector boundary."""

    kind: Literal["rotation_vector_boundary"]
    angle_units: Literal["radians"]
    native_representation: str
    native_matrix_action: Literal["cv2.gemm(R, vector_column, alpha=1, beta=0)"]


class FrameAxisMapping(StrictEvidenceModel):
    """Declared signed-axis mapping for one named transform frame."""

    frame: str
    axes_from_canonical: Annotated[list[SignedAxis], Field(min_length=3, max_length=3)]
    semantics: FrameSemantics


class _CanonicalFrameContract(_EvidenceContract):
    """A contract anchored to the explicitly assigned canonical frame."""

    canonical_frame: CoordinateFrame
    canonical_frame_semantics: FrameSemantics


class FrameChainContract(_CanonicalFrameContract):
    """Named-frame direction contract for robotics and transform graphs."""

    kind: Literal["frame_chain"]
    quaternion_order: Literal["wxyz", "xyzw"]
    vector_action: Literal["active source-point to target-point transform"]
    lookup_direction: Literal[
        "lookupTransform(target, source) returns target_from_source; "
        "its rotation and translation map source point coordinates "
        "into target coordinates"
    ]
    frame_mappings: list[FrameAxisMapping]


class GeographicContract(_CanonicalFrameContract):
    """Local geographic frame and global reference-frame contract."""

    kind: Literal["geographic"]
    local_frame: CoordinateFrame
    local_frame_semantics: FrameSemantics
    local_axes_from_canonical: Annotated[
        list[SignedAxis], Field(min_length=3, max_length=3)
    ]
    local_alignment_authority: Literal[
        "target_documented",
        "application_declared",
        "fixture_only",
        "unspecified",
    ]
    global_frame_name: str
    geodetic_datum: Literal["WGS84"]
    angular_units: Literal["degrees"]
    linear_units: Literal["metres"]
    forward_expression: str
    inverse_expression: str
    quaternion_order: Literal["wxyz", "xyzw"]


class SceneTransportContract(_CanonicalFrameContract):
    """Serialized scene rotation and hierarchy contract."""

    kind: Literal["scene_transport"]
    format_name: str
    format_version: str
    loader_name: str
    loader_version: str
    orientation_field: Literal["node.rotation", "xformOp:orient"]
    meters_per_unit: Optional[PositiveFiniteFloat] = None  # noqa: UP045
    target_frame: CoordinateFrame
    target_frame_semantics: FrameSemantics
    target_frame_role: Literal["caller_assigned_probe", "declared_stage"]
    destination_axes_from_source: Annotated[
        list[SignedAxis], Field(min_length=3, max_length=3)
    ]
    quaternion_order: Literal["wxyz", "xyzw"]
    matrix_action: SceneTransformSemantics
    hierarchy_semantics: SceneTransformSemantics


EvidenceContract = Annotated[
    Union[
        QuaternionBoundaryContract,
        RotationVectorBoundaryContract,
        FrameChainContract,
        GeographicContract,
        SceneTransportContract,
    ],
    Field(discriminator="kind"),
]


class _PrecisionWitness(StrictEvidenceModel):
    """A named observation with a native precision and basis action."""

    case_id: str
    precision: NativeScalarProfile
    basis_images_xyz: BasisImages


class _QuaternionWitness(_PrecisionWitness):
    """A quaternion-valued native observation."""

    coefficient_order: Literal["wxyz", "xyzw"]
    coefficients: Quaternion4


class OrientationWitness(_QuaternionWitness):
    """Native quaternion coefficients and the native action on a basis."""

    kind: Literal["orientation"]
    storage_coefficients_xyzw: Optional[Quaternion4] = None  # noqa: UP045
    probe_image_xyz: Optional[Vector3] = None  # noqa: UP045
    constructor_axis_xyz: Optional[Vector3] = None  # noqa: UP045
    constructor_angle_radians: Optional[float] = None  # noqa: UP045


class CompositionWitness(_QuaternionWitness):
    """Native result for a named, ordered product of two rotations."""

    kind: Literal["composition"]
    left_case_id: str
    right_case_id: str
    product_order: ObservedProductOrder


class _RotationVectorWitness(_PrecisionWitness):
    """A native observation returning an axis-angle rotation vector."""

    output_rotation_vector_xyz: Vector3


class RotationVectorWitness(_RotationVectorWitness):
    """Native Rodrigues route from or to an axis-angle rotation vector."""

    kind: Literal["rotation_vector"]
    input_rotation_vector_xyz: Vector3
    probe_image_xyz: Optional[Vector3] = None  # noqa: UP045


class RotationVectorCompositionWitness(_RotationVectorWitness):
    """Native matrix product and its returned rotation-vector representation."""

    kind: Literal["rotation_vector_composition"]
    left_case_id: str
    right_case_id: str
    product_order: ObservedProductOrder


class FrameTransform(StrictEvidenceModel):
    """One captured rigid transform in a named frame chain."""

    parent_frame: str
    child_frame: str
    quaternion: Quaternion4
    coefficient_order: Literal["wxyz", "xyzw"]
    translation_xyz: Vector3


class FrameMessageWitness(FrameTransform):
    """Serialized transform-message fields and their named frame direction."""

    stamp_sec: int
    stamp_nanosec: int


class PointAction(StrictEvidenceModel):
    """One native transform of a point between named frames."""

    source_frame: str
    target_frame: str
    input_xyz: Vector3
    output_xyz: Vector3


class FrameChainWitness(StrictEvidenceModel):
    """Native lookups, messages, and point actions for a frame chain."""

    kind: Literal["frame_chain"]
    case_id: str
    intermediate_transforms: list[FrameTransform]
    lookups: list[FrameTransform]
    messages: list[FrameMessageWitness]
    point_actions: list[PointAction]
    message_roundtrips: list[FrameTransform]


class GeographicOrientationWitness(StrictEvidenceModel):
    """Local orientation at a geographic origin, expressed in ECEF."""

    kind: Literal["geographic_orientation"]
    case_id: str
    location_case: str
    longitude_degrees: float
    latitude_degrees: float
    height_m: float
    coefficient_order: Literal["wxyz", "xyzw"]
    local_coefficients: Quaternion4
    local_translation_xyz_m: Vector3
    ecef_coefficients: Quaternion4
    recovered_local_coefficients: Quaternion4
    local_basis_images_xyz: Optional[BasisImages] = None  # noqa: UP045
    local_probe_image_xyz: Optional[Vector3] = None  # noqa: UP045
    world_directions_xyz: BasisImages
    world_probe_direction_xyz: Optional[Vector3] = None  # noqa: UP045
    ecef_origin_xyz_m: Vector3
    world_probe_point_ecef_xyz_m: Vector3
    recovered_probe_point_local_xyz_m: Vector3


class SceneNodeWitness(StrictEvidenceModel):
    """Minimal local/world rotation and point evidence for a serialized node."""

    kind: Literal["scene_node"]
    node_id: str
    source_case_id: Optional[str] = None  # noqa: UP045
    parent_id: Optional[str] = None  # noqa: UP045
    prim_path: Optional[str] = None  # noqa: UP045
    local_coefficients: Quaternion4
    coefficient_order: Literal["wxyz", "xyzw"]
    source_coefficients_wxyz: Optional[Quaternion4] = None  # noqa: UP045
    serialized_coefficients: Optional[Quaternion4] = None  # noqa: UP045
    local_translation_xyz: Vector3
    probe_point_local_xyz: Optional[Vector3] = None  # noqa: UP045
    local_matrix_rows: Optional[Matrix4] = None  # noqa: UP045
    world_matrix_rows: Optional[Matrix4] = None  # noqa: UP045
    local_basis_images_xyz: Optional[BasisImages] = None  # noqa: UP045
    world_basis_images_xyz: Optional[BasisImages] = None  # noqa: UP045
    world_probe_point_xyz: Optional[Vector3] = None  # noqa: UP045
    roundtrip_probe_point_xyz: Optional[Vector3] = None  # noqa: UP045
    ordered_ops: Optional[list[str]] = None  # noqa: UP045
    resets_parent_stack: Optional[bool] = None  # noqa: UP045


QuaternionBoundaryWitness = Annotated[
    Union[OrientationWitness, CompositionWitness], Field(discriminator="kind")
]
RotationVectorBoundaryWitness = Annotated[
    Union[RotationVectorWitness, RotationVectorCompositionWitness],
    Field(discriminator="kind"),
]


class EngineQuaternionSerialization(StrictEvidenceModel):
    """Engine quaternion evidence with its exact shared contract and witnesses."""

    layer: Literal["engine-quaternion"]
    contract: QuaternionBoundaryContract
    witnesses: list[QuaternionBoundaryWitness]


class NativeQuaternionApiSerialization(StrictEvidenceModel):
    """Native quaternion API evidence with its exact witness family."""

    layer: Literal["native-quaternion-api"]
    contract: QuaternionBoundaryContract
    witnesses: list[QuaternionBoundaryWitness]


class RotationVectorSerialization(StrictEvidenceModel):
    """Native rotation-vector evidence with its exact contract and witnesses."""

    layer: Literal["native-quaternion-api"]
    contract: RotationVectorBoundaryContract
    witnesses: list[RotationVectorBoundaryWitness]


class FrameChainSerialization(StrictEvidenceModel):
    """Frame-transport evidence with only frame-chain witnesses."""

    layer: Literal["frame-transport"]
    contract: FrameChainContract
    witnesses: list[FrameChainWitness]


class GeographicSerialization(StrictEvidenceModel):
    """Geographic-transport evidence with only geographic witnesses."""

    layer: Literal["geographic-transport"]
    contract: GeographicContract
    witnesses: list[GeographicOrientationWitness]


class SceneTransportSerialization(StrictEvidenceModel):
    """Scene-transport evidence with only scene-node witnesses."""

    layer: Literal["scene-transport"]
    contract: SceneTransportContract
    witnesses: list[SceneNodeWitness]


SemanticSerialization = Union[
    EngineQuaternionSerialization,
    NativeQuaternionApiSerialization,
    RotationVectorSerialization,
    FrameChainSerialization,
    GeographicSerialization,
    SceneTransportSerialization,
]


class CanonicalOrientationInput(StrictEvidenceModel):
    """One shared canonical quaternion input referenced by case ID."""

    case_id: str
    coefficients_wxyz: Quaternion4


class CanonicalInputContract(StrictEvidenceModel):
    algebra: Literal["Hamilton"]
    coefficient_order: Literal["wxyz"]
    rotation_action: Literal["active"]
    vector_action: Literal["q * (0, v) * conjugate(q)"]
    matrix_action: Literal["column vectors"]
    coordinate_system: Literal["right-handed"]
    construction_angle_units: Literal["radians"]
    normalization: str
    frame_mapping: str


class CanonicalInputs(StrictEvidenceModel):
    """Shared test inputs stored once for all crystallized target records."""

    input_set_id: str
    contract_id: str
    contract: CanonicalInputContract
    source_frame: CoordinateFrame
    source_frame_semantics: FrameSemantics
    orientations: list[CanonicalOrientationInput]
    probe_vector_xyz: Vector3
    composition_left_case_id: str
    composition_right_case_id: str
    product_order: Literal["lhs * rhs; rhs acts first"]
