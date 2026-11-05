"""Common evidence records emitted after target-specific capture validation."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from scripts.evidence_schema.primitives import (
    SHA256,
    BasisImages,
    Matrix4,
    Quaternion4,
    StrictEvidenceModel,
    Vector3,
)


class CoordinateFrame(StrictEvidenceModel):
    """Named, explicit coordinate-frame axis meanings."""

    name: str
    handedness: Literal["right-handed", "left-handed"]
    positive_x: str
    positive_y: str
    positive_z: str


class PrecisionTolerance(StrictEvidenceModel):
    """Comparison rule for one native scalar profile."""

    profile: str
    absolute: float
    relative: float


class QuaternionBoundaryContract(StrictEvidenceModel):
    """Quaternion coefficient and shared-basis conversion contract."""

    kind: Literal["quaternion_boundary"]
    canonical_algebra: Literal["Hamilton", "JPL"]
    canonical_order: Literal["wxyz", "xyzw"]
    rotation_action: Literal["active", "passive"]
    source_frame: CoordinateFrame
    destination_frame: CoordinateFrame
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    native_algebra: Literal["Hamilton", "JPL"]
    native_order: Literal["wxyz", "xyzw"]
    native_storage_order: Optional[Literal["wxyz", "xyzw"]] = None  # noqa: UP045
    native_vector_action: str
    conversion_rule: str
    inverse_rule: str
    input_domain: str
    tolerances: list[PrecisionTolerance]
    scope: list[str]
    excludes: list[str]


class RotationVectorBoundaryContract(StrictEvidenceModel):
    """Conversion contract for a native axis-angle/rotation-vector boundary."""

    kind: Literal["rotation_vector_boundary"]
    canonical_algebra: Literal["Hamilton", "JPL"]
    canonical_order: Literal["wxyz", "xyzw"]
    rotation_action: Literal["active", "passive"]
    source_frame: CoordinateFrame
    destination_frame: CoordinateFrame
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    angle_units: Literal["radians"]
    native_representation: str
    native_matrix_action: str
    conversion_rule: str
    inverse_rule: str
    tolerances: list[PrecisionTolerance]
    scope: list[str]
    excludes: list[str]


class FrameAxisMapping(StrictEvidenceModel):
    """Declared signed-axis mapping for one named transform frame."""

    frame: str
    axes_from_canonical: Annotated[list[str], Field(min_length=3, max_length=3)]


class FrameChainContract(StrictEvidenceModel):
    """Named-frame direction contract for robotics and transform graphs."""

    kind: Literal["frame_chain"]
    quaternion_order: Literal["wxyz", "xyzw"]
    vector_action: str
    lookup_direction: str
    frame_mappings: list[FrameAxisMapping]
    scope: list[str]
    excludes: list[str]


class GeographicContract(StrictEvidenceModel):
    """Local geographic frame and global reference-frame contract."""

    kind: Literal["geographic"]
    canonical_frame: CoordinateFrame
    local_frame: CoordinateFrame
    local_axes_from_canonical: Annotated[list[str], Field(min_length=3, max_length=3)]
    global_frame_name: str
    geodetic_datum: str
    forward_expression: str
    inverse_expression: str
    quaternion_order: Literal["wxyz", "xyzw"]
    scope: list[str]
    excludes: list[str]


class SceneTransportContract(StrictEvidenceModel):
    """Serialized scene rotation and hierarchy contract."""

    kind: Literal["scene_transport"]
    format_name: str
    format_version: str
    loader_name: str
    loader_version: str
    canonical_frame: CoordinateFrame
    asset_frame: CoordinateFrame
    destination_axes_from_source: Annotated[
        list[str], Field(min_length=3, max_length=3)
    ]
    quaternion_order: Literal["wxyz", "xyzw"]
    matrix_action: str
    hierarchy_semantics: str
    scope: list[str]
    excludes: list[str]


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

EvidenceLayer = Literal[
    "engine-quaternion",
    "native-quaternion-api",
    "frame-transport",
    "geographic-transport",
    "scene-transport",
]


class EvidenceTarget(StrictEvidenceModel):
    """Runtime identity associated with a captured witness set."""

    product: str
    version: str
    api: str
    native_precision: list[str]
    build_id: Optional[str] = None  # noqa: UP045


class DependencyPin(StrictEvidenceModel):
    """One pinned runtime dependency used by a native capture."""

    name: str
    version: str


class CaptureProvenance(StrictEvidenceModel):
    """Minimal provenance retained with a crystallized target record."""

    capture_adapter: str
    capture_adapter_sha256: SHA256
    input_sha256: SHA256
    contract_sha256: SHA256
    witnesses_sha256: SHA256
    manifest_sha256: Optional[SHA256] = None  # noqa: UP045
    native_runtime: str
    native_binary_sha256: Optional[SHA256] = None  # noqa: UP045
    checked_in_adapter_sha256: Optional[SHA256] = None  # noqa: UP045
    transport_asset_sha256: Optional[SHA256] = None  # noqa: UP045
    dependency_lock_sha256: Optional[SHA256] = None  # noqa: UP045
    dependencies: list[DependencyPin]


class OrientationWitness(StrictEvidenceModel):
    """Native quaternion coefficients and the native action on a basis."""

    kind: Literal["orientation"]
    case_id: str
    precision: str
    coefficient_order: Literal["wxyz", "xyzw"]
    coefficients: Quaternion4
    storage_coefficients_xyzw: Optional[Quaternion4] = None  # noqa: UP045
    basis_images_xyz: BasisImages
    probe_image_xyz: Optional[Vector3] = None  # noqa: UP045
    constructor_axis_xyz: Optional[Vector3] = None  # noqa: UP045
    constructor_angle_radians: Optional[float] = None  # noqa: UP045


class CompositionWitness(StrictEvidenceModel):
    """Native result for a named, ordered product of two rotations."""

    kind: Literal["composition"]
    case_id: str
    left_case_id: str
    right_case_id: str
    product_order: str
    precision: str
    coefficient_order: Literal["wxyz", "xyzw"]
    coefficients: Quaternion4
    basis_images_xyz: BasisImages


class RotationVectorWitness(StrictEvidenceModel):
    """Native Rodrigues route from or to an axis-angle rotation vector."""

    kind: Literal["rotation_vector"]
    case_id: str
    precision: str
    input_rotation_vector_xyz: Vector3
    output_rotation_vector_xyz: Vector3
    basis_images_xyz: BasisImages
    probe_image_xyz: Optional[Vector3] = None  # noqa: UP045


class RotationVectorCompositionWitness(StrictEvidenceModel):
    """Native matrix product and its returned rotation-vector representation."""

    kind: Literal["rotation_vector_composition"]
    case_id: str
    left_case_id: str
    right_case_id: str
    product_order: str
    precision: str
    output_rotation_vector_xyz: Vector3
    basis_images_xyz: BasisImages


class FrameTransform(StrictEvidenceModel):
    """One captured rigid transform in a named frame chain."""

    parent_frame: str
    child_frame: str
    quaternion_xyzw: Quaternion4
    translation_xyz: Vector3


class FrameMessageWitness(StrictEvidenceModel):
    """Serialized transform-message fields and their named frame direction."""

    parent_frame: str
    child_frame: str
    stamp_sec: int
    stamp_nanosec: int
    quaternion_xyzw: Quaternion4
    translation_xyz: Vector3


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
    local_coefficients_xyzw: Quaternion4
    local_translation_xyz_m: Vector3
    ecef_coefficients_xyzw: Quaternion4
    recovered_local_coefficients_xyzw: Quaternion4
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


EvidenceWitness = Annotated[
    Union[
        OrientationWitness,
        CompositionWitness,
        RotationVectorWitness,
        RotationVectorCompositionWitness,
        FrameChainWitness,
        GeographicOrientationWitness,
        SceneNodeWitness,
    ],
    Field(discriminator="kind"),
]


class CanonicalOrientationInput(StrictEvidenceModel):
    """One shared canonical quaternion input referenced by case ID."""

    case_id: str
    coefficients_wxyz: Quaternion4


class CanonicalInputContract(StrictEvidenceModel):
    algebra: Literal["Hamilton", "JPL"]
    coefficient_order: Literal["wxyz", "xyzw"]
    rotation_action: Literal["active", "passive"]
    vector_action: str
    matrix_action: str
    coordinate_system: Literal["right-handed", "left-handed"]
    construction_angle_units: Literal["radians"]
    normalization: str
    frame_mapping: str


class CanonicalInputs(StrictEvidenceModel):
    """Shared test inputs stored once for all crystallized target records."""

    input_set_id: str
    contract_id: str
    contract: CanonicalInputContract
    source_frame: CoordinateFrame
    orientations: list[CanonicalOrientationInput]
    probe_vector_xyz: Vector3
    composition_left_case_id: str
    composition_right_case_id: str
    product_order: str


class CrystallizedEvidence(StrictEvidenceModel):
    """Small, target-neutral evidence record consumed by offline tests."""

    schema_version: Literal[1]
    evidence_id: str
    layer: EvidenceLayer
    target: EvidenceTarget
    contract_id: str
    contract: EvidenceContract
    witness_set_id: str
    provenance: CaptureProvenance


class CrystallizedWitnessSet(StrictEvidenceModel):
    """Immutable selected values referenced by one or more version records."""

    schema_version: Literal[1]
    witness_set_id: str
    witnesses: list[EvidenceWitness]


class CrystallizedCapture(StrictEvidenceModel):
    """Transient reducer result before product-level witness-set deduplication."""

    record: CrystallizedEvidence
    witness_set: CrystallizedWitnessSet


class CrystallizationHeader(StrictEvidenceModel):
    """Validated target contract and provenance supplied to a reducer."""

    evidence_id: str
    layer: EvidenceLayer
    target: EvidenceTarget
    contract_id: str
    contract: EvidenceContract
    witness_set_id: str
    provenance: CaptureProvenance


class EvidenceProduct(StrictEvidenceModel):
    """Versioned collection of crystallized records with shared inputs."""

    schema_version: Literal[1]
    canonical_inputs: CanonicalInputs
    witness_sets: list[CrystallizedWitnessSet]
    records: list[CrystallizedEvidence]
