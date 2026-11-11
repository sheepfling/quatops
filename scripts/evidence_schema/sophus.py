"""Strict experiment configuration for the supporting Sophus SO(3) oracle."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import (
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    model_validator,
)

from scripts.evidence_schema.primitives import (
    SHA256,
    NativeCaptureModel,
    SemanticContractReference,
)


class SophusScalarProfile(NativeCaptureModel):
    capture_target: Literal["Sophus SO3f", "Sophus SO3d"]
    precision: Literal["float32", "float64"]
    witness_target: Annotated[str, Field(min_length=1)]
    scipy_tolerance: FiniteFloat


class SophusProbeFrame(NativeCaptureModel):
    description: Literal["+X right, +Y forward, +Z up; right-handed"]
    handedness: Literal["right-handed"]
    positive_x: Literal["right"]
    positive_y: Literal["forward"]
    positive_z: Literal["up"]
    assignment: Literal["fixture_numeric_assignment"]


class SophusRepresentationContract(NativeCaptureModel):
    algebra: Literal["Hamilton"]
    vector_action: Literal["active"]
    matrix_action: Literal["column vectors"]
    constructor_order: Literal["WXYZ"]
    params_order: Literal["XYZW"]
    composition_order: Literal["lhs * rhs; rhs acts first"]


class SophusSemanticConfig(NativeCaptureModel):
    """Per-version semantics; a structural model does not assert native capture."""

    schema_version: Literal["1.0"]
    evidence_status: Literal["supporting_oracle_only"]
    target: Literal["Sophus"]
    version: Annotated[str, Field(min_length=1)]
    dependency: Annotated[str, Field(min_length=1)]
    representation: SophusRepresentationContract
    probe_frame: SophusProbeFrame
    probe_xyz: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    scalar_profiles: Annotated[
        list[SophusScalarProfile], Field(min_length=1, max_length=2)
    ]

    @model_validator(mode="after")
    def scalar_profiles_are_distinct(self) -> SophusSemanticConfig:
        profiles = {item.capture_target: item for item in self.scalar_profiles}
        if len(profiles) != len(self.scalar_profiles):
            raise ValueError("Sophus scalar targets must be distinct")
        for target, profile in profiles.items():
            name, precision = (
                ("SO3f", "float32") if target == "Sophus SO3f" else ("SO3d", "float64")
            )
            if profile.precision != precision:
                raise ValueError("Sophus scalar target and precision do not match")
            if (
                profile.witness_target
                != f"Sophus {self.version} {name} / {self.dependency}"
            ):
                raise ValueError("Sophus scalar target and witness label do not match")
        if any(item.scipy_tolerance <= 0 for item in self.scalar_profiles):
            raise ValueError("SciPy formula tolerances must be positive")
        return self


class SophusWitnessReference(NativeCaptureModel):
    """Per-version association with one immutable shared numeric payload."""

    raw_capture_sha256: SHA256
    raw_capture_retained: bool
    selected_witness_file: str
    selected_witness_sha256: SHA256
    previous_selected_witness_sha256: Optional[SHA256] = None  # noqa: UP045
    historical_raw_capture_sha256: Optional[SHA256] = None  # noqa: UP045
    comparison: Literal["verified_exact"]


class SophusDirectCapture(NativeCaptureModel):
    """Identity and hashes recorded for one actual native run."""

    product_version: str
    dependency: str
    capture_source_file: str
    capture_source_sha256: SHA256
    native_executable_sha256: SHA256
    input_sha256: SHA256
    compiler: Annotated[str, Field(min_length=1)]
    utc: Annotated[str, Field(min_length=1)]
    exit_code: StrictInt
    raw_capture_sha256: dict[Literal["float32", "float64"], SHA256]
    selected_values: str
    scope_note: str

    @model_validator(mode="after")
    def successful(self) -> SophusDirectCapture:
        if self.exit_code != 0:
            raise ValueError("Sophus native capture did not complete successfully")
        return self


class SophusRecipeArtifacts(NativeCaptureModel):
    """Source and input identities available before a capture is run."""

    model_config = ConfigDict(extra="allow")

    input_sha256: SHA256
    current_source_sha256: SHA256


class SophusRecipeManifest(NativeCaptureModel):
    model_config = ConfigDict(extra="allow")

    semantic_contract: SemanticContractReference
    artifacts: SophusRecipeArtifacts


class SophusBindingArtifacts(SophusRecipeArtifacts):
    """Required binding fields while preserving unrelated historical provenance."""

    direct_capture: SophusDirectCapture
    scalar_profiles: dict[Literal["float32", "float64"], SophusWitnessReference]


class SophusExecution(NativeCaptureModel):
    model_config = ConfigDict(extra="allow")

    was_run: StrictBool

    @model_validator(mode="after")
    def executed(self) -> SophusExecution:
        if not self.was_run:
            raise ValueError("Sophus native capture was not executed")
        return self


class SophusBindingManifest(NativeCaptureModel):
    """Typed view of binding fields; remaining capture provenance stays intact."""

    model_config = ConfigDict(extra="allow")

    target: str
    execution: SophusExecution
    artifacts: SophusBindingArtifacts


class SophusSourceDependency(NativeCaptureModel):
    model_config = ConfigDict(extra="allow")

    product: Literal["Eigen"]
    version: str
    tag: Annotated[str, Field(min_length=1)]
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class SophusSourcePin(NativeCaptureModel):
    model_config = ConfigDict(extra="allow")

    product: Literal["Sophus"]
    version: str
    tag: Annotated[str, Field(min_length=1)]
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    dependency: SophusSourceDependency


StrictFiniteFloat = Annotated[FiniteFloat, Field(strict=True)]
Vector3 = tuple[StrictFiniteFloat, StrictFiniteFloat, StrictFiniteFloat]
Quaternion4 = tuple[
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
    StrictFiniteFloat,
]
Matrix3 = tuple[Vector3, Vector3, Vector3]
OptionalVector3 = Optional[Vector3]
ConstructorRow = tuple[
    str,
    Quaternion4,
    StrictFiniteFloat,
    Quaternion4,
    Matrix3,
    OptionalVector3,
]
RotationRow = tuple[str, Quaternion4, Matrix3, OptionalVector3]

CONSTRUCTION_CASES = (
    "identity",
    "x_quarter",
    "y_quarter",
    "z_quarter",
    "mixed_unequal",
    "mixed_unequal_antipode",
    "nonunit",
    "half_turn_axis",
)
CONSTRUCTOR_COLUMNS = (
    "case_id",
    "input_wxyz",
    "input_norm",
    "native_params_xyzw",
    "native_basis_images_xyz",
    "native_probe_image_xyz_or_null",
)
ROTATION_COLUMNS = (
    "case_id",
    "native_params_xyzw",
    "native_basis_images_xyz",
    "native_probe_image_xyz_or_null",
)


class SophusDiscoveryWitnessSet(NativeCaptureModel):
    """Version-neutral numeric payload; schema 4.0 names its layout only."""

    schema_version: Literal["4.0"]
    evidence_kind: Literal["selected_native_discovery_witnesses"]
    scalar_profile: Literal["float32", "float64"]
    probe_xyz: Vector3
    quaternion_construction_columns: tuple[str, ...]
    quaternion_construction: Annotated[
        tuple[ConstructorRow, ...], Field(min_length=8, max_length=8)
    ]
    rotation_columns: tuple[str, ...]
    composition: Annotated[tuple[RotationRow, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def columns_match_rows(self) -> SophusDiscoveryWitnessSet:
        expected_columns = {
            "quaternion_construction_columns": CONSTRUCTOR_COLUMNS,
            "rotation_columns": ROTATION_COLUMNS,
        }
        if any(
            getattr(self, name) != expected
            for name, expected in expected_columns.items()
        ):
            raise ValueError("Sophus witness columns differ from the typed schema")
        if tuple(row[0] for row in self.quaternion_construction) != CONSTRUCTION_CASES:
            raise ValueError("Sophus constructor witness IDs or order changed")
        if tuple(row[0] for row in self.composition) != (
            "mixed_unequal_times_nonunit",
            "nonunit_times_mixed_unequal",
        ):
            raise ValueError("Sophus composition IDs or order changed")
        return self
