"""Strict experiment configuration for the supporting Sophus SO(3) oracle."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field, FiniteFloat, model_validator

from scripts.evidence_schema.primitives import NativeCaptureModel


class SophusScalarProfile(NativeCaptureModel):
    capture_target: Literal["Sophus SO3f", "Sophus SO3d"]
    precision: Literal["float32", "float64"]
    witness_target: Literal[
        "Sophus 1.24.6 SO3f / Eigen 5.0.1",
        "Sophus 1.24.6 SO3d / Eigen 5.0.1",
    ]
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
    """Semantics and capture profiles owned by the Sophus experiment."""

    schema_version: Literal["1.0"]
    evidence_status: Literal["supporting_oracle_only"]
    target: Literal["Sophus"]
    version: Literal["1.24.6"]
    dependency: Literal["Eigen 5.0.1"]
    representation: SophusRepresentationContract
    probe_frame: SophusProbeFrame
    probe_xyz: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    scalar_profiles: Annotated[
        list[SophusScalarProfile], Field(min_length=2, max_length=2)
    ]

    @model_validator(mode="after")
    def scalar_profiles_are_distinct(self) -> SophusSemanticConfig:
        profiles = {item.capture_target: item for item in self.scalar_profiles}
        if set(profiles) != {"Sophus SO3f", "Sophus SO3d"}:
            raise ValueError("Sophus config must declare SO3f and SO3d exactly once")
        if (
            profiles["Sophus SO3f"].precision != "float32"
            or profiles["Sophus SO3d"].precision != "float64"
        ):
            raise ValueError("Sophus scalar target and precision do not match")
        if (
            profiles["Sophus SO3f"].witness_target != "Sophus 1.24.6 SO3f / Eigen 5.0.1"
            or profiles["Sophus SO3d"].witness_target
            != "Sophus 1.24.6 SO3d / Eigen 5.0.1"
        ):
            raise ValueError("Sophus scalar target and witness label do not match")
        if any(item.scipy_tolerance <= 0 for item in self.scalar_profiles):
            raise ValueError("SciPy formula tolerances must be positive")
        return self


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
    Vector3,
    OptionalVector3,
]
ExpLogRow = tuple[
    str,
    Vector3,
    Quaternion4,
    Matrix3,
    Vector3,
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
EXP_LOG_CASES = (
    "zero",
    "near_zero",
    "mixed_a",
    "mixed_b",
    "near_pi_axis",
    "above_pi_axis",
)
CONSTRUCTOR_COLUMNS = (
    "case_id",
    "input_wxyz",
    "input_norm",
    "native_params_xyzw",
    "native_basis_images_xyz",
    "native_log_xyz",
    "native_probe_image_xyz_or_null",
)
EXP_LOG_COLUMNS = (
    "case_id",
    "tangent_xyz_radians",
    "native_params_xyzw",
    "native_basis_images_xyz",
    "native_log_of_exp_xyz",
    "native_probe_image_xyz_or_null",
)
ROTATION_COLUMNS = (
    "case_id",
    "native_params_xyzw",
    "native_basis_images_xyz",
    "native_probe_image_xyz_or_null",
)


class SophusDiscoveryWitnessSet(NativeCaptureModel):
    """Strict schema for the compact table-shaped retained Sophus JSON."""

    schema_version: Literal["2.0"]
    evidence_kind: Literal["selected_native_discovery_witnesses"]
    target: Literal[
        "Sophus 1.24.6 SO3d / Eigen 5.0.1",
        "Sophus 1.24.6 SO3f / Eigen 5.0.1",
    ]
    probe_frame: Literal["+X right, +Y forward, +Z up; right-handed"]
    probe_xyz: Vector3
    raw_capture_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    quaternion_construction_columns: tuple[str, ...]
    quaternion_construction: Annotated[
        tuple[ConstructorRow, ...], Field(min_length=8, max_length=8)
    ]
    exp_log_columns: tuple[str, ...]
    exp_log: Annotated[tuple[ExpLogRow, ...], Field(min_length=6, max_length=6)]
    rotation_columns: tuple[str, ...]
    composition: Annotated[tuple[RotationRow, ...], Field(min_length=2, max_length=2)]
    inverse: RotationRow
    perturbation_side: Annotated[
        tuple[RotationRow, ...], Field(min_length=2, max_length=2)
    ]

    @model_validator(mode="after")
    def columns_match_rows(self) -> SophusDiscoveryWitnessSet:
        expected_columns = {
            "quaternion_construction_columns": CONSTRUCTOR_COLUMNS,
            "exp_log_columns": EXP_LOG_COLUMNS,
            "rotation_columns": ROTATION_COLUMNS,
        }
        if any(
            getattr(self, name) != expected
            for name, expected in expected_columns.items()
        ):
            raise ValueError("Sophus witness columns differ from the typed schema")
        if tuple(row[0] for row in self.quaternion_construction) != CONSTRUCTION_CASES:
            raise ValueError("Sophus constructor witness IDs or order changed")
        if tuple(row[0] for row in self.exp_log) != EXP_LOG_CASES:
            raise ValueError("Sophus exp/log witness IDs or order changed")
        return self
