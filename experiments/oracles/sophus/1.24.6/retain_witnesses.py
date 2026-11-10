"""Reduce a native Sophus discovery capture to selected typed witnesses."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_serializer,
    model_validator,
)

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.primitives import SemanticContractReference
from scripts.evidence_schema.sophus import SophusScalarProfile, SophusSemanticConfig
from scripts.evidence_schema.utilities import sha256_file

_CONSTRUCTION_CASES = (
    "identity",
    "x_quarter",
    "y_quarter",
    "z_quarter",
    "mixed_unequal",
    "mixed_unequal_antipode",
    "nonunit",
    "half_turn_axis",
)
_RAW_EXP_LOG_CASES = (
    "zero",
    "near_zero",
    "mixed_a",
    "mixed_b",
    "near_pi_axis",
    "above_pi_axis",
    "perturbation",
)
_EXP_LOG_CASES = (
    "zero",
    "near_zero",
    "mixed_a",
    "mixed_b",
    "near_pi_axis",
    "above_pi_axis",
)
_PINNED_INPUT_SHA256 = (
    "713fece6dbad561b864c3f625c740190f92210318af4fa09f4467930a04a3048"
)

StrictFiniteFloat = Annotated[FiniteFloat, Field(strict=True)]
Vector3 = tuple[StrictFiniteFloat, StrictFiniteFloat, StrictFiniteFloat]
Quaternion4 = tuple[
    StrictFiniteFloat, StrictFiniteFloat, StrictFiniteFloat, StrictFiniteFloat
]
Matrix3 = tuple[Vector3, Vector3, Vector3]
VectorActions = tuple[Vector3, Vector3, Vector3, Vector3]


class StrictModel(BaseModel):
    """Reject untyped or unexpected capture fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class NativeQuaternion(StrictModel):
    wxyz: Quaternion4
    coeffs_xyzw: Quaternion4


class NativeRotation(StrictModel):
    case_id: str
    unit_quaternion: NativeQuaternion
    params_xyzw: Quaternion4
    matrix_rows: Matrix3
    native_vector_actions_xyz: VectorActions

    @model_validator(mode="after")
    def coefficient_views_agree(self) -> NativeRotation:
        if self.unit_quaternion.coeffs_xyzw != self.params_xyzw:
            raise ValueError("SO3 params() and Eigen quaternion coeffs disagree")
        if self.unit_quaternion.wxyz != (
            self.unit_quaternion.coeffs_xyzw[3],
            self.unit_quaternion.coeffs_xyzw[0],
            self.unit_quaternion.coeffs_xyzw[1],
            self.unit_quaternion.coeffs_xyzw[2],
        ):
            raise ValueError("Eigen quaternion WXYZ and XYZW views disagree")
        return self


class QuaternionConstruction(StrictModel):
    case_id: str
    input_wxyz: Quaternion4
    input_norm: StrictFiniteFloat
    result: NativeRotation
    log_xyz: Vector3


class ExpLog(StrictModel):
    case_id: str
    tangent_xyz: Vector3
    exp_result: NativeRotation
    log_of_exp_xyz: Vector3


class NativeCapture(StrictModel):
    capture_schema: Literal["quatops-sophus-discovery-v1"]
    target: Literal["Sophus SO3d", "Sophus SO3f"]
    probe_xyz: Vector3
    quaternion_construction: tuple[QuaternionConstruction, ...]
    exp_log: tuple[ExpLog, ...]
    composition: tuple[NativeRotation, NativeRotation]
    inverse: NativeRotation
    perturbation_side: tuple[NativeRotation, NativeRotation]

    @model_validator(mode="after")
    def complete_and_consistent_cases(self) -> NativeCapture:
        def exact_ids(
            name: str, actual: tuple[str, ...], expected: tuple[str, ...]
        ) -> None:
            if len(actual) != len(set(actual)):
                raise ValueError(f"duplicate {name} case id")
            if actual != expected:
                raise ValueError(
                    f"{name} case IDs/order differ from the pinned contract"
                )

        exact_ids(
            "quaternion construction",
            tuple(item.case_id for item in self.quaternion_construction),
            _CONSTRUCTION_CASES,
        )
        exact_ids(
            "exp/log",
            tuple(item.case_id for item in self.exp_log),
            _RAW_EXP_LOG_CASES,
        )
        exact_ids(
            "composition",
            tuple(item.case_id for item in self.composition),
            ("mixed_a_times_mixed_b", "mixed_b_times_mixed_a"),
        )
        exact_ids("inverse", (self.inverse.case_id,), ("inverse_mixed_a",))
        exact_ids(
            "perturbation",
            tuple(item.case_id for item in self.perturbation_side),
            ("delta_times_mixed_a", "mixed_a_times_delta"),
        )
        for item in self.quaternion_construction:
            if item.result.case_id != item.case_id:
                raise ValueError("constructor result case id does not match parent")
        for item in self.exp_log:
            if item.exp_result.case_id != item.case_id:
                raise ValueError("exp result case id does not match parent")
        constructors = {item.case_id: item for item in self.quaternion_construction}
        mixed = constructors["mixed_unequal"].result.params_xyzw
        antipode = constructors["mixed_unequal_antipode"].result.params_xyzw
        mixed_scale = max(abs(value) for value in mixed)
        antipode_scale = max(abs(value) for value in antipode)
        if mixed_scale == 0.0 or antipode_scale == 0.0:
            raise ValueError("antipode constructor storage must be nonzero")
        normalized_dot = sum(
            (left / mixed_scale) * (right / antipode_scale)
            for left, right in zip(mixed, antipode)
        )
        if normalized_dot >= 0.0:
            raise ValueError("antipodal constructor storage must have opposite signs")
        return self

    def validate_pinned_inputs(self, input_path: Path) -> None:
        """Compare the native capture's inputs with the exact pinned TSV corpus."""
        input_bytes = input_path.read_bytes()
        if hashlib.sha256(input_bytes).hexdigest() != _PINNED_INPUT_SHA256:
            raise ValueError(
                "pinned Sophus input TSV hash differs from the recorded corpus"
            )
        probe: tuple[float, ...] | None = None
        quaternions: dict[str, tuple[float, ...]] = {}
        tangents: dict[str, tuple[float, ...]] = {}
        for line_number, raw_line in enumerate(
            input_bytes.decode("utf-8").splitlines(), 1
        ):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            fields = line.split()
            kind, *values = fields
            if kind == "probe":
                if probe is not None or len(values) != 3:
                    raise ValueError(
                        f"invalid or duplicate probe at line {line_number}"
                    )
                probe = tuple(float(value) for value in values)
                if not all(math.isfinite(value) for value in probe):
                    raise ValueError(
                        f"non-finite probe in pinned TSV at line {line_number}"
                    )
                continue
            if kind not in {"quaternion", "tangent"}:
                raise ValueError(f"unknown pinned input kind at line {line_number}")
            expected_width = 5 if kind == "quaternion" else 4
            if len(values) != expected_width:
                raise ValueError(f"invalid {kind} row at line {line_number}")
            case_id, *components = values
            target = quaternions if kind == "quaternion" else tangents
            if case_id in target:
                raise ValueError(f"duplicate {kind} input case id: {case_id}")
            parsed = tuple(float(value) for value in components)
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError(
                    f"non-finite {kind} in pinned TSV at line {line_number}"
                )
            target[case_id] = parsed

        if probe is None or tuple(quaternions) != _CONSTRUCTION_CASES:
            raise ValueError("pinned TSV probe/constructor cases differ from contract")
        raw_tangent_cases = (
            "zero",
            "near_zero",
            "mixed_a",
            "mixed_b",
            "near_pi_axis",
            "above_pi_axis",
            "perturbation",
        )
        if tuple(tangents) != raw_tangent_cases:
            raise ValueError("pinned TSV tangent cases differ from contract")

        def target_precision(values: tuple[float, ...]) -> tuple[float, ...]:
            if self.target == "Sophus SO3f":
                return tuple(
                    struct.unpack("=f", struct.pack("=f", value))[0] for value in values
                )
            return values

        if target_precision(self.probe_xyz) != target_precision(probe):
            raise ValueError("native probe differs from pinned TSV")
        for item in self.quaternion_construction:
            if target_precision(item.input_wxyz) != target_precision(
                quaternions[item.case_id]
            ):
                raise ValueError(
                    f"constructor input differs from pinned TSV: {item.case_id}"
                )
        for item in self.exp_log:
            if target_precision(item.tangent_xyz) != target_precision(
                tangents[item.case_id]
            ):
                raise ValueError(
                    f"exp/log tangent differs from pinned TSV: {item.case_id}"
                )


class RotationWitness(StrictModel):
    case_id: str
    params_xyzw: Quaternion4
    native_basis_images_xyz: Matrix3
    native_probe_image_xyz: Optional[Vector3] = None  # noqa: UP045 -- Pydantic evaluates this on Python 3.9.

    @model_serializer(mode="plain")
    def serialize_row(self) -> list[object]:
        return [
            self.case_id,
            self.params_xyzw,
            self.native_basis_images_xyz,
            self.native_probe_image_xyz,
        ]


class ConstructorWitness(StrictModel):
    case_id: str
    input_wxyz: Quaternion4
    input_norm: float
    result: RotationWitness
    log_xyz: Vector3

    @model_serializer(mode="plain")
    def serialize_row(self) -> list[object]:
        return [
            self.case_id,
            self.input_wxyz,
            self.input_norm,
            self.result.params_xyzw,
            self.result.native_basis_images_xyz,
            self.log_xyz,
            self.result.native_probe_image_xyz,
        ]


class ExpLogWitness(StrictModel):
    case_id: str
    tangent_xyz: Vector3
    exp_result: RotationWitness
    log_of_exp_xyz: Vector3

    @model_serializer(mode="plain")
    def serialize_row(self) -> list[object]:
        return [
            self.case_id,
            self.tangent_xyz,
            self.exp_result.params_xyzw,
            self.exp_result.native_basis_images_xyz,
            self.log_of_exp_xyz,
            self.exp_result.native_probe_image_xyz,
        ]


class DiscoveryWitnesses(StrictModel):
    schema_version: Literal["1.0"]
    evidence_kind: Literal["selected_native_discovery_witnesses"]
    target: Literal[
        "Sophus 1.24.6 SO3d / Eigen 5.0.1",
        "Sophus 1.24.6 SO3f / Eigen 5.0.1",
    ]
    probe_frame: Literal["+X right, +Y forward, +Z up; right-handed"]
    probe_xyz: Vector3
    raw_capture_sha256: str
    quaternion_construction_columns: tuple[str, ...]
    quaternion_construction: tuple[ConstructorWitness, ...]
    exp_log_columns: tuple[str, ...]
    exp_log: tuple[ExpLogWitness, ...]
    composition_columns: tuple[str, ...]
    composition: tuple[RotationWitness, RotationWitness]
    inverse_columns: tuple[str, ...]
    inverse: RotationWitness
    perturbation_columns: tuple[str, ...]
    perturbation_side: tuple[RotationWitness, RotationWitness]


def load_semantic_config() -> SophusSemanticConfig:
    """Load the experiment-owned contract only when its manifest hash agrees."""
    package_dir = Path(__file__).resolve().parent
    repository_root = package_dir.parents[3]
    config_path = package_dir / "semantic_contract.json"
    manifest = read_evidence_document(package_dir / "manifest.json")
    reference = SemanticContractReference.model_validate(
        manifest.get("semantic_contract"), strict=True
    )
    if reference.file != config_path.relative_to(repository_root).as_posix():
        raise ValueError("manifest semantic contract points outside this experiment")
    if sha256_file(config_path) != reference.sha256:
        raise ValueError("experiment semantic contract hash differs from manifest")
    return SophusSemanticConfig.model_validate(
        read_evidence_document(config_path), strict=True
    )


def profile_for_capture(
    config: SophusSemanticConfig, capture: NativeCapture
) -> SophusScalarProfile:
    """Resolve the scalar profile and verify the fixture-owned probe frame."""
    if capture.probe_xyz != tuple(config.probe_xyz):
        raise ValueError("native capture probe differs from the experiment contract")
    profile = next(
        (
            item
            for item in config.scalar_profiles
            if item.capture_target == capture.target
        ),
        None,
    )
    if profile is None:
        raise ValueError(f"semantic config has no profile for {capture.target!r}")
    return profile


def _rotation_witness(
    rotation: NativeRotation,
    *,
    probe_image: bool,
) -> RotationWitness:
    actions = rotation.native_vector_actions_xyz
    return RotationWitness(
        case_id=rotation.case_id,
        params_xyzw=rotation.params_xyzw,
        native_basis_images_xyz=actions[:3],
        native_probe_image_xyz=actions[3] if probe_image else None,
    )


def reduce_capture(
    capture_bytes: bytes, config: SophusSemanticConfig
) -> DiscoveryWitnesses:
    capture = NativeCapture.model_validate_json(capture_bytes, strict=True)
    profile = profile_for_capture(config, capture)
    constructors = {item.case_id: item for item in capture.quaternion_construction}
    exp_log = {item.case_id: item for item in capture.exp_log}
    compositions = {item.case_id: item for item in capture.composition}
    perturbations = {item.case_id: item for item in capture.perturbation_side}

    capture.validate_pinned_inputs(Path(__file__).with_name("capture_inputs.tsv"))

    return DiscoveryWitnesses(
        schema_version="1.0",
        evidence_kind="selected_native_discovery_witnesses",
        target=profile.witness_target,
        probe_frame=config.probe_frame.description,
        probe_xyz=capture.probe_xyz,
        raw_capture_sha256=hashlib.sha256(capture_bytes).hexdigest(),
        quaternion_construction_columns=(
            "case_id",
            "input_wxyz",
            "input_norm",
            "native_params_xyzw",
            "native_basis_images_xyz",
            "native_log_xyz",
            "native_probe_image_xyz_or_null",
        ),
        quaternion_construction=tuple(
            ConstructorWitness(
                case_id=item.case_id,
                input_wxyz=item.input_wxyz,
                input_norm=item.input_norm,
                result=_rotation_witness(
                    item.result,
                    probe_image=item.case_id == "mixed_unequal",
                ),
                log_xyz=item.log_xyz,
            )
            for item in (constructors[case_id] for case_id in _CONSTRUCTION_CASES)
        ),
        exp_log_columns=(
            "case_id",
            "tangent_xyz_radians",
            "native_params_xyzw",
            "native_basis_images_xyz",
            "native_log_of_exp_xyz",
            "native_probe_image_xyz_or_null",
        ),
        exp_log=tuple(
            ExpLogWitness(
                case_id=item.case_id,
                tangent_xyz=item.tangent_xyz,
                exp_result=_rotation_witness(
                    item.exp_result,
                    probe_image=item.case_id == "mixed_a",
                ),
                log_of_exp_xyz=item.log_of_exp_xyz,
            )
            for item in (exp_log[case_id] for case_id in _EXP_LOG_CASES)
        ),
        composition_columns=(
            "case_id",
            "native_params_xyzw",
            "native_basis_images_xyz",
            "native_probe_image_xyz_or_null",
        ),
        composition=(
            _rotation_witness(compositions["mixed_a_times_mixed_b"], probe_image=True),
            _rotation_witness(compositions["mixed_b_times_mixed_a"], probe_image=True),
        ),
        inverse_columns=(
            "case_id",
            "native_params_xyzw",
            "native_basis_images_xyz",
            "native_probe_image_xyz_or_null",
        ),
        inverse=_rotation_witness(capture.inverse, probe_image=False),
        perturbation_columns=(
            "case_id",
            "native_params_xyzw",
            "native_basis_images_xyz",
            "native_probe_image_xyz_or_null",
        ),
        perturbation_side=(
            _rotation_witness(perturbations["delta_times_mixed_a"], probe_image=False),
            _rotation_witness(perturbations["mixed_a_times_delta"], probe_image=False),
        ),
    )


def _format_witnesses(value: DiscoveryWitnesses) -> str:
    return dump_evidence_json(value.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_semantic_config()
    capture_path = args.capture.resolve()
    output_path = args.output.resolve()
    selected = reduce_capture(capture_path.read_bytes(), config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _format_witnesses(selected),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {output_path}")
    print(
        f"Retained {len(selected.quaternion_construction)} constructor and "
        f"{len(selected.exp_log)} exp/log witnesses; raw capture remains external."
    )


if __name__ == "__main__":
    main()
