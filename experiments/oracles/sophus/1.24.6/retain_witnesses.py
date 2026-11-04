"""Reduce a native Sophus discovery capture to selected typed witnesses."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator

from scripts.evidence_format import dump_evidence_json

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
_EXP_LOG_CASES = (
    "zero",
    "near_zero",
    "mixed_a",
    "mixed_b",
    "near_pi_axis",
    "above_pi_axis",
)

Vector3 = tuple[float, float, float]
Quaternion4 = tuple[float, float, float, float]
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
        return self


class QuaternionConstruction(StrictModel):
    case_id: str
    input_wxyz: Quaternion4
    input_norm: float
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


class RotationWitness(StrictModel):
    case_id: str
    params_xyzw: Quaternion4
    native_basis_images_xyz: Matrix3
    native_probe_image_xyz: Vector3 | None = None

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


def reduce_capture(capture_bytes: bytes) -> DiscoveryWitnesses:
    capture = NativeCapture.model_validate_json(capture_bytes, strict=True)
    constructors = {item.case_id: item for item in capture.quaternion_construction}
    exp_log = {item.case_id: item for item in capture.exp_log}
    compositions = {item.case_id: item for item in capture.composition}
    perturbations = {item.case_id: item for item in capture.perturbation_side}

    missing = {
        "quaternion_construction": set(_CONSTRUCTION_CASES) - constructors.keys(),
        "exp_log": set(_EXP_LOG_CASES) - exp_log.keys(),
        "composition": {
            "mixed_a_times_mixed_b",
            "mixed_b_times_mixed_a",
        }
        - compositions.keys(),
        "perturbation_side": {"delta_times_mixed_a", "mixed_a_times_delta"}
        - perturbations.keys(),
    }
    missing = {name: sorted(values) for name, values in missing.items() if values}
    if missing:
        raise ValueError(f"native capture lacks selected witness cases: {missing}")

    return DiscoveryWitnesses(
        schema_version="1.0",
        evidence_kind="selected_native_discovery_witnesses",
        target=(
            "Sophus 1.24.6 SO3f / Eigen 5.0.1"
            if capture.target == "Sophus SO3f"
            else "Sophus 1.24.6 SO3d / Eigen 5.0.1"
        ),
        probe_frame="+X right, +Y forward, +Z up; right-handed",
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
        composition=tuple(
            _rotation_witness(compositions[case_id], probe_image=True)
            for case_id in (
                "mixed_a_times_mixed_b",
                "mixed_b_times_mixed_a",
            )
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
        perturbation_side=tuple(
            _rotation_witness(perturbations[case_id], probe_image=False)
            for case_id in ("delta_times_mixed_a", "mixed_a_times_delta")
        ),
    )


def _format_witnesses(value: DiscoveryWitnesses) -> str:
    return dump_evidence_json(value.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture_path = args.capture.resolve()
    output_path = args.output.resolve()
    selected = reduce_capture(capture_path.read_bytes())
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
