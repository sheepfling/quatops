"""Compile and run selected Eigen quaternion witnesses with a native C++ harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import parse_evidence_document, read_evidence_document
from scripts.evidence_schema.converter_captures import (
    BasisActions,
    CompositionInput,
    EigenCapture,
    EigenComposition,
    EigenConstructedOrientation,
    EigenInputs,
    EigenMatrix,
    EigenObservations,
    EigenOrientation,
    EigenPrecisionProfile,
    NativeConstructedInput,
    QuaternionValues,
    TargetIdentity,
)
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS, Vector3
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    SharedFrameInput,
)
from scripts.platform_probe import executable_suffix


class NativeEigenOrientation(BaseModel):
    """Full quaternion and matrix result from the Eigen C++ harness."""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str
    quaternion: QuaternionValues
    rotation_matrix: EigenMatrix
    native_quaternion_basis_actions_xyz: BasisActions
    native_quaternion_probe_action_xyz: Vector3 | None = None
    matrix_to_quaternion: QuaternionValues


class NativeEigenConstructedOrientation(NativeEigenOrientation):
    """Native AngleAxis result with its explicit construction input."""

    constructor: str
    input_axis_xyz: Vector3
    input_angle_radians: float


class NativeEigenComposition(EigenComposition):
    """Full native matrix view of one noncommuting product."""

    rotation_matrix: EigenMatrix


class NativeEigenProfile(BaseModel):
    """One native Eigen scalar profile before compact witness selection."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dtype: Literal["float32", "float64"]
    scalar_bytes: Literal[4, 8]
    orientations: list[NativeEigenOrientation]
    native_created: list[NativeEigenConstructedOrientation]
    composition: list[NativeEigenComposition]


class NativeEigenCapture(BaseModel):
    """Pydantic envelope for the full native C++ harness output."""

    model_config = ConfigDict(extra="forbid", strict=True)

    eigen_version: str
    version_macros: list[int]
    profiles: list[NativeEigenProfile]


class PreparedEigenOrientation(BaseModel):
    """One canonical quaternion sent to the native stdin protocol."""

    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str
    canonical_wxyz: list[float]


class PreparedEigenInputs(BaseModel):
    """Typed inputs and source provenance for the native Eigen harness."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    contract_id: str
    canonical_input_sha256: str
    preparation_spec_sha256: str
    source_frame: SharedFrameInput
    probe_vector_xyz: Vector3
    scalar_profiles: list[Literal["float32", "float64"]]
    orientations: list[PreparedEigenOrientation]
    native_constructed_orientation: NativeConstructedInput
    composition: CompositionInput


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: BaseModel) -> None:
    path.write_text(
        dump_evidence_json(value.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
        newline="\n",
    )


def retain_profile(profile: NativeEigenProfile) -> EigenPrecisionProfile:
    """Drop redundant matrix views from the permanent witness set."""
    matrix_case_ids = {"mixed_a", "nonprincipal_halfturn"}

    def orientation(row: NativeEigenOrientation) -> EigenOrientation:
        case_id = row.case_id
        return EigenOrientation(
            case_id=case_id,
            quaternion=row.quaternion,
            rotation_matrix=row.rotation_matrix if case_id == "mixed_a" else None,
            native_quaternion_basis_actions_xyz=row.native_quaternion_basis_actions_xyz,
            native_quaternion_probe_action_xyz=row.native_quaternion_probe_action_xyz,
            matrix_to_quaternion=(
                row.matrix_to_quaternion
                if case_id in matrix_case_ids
                or isinstance(row, NativeEigenConstructedOrientation)
                else None
            ),
        )

    constructed = [
        EigenConstructedOrientation(
            case_id=row.case_id,
            constructor=row.constructor,
            quaternion=row.quaternion,
            matrix_to_quaternion=row.matrix_to_quaternion,
            native_quaternion_basis_actions_xyz=row.native_quaternion_basis_actions_xyz,
        )
        for row in profile.native_created
    ]
    compositions = [
        EigenComposition(
            case_id=row.case_id,
            lhs_case_id=row.lhs_case_id,
            rhs_case_id=row.rhs_case_id,
            product_order=row.product_order,
            quaternion=row.quaternion,
            native_quaternion_basis_actions_xyz=row.native_quaternion_basis_actions_xyz,
        )
        for row in profile.composition
    ]
    return EigenPrecisionProfile(
        dtype=profile.dtype,
        scalar_bytes=profile.scalar_bytes,
        orientations=[
            orientation(row)
            for row in profile.orientations
            if row.case_id in NATIVE_RETAINED_CASE_IDS
        ],
        native_created=constructed,
        composition=compositions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--eigen-root",
        type=Path,
        required=True,
        help="Extracted source root containing Eigen/Geometry",
    )
    parser.add_argument(
        "--eigen-archive",
        type=Path,
        required=True,
        help="Downloaded official release archive; verified by SHA-256",
    )
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    target_dir = repository_root / "experiments/converters/eigen/5.0.1"
    source_metadata = repository_root / "experiments/oracles/eigen/5.0.1/source.json"
    eigen_root = options.eigen_root.resolve()
    archive_path = options.eigen_archive.resolve()
    source = read_evidence_document(source_metadata)
    if not (eigen_root / "Eigen/Geometry").is_file():
        raise FileNotFoundError(f"Eigen/Geometry is not under {eigen_root}")
    if sha256_file(archive_path) != source["archive_sha256"]:
        raise ValueError(
            "Eigen release archive SHA-256 does not match the pinned source metadata"
        )
    version_header = eigen_root / "Eigen/Version"
    if (
        f'#define EIGEN_VERSION_STRING "{source["version"]}"'
        not in version_header.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "Extracted Eigen header version does not match the pinned release"
        )

    spec_path = target_dir / "converter_inputs.json"
    spec = EigenInputs.model_validate(read_evidence_document(spec_path), strict=True)
    shared_path = repository_root / spec.shared_canonical_inputs
    shared_bytes = shared_path.read_bytes()
    shared = SharedCanonicalInputCapture.model_validate_json(shared_bytes, strict=True)
    shared_by_id = {row.case_id: row for row in shared.orientations}
    inputs = PreparedEigenInputs(
        schema_version="1.0",
        contract_id=spec.contract_id,
        canonical_input_sha256=hashlib.sha256(shared_bytes).hexdigest(),
        preparation_spec_sha256=sha256_file(spec_path),
        source_frame=shared.source_frame,
        probe_vector_xyz=shared.probe_vector_xyz,
        scalar_profiles=spec.scalar_profiles,
        orientations=[
            PreparedEigenOrientation(
                case_id=case_id,
                canonical_wxyz=shared_by_id[case_id].canonical_wxyz,
            )
            for case_id in spec.retained_case_ids
        ],
        native_constructed_orientation=spec.native_constructed_orientation,
        composition=spec.composition,
    )
    prepared_inputs_bytes = (
        json.dumps(
            inputs.model_dump(mode="json", exclude_none=True),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    prepared_inputs_sha256 = hashlib.sha256(prepared_inputs_bytes).hexdigest()
    output_dir = (
        options.output_dir
        or repository_root / "tests/golden/oracles/eigen/5.0.1/converter"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compiler = shutil.which(options.compiler)
    if compiler is None:
        raise FileNotFoundError(f"C++ compiler not found: {options.compiler}")
    compiler_path = Path(compiler).resolve()
    compiler_version = subprocess.run(
        [str(compiler_path), "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    cpp_source = target_dir / "capture_native.cpp"

    with tempfile.TemporaryDirectory(
        prefix="quatops-eigen-5.0.1-capture-"
    ) as temp_name:
        temp = Path(temp_name)
        input_text_path = temp / "native-input.tsv"
        executable = temp / f"capture_eigen{executable_suffix()}"
        lines = [
            str(len(inputs.orientations)),
            " ".join(format(float(value), ".17g") for value in inputs.probe_vector_xyz),
        ]
        lines.extend(
            row.case_id
            + " "
            + " ".join(format(float(value), ".17g") for value in row.canonical_wxyz)
            for row in inputs.orientations
        )
        native_axis = inputs.native_constructed_orientation
        lines.append(
            (native_axis.case_id or "native_axis_angle")
            + " "
            + " ".join(
                format(float(value), ".17g")
                for value in (*native_axis.axis_xyz, native_axis.angle_radians)
            )
        )
        input_text = "\n".join(lines) + "\n"
        input_text_path.write_text(input_text, encoding="ascii", newline="\n")
        compile_command = [
            str(compiler_path),
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            "-I",
            str(eigen_root),
            str(cpp_source),
            "-o",
            str(executable),
        ]
        compile_result = subprocess.run(
            compile_command, check=False, text=True, capture_output=True
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                f"Eigen capture compile failed:\n{compile_result.stdout}\n{compile_result.stderr}"
            )
        executable_sha256 = sha256_file(executable)
        native_run = subprocess.run(
            [str(executable), str(input_text_path)],
            check=True,
            text=True,
            capture_output=True,
        )
        native_capture = NativeEigenCapture.model_validate(
            parse_evidence_document(native_run.stdout), strict=True
        )
        native_input_sha256 = sha256_file(input_text_path)

    if native_capture.eigen_version != source["version"]:
        raise RuntimeError(
            f"Native harness reports Eigen {native_capture.eigen_version}, expected {source['version']}"
        )
    witnesses_path = output_dir / "witnesses.json"
    witnesses = EigenCapture(
        schema_version="1.0",
        evidence_kind="native_runtime_selected_converter_witnesses",
        target=TargetIdentity(
            name="Eigen",
            version=native_capture.eigen_version,
            version_macros=native_capture.version_macros,
        ),
        observations=EigenObservations(
            profiles=[retain_profile(profile) for profile in native_capture.profiles]
        ),
    )
    write_json(witnesses_path, witnesses)
    capture_runner_sha256 = sha256_file(Path(__file__))
    manifest = {
        "schema_version": "1.0",
        "semantic_contract": {
            "file": (target_dir / "semantic_contract.json")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha256_file(target_dir / "semantic_contract.json"),
        },
        "review_state": "Pinned native Eigen harness capture; repository replay acceptance is recorded by tests",
        "execution": {
            "was_run": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "target": {
            "name": "Eigen",
            "version": native_capture.eigen_version,
            "header": "Eigen/Geometry",
            "scalar_profiles": ["float32", "float64"],
        },
        "native_build": {
            "compiler_path": str(compiler_path),
            "compiler_version": compiler_version,
            "cxx_standard": "C++17",
            "compiler_flags": ["-O2", "-DNDEBUG"],
            "native_executable_sha256": executable_sha256,
            "version_macros": native_capture.version_macros,
        },
        "selection": {
            "orientation_cases_per_scalar_profile": len(NATIVE_RETAINED_CASE_IDS),
            "native_created_orientations_per_scalar_profile": 1,
            "noncommuting_product_orders_per_scalar_profile": 2,
            "native_basis_images_per_orientation": 3,
            "native_mixed_probe_case_id": "mixed_a",
            "matrix_to_quaternion_case_ids": ["mixed_a", "nonprincipal_halfturn"],
            "matrix_layout_case_id": "mixed_a",
        },
        "provenance": {
            "capture_runner_sha256": capture_runner_sha256,
            "reproduction_capture_runner_sha256": capture_runner_sha256,
            "native_cpp_source_sha256": sha256_file(cpp_source),
            "reproduction_native_cpp_source_sha256": sha256_file(cpp_source),
            "contract_sha256": sha256_file(target_dir / "semantic_contract.json"),
            "converter_input_spec_sha256": sha256_file(spec_path),
            "selected_inputs_sha256": inputs.canonical_input_sha256,
            "prepared_inputs_sha256": prepared_inputs_sha256,
            "native_input_stream_sha256": native_input_sha256,
            "witnesses_sha256": sha256_file(witnesses_path),
            "eigen_source_archive_sha256": sha256_file(archive_path),
            "eigen_source_metadata_sha256": sha256_file(source_metadata),
            "eigen_version_header_sha256": sha256_file(version_header),
            "compile_command": compile_command,
            "native_run_command": [str(executable), str(input_text_path)],
            "exact_invocation": (
                f"{sys.executable} {Path(__file__).resolve()} --eigen-root "
                f"{eigen_root} --eigen-archive {archive_path} "
                f"--compiler {options.compiler} --repository-root {repository_root}"
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        dump_evidence_json(manifest), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "eigen_version": manifest["target"]["version"],
                "compiler": compiler_version.splitlines()[0],
                "selection": manifest["selection"],
            }
        )
    )


if __name__ == "__main__":
    main()
