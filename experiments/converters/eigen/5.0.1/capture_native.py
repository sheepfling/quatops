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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "tests/golden/oracles/eigen/5.0.1/converter"
SOURCE_METADATA = ROOT / "experiments/oracles/eigen/5.0.1/source.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    from scripts.evidence_json import parse_evidence_document, read_evidence_document

    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    eigen_root = options.eigen_root.resolve()
    archive_path = options.eigen_archive.resolve()
    source = read_evidence_document(SOURCE_METADATA)
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

    spec_path = HERE / "converter_inputs.json"
    spec = read_evidence_document(spec_path)
    shared_path = ROOT / spec["shared_canonical_inputs"]
    shared_bytes = shared_path.read_bytes()
    shared = parse_evidence_document(shared_bytes)
    shared_by_id = {row["case_id"]: row for row in shared["orientations"]}
    inputs = {
        "schema_version": "1.0",
        "contract_id": spec["contract_id"],
        "canonical_input_sha256": hashlib.sha256(shared_bytes).hexdigest(),
        "preparation_spec_sha256": sha256_file(spec_path),
        "source_frame": shared["source_frame"],
        "probe_vector_xyz": shared["probe_vector_xyz"],
        "scalar_profiles": spec["scalar_profiles"],
        "orientations": [
            {
                "case_id": case_id,
                "canonical_wxyz": shared_by_id[case_id]["canonical_wxyz"],
            }
            for case_id in spec["retained_case_ids"]
        ],
        "native_constructed_orientation": spec["native_constructed_orientation"],
        "composition": spec["composition"],
    }
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "inputs.json"
    write_json(inputs_path, inputs)

    compiler = shutil.which(options.compiler)
    if compiler is None:
        raise FileNotFoundError(f"C++ compiler not found: {options.compiler}")
    compiler_path = Path(compiler).resolve()
    compiler_version = subprocess.run(
        [str(compiler_path), "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    cpp_source = HERE / "capture_native.cpp"

    with tempfile.TemporaryDirectory(
        prefix="quatops-eigen-5.0.1-capture-"
    ) as temp_name:
        temp = Path(temp_name)
        input_text_path = temp / "native-input.tsv"
        executable = temp / (
            "capture_eigen.exe" if platform.system() == "Windows" else "capture_eigen"
        )
        lines = [
            str(len(inputs["orientations"])),
            " ".join(
                format(float(value), ".17g") for value in inputs["probe_vector_xyz"]
            ),
        ]
        lines.extend(
            row["case_id"]
            + " "
            + " ".join(format(float(value), ".17g") for value in row["canonical_wxyz"])
            for row in inputs["orientations"]
        )
        native_axis = inputs["native_constructed_orientation"]
        lines.append(
            native_axis["case_id"]
            + " "
            + " ".join(
                format(float(value), ".17g")
                for value in (*native_axis["axis_xyz"], native_axis["angle_radians"])
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
        native_capture = parse_evidence_document(native_run.stdout)
        native_input_sha256 = sha256_file(input_text_path)

    if native_capture["eigen_version"] != source["version"]:
        raise RuntimeError(
            f"Native harness reports Eigen {native_capture['eigen_version']}, expected {source['version']}"
        )
    witnesses_path = output_dir / "witnesses.json"
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_runtime_selected_converter_witnesses",
        "target": {
            "name": "Eigen",
            "version": native_capture["eigen_version"],
            "version_macros": native_capture["version_macros"],
        },
        "observations": {"profiles": native_capture["profiles"]},
    }
    write_json(witnesses_path, witnesses)
    manifest = {
        "schema_version": "1.0",
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
            "version": native_capture["eigen_version"],
            "header": "Eigen/Geometry",
            "scalar_profiles": ["float32", "float64"],
        },
        "native_build": {
            "compiler_path": str(compiler_path),
            "compiler_version": compiler_version,
            "cxx_standard": "C++17",
            "compiler_flags": ["-O2", "-DNDEBUG"],
            "native_executable_sha256": executable_sha256,
            "version_macros": native_capture["version_macros"],
        },
        "selection": {
            "orientation_cases_per_scalar_profile": len(inputs["orientations"]),
            "native_created_orientations_per_scalar_profile": 1,
            "noncommuting_product_orders_per_scalar_profile": 2,
            "native_vector_actions_per_orientation": 4,
        },
        "provenance": {
            "capture_runner_sha256": sha256_file(Path(__file__)),
            "native_cpp_source_sha256": sha256_file(cpp_source),
            "contract_sha256": sha256_file(output_dir / "contract.json"),
            "converter_input_spec_sha256": sha256_file(spec_path),
            "selected_inputs_sha256": inputs["canonical_input_sha256"],
            "fixture_inputs_sha256": sha256_file(inputs_path),
            "native_input_stream_sha256": native_input_sha256,
            "witnesses_sha256": sha256_file(witnesses_path),
            "eigen_source_archive_sha256": sha256_file(archive_path),
            "eigen_source_metadata_sha256": sha256_file(SOURCE_METADATA),
            "eigen_version_header_sha256": sha256_file(version_header),
            "compile_command": compile_command,
            "native_run_command": [str(executable), str(input_text_path)],
            "exact_invocation": (
                f"{sys.executable} {Path(__file__).resolve()} --eigen-root "
                f"{eigen_root} --eigen-archive {archive_path} "
                f"--compiler {options.compiler}"
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
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
