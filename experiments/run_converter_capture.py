"""Run portable converter captures and validate their native witnesses."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import (
    CesiumCapture,
    CesiumContract,
    CesiumInputs,
    ThreeGltfCapture,
    ThreeGltfContract,
    ThreeGltfInputs,
)
from scripts.evidence_schema.utilities import sha256_file
from scripts.platform_probe import is_windows


@dataclass(frozen=True)
class CapturePlan:
    node_project: Path
    input_generator: Path
    native_capture: Path
    target_dir: Path


PLANS = {
    "cesium": CapturePlan(
        node_project=Path("experiments/converters/cesium/1.124.0"),
        input_generator=Path("experiments/converters/cesium/1.124.0/create_inputs.py"),
        native_capture=Path("experiments/converters/cesium/1.124.0/capture_native.mjs"),
        target_dir=Path("experiments/converters/cesium/1.124.0"),
    ),
    "gltf": CapturePlan(
        node_project=Path("experiments/transports/gltf/2.0.0/threejs/0.187.0"),
        input_generator=Path("experiments/converters/gltf/2.0.0/create_asset.py"),
        native_capture=Path("experiments/converters/gltf/2.0.0/capture_threejs.mjs"),
        target_dir=Path("experiments/converters/gltf/2.0.0"),
    ),
}


def run(
    command: list[str],
    *,
    repository_root: Path,
    env: dict[str, str],
    cwd: Path | None = None,
) -> None:
    subprocess.run(command, cwd=cwd or repository_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=[*sorted(PLANS), "opencv"])
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="New or empty native capture directory"
    )
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repository_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    target = options.target
    if target == "opencv":
        if options.output:
            parser.error("--output is supported for cesium and gltf only")
        run_opencv_capture(repository_root, env)
        return
    plan = PLANS[target]
    python_command = [sys.executable]

    if is_windows():
        npm_ci = "npm.cmd ci"
        npm_command = ["cmd.exe", "/d", "/s", "/c", npm_ci]
    else:
        npm_command = ["npm", "ci"]
    run(
        npm_command,
        repository_root=repository_root,
        env=env,
        cwd=repository_root / plan.node_project,
    )
    run(
        [
            *python_command,
            str(repository_root / plan.input_generator),
            "--repository-root",
            str(repository_root),
        ],
        repository_root=repository_root,
        env=env,
    )
    capture_dir = (
        options.output.resolve()
        if options.output
        else Path(tempfile.mkdtemp(prefix=f"quatops-{target}-capture-"))
    )
    run(
        ["node", str(repository_root / plan.native_capture), str(capture_dir)],
        repository_root=repository_root,
        env=env,
    )
    validate_capture(target, repository_root / plan.target_dir, capture_dir)
    print(f"Validated native witnesses and provenance in {capture_dir}")


def validate_capture(target: str, source: Path, capture: Path) -> None:
    """Check the declared target, input cases, native values, and artifact digests."""
    manifest = read_evidence_document(capture / "manifest.json")
    witnesses_path = capture / "witnesses.json"
    document = read_evidence_document(witnesses_path)
    contract_path = source / "semantic_contract.json"
    input_path = source / "converter_inputs.json"
    contract = read_evidence_document(contract_path)
    inputs = read_evidence_document(input_path)
    if target == "cesium":
        result = CesiumCapture.model_validate(document, strict=True)
        CesiumContract.model_validate(contract, strict=True)
        recipe = CesiumInputs.model_validate(inputs, strict=True)
        ids = [row.case_id for row in result.observations]
        expected = [row.case_id for row in recipe.cases]
        if len(expected) != len(set(expected)) or ids != expected:
            raise ValueError(
                "Cesium native witness cases differ from their input recipe"
            )
        for row in result.observations:
            special = row.case_id == "mixed_a@geo-03"
            outputs = row.outputs
            if (
                (outputs.local_basis_and_probe_images is not None) != special
                or len(outputs.world_directions_xyz.values) != (4 if special else 3)
                or len(outputs.world_points_xyz_m.values) != 1
                or len(outputs.point_roundtrip_local_xyz_m.values) != 1
            ):
                raise ValueError("Cesium capture is missing selected native actions")
        expected_name, expected_version = "CesiumJS", "1.124.0"
        input_digest = manifest["inputs"]["sha256"]
    else:
        result = ThreeGltfCapture.model_validate(document, strict=True)
        ThreeGltfContract.model_validate(contract, strict=True)
        ThreeGltfInputs.model_validate(inputs, strict=True)
        if result.native_entry_points != [
            "GLTFLoader.parse",
            "Object3D.updateMatrixWorld",
            "Vector3.applyMatrix4",
            "Matrix4.invert",
        ]:
            raise ValueError(
                "glTF native capture entry points differ from the contract"
            )
        if [row.node for row in result.observations] != [
            "QuatOpsParent",
            "QuatOpsChild",
        ]:
            raise ValueError("glTF capture must contain the parent and child nodes")
        child = result.observations[1].outputs
        if (
            child.world_basis_images_xyz is None
            or child.world_probe_point_xyz is None
            or child.roundtrip_probe_point_xyz is None
        ):
            raise ValueError(
                "glTF child capture is missing native world or inverse actions"
            )
        expected_name, expected_version = "Three.js GLTFLoader", "0.187.0"
        input_digest = manifest["asset"]["input_sha256"]
        if manifest["asset"]["sha256"] != sha256_file(
            source / "quatops_hierarchy.gltf"
        ):
            raise ValueError("glTF asset digest differs from the native capture")
    if (
        manifest["target"]["name"] != expected_name
        or manifest["target"]["package_version"] != expected_version
    ):
        raise ValueError("native target differs from the pinned contract")
    lockfile = (
        source / "package-lock.json"
        if target == "cesium"
        else source.parents[2]
        / "transports/gltf/2.0.0/threejs/0.187.0/package-lock.json"
    )
    if (
        manifest["semantic_contract"]["sha256"] != sha256_file(contract_path)
        or input_digest != sha256_file(input_path)
        or manifest["observations"]["sha256"] != sha256_file(witnesses_path)
        or manifest["observations"]["count"] != len(result.observations)
        or manifest["exporter"]["sha256"]
        != sha256_file(
            source
            / ("capture_native.mjs" if target == "cesium" else "capture_threejs.mjs")
        )
        or manifest["numeric_helper_sha256"]
        != sha256_file(source.parents[1] / "shared/numeric_evidence.mjs")
        or manifest["package_lock_sha256"] != sha256_file(lockfile)
    ):
        raise ValueError("native capture provenance digest differs from its artifact")


def run_opencv_capture(repository_root: Path, env: dict[str, str]) -> None:
    """Prepare and capture OpenCV in its pinned target environment."""
    project = "experiments/converters/python313"
    command_prefix = [
        "uv",
        "run",
        "--locked",
        "--project",
        project,
        "--extra",
        "opencv",
        "--python",
        "3.13.14",
        "python",
    ]
    with tempfile.TemporaryDirectory(prefix="quatops-opencv-input-") as temporary:
        input_path = Path(temporary) / "prepared-inputs.json"
        run(
            [
                *command_prefix,
                "experiments/converters/opencv/4.13.0/prepare_inputs.py",
                "--repository-root",
                str(repository_root),
                "--output",
                str(input_path),
            ],
            repository_root=repository_root,
            env=env,
        )
        run(
            [
                *command_prefix,
                "experiments/converters/opencv/4.13.0/capture_native.py",
                "--repository-root",
                str(repository_root),
                "--inputs",
                str(input_path),
            ],
            repository_root=repository_root,
            env=env,
        )


if __name__ == "__main__":
    main()
