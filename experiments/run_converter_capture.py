"""Run portable Node-backed converter captures and retain selected witnesses."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scripts.platform_probe import is_windows


@dataclass(frozen=True)
class CapturePlan:
    node_project: Path
    input_generator: Path
    native_capture: Path
    retainer_module: str


PLANS = {
    "cesium": CapturePlan(
        node_project=Path("experiments/converters/cesium/1.124.0"),
        input_generator=Path("experiments/converters/cesium/1.124.0/create_inputs.py"),
        native_capture=Path("experiments/converters/cesium/1.124.0/capture_native.mjs"),
        retainer_module="scripts.retain_cesium_converter_witnesses",
    ),
    "gltf": CapturePlan(
        node_project=Path("experiments/transports/gltf/2.0.0/threejs/0.187.0"),
        input_generator=Path("experiments/converters/gltf/2.0.0/create_asset.py"),
        native_capture=Path("experiments/converters/gltf/2.0.0/capture_threejs.mjs"),
        retainer_module="scripts.retain_gltf_converter_witnesses",
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
    options = parser.parse_args()
    repository_root = options.repository_root.resolve(strict=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repository_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    target = options.target
    if target == "opencv":
        run_opencv_capture(repository_root, env)
        return
    plan = PLANS[target]
    python_command = ["uv", "run", "--python", "3.13.14", "--extra", "dev", "python"]

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
    with tempfile.TemporaryDirectory(prefix=f"quatops-{target}-capture-") as temporary:
        capture_dir = Path(temporary)
        run(
            ["node", str(repository_root / plan.native_capture), str(capture_dir)],
            repository_root=repository_root,
            env=env,
        )
        run(
            [
                *python_command,
                "-m",
                plan.retainer_module,
                str(capture_dir),
                "--repository-root",
                str(repository_root),
            ],
            repository_root=repository_root,
            env=env,
        )


def run_opencv_capture(repository_root: Path, env: dict[str, str]) -> None:
    """Prepare and capture OpenCV in its pinned target environment."""
    project = "experiments/converters/opencv/4.13.0"
    command_prefix = [
        "uv",
        "run",
        "--locked",
        "--project",
        project,
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
