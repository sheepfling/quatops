"""Build and run the pinned ROS 2 capture through Docker on any host."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

IMAGE = "quatops-ros2-tf2:jazzy"
PACKAGE = Path("experiments/converters/ros2/0.36.23")
COMMAND = """\
set -e
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/ros-log build --base-paths experiments/converters/ros2/0.36.23/native \
  --build-base /tmp/ros-build --install-base /tmp/ros-install
source /tmp/ros-install/setup.bash
PYTHONPATH=/workspace /opt/quatops-evidence/bin/python experiments/converters/ros2/0.36.23/capture_native.py \
  --repository-root /workspace --inputs experiments/converters/ros2/0.36.23/chain_inputs.json \
  --output /captures --executable /tmp/ros-install/quatops_ros_capture/lib/quatops_ros_capture/capture_ros \
  --compare-witnesses tests/golden/oracles/ros2/0.36.23/witnesses.json --runtime-image "$1"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New or empty capture directory")
    options = parser.parse_args()
    root = options.repository_root.resolve(strict=True)
    package = root / PACKAGE
    dockerfile = package / "Dockerfile"
    base_image = next(
        line.split()[1]
        for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.startswith("FROM ")
    )
    output = (
        options.output.resolve()
        if options.output
        else Path(tempfile.mkdtemp(prefix="quatops-ros2-capture-"))
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Capture directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "build", "--tag", IMAGE, "--file", str(dockerfile), str(package)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--mount",
            f"type=bind,source={root},target=/workspace,readonly",
            "--mount",
            f"type=bind,source={output},target=/captures",
            IMAGE,
            COMMAND,
            "quatops-ros-capture",
            base_image,
        ],
        cwd=root,
        check=True,
    )
    print(f"Validated native ROS witnesses and provenance in {output}")


if __name__ == "__main__":
    main()
