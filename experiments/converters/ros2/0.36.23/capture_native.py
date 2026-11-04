"""Run the native ROS 2 tf2 capture and record pinned runtime provenance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from experiments.oracles.shared.capture_io import (
    Capture,
    Observation,
    contract,
    numeric,
    sha256,
)


def version_for(package: str) -> tuple[str, Path]:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        path = Path(prefix) / "share" / package / "package.xml"
        if path.is_file():
            version = ET.parse(path).getroot().findtext("version")
            if version:
                return version, path
    raise RuntimeError(f"Cannot resolve {package} package identity. Source ROS first.")


def deb_version(package: str) -> str:
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Version}", package],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def pack(value: Any, label: str) -> Any:
    if isinstance(value, dict):
        return {key: pack(item, label) for key, item in value.items()}
    if isinstance(value, list) and (
        not value or isinstance(value[0], (int, float, list))
    ):
        return numeric(value, label)
    if isinstance(value, float):
        return numeric(value, label)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    package_dir = repository_root / "experiments/converters/ros2/0.36.23"
    corpus = args.corpus or (
        repository_root / "experiments/oracles/shared/orientation_inputs_v1.json"
    )

    package_versions = [
        {
            "name": name,
            "version": version_for(name)[0],
            "package_xml_sha256": sha256(version_for(name)[1]),
        }
        for name in ("tf2", "tf2_geometry_msgs", "geometry_msgs")
    ]
    system_versions = [
        {"name": name, "version": deb_version(name)}
        for name in ("nlohmann-json3-dev", "ros-jazzy-tf2-geometry-msgs")
    ]
    capture = Capture(
        "ros2_tf2",
        "ROS 2 tf2",
        package_versions[0]["version"],
        corpus,
        args.output,
        package_dir / "capture_native.py",
        {
            "input": "native tf2Scalar; geometry_msgs double fields",
            "arithmetic": "tf2Scalar as measured by compiled executable",
            "output": "native scalar and message doubles; inspect native_scalar_bytes",
            "serialized": "JSON plus native scalar IEEE bits",
        },
        dependencies=[
            {"name": "ROS_DISTRO", "version": os.environ.get("ROS_DISTRO")},
            {"name": "container image", "version": args.runtime_image},
            *package_versions,
            *system_versions,
        ],
        native_binary=args.executable,
        limitations=[
            "Static same-time BufferCore chain only; no DDS/live network, interpolation, or timestamp extrapolation.",
            "Message field conversion is captured, not DDS/CDR wire serialization.",
            "No QuatOps replay is performed by this native capture.",
        ],
    )

    run = subprocess.run(
        [str(args.executable.resolve()), str(corpus.resolve())],
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = args.output / "native-stdout.json"
    stderr = args.output / "native-stderr.log"
    stdout.write_text(run.stdout, encoding="utf-8", newline="\n")
    stderr.write_text(run.stderr, encoding="utf-8", newline="\n")
    if run.returncode:
        raise RuntimeError(
            f"Native ROS exporter exited {run.returncode}; see {stdout} and {stderr}"
        )

    result = json.loads(run.stdout)
    label = {4: "binary32", 8: "binary64"}.get(result["native_scalar_bytes"])
    if label is None:
        raise ValueError("Unknown native scalar size")
    if result["native_scalar_bytes"] != 8:
        raise ValueError(
            "This wrapper requires double tf2Scalar; capture 32-bit fields separately"
        )
    for item in result["observations"]:
        capture.add(
            Observation(
                item["case_id"],
                item["operation_id"],
                item["native_entry_point"],
                contract(
                    order="xyzw",
                    frame=(
                        "lookup target_from_source; transform maps source point "
                        "coordinates into target"
                    ),
                ),
                pack(item["native_inputs"], label),
                pack(item["native_outputs"], label),
            )
        )
    capture.artifacts.extend(
        [
            (stdout, "native_process_output"),
            (stderr, "native_process_stderr"),
            (package_dir / "native/src/capture_ros.cpp", "native_exporter_source"),
            (package_dir / "native/CMakeLists.txt", "build_recipe"),
            (package_dir / "Dockerfile", "runtime_image_recipe"),
            (
                repository_root / "experiments/oracles/shared/capture_io.py",
                "capture_envelope_helper",
            ),
        ]
    )
    print(capture.finish(["EULER", "ALGEBRA", "MATRIX", "ACTION", "FRAME", "FORMAT"]))


if __name__ == "__main__":
    main()
