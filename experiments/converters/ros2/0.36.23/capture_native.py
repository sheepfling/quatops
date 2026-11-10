"""Run and validate selected native ROS 2 frame-chain witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evidence_json import dump_evidence_json, read_evidence_document
from scripts.evidence_schema.converter_captures import Ros2Contract
from scripts.evidence_schema.ros_capture import RosChainInputs, RosDirectCapture


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version_for(package: str) -> tuple[str, str]:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        path = Path(prefix) / "share" / package / "package.xml"
        if path.is_file():
            version = ET.parse(path).getroot().findtext("version")
            if version:
                return version, sha256(path)
    raise RuntimeError(f"Cannot resolve {package} package identity. Source ROS first.")


def pack(value: Any) -> Any:
    """Tag native numeric arrays without deriving or rounding their values."""
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, list):
        return {"values": value, "dtype": "binary64"}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--compare-witnesses", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    package = root / "experiments/converters/ros2/0.36.23"
    inputs_path = args.inputs or package / "chain_inputs.json"
    inputs = RosChainInputs.model_validate(
        read_evidence_document(inputs_path), strict=True
    )
    contract_path = package / "semantic_contract.json"
    contract = Ros2Contract.model_validate(
        read_evidence_document(contract_path), strict=True
    )
    packages = {
        name: version_for(name)
        for name in ("tf2", "tf2_geometry_msgs", "geometry_msgs")
    }
    if packages["tf2"][0] != contract.target.tf2_version:
        raise ValueError("ROS semantic contract and native runtime versions differ")

    args.output.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        [str(args.executable.resolve()), str(inputs_path.resolve())],
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = args.output / "native-stdout.json"
    stdout.write_text(run.stdout, encoding="utf-8", newline="\n")
    (args.output / "native-stderr.log").write_text(
        run.stderr, encoding="utf-8", newline="\n"
    )
    if run.returncode:
        raise RuntimeError(
            f"Native ROS exporter exited {run.returncode}; see {args.output}"
        )
    capture = RosDirectCapture.model_validate(
        read_evidence_document(stdout), strict=True
    )
    capture.validate_inputs(inputs)
    if capture.native_scalar_bytes != contract.target.native_scalar_bytes:
        raise ValueError("ROS semantic contract and native scalar widths differ")
    native_inputs = capture.native_inputs.model_dump(mode="json")
    del native_inputs["world_from_base_rpy"]
    del native_inputs["base_from_sensor_rpy"]
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_ros2_tf2_buffer_core_capture",
        "selected_from_full_observation_count": len(capture.observations),
        "native_inputs": pack(native_inputs),
        "observations": [
            {
                "operation_id": row.operation_id,
                "native_outputs": pack(
                    row.native_outputs.model_dump(mode="json", exclude_none=True)
                ),
            }
            for row in capture.observations
        ],
    }
    if args.compare_witnesses:
        historical = read_evidence_document(args.compare_witnesses)
        # The historical fixture was selected from a larger discovery run.
        historical.pop("selected_from_full_observation_count")
        selected = dict(witnesses)
        selected.pop("selected_from_full_observation_count")
        if json.dumps(historical, sort_keys=True) != json.dumps(
            selected, sort_keys=True
        ):
            raise ValueError("Direct native values differ from the selected witnesses")
    witness_path = args.output / "witnesses.json"
    witness_path.write_text(
        dump_evidence_json(witnesses), encoding="utf-8", newline="\n"
    )
    files = {
        "input": inputs_path,
        "contract": contract_path,
        "native_source": package / "native/src/capture_ros.cpp",
        "runner": Path(__file__),
        "schema": root / "scripts/evidence_schema/ros_capture.py",
        "dockerfile": package / "Dockerfile",
        "build_recipe": package / "native/CMakeLists.txt",
        "native_binary": args.executable,
        "native_stdout": stdout,
        "witnesses": witness_path,
    }
    provenance = {
        "kind": "direct_native_ros2_tf2_capture",
        "utc": datetime.now(timezone.utc).isoformat(),
        "runtime_image": args.runtime_image,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "packages": {
            name: {"version": version, "package_xml_sha256": digest}
            for name, (version, digest) in packages.items()
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "native_build": {
            key: getattr(capture, key)
            for key in ("native_scalar_bytes", "compiler", "cpp_standard", "flags")
        },
        "sha256": {name: sha256(path) for name, path in files.items()},
        "scope": "static same-time BufferCore chain; no DDS, interpolation or hardware transport",
        "comparison_witness_sha256": sha256(args.compare_witnesses)
        if args.compare_witnesses
        else None,
    }
    (args.output / "manifest.json").write_text(
        dump_evidence_json(provenance), encoding="utf-8", newline="\n"
    )
    print(f"Wrote selected native witnesses and provenance to {args.output}")


if __name__ == "__main__":
    main()
