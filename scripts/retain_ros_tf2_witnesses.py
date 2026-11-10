"""Select the native ROS/tf2 frame-chain witnesses from a completed capture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scripts.evidence_json import (
    dump_evidence_json,
    parse_evidence_document,
    read_evidence_document,
)
from scripts.evidence_schema.converter_captures import Ros2Contract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path, help="Full native capture folder")
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    package_dir = repository_root / "experiments/converters/ros2/0.36.23"
    source = package_dir / "native/src/capture_ros.cpp"
    runner = package_dir / "capture_native.py"
    dockerfile = package_dir / "Dockerfile"
    cmake = package_dir / "native/CMakeLists.txt"
    capture_helper = repository_root / "experiments/oracles/shared/capture_io.py"
    capture_schema = repository_root / "scripts/evidence_schema/ros_capture.py"
    retention_helper = repository_root / "scripts/evidence_json.py"
    retention_formatter = repository_root / "scripts/evidence_format.py"
    corpus = package_dir / "chain_inputs.json"
    destination = repository_root / "tests/golden/oracles/ros2/0.36.23"
    capture_dir = args.capture_dir.resolve()
    manifest = read_evidence_document(capture_dir / "manifest.json")
    native = read_evidence_document(capture_dir / "native-stdout.json")
    observations = [
        parse_evidence_document(line)
        for line in (capture_dir / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = [
        row
        for row in observations
        if row["operation_id"]
        in {"lookup-world-from-sensor", "lookup-sensor-from-world"}
    ]
    if len(selected) != 2:
        raise ValueError(f"expected two ROS frame-chain lookups, found {len(selected)}")
    for row in selected:
        if row["case_id"] != "world-base-sensor":
            raise ValueError("unexpected frame-chain witness")
    by_operation = {row["operation_id"]: row for row in selected}
    forward = by_operation["lookup-world-from-sensor"]

    def transform(row: dict) -> dict:
        return {
            "quaternion_xyzw": row["quaternion_xyzw"],
            "translation_xyz": row["translation_xyz"],
        }

    def observation(row: dict) -> dict:
        outputs = {"lookup": transform(row["native_outputs"]["lookup"])}
        if row["operation_id"] == "lookup-world-from-sensor":
            outputs.update(
                {
                    "lookup_message": row["native_outputs"]["lookup_message"],
                    "message_roundtrip": transform(
                        row["native_outputs"]["message_roundtrip"]
                    ),
                    "probe_in_target_xyz": row["native_outputs"]["probe_in_target_xyz"],
                }
            )
        return {"operation_id": row["operation_id"], "native_outputs": outputs}

    inputs = forward["native_inputs"]
    retained_inputs = {
        "world_from_base": transform(inputs["world_from_base"]),
        "base_from_sensor": transform(inputs["base_from_sensor"]),
        "probe_in_source_xyz": inputs["probe_in_source_xyz"],
        "source_frame": inputs["source_frame"],
        "stamp_sec": inputs["stamp_sec"],
        "target_frame": inputs["target_frame"],
    }
    selected = {
        "native_inputs": retained_inputs,
        "observations": [observation(row) for row in selected],
    }

    destination.mkdir(parents=True, exist_ok=True)
    witnesses_path = destination / "witnesses.json"
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_ros2_tf2_buffer_core_capture",
        "selected_from_full_observation_count": len(observations),
        **selected,
    }
    witnesses_path.write_text(
        dump_evidence_json(witnesses), encoding="utf-8", newline="\n"
    )

    contract_path = package_dir / "semantic_contract.json"
    contract = Ros2Contract.model_validate(
        read_evidence_document(contract_path), strict=True
    )
    if contract.target.tf2_version != manifest["product"]["version"]:
        raise ValueError("ROS semantic contract and native capture versions differ")
    if contract.target.native_scalar_bytes != native["native_scalar_bytes"]:
        raise ValueError("ROS semantic contract and native scalar widths differ")
    (destination / "manifest.json").write_text(
        dump_evidence_json(
            {
                "schema_version": "1.0",
                "semantic_contract": {
                    "file": contract_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256(contract_path),
                },
                "evidence_kind": "native_runtime_selected_witnesses",
                "review_state": "native tf2 capture retained; QuatOps frame-chain replay passed",
                "product": manifest["product"],
                "execution": manifest["execution"],
                "precision": manifest["precision"],
                "native_build": {
                    key: native[key]
                    for key in (
                        "native_scalar_bytes",
                        "compiler",
                        "cpp_standard",
                        "flags",
                    )
                },
                "capture_provenance": {
                    "full_observation_sha256": sha256(
                        capture_dir / "observations.jsonl"
                    ),
                    "native_stdout_sha256": sha256(capture_dir / "native-stdout.json"),
                    "capture_manifest_sha256": sha256(capture_dir / "manifest.json"),
                    "corpus_sha256": sha256(corpus),
                    "exporter_source_sha256": sha256(source),
                    "capture_runner_sha256": sha256(runner),
                    "capture_helper_sha256": sha256(capture_helper),
                    "dockerfile_sha256": sha256(dockerfile),
                    "build_recipe_sha256": sha256(cmake),
                    "selected_witnesses_sha256": sha256(witnesses_path),
                    "reproduction": {
                        "exporter_source_sha256": sha256(source),
                        "runner_sha256": sha256(runner),
                        "capture_helper_sha256": sha256(capture_helper),
                        "capture_schema_sha256": sha256(capture_schema),
                        "retention_script_sha256": sha256(Path(__file__)),
                        "retention_helper_sha256": sha256(retention_helper),
                        "retention_formatter_sha256": sha256(retention_formatter),
                        "dockerfile_sha256": sha256(dockerfile),
                        "build_recipe_sha256": sha256(cmake),
                        "corpus_sha256": sha256(corpus),
                        "input_serialization_note": (
                            "The frame-chain recipe is the capture input; "
                            "historical captures retain their own input digest."
                        ),
                    },
                },
                "selection": {
                    "retained_observations": len(selected["observations"]),
                    "full_capture_observations": len(observations),
                    "policy": (
                        "retain both lookup directions; store shared edge inputs "
                        "once and keep one forward message, point action, and round trip"
                    ),
                    "numeric_values": (
                        "round-trippable finite binary64 decimals, including signed zero"
                    ),
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"retained {len(selected['observations'])} native ROS/tf2 witnesses at {destination}"
    )


if __name__ == "__main__":
    main()
