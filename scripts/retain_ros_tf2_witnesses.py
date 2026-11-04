"""Select the native ROS/tf2 frame-chain witnesses from a completed capture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from evidence_json import (
    compact_numeric_evidence,
    dump_evidence_json,
    parse_evidence_document,
    read_evidence_document,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/converters/ros2/0.36.23"
SOURCE = HERE / "native/src/capture_ros.cpp"
RUNNER = HERE / "capture_native.py"
DOCKERFILE = HERE / "Dockerfile"
CMAKE = HERE / "native/CMakeLists.txt"
CAPTURE_HELPER = ROOT / "experiments/oracles/shared/capture_io.py"
RETENTION_HELPER = ROOT / "scripts/evidence_json.py"
RETENTION_FORMATTER = ROOT / "scripts/evidence_format.py"
CORPUS = ROOT / "experiments/oracles/shared/orientation_inputs_v1.json"
DESTINATION = ROOT / "tests/golden/oracles/ros2/0.36.23"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path, help="Full native capture folder")
    args = parser.parse_args()
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
    selected = compact_numeric_evidence(
        {
            "native_inputs": retained_inputs,
            "observations": [observation(row) for row in selected],
        }
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    witnesses_path = DESTINATION / "witnesses.json"
    witnesses = {
        "schema_version": "1.0",
        "evidence_kind": "native_ros2_tf2_buffer_core_capture",
        "selected_from_full_observation_count": len(observations),
        **selected,
    }
    witnesses_path.write_text(
        dump_evidence_json(witnesses), encoding="utf-8", newline="\n"
    )

    maps = {
        "canonical_axes": {
            "handedness": "right",
            "positive_x": "right/east",
            "positive_y": "forward/north",
            "positive_z": "up",
        },
        "ros_world_from_canonical_world": ["x", "y", "z"],
        "ros_base_from_canonical_base": ["y", "-x", "z"],
        "ros_sensor_optical_from_canonical_sensor": ["x", "-z", "y"],
        "interpretation": (
            "Each signed-axis map maps canonical frame-coordinate components "
            "into the named ROS frame. The maps are fixture declarations, not "
            "inferred from frame IDs."
        ),
    }
    contract = {
        "contract_id": "quatops-ros2-tf2-frame-chain-v1",
        "target": {
            "name": "ROS 2 tf2",
            "distro": "jazzy",
            "tf2_version": manifest["product"]["version"],
            "native_scalar_bytes": native["native_scalar_bytes"],
            "native_scalar_dtype": "float64",
        },
        "native_api": [
            "tf2::BufferCore::setTransform",
            "tf2::BufferCore::lookupTransform",
            "tf2::Transform",
            "geometry_msgs/msg/TransformStamped",
            "tf2::fromMsg",
        ],
        "rotation": {
            "algebra": "Hamilton",
            "native_order": "xyzw",
            "action": "active source-point to target-point transform",
            "units": "radians",
        },
        "frames": maps,
        "frame_direction": (
            "lookupTransform(target, source) returns target_from_source; its "
            "rotation and translation map source point coordinates into target "
            "coordinates"
        ),
        "conversion": (
            "For each edge, convert native target_from_source quaternion with "
            "QuaternionFrameMap(world_map=target_map.inverse(), "
            "body_map=source_map.inverse()).from_destination_wxyz(); map "
            "translation with target_map.inverse().vector(). Compose labelled "
            "edges world_from_base and base_from_sensor in that order."
        ),
        "inverse_conversion": (
            "Apply the same operation to the reverse lookup using its declared "
            "target/source maps; equivalently invert the canonical forward "
            "rigid transform."
        ),
        "input_domain": (
            "unit rotations, finite translations, static transforms at one "
            "exact timestamp; native double tf2Scalar"
        ),
        "scope": [
            "named-frame direction",
            "two-edge noncommuting chain",
            "both lookup directions",
            "TransformStamped message fields and fromMsg round trip",
            "point action",
        ],
        "exclusions": [
            "DDS/CDR wire encoding",
            "live network",
            "time interpolation/extrapolation",
            "dynamic broadcasters",
            "CovarianceStamped",
            "ROS model-forward assumptions",
        ],
        "native_witnesses": "witnesses.json",
    }
    (DESTINATION / "contract.json").write_text(
        dump_evidence_json(contract), encoding="utf-8", newline="\n"
    )
    (DESTINATION / "manifest.json").write_text(
        dump_evidence_json(
            {
                "schema_version": "1.0",
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
                    "corpus_sha256": sha256(CORPUS),
                    "exporter_source_sha256": sha256(SOURCE),
                    "capture_runner_sha256": sha256(RUNNER),
                    "capture_helper_sha256": sha256(CAPTURE_HELPER),
                    "dockerfile_sha256": sha256(DOCKERFILE),
                    "build_recipe_sha256": sha256(CMAKE),
                    "selected_witnesses_sha256": sha256(witnesses_path),
                    "reproduction": {
                        "exporter_source_sha256": sha256(SOURCE),
                        "runner_sha256": sha256(RUNNER),
                        "capture_helper_sha256": sha256(CAPTURE_HELPER),
                        "retention_script_sha256": sha256(Path(__file__)),
                        "retention_helper_sha256": sha256(RETENTION_HELPER),
                        "retention_formatter_sha256": sha256(RETENTION_FORMATTER),
                        "dockerfile_sha256": sha256(DOCKERFILE),
                        "build_recipe_sha256": sha256(CMAKE),
                        "corpus_sha256": sha256(CORPUS),
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
                        "round-trippable decimals and dtype; exact bits only for "
                        "signed-zero or nonfinite values"
                    ),
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"retained {len(selected['observations'])} native ROS/tf2 witnesses at {DESTINATION}"
    )


if __name__ == "__main__":
    main()
