"""Pinned ROS 2 tf2 frame-chain capture replayed through QuatOps maps."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from quatops.engines import QuaternionFrameMap, SignedAxisMap
from quatops.python import hamilton
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
FIXTURE = ROOT / "tests/golden/oracles/ros2/0.36.23"
CONTRACT = json.loads((FIXTURE / "contract.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
WITNESSES = json.loads((FIXTURE / "witnesses.json").read_text(encoding="utf-8"))
ROWS = {item["operation_id"]: item for item in WITNESSES["observations"]}
WORLD = SignedAxisMap(("x", "y", "z"))
BASE = SignedAxisMap(("y", "-x", "z"))
SENSOR = SignedAxisMap(("x", "-z", "y"))
ATOL = 4e-12


def values(item):
    """Unwrap the capture envelope, preserving numeric values as captured."""
    if isinstance(item, dict) and "values" in item and "dtype" in item:
        return item["values"]
    if isinstance(item, dict):
        return {key: values(value) for key, value in item.items()}
    if isinstance(item, list):
        return [values(value) for value in item]
    return item


def native_wxyz(xyzw):
    q = np.asarray(values(xyzw), dtype=float)
    return (q[3], q[0], q[1], q[2])


def as_array(item):
    return np.asarray(values(item), dtype=float)


def map_native_quaternion(xyzw, target_frame_map, source_frame_map):
    frame_map = QuaternionFrameMap(target_frame_map, source_frame_map)
    return np.asarray(frame_map.from_destination_wxyz(native_wxyz(xyzw)))


def map_native_translation(xyz, target_frame_map):
    return np.asarray(target_frame_map.inverse().vector(values(xyz)), dtype=float)


def close(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ATOL, err_msg=message)


def test_ros_capture_is_native_pinned_and_selected_values_are_lossless():
    assert MANIFEST["evidence_kind"] == "native_runtime_selected_witnesses"
    assert MANIFEST["product"]["name"] == "ROS 2 tf2"
    assert MANIFEST["product"]["version"] == "0.36.23"
    assert MANIFEST["execution"]["was_run"] is True
    assert (
        MANIFEST["review_state"]
        == "native tf2 capture retained; QuatOps frame-chain replay passed"
    )
    assert {
        item["version"]
        for item in MANIFEST["product"]["dependencies"]
        if item["name"] == "ROS_DISTRO"
    } == {"jazzy"}
    assert MANIFEST["native_build"]["native_scalar_bytes"] == 8
    assert MANIFEST["native_build"]["cpp_standard"] == 202002
    assert MANIFEST["selection"]["full_capture_observations"] == 102
    assert MANIFEST["selection"]["retained_observations"] == 2
    assert set(ROWS) == {"lookup-world-from-sensor", "lookup-sensor-from-world"}
    assert CONTRACT["target"]["native_scalar_dtype"] == "float64"
    assert CONTRACT["frames"]["ros_world_from_canonical_world"] == ["x", "y", "z"]
    assert CONTRACT["frames"]["ros_base_from_canonical_base"] == ["y", "-x", "z"]
    assert CONTRACT["frames"]["ros_sensor_optical_from_canonical_sensor"] == [
        "x",
        "-z",
        "y",
    ]

    provenance = MANIFEST["capture_provenance"]
    assert (
        provenance["selected_witnesses_sha256"]
        == hashlib.sha256((FIXTURE / "witnesses.json").read_bytes()).hexdigest()
    )
    reproduction = provenance["reproduction"]
    assert len(reproduction["captured_replay_corpus_sha256"]) == 64
    assert reproduction["input_serialization_note"].startswith(
        "JSON formatting changed only"
    )
    file_hashes = {
        "exporter_source_sha256": ROOT
        / "experiments/converters/ros2/0.36.23/native/src/capture_ros.cpp",
        "runner_sha256": ROOT / "experiments/converters/ros2/0.36.23/capture_native.py",
        "capture_helper_sha256": ROOT / "experiments/oracles/shared/capture_io.py",
        "retention_script_sha256": ROOT / "scripts/retain_ros_tf2_witnesses.py",
        "retention_helper_sha256": ROOT / "scripts/evidence_json.py",
        "retention_formatter_sha256": ROOT / "scripts/evidence_format.py",
        "dockerfile_sha256": ROOT / "experiments/converters/ros2/0.36.23/Dockerfile",
        "build_recipe_sha256": ROOT
        / "experiments/converters/ros2/0.36.23/native/CMakeLists.txt",
        "corpus_sha256": ROOT / "experiments/oracles/shared/orientation_inputs_v1.json",
    }
    for field, path in file_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reproduction[field]


def _canonical_chain():
    inputs = WITNESSES["native_inputs"]
    world_from_base = inputs["world_from_base"]
    base_from_sensor = inputs["base_from_sensor"]

    q_world_base = map_native_quaternion(
        world_from_base["quaternion_xyzw"], WORLD, BASE
    )
    q_base_sensor = map_native_quaternion(
        base_from_sensor["quaternion_xyzw"], BASE, SENSOR
    )
    t_world_base = map_native_translation(world_from_base["translation_xyz"], WORLD)
    t_base_sensor = map_native_translation(base_from_sensor["translation_xyz"], BASE)

    q_world_sensor = np.asarray(hamilton.multiply(q_world_base, q_base_sensor))
    t_world_sensor = (
        np.asarray(hamilton.rotate(q_world_base, t_base_sensor)) + t_world_base
    )
    return (
        q_world_base,
        q_base_sensor,
        t_world_base,
        t_base_sensor,
        q_world_sensor,
        t_world_sensor,
    )


def test_native_tf2_lookup_direction_composition_and_point_action_match_quatops():
    q_wb, q_bs, _t_wb, _t_bs, q_ws, t_ws = _canonical_chain()
    assert abs(float(np.dot(q_wb, q_bs))) < 0.999999

    forward = ROWS["lookup-world-from-sensor"]
    reverse = ROWS["lookup-sensor-from-world"]
    native_fwd = forward["native_outputs"]["lookup"]
    native_rev = reverse["native_outputs"]["lookup"]

    fwd_q = map_native_quaternion(native_fwd["quaternion_xyzw"], WORLD, SENSOR)
    fwd_t = map_native_translation(native_fwd["translation_xyz"], WORLD)
    close(fwd_q, q_ws, "tf2 lookupTransform(world, sensor) quaternion")
    close(fwd_t, t_ws, "tf2 lookupTransform(world, sensor) translation")

    expected_reverse_q = np.asarray(hamilton.inverse(q_ws))
    expected_reverse_t = np.asarray(hamilton.rotate(expected_reverse_q, -t_ws))
    rev_q = map_native_quaternion(native_rev["quaternion_xyzw"], SENSOR, WORLD)
    rev_t = map_native_translation(native_rev["translation_xyz"], SENSOR)
    close(rev_q, expected_reverse_q, "tf2 reverse lookup quaternion")
    close(rev_t, expected_reverse_t, "tf2 reverse lookup translation")

    inputs = WITNESSES["native_inputs"]
    probe_ros = as_array(inputs["probe_in_source_xyz"])
    probe_sensor = np.asarray(SENSOR.inverse().vector(probe_ros))
    target_world = np.asarray(
        WORLD.vector(np.asarray(hamilton.rotate(q_ws, probe_sensor)) + t_ws)
    )
    close(
        forward["native_outputs"]["probe_in_target_xyz"]["values"],
        target_world,
        "native tf2 forward vector action",
    )

    # The two edge rotations are noncommuting, so reversed composition is exposed.
    wrong_order = np.asarray(hamilton.multiply(q_bs, q_wb))
    assert (
        np.max(
            np.abs(
                np.asarray(hamilton.rotate(wrong_order, probe_sensor))
                - np.asarray(hamilton.rotate(q_ws, probe_sensor))
            )
        )
        > 1e-2
    )


def test_tf2_message_fields_and_native_from_message_roundtrip_are_captured():
    row = ROWS["lookup-world-from-sensor"]
    message = row["native_outputs"]["lookup_message"]
    native_lookup = row["native_outputs"]["lookup"]
    roundtrip = row["native_outputs"]["message_roundtrip"]
    assert message["parent_frame"] == "world"
    assert message["child_frame"] == "sensor"
    assert message["stamp_sec"] == 42 and message["stamp_nanosec"] == 0
    close(
        values(message["rotation_xyzw"]),
        values(native_lookup["quaternion_xyzw"]),
        "message quaternion fields",
    )
    close(
        values(message["translation_xyz"]),
        values(native_lookup["translation_xyz"]),
        "message translation fields",
    )
    close(
        values(roundtrip["quaternion_xyzw"]),
        values(native_lookup["quaternion_xyzw"]),
        "tf2::fromMsg quaternion round trip",
    )
    close(
        values(roundtrip["translation_xyz"]),
        values(native_lookup["translation_xyz"]),
        "message transform translation round trip",
    )
