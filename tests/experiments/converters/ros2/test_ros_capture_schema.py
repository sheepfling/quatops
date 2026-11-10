"""SDK-free validation tests for all ROS native stdout operation variants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.evidence_schema.ros_capture import (
    RosChainInputs,
    RosFullCapture,
    RosFullTransform,
    reject_lost_nonfinite_values,
    validate_ros_capture_inputs,
)


def _transform() -> dict[str, object]:
    return {
        "quaternion_xyzw": [0, 0, 0, 1],
        "translation_xyz": [0, 0, 0],
        "matrix_rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }


def _lookup(operation: str, target: str, source: str) -> dict[str, object]:
    transform = _transform()
    return {
        "case_id": "world-base-sensor",
        "operation_id": operation,
        "native_entry_point": "BufferCore.lookupTransform",
        "native_inputs": {
            "target_frame": target,
            "source_frame": source,
            "stamp_sec": 42,
            "world_from_base": _transform(),
            "base_from_sensor": _transform(),
            "probe_in_source_xyz": [2, -3, 5],
            "world_from_base_rpy": [0, 0, 0],
            "base_from_sensor_rpy": [0, 0, 0],
        },
        "native_outputs": {
            "lookup_message": {
                "parent_frame": target,
                "child_frame": source,
                "stamp_sec": 42,
                "stamp_nanosec": 0,
                "translation_xyz": transform["translation_xyz"],
                "rotation_xyzw": transform["quaternion_xyzw"],
            },
            "lookup": transform,
            "message_roundtrip": transform,
            "probe_in_target_xyz": [2, -3, 5],
            "direct_world_from_sensor": _transform(),
            "direct_sensor_from_world": _transform(),
            "world_probe_sequential": [2, -3, 5],
        },
    }


def _capture() -> dict[str, object]:
    return {
        "native_scalar_bytes": 8,
        "compiler": "test compiler",
        "cpp_standard": 202002,
        "flags": "test flags",
        "observations": [
            _lookup("lookup-world-from-sensor", "world", "sensor"),
            _lookup("lookup-sensor-from-world", "sensor", "world"),
        ],
    }


def _inputs() -> RosChainInputs:
    return RosChainInputs.model_validate(
        {
            "case_id": "world-base-sensor",
            "world_from_base": {"rpy_radians": [0, 0, 0], "translation_xyz": [0, 0, 0]},
            "base_from_sensor": {
                "rpy_radians": [0, 0, 0],
                "translation_xyz": [0, 0, 0],
            },
            "probe_in_source_xyz": [2, -3, 5],
            "stamp_sec": 42,
        },
        strict=True,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update({"world_from_base_rpy": [0.5, 0, 0]}),
        lambda row: row["world_from_base"].update({"translation_xyz": [1, 0, 0]}),
        lambda row: row.update({"probe_in_source_xyz": [1, 2, 3]}),
    ],
)
def test_ros_capture_links_both_lookups_to_the_frame_chain_recipe(mutate):
    capture = RosFullCapture.model_validate(_capture(), strict=True)
    validate_ros_capture_inputs(capture, _inputs())
    document = _capture()
    for row in document["observations"]:
        mutate(row["native_inputs"])
    capture = RosFullCapture.model_validate(document, strict=True)
    with pytest.raises(ValueError, match="differ from the declared recipe"):
        validate_ros_capture_inputs(capture, _inputs())


def test_ros_message_roundtrip_accepts_rounding_and_antipodal_sign():
    document = _capture()
    document["observations"][0]["native_outputs"]["lookup_message"]["rotation_xyzw"] = [
        1e-16,
        0,
        0,
        1,
    ]
    document["observations"][1]["native_outputs"]["lookup_message"]["rotation_xyzw"] = [
        0,
        0,
        0,
        -1,
    ]
    RosFullCapture.model_validate(document, strict=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["observations"][0]["native_inputs"].update(
            {"target_frame": "sensor"}
        ),
        lambda doc: doc["observations"][0].update({"case_id": "other-lookup"}),
        lambda doc: doc["observations"][0]["native_outputs"]["lookup_message"].update(
            {"child_frame": "base"}
        ),
        lambda doc: doc["observations"][0]["native_outputs"]["lookup_message"].update(
            {"rotation_xyzw": [0.1, 0, 0, 1]}
        ),
        lambda doc: doc["observations"][0]["native_outputs"]["lookup"][
            "translation_xyz"
        ].__setitem__(0, None),
        lambda doc: doc["observations"].append(doc["observations"][0].copy()),
        lambda doc: doc["observations"].pop(),
    ],
)
def test_invalid_ros_stdout_is_rejected(mutate):
    document = _capture()
    mutate(document)
    with pytest.raises(ValidationError):
        RosFullCapture.model_validate(document, strict=True)


def test_native_null_reports_lost_nonfinite_information():
    with pytest.raises(ValueError, match="original non-finite value is lost"):
        reject_lost_nonfinite_values({"native_outputs": {"probe": [0, None, 1]}})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_direct_ros_matrix_models_reject_nonfinite_values(bad):
    transform = _transform()
    transform["matrix_rows"][0][1] = bad
    with pytest.raises(ValidationError):
        RosFullTransform.model_validate(transform, strict=True)
