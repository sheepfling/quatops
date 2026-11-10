"""Validate the narrow ROS stdout without requiring a ROS installation."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from scripts.evidence_schema.ros_capture import RosChainInputs, RosDirectCapture
from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT


def _untag(value):
    if isinstance(value, dict):
        if set(value) == {"values", "dtype"}:
            return value["values"]
        return {key: _untag(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_untag(item) for item in value]
    return value


def _documents():
    package = ROOT / "experiments/converters/ros2/0.36.23"
    inputs = json.loads((package / "chain_inputs.json").read_text(encoding="utf-8"))
    fixture = json.loads(
        (ROOT / "tests/golden/oracles/ros2/0.36.23/witnesses.json").read_text(
            encoding="utf-8"
        )
    )
    native_inputs = _untag(fixture["native_inputs"])
    native_inputs["world_from_base_rpy"] = inputs["world_from_base"]["rpy_radians"]
    native_inputs["base_from_sensor_rpy"] = inputs["base_from_sensor"]["rpy_radians"]
    capture = {
        "native_scalar_bytes": 8,
        "compiler": "test",
        "cpp_standard": 202002,
        "flags": "test",
        "case_id": inputs["case_id"],
        "native_inputs": native_inputs,
        "observations": _untag(fixture["observations"]),
    }
    return capture, RosChainInputs.model_validate(inputs, strict=True)


@pytest.mark.parametrize("field", ["world_from_base_rpy", "probe_in_source_xyz"])
def test_ros_capture_must_match_input_recipe(field):
    document, inputs = _documents()
    RosDirectCapture.model_validate(document, strict=True).validate_inputs(inputs)
    document["native_inputs"][field][0] += 0.5
    with pytest.raises(ValueError, match="differ from the declared recipe"):
        RosDirectCapture.model_validate(document, strict=True).validate_inputs(inputs)


def test_ros_message_accepts_rounding_and_antipodal_sign():
    document, _ = _documents()
    message = document["observations"][0]["native_outputs"]["lookup_message"]
    message["rotation_xyzw"] = [-x for x in message["rotation_xyzw"]]
    message["rotation_xyzw"][0] += 1e-16
    RosDirectCapture.model_validate(document, strict=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["native_inputs"].update(target_frame="sensor"),
        lambda d: d.update(case_id="other"),
        lambda d: d["observations"][0]["native_outputs"]["lookup_message"].update(
            child_frame="base"
        ),
        lambda d: d["observations"][0]["native_outputs"]["lookup_message"].update(
            rotation_xyzw=[0.1, 0, 0, 1]
        ),
        lambda d: d["observations"][0]["native_outputs"]["lookup"][
            "translation_xyz"
        ].__setitem__(0, None),
        lambda d: d["observations"][0]["native_outputs"].pop("probe_in_target_xyz"),
        lambda d: d["observations"].append(copy.deepcopy(d["observations"][0])),
        lambda d: d["observations"].pop(),
        lambda d: d.update(native_scalar_bytes=4),
    ],
)
def test_invalid_ros_stdout_is_rejected(mutation):
    document, _ = _documents()
    mutation(document)
    with pytest.raises(ValidationError):
        RosDirectCapture.model_validate(document, strict=True)
