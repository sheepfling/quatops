"""SDK-free validation tests for all ROS native stdout operation variants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.evidence_json import parse_evidence_document
from scripts.evidence_schema.ros_capture import (
    RosFullCapture,
    RosFullTransform,
    RosMatrixInputs,
    RosSelectedCorpus,
    reject_lost_nonfinite_values,
    validate_ros_capture_corpus_link,
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
            {
                "case_id": "euler-0",
                "operation_id": "setRPY-message-action",
                "native_entry_point": "Quaternion.setRPY",
                "native_inputs": {
                    "roll_pitch_yaw_radians": [0, 0, 0],
                    "probe_vectors_xyz": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [2.0, -3.0, 5.0],
                    ],
                },
                "native_outputs": {
                    "quaternion_xyzw": [0, 0, 0, 1],
                    "matrix_rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "readback_rpy": [0, 0, 0],
                    "message_xyzw": [0, 0, 0, 1],
                    "message_roundtrip_xyzw": [0, 0, 0, 1],
                    "basis_and_probe_images": [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [2, -3, 5],
                    ],
                },
            },
            {
                "case_id": "product-general",
                "operation_id": "products",
                "native_entry_point": "Quaternion.operator*",
                "native_inputs": {"a_xyzw": [1, 2, 3, 4], "b_xyzw": [5, 6, 7, 8]},
                "native_outputs": {"ab_xyzw": [9, 8, 7, 6], "ba_xyzw": [6, 7, 8, 9]},
            },
            {
                "case_id": "matrix-identity",
                "operation_id": "matrix-to-quaternion",
                "native_entry_point": "Matrix3x3.getRotation",
                "native_inputs": {"matrix_rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                "native_outputs": {"quaternion_xyzw": [0, 0, 0, 1]},
            },
            _lookup("lookup-world-from-sensor", "world", "sensor"),
            _lookup("lookup-sensor-from-world", "sensor", "world"),
        ],
    }


def test_full_ros_stdout_schema_accepts_each_operation_and_general_products():
    capture = RosFullCapture.model_validate(_capture(), strict=True)
    assert len(capture.observations) == 5


def _selected_corpus(*, extra_euler: bool = False) -> RosSelectedCorpus:
    euler_inputs = [
        {
            "case_id": "euler-0",
            "xyz": [0, 0, 0],
            "actual_bits_hex": ["0", "0", "0"],
            "family": "general",
            "numeric_format": "binary64",
            "order_label": None,
            "units": "radians",
        }
    ]
    if extra_euler:
        euler_inputs.append(
            {
                **euler_inputs[0],
                "case_id": "euler-1",
                "xyz": [0.25, -0.5, 1.0],
            }
        )
    return RosSelectedCorpus.model_validate(
        {
            "euler_inputs": euler_inputs,
            "multiplication_pairs": [
                {
                    "case_id": "product-general",
                    "a_wxyz": [4, 1, 2, 3],
                    "b_wxyz": [8, 5, 6, 7],
                    "family": "algebra",
                }
            ],
            "matrix_inputs": [
                {
                    "case_id": "matrix-identity",
                    "domain": "proper_rotation",
                    "rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                {
                    "case_id": "matrix-shear",
                    "domain": "invalid_rotation",
                    "rows": [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                },
            ],
            "vectors_xyz": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, -3, 5]],
        },
        strict=True,
    )


def _linked_capture(*, extra_euler: bool = False) -> dict[str, object]:
    document = _capture()
    document["observations"][2]["native_inputs"]["matrix_rows"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    document["observations"][1]["native_inputs"] = {
        "a_xyzw": [1.0, 2.0, 3.0, 4.0],
        "b_xyzw": [5.0, 6.0, 7.0, 8.0],
    }
    if extra_euler:
        document["observations"][0]["native_inputs"]["probe_vectors_xyz"] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, -3.0, 5.0],
        ]
        document["observations"][0]["native_outputs"]["basis_and_probe_images"] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        row = document["observations"][0].copy()
        row["case_id"] = "euler-1"
        row["native_inputs"] = {
            "roll_pitch_yaw_radians": [0.25, -0.5, 1.0],
            "probe_vectors_xyz": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -3.0, 5.0],
            ],
        }
        row["native_outputs"] = dict(row["native_outputs"])
        row["native_outputs"]["basis_and_probe_images"] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        document["observations"].insert(1, row)
    return document


def test_ros_capture_links_echoed_inputs_to_selected_corpus_and_probes():
    capture = RosFullCapture.model_validate(_linked_capture(), strict=True)
    validate_ros_capture_corpus_link(capture, _selected_corpus())


def test_ros_capture_rejects_probe_count_different_from_selected_corpus():
    document = _linked_capture()
    document["observations"][0]["native_outputs"]["basis_and_probe_images"].pop()
    capture = RosFullCapture.model_validate(document, strict=True)
    with pytest.raises(ValueError, match="wrong probe count"):
        validate_ros_capture_corpus_link(capture, _selected_corpus())


@pytest.mark.parametrize(
    "probes",
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [5.0, -3.0, 2.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, -3.0, 5.0]],
    ],
)
def test_ros_capture_rejects_same_count_probe_value_or_order_mismatch(probes):
    document = _linked_capture()
    document["observations"][0]["native_inputs"]["probe_vectors_xyz"] = probes
    capture = RosFullCapture.model_validate(document, strict=True)
    with pytest.raises(ValueError, match="probe inputs.*selected corpus"):
        validate_ros_capture_corpus_link(capture, _selected_corpus())


def test_ros_selected_corpus_rejects_duplicate_case_ids():
    corpus = _selected_corpus().model_dump(mode="python")
    corpus["euler_inputs"].append(dict(corpus["euler_inputs"][0]))
    with pytest.raises(ValidationError, match="case IDs must be nonempty and unique"):
        RosSelectedCorpus.model_validate(corpus, strict=True)


@pytest.mark.parametrize(
    "operation_index,field,value",
    [
        (0, "roll_pitch_yaw_radians", [0, 1, 0]),
        (1, "a_xyzw", [9, 2, 3, 4]),
        (2, "matrix_rows", [[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_ros_capture_rejects_corpus_input_mismatch(operation_index, field, value):
    document = _linked_capture()
    document["observations"][operation_index]["native_inputs"][field] = value
    capture = RosFullCapture.model_validate(document, strict=True)
    with pytest.raises(ValueError, match="corpus row"):
        validate_ros_capture_corpus_link(capture, _selected_corpus())


def test_ros_capture_link_uses_variable_corpus_sizes_and_only_selected_proper_matrices():
    capture = RosFullCapture.model_validate(
        _linked_capture(extra_euler=True), strict=True
    )
    validate_ros_capture_corpus_link(
        capture,
        _selected_corpus(extra_euler=True).model_copy(
            update={"vectors_xyz": [[1, 0, 0], [0, 1, 0], [2, -3, 5]]}
        ),
    )
    extra_matrix = _selected_corpus().model_dump(mode="python")
    extra_matrix["matrix_inputs"].append(
        {
            "case_id": "matrix-extra",
            "domain": "proper_rotation",
            "rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    )
    corpus = RosSelectedCorpus.model_validate(extra_matrix, strict=True)
    parsed = RosFullCapture.model_validate(_linked_capture(), strict=True)
    with pytest.raises(ValueError, match="operation IDs do not match"):
        validate_ros_capture_corpus_link(parsed, corpus)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["observations"][0]["native_outputs"].pop("readback_rpy"),
        lambda doc: doc["observations"][1]["native_outputs"].update({"unexpected": 1}),
        lambda doc: doc["observations"][2]["native_inputs"].update(
            {"matrix_rows": [[1, 0], [0, 1]]}
        ),
        lambda doc: doc["observations"][1]["native_inputs"].update(
            {"a_xyzw": [True, 2, 3, 4]}
        ),
        lambda doc: doc["observations"][3]["native_inputs"].update(
            {"target_frame": "sensor"}
        ),
        lambda doc: doc["observations"][3].update({"case_id": "other-lookup"}),
        lambda doc: doc["observations"][3]["native_outputs"]["lookup"][
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


def test_unknown_operation_and_raw_nonstandard_or_overflow_numbers_are_rejected():
    document = _capture()
    document["observations"][0]["operation_id"] = "unknown-operation"
    with pytest.raises(ValidationError):
        RosFullCapture.model_validate(document, strict=True)
    for raw in ('{"value": NaN}', '{"value": Infinity}', '{"value": 1e400}'):
        with pytest.raises(ValueError):
            parse_evidence_document(raw)


def test_duplicate_json_keys_are_rejected_before_model_validation():
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        parse_evidence_document('{"value": 1, "value": 2}')


def test_native_null_reports_lost_nonfinite_information():
    with pytest.raises(ValueError, match="original non-finite value is lost"):
        reject_lost_nonfinite_values({"native_outputs": {"probe": [0, None, 1]}})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_direct_ros_matrix_models_reject_nonfinite_values(bad):
    with pytest.raises(ValidationError):
        RosMatrixInputs.model_validate(
            {"matrix_rows": [[1.0, bad, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            strict=True,
        )
    transform = _transform()
    transform["matrix_rows"][0][1] = bad
    with pytest.raises(ValidationError):
        RosFullTransform.model_validate(transform, strict=True)
