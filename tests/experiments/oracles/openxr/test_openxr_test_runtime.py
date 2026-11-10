"""Validate and replay the retained Khronos test-runtime frame-chain evidence."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evidence_json import parse_evidence_document, read_evidence_document
from scripts.evidence_schema.openxr import (
    OpenXRTestRuntimeCapture,
    finite_max_abs,
)
from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

ROOT = REPOSITORY_ROOT
CAPTURE_DIR = ROOT / "experiments/oracles/openxr/1.1.65"
MANIFEST_PATH = CAPTURE_DIR / "manifest.json"
WITNESS_PATH = CAPTURE_DIR / "witnesses/test_runtime_frame_chain.json"
VERIFIER_PATH = CAPTURE_DIR / "verify_test_runtime.py"


def _capture_document() -> dict:
    return read_evidence_document(WITNESS_PATH)


def test_openxr_producer_hash_and_current_replay_identity_are_distinct() -> None:
    manifest = read_evidence_document(MANIFEST_PATH)
    assert manifest["pin_file"] == "source.json"
    assert (
        hashlib.sha256((CAPTURE_DIR / "source.json").read_bytes()).hexdigest()
        == (manifest["artifacts"]["capture_sources_sha256"]["source.json"])
    )
    historical_hash = manifest["artifacts"]["capture_sources_sha256"][
        "verify_test_runtime.py"
    ]
    current_hash = manifest["replay"]["verifier_script_sha256"]
    assert (
        hashlib.sha256(
            read_provenance_source(
                ROOT, manifest["artifacts"]["historical_verifier_archive"]
            )
        ).hexdigest()
        == historical_hash
    )
    assert hashlib.sha256(VERIFIER_PATH.read_bytes()).hexdigest() == current_hash
    assert historical_hash != current_hash
    assert (
        manifest["replay"]["test_module"] == Path(__file__).relative_to(ROOT).as_posix()
    )
    witness_hash = hashlib.sha256(WITNESS_PATH.read_bytes()).hexdigest()
    assert (
        witness_hash
        == manifest["artifacts"]["test_runtime_frame_chain"]["witness_sha256"]
    )


def test_capture_schema_accepts_the_retained_fixture() -> None:
    capture = OpenXRTestRuntimeCapture.model_validate(_capture_document(), strict=True)
    assert capture.runtime.product == "Khronos OpenXR-SDK-Source test_runtime"
    assert capture.runtime.hardware_runtime is False
    assert set(capture.observations) == {"A_from_B", "B_from_C", "A_from_C"}


def test_capture_schema_accepts_native_stdout_without_retained_envelope() -> None:
    capture = _capture_document()
    capture.pop("schema_version")

    parsed = OpenXRTestRuntimeCapture.model_validate(capture, strict=True)
    assert parsed.schema_version is None


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_json_reader_rejects_nonfinite_tokens_and_overflow(token: str) -> None:
    with pytest.raises(ValueError):
        parse_evidence_document(f'{{"number":{token}}}')


@pytest.mark.parametrize("invalid", ["1.25", True])
@pytest.mark.parametrize(
    "path",
    [
        ("inputs_pose_in_reference_space", "A", "position_xyz_m", 0),
        ("inputs_pose_in_reference_space", "A", "orientation_xyzw", 0),
        ("observations", "A_from_B", "pose", "position_xyz_m", 0),
        ("observations", "A_from_B", "pose", "orientation_xyzw", 0),
    ],
)
def test_pose_values_reject_strings_and_booleans(
    invalid: object, path: tuple[str | int, ...]
) -> None:
    capture = _capture_document()
    current = capture
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = invalid

    with pytest.raises(ValidationError):
        OpenXRTestRuntimeCapture.model_validate(capture, strict=True)


def test_capture_schema_rejects_nonfinite_values() -> None:
    capture = _capture_document()
    capture["observations"]["A_from_B"]["pose"]["position_xyz_m"][0] = float("inf")

    with pytest.raises(ValidationError):
        OpenXRTestRuntimeCapture.model_validate(capture, strict=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda capture: capture["inputs_pose_in_reference_space"]["A"][
            "orientation_xyzw"
        ].pop(),
        lambda capture: capture["observations"]["A_from_B"].update({"unexpected": 1}),
        lambda capture: capture.update({"schema_version": "2.0"}),
        lambda capture: capture.update({"unrecognized": True}),
    ],
)
def test_capture_schema_rejects_wrong_shapes_extras_and_versions(mutate) -> None:
    capture = _capture_document()
    mutate(capture)

    with pytest.raises(ValidationError):
        OpenXRTestRuntimeCapture.model_validate(capture, strict=True)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, float("nan")],
        [float("nan"), 0.0],
        [float("inf"), float("-inf")],
        [0.0, float("inf")],
    ],
)
def test_calculated_error_guard_rejects_nonfinite_results(values) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        finite_max_abs(values, "test error")


def test_calculated_error_guard_rejects_overflowed_error() -> None:
    overflowed_error = 1e308 - (-1e308)
    with pytest.raises(ValueError, match="non-finite"):
        finite_max_abs([overflowed_error], "test error")


def test_calculated_error_guard_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        finite_max_abs([], "test error")


def test_test_runtime_witness_replays_without_hardware_claims() -> None:
    pytest.importorskip("scipy")
    environment = os.environ.copy()
    source_paths = [str(ROOT), str(ROOT / "src")]
    environment["PYTHONPATH"] = os.pathsep.join(
        source_paths + list(filter(None, (environment.get("PYTHONPATH"),)))
    )
    result = subprocess.run(
        [sys.executable, str(VERIFIER_PATH), str(WITNESS_PATH)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = read_evidence_document_from_text(result.stdout)
    assert summary["result"] == "pass"
    assert summary["quatops_executed"] is False
    assert summary["reference"] == "SciPy Rotation"


def read_evidence_document_from_text(payload: str) -> dict:
    return parse_evidence_document(payload)
