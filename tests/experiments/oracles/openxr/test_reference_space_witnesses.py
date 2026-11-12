"""Replay explicitly scoped native VIEW and LOCAL_FLOOR static offsets."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.openxr import OpenXRTestRuntimeCapture
from scripts.evidence_schema.openxr_spaces import validate_space_capture
from tests.support.paths import REPOSITORY_ROOT

PACKAGE = REPOSITORY_ROOT / "experiments/oracles/openxr/1.1.65"


def _document(space: str) -> dict:
    manifest = read_evidence_document(PACKAGE / "reference_spaces_manifest.json")
    entry = manifest["observations"][space]
    path = PACKAGE / entry["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    return read_evidence_document(path)


def test_reference_space_capture_identity() -> None:
    manifest = read_evidence_document(PACKAGE / "reference_spaces_manifest.json")
    pins = read_evidence_document(PACKAGE / "source.json")
    assert manifest["sdk_commit"] == pins["commit"]
    assert manifest["runtime_commit"] == pins["test_runtime"]["commit"]
    assert manifest["hardware_runtime"] is False
    assert manifest["quatops_executed"] is False
    assert manifest["local_legacy_output_reproduced"] is True
    assert set(manifest["observations"]) == {"VIEW", "LOCAL_FLOOR"}
    assert (
        hashlib.sha256((PACKAGE / manifest["capture_source"]).read_bytes()).hexdigest()
        == manifest["capture_source_sha256"]
    )


@pytest.mark.parametrize("space", ["VIEW", "LOCAL_FLOOR"])
def test_reference_space_static_offsets_match_independent_formula(space: str) -> None:
    capture = OpenXRTestRuntimeCapture.model_validate(_document(space), strict=True)
    assert capture.reference_space_type == space
    assert capture.api_version == [1, 1, 65]
    assert capture.locate_time_xr_time == 1
    assert set(capture.inputs_pose_in_reference_space) == {"A", "B", "C"}
    assert set(capture.observations) == {"A_from_B", "B_from_C", "A_from_C"}
    for name, observed in capture.observations.items():
        assert name == f"{observed.base}_from_{observed.target}"
        base = capture.inputs_pose_in_reference_space[observed.base]
        target = capture.inputs_pose_in_reference_space[observed.target]
        inverse_base = Rotation.from_quat(base.orientation_xyzw).inv()
        expected_rotation = inverse_base * Rotation.from_quat(target.orientation_xyzw)
        expected_translation = inverse_base.apply(
            np.array(target.position_xyz_m) - base.position_xyz_m
        )
        actual = Rotation.from_quat(observed.pose.orientation_xyzw)
        np.testing.assert_allclose(
            actual.as_matrix(), expected_rotation.as_matrix(), atol=2e-6, rtol=0
        )
        np.testing.assert_allclose(
            observed.pose.position_xyz_m, expected_translation, atol=2e-6, rtol=0
        )
        assert observed.location_flags == 15
    ab, bc, ac = (
        capture.observations[key].pose for key in ("A_from_B", "B_from_C", "A_from_C")
    )
    rab, rbc, rac = (Rotation.from_quat(pose.orientation_xyzw) for pose in (ab, bc, ac))
    probe = np.array([0.5, -1.25, 2.0])
    chained = rab.apply(rbc.apply(probe) + bc.position_xyz_m) + ab.position_xyz_m
    direct = rac.apply(probe) + ac.position_xyz_m
    np.testing.assert_allclose(chained, direct, atol=2e-6, rtol=0)


@pytest.mark.parametrize("invalid", ["STAGE", "universal-world", "", 1])
def test_only_explicit_captured_space_types_are_accepted(invalid: object) -> None:
    document = deepcopy(_document("VIEW"))
    document["reference_space_type"] = invalid
    with pytest.raises(ValidationError):
        OpenXRTestRuntimeCapture.model_validate(document, strict=True)


def test_test_runtime_claim_cannot_be_relabelled_as_hardware() -> None:
    document = deepcopy(_document("LOCAL_FLOOR"))
    document["runtime"]["hardware_runtime"] = True
    with pytest.raises(ValidationError):
        OpenXRTestRuntimeCapture.model_validate(document, strict=True)


@pytest.mark.parametrize("space", ["VIEW", "LOCAL_FLOOR"])
def test_typed_supplemental_coverage(space: str) -> None:
    version, actual, witnesses = validate_space_capture(
        REPOSITORY_ROOT, PACKAGE / "witnesses" / f"test_runtime_{space.lower()}.json"
    )
    assert (version, actual) == ("1.1.65", space)
    assert witnesses == {"A_from_B", "B_from_C", "A_from_C"}


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "space",
        "direction",
        "numeric",
        "missing",
        "producer",
        "hardware",
        "identity",
    ],
)
def test_supplemental_coverage_rejects_misattribution_and_bad_actions(
    tmp_path: Path, mutation: str
) -> None:
    package = tmp_path / "experiments/oracles/openxr/1.1.65"
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
    manifest_path = package / "reference_spaces_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = package / manifest["observations"]["VIEW"]["file"]
    value = json.loads(path.read_text())
    if mutation == "version":
        value["api_version"] = [1, 1, 64]
    elif mutation == "space":
        value["reference_space_type"] = "LOCAL_FLOOR"
    elif mutation == "direction":
        value["observations"]["A_from_B"]["base"] = "C"
    elif mutation == "numeric":
        value["observations"]["A_from_B"]["pose"]["position_xyz_m"][0] += 1
    elif mutation == "missing":
        del value["observations"]["A_from_C"]
    elif mutation == "producer":
        manifest["capture_source_sha256"] = "0" * 64
    elif mutation == "identity":
        for pose in list(value["inputs_pose_in_reference_space"].values()) + [
            row["pose"] for row in value["observations"].values()
        ]:
            pose["orientation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
            pose["position_xyz_m"] = [0.0, 0.0, 0.0]
    else:
        value["runtime"]["hardware_runtime"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest["observations"]["VIEW"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_space_capture(tmp_path, path)
