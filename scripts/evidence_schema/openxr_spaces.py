"""Typed provenance and numeric validation for supplemental OpenXR spaces.

This is static mock-runtime evidence, not a hardware/adapter acceptance path.
The legacy LOCAL semantic contract and validation path remain independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.openxr import OpenXRTestRuntimeCapture, finite_max_abs
from scripts.evidence_schema.primitives import NativeCaptureModel
from scripts.evidence_schema.utilities import sha256_file

Space = Literal["VIEW", "LOCAL_FLOOR"]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SpaceWitnessReference(NativeCaptureModel):
    file: str
    sha256: Digest


class SpaceCaptureProvenance(NativeCaptureModel):
    schema_version: Literal["1.0"]
    scope: Literal[
        "same-reference-type static offset chains; no tracking, floor-height or physical alignment evidence"
    ]
    runtime_kind: Literal[
        "Khronos test runtime; direct DLL negotiation, no system registration"
    ]
    hardware_runtime: Literal[False]
    quatops_executed: Literal[False]
    api_version: list[int]
    captured_utc: str
    sdk_commit: Literal["24a1f2b92d212501a2b1b8d30ffd53abf55fcfa8"]
    runtime_commit: Literal["a9f14c416e383d05c142f5a2539a91fe3931b52a"]
    capture_source: Literal["capture_test_runtime.cpp"]
    capture_source_sha256: Digest
    runtime_source_sha256: Literal[
        "1ab9f9e7ae170c7a1896ae51c6535a60f9eae5c8d2eef52bddd4ae5e369c50be"
    ]
    runtime_binary_sha256: Digest
    capture_binary_sha256: Digest
    compiler: str
    observations: dict[Space, SpaceWitnessReference]
    local_legacy_output_reproduced: Literal[True]
    platform: str
    architecture: str
    build_commands: list[list[str]]
    capture_commands: list[list[str]]
    synthetic_xr_time: Literal[1]


def validate_space_capture(
    root: Path, witness_path: Path
) -> tuple[str, Space, set[str]]:
    """Return only the version/type/relation IDs established by this capture.

    Callers must retain adapter_conformance_checked=False and the static-offset
    limitation. A same-type VIEW chain cannot establish viewer tracking, and a
    LOCAL_FLOOR chain cannot establish floor alignment or cross-type relations.
    """
    package = root.resolve() / "experiments/oracles/openxr/1.1.65"
    provenance = SpaceCaptureProvenance.model_validate(
        read_evidence_document(package / "reference_spaces_manifest.json"), strict=True
    )
    if provenance.api_version != [1, 1, 65] or set(provenance.observations) != {
        "VIEW",
        "LOCAL_FLOOR",
    }:
        raise ValueError("OpenXR supplemental capture identities differ")
    if (
        sha256_file(package / provenance.capture_source)
        != provenance.capture_source_sha256
    ):
        raise ValueError("OpenXR supplemental producer digest differs")
    chosen: Space | None = None
    for space, entry in provenance.observations.items():
        expected = package / "witnesses" / f"test_runtime_{space.lower()}.json"
        if (package / entry.file).resolve() != expected.resolve():
            raise ValueError("OpenXR supplemental witness path differs")
        if expected.resolve() == witness_path.resolve():
            chosen = space
            if sha256_file(expected) != entry.sha256:
                raise ValueError("OpenXR supplemental witness digest differs")
    if chosen is None:
        raise ValueError("Unknown OpenXR supplemental witness")
    capture = OpenXRTestRuntimeCapture.model_validate(
        read_evidence_document(witness_path), strict=True
    )
    if (
        capture.reference_space_type != chosen
        or capture.api_version != provenance.api_version
        or capture.locate_time_xr_time != provenance.synthetic_xr_time
    ):
        raise ValueError("OpenXR supplemental capture differs from provenance")
    if set(capture.inputs_pose_in_reference_space) != {"A", "B", "C"} or set(
        capture.observations
    ) != {"A_from_B", "B_from_C", "A_from_C"}:
        raise ValueError("OpenXR supplemental relation set differs")
    # Pin the asymmetric float32 fixture in capture_test_runtime.cpp, not merely
    # self-consistency of attacker-replaced input/output observations.
    expected_inputs: dict[
        Literal["A", "B", "C"],
        tuple[tuple[float, float, float, float], tuple[float, float, float]],
    ] = {
        "A": ((0.5, 0.5, 0.0, 2**-0.5), (1.25, -2.0, 0.5)),
        "B": (
            (0.0, 0.3535533905932737622, 0.3535533905932737622, 0.8660254037844386468),
            (-0.75, 1.5, 2.25),
        ),
        "C": (
            (
                -0.2705980500730984922,
                0.0,
                -0.2705980500730984922,
                0.9238795325112867561,
            ),
            (3.0, 0.25, -1.0),
        ),
    }
    for name, (quaternion, position) in expected_inputs.items():
        pose = capture.inputs_pose_in_reference_space[name]
        if not np.allclose(
            pose.orientation_xyzw,
            np.asarray(quaternion, dtype=np.float32),
            atol=1e-9,
            rtol=0,
        ) or not np.array_equal(pose.position_xyz_m, position):
            raise ValueError("OpenXR supplemental producer fixture differs")
    ra = Rotation.from_quat(
        capture.inputs_pose_in_reference_space["A"].orientation_xyzw
    )
    rb = Rotation.from_quat(
        capture.inputs_pose_in_reference_space["B"].orientation_xyzw
    )
    if float(np.max(np.abs((ra * rb).as_matrix() - (rb * ra).as_matrix()))) < 0.1:
        raise ValueError("OpenXR supplemental fixture must be noncommuting")
    for key, observed in capture.observations.items():
        if key != f"{observed.base}_from_{observed.target}":
            raise ValueError("OpenXR supplemental relation direction differs")
        base = capture.inputs_pose_in_reference_space[observed.base]
        target = capture.inputs_pose_in_reference_space[observed.target]
        for pose in (base, target, observed.pose):
            if abs(float(np.linalg.norm(pose.orientation_xyzw)) - 1.0) > 1e-6:
                raise ValueError("OpenXR supplemental quaternion is not unit length")
        inverse_base = Rotation.from_quat(base.orientation_xyzw).inv()
        expected_rotation = inverse_base * Rotation.from_quat(target.orientation_xyzw)
        expected_translation = inverse_base.apply(
            np.asarray(target.position_xyz_m) - base.position_xyz_m
        )
        actual_rotation = Rotation.from_quat(observed.pose.orientation_xyzw)
        rotation_error = finite_max_abs(
            (expected_rotation.as_matrix() - actual_rotation.as_matrix()).flat,
            "OpenXR rotation",
        )
        translation_error = finite_max_abs(
            (expected_translation - observed.pose.position_xyz_m).flat,
            "OpenXR translation",
        )
        if max(rotation_error, translation_error) > 2e-6:
            raise ValueError("OpenXR supplemental numeric action differs")
    return "1.1.65", chosen, set(capture.observations)
