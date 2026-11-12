"""Convert a canonical hierarchical orientation into a minimal glTF asset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quatops.experimental.engines import QuaternionFrameMap, SignedAxisMap
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import ThreeGltfInputs
from scripts.evidence_schema.shared_inputs import SharedCanonicalInputCapture


def build_artifacts(repository_root: Path) -> tuple[ThreeGltfInputs, dict]:
    """Build the captured converter input and static hierarchy from shared cases."""
    canonical_inputs = (
        repository_root / "experiments/interoperability/v1/canonical_inputs.json"
    )
    source = SharedCanonicalInputCapture.model_validate(
        read_evidence_document(canonical_inputs), strict=True
    )
    values = {row.case_id: row.canonical_wxyz for row in source.orientations}
    # glTF test-frame declaration: +X right, +Y up, -Z forward. This is the
    # explicit mapping used by this artifact, not a guessed imported-model axis.
    frame = SignedAxisMap(("x", "z", "-y"))
    orientation_map = QuaternionFrameMap(frame, frame)
    parent_t = (3.0, -2.0, 1.0)
    child_t = (1.25, -2.5, 3.75)
    rows = []
    for name, source_q, source_t in (
        ("parent", values["quarter_z"], parent_t),
        ("child", values["mixed_a"], child_t),
    ):
        dest_wxyz = orientation_map.to_destination_wxyz(source_q)
        rows.append(
            {
                "node": name,
                "canonical_wxyz": list(source_q),
                "gltf_xyzw": [dest_wxyz[1], dest_wxyz[2], dest_wxyz[3], dest_wxyz[0]],
                "canonical_translation_xyz": list(source_t),
                "gltf_translation_xyz": list(frame.vector(source_t)),
            }
        )
    payload = {
        "schema_version": "1.0",
        "kind": "quatops_generated_gltf_converter_input",
        "canonical_input_sha256": hashlib.sha256(
            canonical_inputs.read_bytes()
        ).hexdigest(),
        "frame_contract": {
            "canonical": {
                "handedness": "right",
                "positive_x": "right",
                "positive_y": "forward",
                "positive_z": "up",
            },
            "gltf_probe_frame": {
                "handedness": "right",
                "positive_x": "right",
                "positive_y": "up",
                "positive_z": "backward",
                "forward": "-Z",
            },
            "destination_axes_from_canonical": list(frame.destination_axes),
            "quaternion": "Hamilton active; glTF rotation property stores XYZW",
            "translation": "gltf_translation = B * canonical_translation",
            "scope": "static parent/child TRS artifact; no model import-axis assumption",
        },
        "probe_vectors_xyz": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            source.probe_vector_xyz,
        ],
        "gltf_probe_vectors_xyz": [
            list(frame.vector(v))
            for v in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                source.probe_vector_xyz,
            )
        ],
        "nodes": rows,
    }
    typed_payload = ThreeGltfInputs.model_validate(payload, strict=True)
    asset = {
        "asset": {
            "version": "2.0",
            "generator": "QuatOps explicit-frame converter witness",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "QuatOpsParent",
                "translation": rows[0]["gltf_translation_xyz"],
                "rotation": rows[0]["gltf_xyzw"],
                "children": [1],
            },
            {
                "name": "QuatOpsChild",
                "translation": rows[1]["gltf_translation_xyz"],
                "rotation": rows[1]["gltf_xyzw"],
            },
        ],
    }
    return typed_payload, asset


def serialize_artifact(value: object) -> bytes:
    """Match the JSON bytes used for the native loader capture."""
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")
