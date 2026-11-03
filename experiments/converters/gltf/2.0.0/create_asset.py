"""Convert a canonical hierarchical orientation into a minimal glTF asset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from quatops.engines import QuaternionFrameMap, SignedAxisMap

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CANONICAL_INPUTS = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
HERE = Path(__file__).parent
INPUTS_OUT = HERE / "converter_inputs.json"
ASSET_OUT = HERE / "quatops_hierarchy.gltf"


def main() -> None:
    from scripts.evidence_json import read_evidence_document

    source = read_evidence_document(CANONICAL_INPUTS)
    values = {row["case_id"]: row["canonical_wxyz"] for row in source["orientations"]}
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
            CANONICAL_INPUTS.read_bytes()
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
            source["probe_vector_xyz"],
        ],
        "gltf_probe_vectors_xyz": [
            list(frame.vector(v))
            for v in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                source["probe_vector_xyz"],
            )
        ],
        "nodes": rows,
    }
    INPUTS_OUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
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
    ASSET_OUT.write_text(
        json.dumps(asset, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote QuatOps inputs and glTF asset: {ASSET_OUT}")


if __name__ == "__main__":
    main()
