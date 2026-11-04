"""Create a minimal USDA hierarchy with QuatOps-converted orientations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

CANONICAL_INPUTS = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
HERE = Path(__file__).parent
INPUTS_OUT = HERE / "converter_inputs.json"
STAGE_OUT = HERE / "quatops_hierarchy.usda"


def number(value: float) -> str:
    return format(float(value), ".17g")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from quatops.engines import QuaternionFrameMap, SignedAxisMap
    from scripts.evidence_format import dump_evidence_json

    corpus = json.loads(CANONICAL_INPUTS.read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row["canonical_wxyz"] for row in corpus["orientations"]}
    # Explicit right-handed USD test frame: X right, Y up, -Z forward.
    frame = SignedAxisMap(("x", "z", "-y"))
    convert = QuaternionFrameMap(frame, frame)
    specs = (
        ("parent", "quarter_z", (3.0, -2.0, 1.0)),
        ("child", "mixed_a", (1.25, -2.5, 3.75)),
        ("reset_child", "quarter_x", (-0.75, 1.5, 2.25)),
    )
    rows = []
    for node, case_id, translation in specs:
        source_q = by_id[case_id]
        target_q = convert.to_destination_wxyz(source_q)
        rows.append(
            {
                "node": node,
                "canonical_case_id": case_id,
                "canonical_wxyz": list(source_q),
                "usd_wxyz": list(target_q),
                "canonical_translation_xyz": list(translation),
                "usd_translation_xyz": list(frame.vector(translation)),
            }
        )
    payload = {
        "schema_version": "1.0",
        "kind": "quatops_generated_openusd_converter_input",
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
            "usd_stage": {
                "handedness": "right",
                "upAxis": "Y",
                "positive_x": "right",
                "positive_y": "up",
                "positive_z": "backward",
                "forward": "-Z",
            },
            "destination_axes_from_source": list(frame.destination_axes),
            "quaternion": "Hamilton active; USDA quatd real-first WXYZ",
            "matrix_action": "OpenUSD Gf/UsdGeom row vectors, captured and compared as row-major arrays",
        },
        "probe_vectors_xyz": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            corpus["probe_vector_xyz"],
        ],
        "usd_probe_vectors_xyz": [
            list(frame.vector(v))
            for v in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                corpus["probe_vector_xyz"],
            )
        ],
        "nodes": rows,
        "stack_contract": {
            "parent_ops": ["xformOp:translate", "xformOp:orient"],
            "child_ops": ["xformOp:translate", "xformOp:orient"],
            "reset_ops": ["!resetXformStack!", "xformOp:translate", "xformOp:orient"],
        },
    }
    INPUTS_OUT.write_text(dump_evidence_json(payload), encoding="utf-8", newline="\n")

    def orient(row):
        return ", ".join(number(v) for v in row["usd_wxyz"])

    def translate(row):
        return ", ".join(number(v) for v in row["usd_translation_xyz"])

    parent, child, reset = rows
    stage = f"""#usda 1.0
(
    defaultPrim = "Scene"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Scene"
{{
    def Xform "Parent"
    {{
        quatd xformOp:orient = ({orient(parent)})
        double3 xformOp:translate = ({translate(parent)})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

        def Xform "Child"
        {{
            quatd xformOp:orient = ({orient(child)})
            double3 xformOp:translate = ({translate(child)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}

        def Xform "ResetChild"
        {{
            quatd xformOp:orient = ({orient(reset)})
            double3 xformOp:translate = ({translate(reset)})
            uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate", "xformOp:orient"]
        }}
    }}
}}
"""
    STAGE_OUT.write_text(stage, encoding="utf-8", newline="\n")
    print(f"wrote QuatOps converter stage: {STAGE_OUT}")


if __name__ == "__main__":
    main()
