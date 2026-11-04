"""Capture the selected Blender quaternion-converter witnesses."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

import bpy
from mathutils import Quaternion, Vector

ORIENTATIONS = (
    ("identity", (1.0, 0.0, 0.0, 0.0)),
    ("quarter_x", (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)),
    ("quarter_y", (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)),
    ("quarter_z", (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)),
    ("mixed_a", (0.8, 0.2, -0.4, 0.4)),
    ("mixed_b", (0.7, -0.3, 0.2, 0.6164414002968976)),
    ("antipode_mixed_a", (-0.8, -0.2, 0.4, -0.4)),
    (
        "nonprincipal_halfturn",
        (0.0, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    ),
)
PROBE = Vector((0.25, -1.5, 2.75))
AXES = (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))
DESTINATION_AXES = ("x", "y", "z")
TARGET_VERSION = "5.1.2"


class OrientationCase(TypedDict):
    case_id: str
    canonical_wxyz: list[float]
    native_quaternion: list[float]
    native_rotated_positive_basis_vectors_xyz: list[list[float]]
    native_rotated_probe_vector_xyz: list[float]


def vector_values(value: Vector) -> list[float]:
    return [float(component) for component in value]


def rotated_axes(quaternion: Quaternion) -> list[list[float]]:
    return [vector_values(quaternion @ axis) for axis in AXES]


def capture_canonical_interop() -> dict[str, object]:
    native = {case_id: Quaternion(wxyz) for case_id, wxyz in ORIENTATIONS}
    orientation_cases: list[OrientationCase] = [
        {
            "case_id": case_id,
            "canonical_wxyz": list(wxyz),
            "native_quaternion": vector_values(native[case_id]),
            "native_rotated_positive_basis_vectors_xyz": rotated_axes(native[case_id]),
            "native_rotated_probe_vector_xyz": vector_values(native[case_id] @ PROBE),
        }
        for case_id, wxyz in ORIENTATIONS
    ]

    def composition(left_id: str, right_id: str) -> dict[str, object]:
        product = native[left_id] @ native[right_id]
        return {
            "left_case_id": left_id,
            "right_case_id": right_id,
            "product_order": "lhs * rhs; rhs acts first",
            "native_product_quaternion": vector_values(product),
            "native_rotated_positive_basis_vectors_xyz": rotated_axes(product),
            "native_rotated_probe_vector_xyz": vector_values(product @ PROBE),
        }

    return {
        "contract_id": "quatops-canonical-hamilton-wxyz-active-rh-v1",
        "native_quaternion_order": "wxyz",
        "frame_map_destination_axes_from_source": list(DESTINATION_AXES),
        "probe_vector_canonical_xyz": vector_values(PROBE),
        "probe_vector_native_xyz": vector_values(PROBE),
        "orientation_cases": orientation_cases,
        "composition": composition("mixed_a", "mixed_b"),
        "reverse_composition": composition("mixed_b", "mixed_a"),
    }


def build_fixture() -> dict[str, object]:
    version = bpy.app.version_string
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii")
    return {
        "schema_version": 1,
        "evidence_kind": "native_blender_runtime_capture",
        "capture_method": "Blender background mode with factory startup and embedded Python script",
        "engine": {
            "name": "Blender",
            "version": version,
            "major": int(bpy.app.version[0]),
            "minor": int(bpy.app.version[1]),
            "patch": int(bpy.app.version[2]),
            "build_hash": build_hash,
            "build_branch": str(
                bpy.app.build_branch.decode("ascii")
                if isinstance(bpy.app.build_branch, bytes)
                else bpy.app.build_branch
            ),
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        },
        "source_frame": {
            "handedness": "right-handed",
            "positive_x": "right",
            "positive_y": "forward",
            "positive_z": "up",
            "declaration": "explicit probe frame; not inferred from Blender or an imported object's local axes",
        },
        "blender_world_frame_probe": {
            "positive_x_image": [1.0, 0.0, 0.0],
            "positive_y_image": [0.0, 1.0, 0.0],
            "positive_z_image": [0.0, 0.0, 1.0],
            "declared_probe_semantics": "+X right, +Y forward, +Z up for this fixture",
            "semantic_imported_object_forward_axis": "not assumed",
        },
        "source_to_blender_axis_map": {
            "destination_axes_from_source": list(DESTINATION_AXES),
            "determinant": 1,
        },
        "canonical_interop": capture_canonical_interop(),
    }


def parse_output_path() -> Path:
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(script_args)
    version = bpy.app.version_string
    if TARGET_VERSION != version:
        raise RuntimeError(
            f"capture targets Blender {TARGET_VERSION!r}, but runtime is {version!r}"
        )
    return args.output or Path(tempfile.gettempdir()) / "quatops-blender-witnesses.json"


def main() -> None:
    output = parse_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_fixture()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Wrote native Blender quaternion witnesses:", output)
    print("Blender:", payload["engine"]["version"], payload["engine"]["build_hash"])
    print("Orientation cases:", len(payload["canonical_interop"]["orientation_cases"]))


if __name__ == "__main__":
    main()
