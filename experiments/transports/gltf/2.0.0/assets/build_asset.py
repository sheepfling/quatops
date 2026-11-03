"""Build the deterministic glTF 2.0 rotation transport input using stdlib."""

import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).parent


def axis_quaternion(axis, angle):
    length = math.sqrt(sum(component * component for component in axis))
    scale = math.sin(angle / 2.0) / length
    return [*(component * scale for component in axis), math.cos(angle / 2.0)]


def append_f32(blob, values):
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(struct.pack(f"<{len(values)}f", *values))
    return offset, len(values) * 4


def main():
    times = [0.0, 0.7, 1.9]
    rotations = [
        axis_quaternion((1.0, 0.0, 0.0), 0.35),
        axis_quaternion((0.0, 1.0, 0.0), 1.2),
        axis_quaternion((0.0, 0.0, 1.0), 2.2),
    ]
    cubic_tangents = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.28, -0.03],
        [0.025, -0.04, 0.12, -0.06],
        [-0.07, 0.03, 0.02, -0.09],
        [-0.08, 0.02, 0.035, -0.17],
        [0.0, 0.0, 0.0, 0.0],
    ]
    cubic_values = []
    for index, value in enumerate(rotations):
        cubic_values.extend(
            (cubic_tangents[index * 2], value, cubic_tangents[index * 2 + 1])
        )

    binary = bytearray()
    buffer_views = []
    accessors = []

    def accessor(
        values, count, type_name, min_values=None, max_values=None, target=None
    ):
        flat = (
            [component for element in values for component in element]
            if type_name != "SCALAR"
            else values
        )
        byte_offset, byte_length = append_f32(binary, flat)
        view_index = len(buffer_views)
        view = {"buffer": 0, "byteOffset": byte_offset, "byteLength": byte_length}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        item = {
            "bufferView": view_index,
            "componentType": 5126,
            "count": count,
            "type": type_name,
        }
        if min_values is not None:
            item["min"] = min_values
        if max_values is not None:
            item["max"] = max_values
        accessors.append(item)
        return len(accessors) - 1

    time_accessor = accessor(times, len(times), "SCALAR", [min(times)], [max(times)])
    step_accessor = accessor(rotations, len(rotations), "VEC4")
    linear_accessor = accessor(rotations, len(rotations), "VEC4")
    cubic_accessor = accessor(cubic_values, len(cubic_values), "VEC4")
    position_accessor = accessor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        3,
        "VEC3",
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        target=34962,
    )

    matrix_angle = math.pi / 3.0
    cosine = math.cos(matrix_angle)
    sine = math.sin(matrix_angle)
    matrix_node = [
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        cosine,
        sine,
        0.0,
        0.0,
        -0.5 * sine,
        0.5 * cosine,
        0.0,
        2.0,
        1.0,
        -1.0,
        1.0,
    ]
    parent_rotation = axis_quaternion((0.0, 0.0, 1.0), 0.4)

    asset = {
        "asset": {"version": "2.0", "generator": "QuatOps stdlib fixture builder"},
        "scene": 0,
        "scenes": [{"nodes": [0, 6, 7, 8, 9]}],
        "nodes": [
            {
                "name": "parent_frame",
                "translation": [1.25, -2.5, 0.75],
                "rotation": parent_rotation,
                "children": [1, 2, 3, 4, 5],
            },
            {"name": "animated_step", "mesh": 0, "rotation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "animated_linear", "mesh": 0, "rotation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "animated_cubic", "mesh": 0, "rotation": [0.0, 0.0, 0.0, 1.0]},
            {
                "name": "static_trs",
                "mesh": 0,
                "translation": [2.0, 1.0, -1.0],
                "rotation": axis_quaternion((1.0, 0.0, 0.0), math.pi / 3.0),
                "scale": [2.0, 1.0, 0.5],
            },
            {"name": "static_matrix", "mesh": 0, "matrix": matrix_node},
            {"name": "source_axis_origin", "mesh": 0, "translation": [0.0, 0.0, 0.0]},
            {"name": "source_axis_x", "mesh": 0, "translation": [1.0, 0.0, 0.0]},
            {"name": "source_axis_y", "mesh": 0, "translation": [0.0, 1.0, 0.0]},
            {"name": "source_axis_z", "mesh": 0, "translation": [0.0, 0.0, 1.0]},
        ],
        "meshes": [
            {
                "name": "proof_triangle",
                "primitives": [{"attributes": {"POSITION": position_accessor}}],
            }
        ],
        "animations": [
            {
                "name": "rotation_step",
                "samplers": [
                    {
                        "input": time_accessor,
                        "output": step_accessor,
                        "interpolation": "STEP",
                    }
                ],
                "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}],
            },
            {
                "name": "rotation_linear",
                "samplers": [
                    {
                        "input": time_accessor,
                        "output": linear_accessor,
                        "interpolation": "LINEAR",
                    }
                ],
                "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}],
            },
            {
                "name": "rotation_cubic",
                "samplers": [
                    {
                        "input": time_accessor,
                        "output": cubic_accessor,
                        "interpolation": "CUBICSPLINE",
                    }
                ],
                "channels": [{"sampler": 0, "target": {"node": 3, "path": "rotation"}}],
            },
        ],
        "buffers": [{"uri": "rotation_transport.bin", "byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    (HERE / "rotation_transport.bin").write_bytes(binary)
    (HERE / "rotation_transport.gltf").write_text(
        json.dumps(asset, indent=2, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
