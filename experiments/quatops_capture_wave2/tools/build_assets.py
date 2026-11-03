"""Build deterministic FORMAT inputs; these are not native golden outputs."""
from __future__ import annotations

import base64
import copy
import json
import math
from pathlib import Path
import struct

from shared.capture_io import ROOT, write_json

TIMES = [0.0, 0.75, 2.5]
Q = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], [0.5, 0.5, 0.5, 0.5]]
TAN_IN = [[0.0]*4, [0.06, -0.04, 0.03, 0.01], [-0.02, 0.07, -0.05, 0.02]]
TAN_OUT = [[0.12, -0.08, 0.05, 0.03], [-0.03, 0.05, 0.08, -0.02], [0.0]*4]


def glb(document: dict, binary: bytes) -> bytes:
    doc = copy.deepcopy(document)
    if binary:
        doc["buffers"][0].pop("uri", None)
    raw = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    raw += b" " * (-len(raw) % 4)
    body = struct.pack("<II", len(raw), 0x4E4F534A) + raw
    if binary:
        padded = binary + b"\0" * (-len(binary) % 4)
        body += struct.pack("<II", len(padded), 0x004E4942) + padded
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def save_asset(name: str, doc: dict, binary: bytes = b"") -> None:
    directory = ROOT / "assets/gltf"
    # An asymmetric tetrahedron gives real loaders observable geometry and basis probes.
    vertices = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]
    indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    positions = struct.pack("<12f", *vertices)
    triangles = struct.pack("<12H", *indices)
    binary += b"\0" * (-len(binary) % 4)
    offset = len(binary)
    views = doc.setdefault("bufferViews", [])
    pos_view = len(views)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(positions), "target": 34962})
    index_view = len(views)
    views.append({"buffer": 0, "byteOffset": offset + len(positions), "byteLength": len(triangles), "target": 34963})
    accessors = doc.setdefault("accessors", [])
    pos_accessor = len(accessors)
    accessors.append({"bufferView": pos_view, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0,0,0], "max": [1,2,3]})
    index_accessor = len(accessors)
    accessors.append({"bufferView": index_view, "componentType": 5123, "count": 12, "type": "SCALAR"})
    doc["meshes"] = [{"name": "AsymmetricProbe", "primitives": [{"attributes": {"POSITION": pos_accessor}, "indices": index_accessor}]}]
    for node in doc["nodes"][1:]:
        node["mesh"] = 0
    binary += positions + triangles
    doc["buffers"] = [{"byteLength": len(binary), "uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode()}]
    write_json(directory / f"{name}.gltf", doc)
    (directory / f"{name}.glb").write_bytes(glb(doc, binary))


def base_document() -> dict:
    return {"asset": {"version": "2.0", "generator": "QuatOps orientation capture inputs v2 (spec-derived, not native)"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Parent", "translation": [3.0, 5.0, -7.0],
                   "rotation": [0, 0, math.sqrt(0.5), math.sqrt(0.5)], "children": [1]},
                  {"name": "Probe", "translation": [1.25, -2.5, 3.75]}]}


def main() -> None:
    doc = base_document()
    root = math.sqrt(30)
    doc["nodes"][1]["rotation"] = [2/root, 3/root, 4/root, 1/root]
    # Independent rational matrix for the same rotation, in glTF column-major order.
    doc["nodes"].append({"name": "ProbeMatrix", "matrix": [
        -2/3, 2/3, 1/3, 0, 2/15, -1/3, 14/15, 0,
        11/15, 2/3, 2/15, 0, 1.25, -2.5, 3.75, 1]})
    doc["nodes"][0]["children"].append(2)
    save_asset("static_hierarchy", doc)
    variants = [("step", "STEP", "original"), ("linear", "LINEAR", "original"),
                ("cubic", "CUBICSPLINE", "original"),
                ("cubic_all_negated", "CUBICSPLINE", "all"),
                ("cubic_middle_negated", "CUBICSPLINE", "middle")]
    for name, interpolation, flip in variants:
        values: list[float] = []
        for i, quat in enumerate(Q):
            sign = -1 if flip == "all" or (flip == "middle" and i == 1) else 1
            vectors = [TAN_IN[i], quat, TAN_OUT[i]] if interpolation == "CUBICSPLINE" else [quat]
            values.extend(sign * x for vector in vectors for x in vector)
        time_bytes = struct.pack("<3f", *TIMES)
        data_bytes = struct.pack("<" + "f"*len(values), *values)
        binary = time_bytes + data_bytes
        doc = base_document()
        doc["buffers"] = [{"byteLength": len(binary), "uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode()}]
        doc["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(time_bytes)},
                              {"buffer": 0, "byteOffset": len(time_bytes), "byteLength": len(data_bytes)}]
        doc["accessors"] = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR", "min": [0], "max": [2.5]},
                            {"bufferView": 1, "componentType": 5126, "count": len(values)//4, "type": "VEC4"}]
        doc["animations"] = [{"name": name, "samplers": [{"input": 0, "output": 1, "interpolation": interpolation}],
                              "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]}]
        save_asset(name, doc, binary)
    # Text-authored USD is an input fixture. It is not USD-validated until Usd.Stage.Open succeeds.
    text = '''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
    timeCodesPerSecond = 1
    startTimeCode = 0
    endTimeCode = 2.5
)
def Xform "World" {
    double3 xformOp:translate = (3, 5, -7)
    double xformOp:rotateZ = 90
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]
    def Xform "Pose" {
        double3 xformOp:translate = (1.25, -2.5, 3.75)
        quatd xformOp:orient = (0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }
    def Xform "Reordered" {
        double3 xformOp:translate = (1.25, -2.5, 3.75)
        quatd xformOp:orient = (0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:translate"]
    }
    def Xform "Reset" {
        double3 xformOp:translate = (1.25, -2.5, 3.75)
        quatd xformOp:orient = (1, 0, 0, 0)
        uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate", "xformOp:orient"]
    }
    def Xform "Pivot" {
        double3 xformOp:translate:pivot = (2, 3, 4)
        double3 xformOp:rotateXYZ = (10, 20, 30)
        uniform token[] xformOpOrder = ["xformOp:translate:pivot", "xformOp:rotateXYZ", "!invert!xformOp:translate:pivot"]
    }
    def Xform "Animated" {
        quatd xformOp:orient.timeSamples = {
            0: (1, 0, 0, 0),
            0.75: (0.7071067811865476, 0, 0, 0.7071067811865476),
            2.5: (0.5, 0.5, 0.5, 0.5)
        }
        uniform token[] xformOpOrder = ["xformOp:orient"]
    }
}
'''
    (ROOT / "assets/usd/transform_stack_z.usda").write_text(text, encoding="utf-8")
    (ROOT / "assets/usd/transform_stack_y.usda").write_text(text.replace('upAxis = "Z"', 'upAxis = "Y"'), encoding="utf-8")
    write_json(ROOT / "assets/sampling.json", {
        "times_seconds": [0, 0.1875, 0.375, 0.749999, 0.75, 0.750001, 1.1875, 1.625, 2.499999, 2.5],
        "mode": "sample absolute times with looping disabled", "gltf_input_component_type": "FLOAT (binary32)",
        "kind": "spec_contract_inputs_not_native_goldens"})
    write_json(ROOT / "corpus/cesium_geographic_inputs.json", {
        "schema_version": "1.0", "ellipsoid": "WGS84", "kind": "inputs_not_expected_outputs",
        "locations": [{"case_id": f"geo-{i:02d}", "longitude_degrees": lon, "latitude_degrees": lat, "height_m": h}
            for i, (lon, lat, h) in enumerate([(0, 0, 0), (90, 0, 0), (180, 0, 0), (-75, 40, 1000),
                (45, -45, 400000), (179.999999, 35, 0), (-179.999999, 35, 0),
                (0, 89.999999, 0), (0, -89.999999, 0), (0, 90, 0), (0, -90, 0)])],
        "heading_pitch_roll_radians": [[0, 0, 0], [0.3, -0.4, 0.7], [-1.2, 0.8, -0.5]],
        "local_point_xyz_m": [2, -3, 5], "local_direction_xyz": [0.3, -0.2, 0.7],
        "exact_pole_note": "Native fallback branch, not universal east/north directions at the poles"})


if __name__ == "__main__":
    main()
