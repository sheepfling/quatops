"""Capture static glTF/GLB scene graphs from the installed Trimesh consumer."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import trimesh

from shared.capture_io import ROOT, Capture, Observation, contract, numeric, parser, sha256


def header(path: Path) -> dict:
    if path.suffix == ".gltf":
        return json.loads(path.read_text(encoding="utf-8"))
    data = path.read_bytes()
    size, chunk = struct.unpack_from("<II", data, 12)
    if chunk != 0x4E4F534A:
        raise ValueError("First GLB chunk must be JSON")
    return json.loads(data[20:20+size])


def main() -> None:
    cli = parser(__doc__ or "Trimesh glTF")
    cli.add_argument("--assets", type=Path, default=ROOT / "assets/gltf")
    args = cli.parse_args()
    cap = Capture("gltf_trimesh", "Trimesh glTF loader", trimesh.__version__, args.corpus, args.output,
        Path(__file__), {"input": "JSON binary64 values and buffer binary32", "arithmetic": "NumPy/native library implementation",
        "output": "NumPy array dtype per field", "serialized": "JSON plus explicit IEEE bits"},
        oracle_family="trimesh", dependencies=[{"name": "numpy", "version": np.__version__},
            {"name": "trimesh.exchange.gltf source", "sha256": sha256(Path(trimesh.exchange.gltf.__file__))}],
        limitations=["Static pose only. No Trimesh animation sampling claim.",
            "Trimesh bundled JSON-schema validation is not execution of Khronos glTF-Validator.",
            "No QuatOps adapter replay; interpreted scene graph is raw consumer evidence."])
    for path in sorted(args.assets.iterdir()):
        if path.suffix not in {".gltf", ".glb"}:
            continue
        doc = header(path)
        trimesh.exchange.gltf.validate(doc)
        scene = trimesh.load_scene(path, process=False)
        nodes = []
        for name in sorted(scene.graph.nodes):
            matrix, geometry = scene.graph[name]
            nodes.append({"node_name": name, "geometry_name": geometry,
                "world_matrix_rows": numeric(np.asarray(matrix).tolist()),
                "world_matrix_numpy_dtype": str(np.asarray(matrix).dtype)})
        cap.add(Observation(path.name, "gltf-static-load", "trimesh.load_scene; SceneGraph.__getitem__; trimesh.exchange.gltf.validate",
            contract(units="metres; quaternion dimensionless", order="asset xyzw", frame="node-local to scene/world"),
            {"asset": path.name, "sha256": sha256(path), "process": False},
            {"nodes": nodes, "geometry_count": len(scene.geometry), "bundled_json_schema": "passed",
             "animation_count_in_asset": len(doc.get("animations", [])), "animation_sampled": False}))
        cap.artifacts.append((path.resolve(), "input_asset"))
    print(cap.finish(["FORMAT", "MATRIX", "FRAME"]))


if __name__ == "__main__":
    main()
