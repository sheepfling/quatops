"""Open the QuatOps-authored USDA file and capture native OpenUSD evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


HERE = Path(__file__).parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(value):
    return [float(x) for x in value]


def matrix_rows(matrix):
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def q_wxyz(quaternion):
    return [float(quaternion.GetReal()), *[float(x) for x in quaternion.GetImaginary()]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=HERE / "quatops_hierarchy.usda")
    parser.add_argument("--inputs", type=Path, default=HERE / "converter_inputs.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_path = args.stage.resolve()
    inputs_path = args.inputs.resolve()
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open {stage_path}")
    cache = UsdGeom.XformCache()
    probe_vectors = [Gf.Vec3d(*row) for row in inputs["usd_probe_vectors_xyz"]]
    observations = []
    for node, prim_path in (("parent", "/Scene/Parent"), ("child", "/Scene/Parent/Child"), ("reset_child", "/Scene/Parent/ResetChild")):
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        local = xformable.GetLocalTransformation()
        world = cache.GetLocalToWorldTransform(prim)
        ops = xformable.GetOrderedXformOps()
        orient = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
        qnative = orient.Get()
        inverse = world.GetInverse()
        observations.append({
            "node": node,
            "prim_path": prim_path,
            "native_entry_points": ["Usd.Stage.Open", "UsdGeom.Xformable.GetOrderedXformOps", "GetLocalTransformation", "UsdGeom.XformCache.GetLocalToWorldTransform", "Gf.Matrix4d.Transform", "Gf.Matrix4d.TransformDir"],
            "outputs": {
                "orient_op_wxyz": q_wxyz(qnative),
                "local_matrix_rows": matrix_rows(local),
                "world_matrix_rows": matrix_rows(world),
                "ordered_xform_ops": [op.GetOpName() for op in ops],
                "resets_xform_stack": bool(xformable.GetResetXformStack()),
                "local_points_xyz": [vec(local.Transform(p)) for p in probe_vectors],
                "world_points_xyz": [vec(world.Transform(p)) for p in probe_vectors],
                "world_directions_xyz": [vec(world.TransformDir(p)) for p in probe_vectors],
                "world_to_local_point_roundtrip_xyz": [vec(inverse.Transform(world.Transform(p))) for p in probe_vectors],
            },
        })
    package_dir = Path(__import__("pxr").__file__).parent
    binaries = {}
    for relative in ("usd_ms.dll", "Gf/_gf.pyd", "Usd/_usd.pyd", "UsdGeom/_usdGeom.pyd"):
        binary = package_dir / relative
        binaries[relative] = sha(binary)
    output_path = args.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    observations_path = output_path / "observations.json"
    observations_path.write_text(json.dumps(observations, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": "1.0", "evidence_kind": "native_openusd_quatops_converter_capture",
        "target": {"name": "OpenUSD", "distribution": "usd-core", "distribution_version": version("usd-core"), "openusd_version_tuple": list(Usd.GetVersion())},
        "stage_metadata": {"upAxis": str(UsdGeom.GetStageUpAxis(stage)), "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)), "defaultPrim": str(stage.GetDefaultPrim().GetPath())},
        "execution": {"was_run": True, "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "architecture": platform.machine()},
        "native_binaries_sha256": binaries,
        "inputs": {"stage_sha256": sha(stage_path), "converter_input_sha256": sha(inputs_path)},
        "exporter": {"path": str(Path(__file__).name), "sha256": sha(Path(__file__))},
        "generator_sha256": sha(HERE / "create_stage.py"),
        "lockfile_sha256": sha(HERE.parents[2] / "transports/openusd/26.11/uv.lock"),
        "canonical_inputs_sha256": inputs["canonical_input_sha256"],
        "observations": {"path": "observations.json", "sha256": sha(observations_path), "count": len(observations)},
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"captured {len(observations)} native OpenUSD stage observations at {output_path}")


if __name__ == "__main__":
    main()
