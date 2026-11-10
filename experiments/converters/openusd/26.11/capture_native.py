"""Open the QuatOps-authored USDA file and capture native OpenUSD evaluation."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from importlib import resources
from importlib.metadata import version
from pathlib import Path

from pxr import Gf, Usd, UsdGeom
from pydantic import BaseModel, ConfigDict

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_schema.converter_captures import OpenUsdInputs
from scripts.evidence_schema.primitives import Matrix4, Quaternion4, Vector3


class NativeOpenUsdOutputs(BaseModel):
    """Typed raw OpenUSD API observations before witness reduction."""

    model_config = ConfigDict(extra="forbid", strict=True)

    orient_op_wxyz: Quaternion4
    local_matrix_rows: Matrix4
    world_matrix_rows: Matrix4
    ordered_xform_ops: list[str]
    resets_xform_stack: bool
    local_points_xyz: list[Vector3]
    world_points_xyz: list[Vector3]
    world_directions_xyz: list[Vector3]
    world_to_local_point_roundtrip_xyz: list[Vector3]


class NativeOpenUsdObservation(BaseModel):
    """One captured prim and the exact native operations read from it."""

    model_config = ConfigDict(extra="forbid", strict=True)

    node: str
    prim_path: str
    native_entry_points: list[str]
    outputs: NativeOpenUsdOutputs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(value: Iterable[float]) -> list[float]:
    return [float(x) for x in value]


def matrix_rows(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def q_wxyz(quaternion: Gf.Quatd) -> list[float]:
    return [float(quaternion.GetReal()), *[float(x) for x in quaternion.GetImaginary()]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    target_dir = repository_root / "experiments/converters/openusd/26.11"
    stage_path = (args.stage or target_dir / "quatops_hierarchy.usda").resolve()
    inputs_path = (args.inputs or target_dir / "converter_inputs.json").resolve()
    inputs = OpenUsdInputs.model_validate_json(inputs_path.read_bytes(), strict=True)
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open {stage_path}")
    cache = UsdGeom.XformCache()
    probe_vectors = [Gf.Vec3d(*row) for row in inputs.usd_probe_vectors_xyz]
    raw_observations: list[dict[str, object]] = []
    for node, prim_path in (
        ("parent", "/Scene/Parent"),
        ("child", "/Scene/Parent/Child"),
        ("reset_child", "/Scene/Parent/ResetChild"),
    ):
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        local = xformable.GetLocalTransformation()
        world = cache.GetLocalToWorldTransform(prim)
        ops = xformable.GetOrderedXformOps()
        orient = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
        qnative = orient.Get()
        inverse = world.GetInverse()
        raw_observations.append(
            {
                "node": node,
                "prim_path": prim_path,
                "native_entry_points": [
                    "Usd.Stage.Open",
                    "UsdGeom.Xformable.GetOrderedXformOps",
                    "GetLocalTransformation",
                    "UsdGeom.XformCache.GetLocalToWorldTransform",
                    "Gf.Matrix4d.Transform",
                    "Gf.Matrix4d.TransformDir",
                ],
                "outputs": {
                    "orient_op_wxyz": q_wxyz(qnative),
                    "local_matrix_rows": matrix_rows(local),
                    "world_matrix_rows": matrix_rows(world),
                    "ordered_xform_ops": [op.GetOpName() for op in ops],
                    "resets_xform_stack": bool(xformable.GetResetXformStack()),
                    "local_points_xyz": [
                        vec(local.Transform(p)) for p in probe_vectors
                    ],
                    "world_points_xyz": [
                        vec(world.Transform(p)) for p in probe_vectors
                    ],
                    "world_directions_xyz": [
                        vec(world.TransformDir(p)) for p in probe_vectors
                    ],
                    "world_to_local_point_roundtrip_xyz": [
                        vec(inverse.Transform(world.Transform(p)))
                        for p in probe_vectors
                    ],
                },
            }
        )
    observations = [
        NativeOpenUsdObservation.model_validate(row, strict=True)
        for row in raw_observations
    ]
    package_dir = Path(str(resources.files("pxr")))
    binaries = {}
    for relative in (
        "usd_ms.dll",
        "Gf/_gf.pyd",
        "Usd/_usd.pyd",
        "UsdGeom/_usdGeom.pyd",
    ):
        binary = package_dir / relative
        binaries[relative] = sha(binary)
    output_path = args.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    semantic_contract_path = target_dir / "semantic_contract.json"
    (output_path / "contract.json").write_bytes(semantic_contract_path.read_bytes())
    observations_path = output_path / "observations.json"
    observations_path.write_text(
        dump_evidence_json([row.model_dump(mode="json") for row in observations]),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "1.0",
        "evidence_kind": "native_openusd_quatops_converter_capture",
        "semantic_contract": {
            "file": (target_dir / "semantic_contract.json")
            .relative_to(repository_root)
            .as_posix(),
            "sha256": sha(target_dir / "semantic_contract.json"),
        },
        "target": {
            "name": "OpenUSD",
            "distribution": "usd-core",
            "distribution_version": version("usd-core"),
            "openusd_version_tuple": list(Usd.GetVersion()),
        },
        "stage_metadata": {
            "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
            "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "defaultPrim": str(stage.GetDefaultPrim().GetPath()),
        },
        "execution": {
            "was_run": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "native_binaries_sha256": binaries,
        "inputs": {
            "stage_sha256": sha(stage_path),
            "converter_input_sha256": sha(inputs_path),
        },
        "exporter": {"path": str(Path(__file__).name), "sha256": sha(Path(__file__))},
        "generator_sha256": sha(target_dir / "create_stage.py"),
        "lockfile_sha256": sha(target_dir / "uv.lock"),
        "canonical_inputs_sha256": inputs.canonical_input_sha256,
        "observations": {
            "path": "observations.json",
            "sha256": sha(observations_path),
            "count": len(observations),
        },
    }
    (output_path / "manifest.json").write_text(
        dump_evidence_json(manifest), encoding="utf-8", newline="\n"
    )
    print(
        f"captured {len(observations)} native OpenUSD stage observations at {output_path}"
    )


if __name__ == "__main__":
    main()
