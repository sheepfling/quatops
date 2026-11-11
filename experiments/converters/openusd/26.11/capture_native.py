"""Open the QuatOps-authored USDA file and capture native OpenUSD evaluation."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib import resources
from importlib.metadata import version
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_schema.converter_captures import (
    OpenUsdCapture,
    OpenUsdInputs,
    OpenUsdNodeObservation,
    OpenUsdOutputs,
)
from scripts.evidence_schema.utilities import sha256_file


def vec(value: Iterable[float]) -> list[float]:
    return [float(x) for x in value]


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
    if version("usd-core") != "26.11":
        raise ValueError("This capture requires usd-core 26.11")
    stage_path = (args.stage or target_dir / "quatops_hierarchy.usda").resolve()
    inputs_path = (args.inputs or target_dir / "converter_inputs.json").resolve()
    inputs = OpenUsdInputs.model_validate_json(inputs_path.read_bytes(), strict=True)
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open {stage_path}")
    cache = UsdGeom.XformCache()
    probe_vectors = [Gf.Vec3d(*row) for row in inputs.usd_probe_vectors_xyz]
    observations: list[OpenUsdNodeObservation] = []
    ordered_xform_ops: list[str] | None = None
    for node, prim_path in (
        ("parent", "/Scene/Parent"),
        ("child", "/Scene/Parent/Child"),
        ("reset_child", "/Scene/Parent/ResetChild"),
    ):
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.GetLocalTransformation()
        world = cache.GetLocalToWorldTransform(prim)
        ops = xformable.GetOrderedXformOps()
        orient = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
        qnative = orient.Get()
        names = [op.GetOpName() for op in ops]
        if ordered_xform_ops is not None and ordered_xform_ops != names:
            raise ValueError("authored nodes must have the same transform-op order")
        ordered_xform_ops = names
        world_probe = world.Transform(probe_vectors[-1])
        observations.append(
            OpenUsdNodeObservation(
                node=node,
                prim_path=prim_path,
                outputs=OpenUsdOutputs(
                    orient_op_wxyz=q_wxyz(qnative),
                    resets_xform_stack=bool(xformable.GetResetXformStack()),
                    world_basis_images_xyz=[
                        vec(world.TransformDir(p)) for p in probe_vectors[:3]
                    ],
                    world_probe_point_xyz=vec(world_probe),
                    roundtrip_probe_point_xyz=vec(
                        world.GetInverse().Transform(world_probe)
                    )
                    if node == "child"
                    else None,
                ),
            )
        )
    witnesses = OpenUsdCapture(
        schema_version="1.0",
        evidence_kind="native_openusd_quatops_converter_capture",
        native_entry_points=[
            "Usd.Stage.Open",
            "UsdGeom.Xformable.GetOrderedXformOps",
            "GetLocalTransformation",
            "UsdGeom.XformCache.GetLocalToWorldTransform",
            "Gf.Matrix4d.Transform",
            "Gf.Matrix4d.TransformDir",
        ],
        ordered_xform_ops=ordered_xform_ops or [],
        observations=observations,
    )
    package_dir = Path(str(resources.files("pxr")))
    binaries = {}
    for relative in (
        "usd_ms.dll",
        "Gf/_gf.pyd",
        "Usd/_usd.pyd",
        "UsdGeom/_usdGeom.pyd",
    ):
        binary = package_dir / relative
        binaries[relative] = sha256_file(binary)
    output_path = args.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    semantic_contract_path = target_dir / "semantic_contract.json"
    (output_path / "contract.json").write_bytes(semantic_contract_path.read_bytes())
    observations_path = output_path / "witnesses.json"
    observations_path.write_text(
        dump_evidence_json(witnesses.model_dump(mode="json", exclude_none=True)),
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
            "sha256": sha256_file(target_dir / "semantic_contract.json"),
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
            "stage_sha256": sha256_file(stage_path),
            "converter_input_sha256": sha256_file(inputs_path),
        },
        "exporter": {
            "path": str(Path(__file__).name),
            "sha256": sha256_file(Path(__file__)),
        },
        "digest_helper_sha256": sha256_file(
            repository_root / "scripts/evidence_schema/utilities.py"
        ),
        "generator_sha256": sha256_file(target_dir / "create_stage.py"),
        "lockfile_sha256": sha256_file(
            repository_root / "experiments/converters/python/uv.lock"
        ),
        "canonical_inputs_sha256": inputs.canonical_input_sha256,
        "observations": {
            "path": "witnesses.json",
            "sha256": sha256_file(observations_path),
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
