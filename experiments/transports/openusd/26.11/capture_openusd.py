"""Capture OpenUSD Gf quaternion APIs and UsdGeom transform evaluation."""

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

HERE = Path(__file__).parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=HERE.parents[2] / "oracles" / "shared" / "orientation_inputs_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--stage",
        default=HERE / "assets" / "rotation_stack.usda",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=HERE.parents[3]
        / "tests"
        / "golden"
        / "transports"
        / "openusd"
        / "26.11"
        / "native_rotation.json",
        type=Path,
    )
    return parser.parse_args()


def vec3(value, scalar_type=Gf.Vec3d):
    return scalar_type(*(float(component) for component in value))


def quat_values(quaternion):
    return [
        float(quaternion.GetReal()),
        *[float(value) for value in quaternion.GetImaginary()],
    ]


def matrix_rows(matrix):
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def create_input_stage(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")

    parent = UsdGeom.Xform.Define(stage, "/World/Parent")
    parent_translate = parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    parent_translate.Set(Gf.Vec3d(1.25, -2.0, 0.5))
    parent_orient = parent.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    parent_orient.Set(Gf.Quatd(0.9238795325112867, Gf.Vec3d(0.0, 0.0, 0.3826834323650898)))

    child = UsdGeom.Xform.Define(stage, "/World/Parent/Child")
    child_rotate = child.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    child_rotate.Set(Gf.Vec3d(25.0, -35.0, 70.0))
    child_translate = child.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    child_translate.Set(Gf.Vec3d(-0.75, 1.5, 2.25))

    reset = UsdGeom.Xform.Define(stage, "/World/Parent/ResetChild")
    reset.SetResetXformStack(True)
    reset_translate = reset.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    reset_translate.Set(Gf.Vec3d(3.0, -1.0, 2.0))
    reset_orient = reset.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    reset_orient.Set(Gf.Quatd(0.8660254037844386, Gf.Vec3d(0.5, 0.0, 0.0)))

    axis_probes = {}
    for name, axis in (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0))):
        probe = UsdGeom.Xform.Define(stage, f"/World/Axis{name}")
        translate = probe.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate.Set(vec3(axis))
        axis_probes[name] = probe

    stage.GetRootLayer().Save()
    return stage, world, parent, child, reset, axis_probes


def capture_quaternions(corpus):
    algebra = []
    for case in corpus["quaternion_inputs"]:
        w, x, y, z = case["wxyz"]
        qd = Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))
        qf = Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
        qf_fixed = Gf.Quatf(0.5, Gf.Vec3f(0.25, -0.75, 1.25))
        normalized_copy = Gf.Quatd(qd)
        normalized_copy.Normalize()
        normalized_float_copy = Gf.Quatf(qf)
        normalized_float_copy.Normalize()
        product = qd * Gf.Quatd(0.5, Gf.Vec3d(0.25, -0.75, 1.25))
        float_product = qf * qf_fixed
        inverse = qd.GetInverse()
        float_inverse = qf.GetInverse()
        algebra.append(
            {
                "case_id": case["case_id"],
                "domain": case["domain"],
                "input_wxyz": [float(value) for value in case["wxyz"]],
                "quatd_wxyz": quat_values(qd),
                "quatf_wxyz": quat_values(qf),
                "quatf_length": float(qf.GetLength()),
                "quatf_normalized_wxyz": quat_values(qf.GetNormalized()),
                "quatf_normalize_in_place_wxyz": quat_values(normalized_float_copy),
                "quatf_conjugate_wxyz": quat_values(qf.GetConjugate()),
                "quatf_inverse_wxyz": quat_values(float_inverse),
                "quatf_product_with_fixed_wxyz": quat_values(float_product),
                "quatd_length": float(qd.GetLength()),
                "quatd_normalized_wxyz": quat_values(qd.GetNormalized()),
                "quatd_normalize_in_place_wxyz": quat_values(normalized_copy),
                "quatd_conjugate_wxyz": quat_values(qd.GetConjugate()),
                "quatd_inverse_wxyz": quat_values(inverse),
                "quatd_product_with_fixed_wxyz": quat_values(product),
                "quatd_transform_vectors_xyz": [
                    {
                        "input_xyz": vector,
                        "transformed_xyz": [
                            float(value)
                            for value in qd.Transform(Gf.Vec3d(*(float(x) for x in vector)))
                        ],
                        "inverse_transformed_xyz": [
                            float(value)
                            for value in inverse.Transform(Gf.Vec3d(*(float(x) for x in vector)))
                        ],
                    }
                    for vector in corpus["vectors_xyz"]
                ],
                "quatf_transform_vectors_xyz": [
                    {
                        "input_xyz": vector,
                        "transformed_xyz": [
                            float(value)
                            for value in qf.Transform(
                                Gf.Vec3f(*(float(x) for x in vector))
                            )
                        ],
                        "inverse_transformed_xyz": [
                            float(value)
                            for value in float_inverse.Transform(
                                Gf.Vec3f(*(float(x) for x in vector))
                            )
                        ],
                    }
                    for vector in corpus["vectors_xyz"]
                ],
            }
        )
    return algebra


def capture_axis_angles(corpus):
    rows = []
    for case in corpus["axis_angle_inputs"]:
        axis = vec3(case["axis_xyz"])
        rotation = Gf.Rotation(axis, float(case["angle_radians"]) * 180.0 / 3.141592653589793)
        quaternion = rotation.GetQuat()
        matrix = Gf.Matrix4d(1.0).SetRotate(rotation)
        rows.append(
            {
                "case_id": case["case_id"],
                "input_axis_xyz": case["axis_xyz"],
                "input_angle_radians": case["angle_radians"],
                "input_angle_degrees": float(case["angle_radians"]) * 180.0 / 3.141592653589793,
                "rotation_get_angle_degrees": float(rotation.GetAngle()),
                "rotation_get_axis_xyz": [float(value) for value in rotation.GetAxis()],
                "rotation_get_quaternion_wxyz": quat_values(quaternion),
                "rotation_matrix_rows": matrix_rows(matrix),
                "rotation_matrix_transform_dir_xyz": [
                    float(value) for value in matrix.TransformDir(Gf.Vec3d(0.31, -0.47, 0.83))
                ],
            }
        )
    return rows


def capture_slerp(corpus):
    rows = []
    for pair in corpus["interpolation_pairs"]:
        qa = Gf.Quatd(
            float(pair["a_wxyz"][0]), Gf.Vec3d(*(float(value) for value in pair["a_wxyz"][1:]))
        )
        qb = Gf.Quatd(
            float(pair["b_wxyz"][0]), Gf.Vec3d(*(float(value) for value in pair["b_wxyz"][1:]))
        )
        for amount in corpus["interpolation_weights"]:
            sampled = Gf.Slerp(float(amount), qa, qb)
            rows.append(
                {
                    "case_id": pair["case_id"],
                    "input_a_wxyz": pair["a_wxyz"],
                    "input_b_wxyz": pair["b_wxyz"],
                    "t": float(amount),
                    "slerp_wxyz": quat_values(sampled),
                    "slerp_norm": float(sampled.GetLength()),
                }
            )
    return rows


def capture_stage(stage, world, parent, child, reset, axis_probes):
    cache = UsdGeom.XformCache()
    stage_observations = {
        "stage_up_axis_metadata": str(UsdGeom.GetStageUpAxis(stage)),
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "axis_probe_world_translations": {},
        "parent": {},
        "child": {},
        "reset_child": {},
    }
    for name, probe in axis_probes.items():
        matrix = cache.GetLocalToWorldTransform(probe.GetPrim())
        stage_observations["axis_probe_world_translations"][name] = [
            float(value) for value in matrix.ExtractTranslation()
        ]

    for label, prim_schema in (("parent", parent), ("child", child), ("reset_child", reset)):
        prim = prim_schema.GetPrim()
        local = prim_schema.GetLocalTransformation()
        world_matrix = cache.GetLocalToWorldTransform(prim)
        stage_observations[label] = {
            "ordered_xform_ops": [
                {
                    "name": op.GetOpName(),
                    "type": str(op.GetOpType()),
                    "value": str(op.Get()),
                    "precision": str(op.GetPrecision()),
                }
                for op in prim_schema.GetOrderedXformOps()
            ],
            "resets_xform_stack": bool(prim_schema.GetResetXformStack()),
            "local_matrix_rows": matrix_rows(local),
            "world_matrix_rows": matrix_rows(world_matrix),
            "world_transform_point_xyz": [
                float(value) for value in world_matrix.Transform(Gf.Vec3d(0.31, -0.47, 0.83))
            ],
            "world_transform_direction_xyz": [
                float(value)
                for value in world_matrix.TransformDir(Gf.Vec3d(0.31, -0.47, 0.83))
            ],
        }
    return stage_observations


def main():
    options = parse_args()
    corpus_bytes = options.corpus.resolve().read_bytes()
    corpus = json.loads(corpus_bytes.decode("utf-8"))
    stage, world, parent, child, reset, axis_probes = create_input_stage(options.stage)
    stage_bytes = options.stage.read_bytes()

    package_dir = Path(__import__("pxr").__file__).parent
    native_files = []
    for relative in (
        "usd_ms.dll",
        "Gf/_gf.pyd",
        "Usd/_usd.pyd",
        "UsdGeom/_usdGeom.pyd",
    ):
        path = package_dir / relative
        if path.exists():
            native_files.append({"path": relative, "sha256": sha256_file(path)})
    if len(native_files) != 4:
        raise RuntimeError(f"Could not find every expected OpenUSD binary under {package_dir}")

    lock_path = HERE / "uv.lock"
    output = {
        "schema_version": "1.0",
        "evidence_kind": "native_openusd_gf_and_usdgeom_runtime_capture",
        "runtime": {
            "distribution": "usd-core",
            "distribution_version": version("usd-core"),
            "openusd_version_tuple": list(Usd.GetVersion()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "native_binary_files": native_files,
        },
        "provenance": {
            "input_corpus_sha256": sha256(corpus_bytes),
            "input_stage_sha256": sha256(stage_bytes),
            "lockfile_sha256": sha256_file(lock_path),
            "exporter_sha256": sha256_file(Path(__file__)),
            "exact_invocation": "uv run --locked python capture_openusd.py --corpus <orientation_inputs_v1.json> --stage <rotation_stack.usda> --output <native_rotation.json>",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "quaternion_algebra": len(corpus["quaternion_inputs"]),
            "axis_angle_and_rotation": len(corpus["axis_angle_inputs"]),
            "slerp": len(corpus["interpolation_pairs"]) * len(corpus["interpolation_weights"]),
            "stage_prims": 7,
        },
        "observations": {
            "quaternion_algebra": capture_quaternions(corpus),
            "axis_angle_and_rotation": capture_axis_angles(corpus),
            "slerp": capture_slerp(corpus),
            "stage": capture_stage(stage, world, parent, child, reset, axis_probes),
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
