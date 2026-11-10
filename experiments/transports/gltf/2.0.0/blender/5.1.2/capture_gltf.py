"""Capture Blender 5.1.2's native glTF importer and animation evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy


@dataclass(frozen=True, slots=True)
class CurveKey:
    interpolation: str
    handle_left_type: str
    handle_right_type: str


@dataclass(frozen=True, slots=True)
class CurveRow:
    data_path: str
    array_index: int
    keys: list[CurveKey]
    key_frames: list[float]


@dataclass(frozen=True, slots=True)
class FcurveSummary:
    data_path: str
    array_indices: list[int]
    interpolations_by_component: list[list[str]]


@dataclass(frozen=True, slots=True)
class CubicFcurveSummary(FcurveSummary):
    left_handle_types_by_component: list[list[str]]
    right_handle_types_by_component: list[list[str]]


@dataclass(frozen=True, slots=True)
class CapturedActionCurves:
    summary: FcurveSummary
    key_frames: list[float]


@dataclass(frozen=True, slots=True)
class StaticTransform:
    matrix_basis_rows: list[list[float]]
    matrix_world_rows: list[list[float]]


@dataclass(frozen=True, slots=True)
class GltfTransport:
    name: str
    version: str
    source_coordinates: str


@dataclass(frozen=True, slots=True)
class BlenderLoader:
    name: str
    version: str
    version_tuple: list[int]
    build_hash: str
    build_date: str
    build_platform: str
    python_version: str
    executable: str
    executable_sha256: str
    importer_source_root: str
    importer_source_sha256: dict[str, str]
    rotation_component_order: str


@dataclass(frozen=True, slots=True)
class OperatorArguments:
    filepath: str


@dataclass(frozen=True, slots=True)
class CaptureProvenance:
    gltf_sha256: str
    binary_sha256: str
    exporter_sha256: str
    exact_invocation: str
    operator_arguments: OperatorArguments
    other_operator_properties: str
    captured_at_utc: str


@dataclass(frozen=True, slots=True)
class SceneUnitSettings:
    system: str
    scale_length: float
    system_rotation: str


@dataclass(frozen=True, slots=True)
class SceneSettings:
    fps: int
    fps_base: float
    frame_start: int
    frame_end: int
    unit_settings: SceneUnitSettings


@dataclass(frozen=True, slots=True)
class AxisImages:
    x: list[float]
    y: list[float]
    z: list[float]


@dataclass(frozen=True, slots=True)
class AnimationSample:
    sample: str
    time_seconds: float
    quaternion_wxyz: list[float]


@dataclass(frozen=True, slots=True)
class AnimationObservation:
    interpolation: str
    action_name: str
    target_object: str
    key_times_seconds_from_imported_fcurves: list[float]
    fcurves: FcurveSummary
    samples: list[AnimationSample]


@dataclass(frozen=True, slots=True)
class GltfObservations:
    static_transforms: dict[str, StaticTransform]
    source_translation_axis_images_in_blender_xyz: AxisImages
    animations: list[AnimationObservation]


@dataclass(frozen=True, slots=True)
class BlenderGltfCapture:
    schema_version: str
    evidence_kind: str
    transport: GltfTransport
    loader: BlenderLoader
    provenance: CaptureProvenance
    scene: SceneSettings
    observations: GltfObservations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_value(value: str | bytes) -> str:
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )


def matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def quaternion_wxyz(quaternion: Any) -> list[float]:
    return [
        float(quaternion.w),
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
    ]


def object_transform(object_: Any) -> StaticTransform:
    bpy.context.view_layer.update()
    evaluated = object_.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return StaticTransform(
        matrix_basis_rows=matrix_rows(evaluated.matrix_basis),
        matrix_world_rows=matrix_rows(evaluated.matrix_world),
    )


def action_curves(action: Any, *, include_handles: bool) -> CapturedActionCurves:
    rows: list[CurveRow] = []
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                bag = strip.channelbag(slot)
                if bag is None:
                    continue
                for curve in bag.fcurves:
                    keys = [
                        CurveKey(
                            interpolation=point.interpolation,
                            handle_left_type=point.handle_left_type,
                            handle_right_type=point.handle_right_type,
                        )
                        for point in curve.keyframe_points
                    ]
                    rows.append(
                        CurveRow(
                            data_path=curve.data_path,
                            array_index=curve.array_index,
                            keys=keys,
                            key_frames=[
                                float(point.co.x) for point in curve.keyframe_points
                            ],
                        )
                    )
    if not rows:
        raise RuntimeError(f"No F-curves in action {action.name!r}")
    key_frames = rows[0].key_frames
    if any(row.key_frames != key_frames for row in rows[1:]):
        raise RuntimeError(f"Quaternion F-curve key times differ in {action.name!r}")
    common = FcurveSummary(
        data_path=rows[0].data_path,
        array_indices=[row.array_index for row in rows],
        interpolations_by_component=[
            [key.interpolation for key in row.keys] for row in rows
        ],
    )
    summary: FcurveSummary = common
    if include_handles:
        summary = CubicFcurveSummary(
            data_path=common.data_path,
            array_indices=common.array_indices,
            interpolations_by_component=common.interpolations_by_component,
            left_handle_types_by_component=[
                [key.handle_left_type for key in row.keys] for row in rows
            ],
            right_handle_types_by_component=[
                [key.handle_right_type for key in row.keys] for row in rows
            ],
        )
    return CapturedActionCurves(summary=summary, key_frames=key_frames)


def install_action(object_: Any, action: Any) -> None:
    if object_.animation_data is None:
        object_.animation_data_create()
    animation_data = object_.animation_data
    if animation_data is None:
        raise RuntimeError("Could not create animation data for imported action")
    animation_data.action = action
    if len(action.slots):
        animation_data.action_slot = action.slots[0]


def require_object(name: str) -> Any:
    object_ = bpy.data.objects.get(name)
    if object_ is None:
        raise RuntimeError(f"Blender importer did not create object {name!r}")
    return object_


def set_scene_time(scene: Any, seconds: float) -> float:
    # The glTF importer maps t=0 to Blender frame 0, even when scene.frame_start
    # remains 1. Sample the imported action's absolute frame coordinate.
    frame_value = seconds * scene.render.fps / scene.render.fps_base
    whole_frame = math.floor(frame_value)
    scene.frame_set(whole_frame, subframe=frame_value - whole_frame)
    bpy.context.view_layer.update()
    return frame_value


def main() -> None:
    if "--" not in sys.argv:
        raise RuntimeError("Expected -- followed by --asset PATH --output PATH")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    asset_path = Path(options.asset).resolve()
    output_path = Path(options.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(asset_path))

    names = [
        "parent_frame",
        "static_trs",
        "static_matrix",
        "source_axis_origin",
        "source_axis_x",
        "source_axis_y",
        "source_axis_z",
    ]
    objects = {name: require_object(name) for name in names}
    static_names = {"parent_frame", "static_trs", "static_matrix"}
    static: dict[str, StaticTransform] = {
        name: object_transform(objects[name]) for name in names if name in static_names
    }

    origin = objects["source_axis_origin"].matrix_world.translation.copy()

    def axis_image(axis: str) -> list[float]:
        return [
            float(component)
            for component in (
                objects[f"source_axis_{axis}"].matrix_world.translation.copy() - origin
            )
        ]

    axis_images = AxisImages(
        x=axis_image("x"),
        y=axis_image("y"),
        z=axis_image("z"),
    )

    animations: list[AnimationObservation] = []
    for interpolation, node_name, action_name in (
        ("STEP", "animated_step", "rotation_step"),
        ("LINEAR", "animated_linear", "rotation_linear"),
        ("CUBICSPLINE", "animated_cubic", "rotation_cubic"),
    ):
        object_ = bpy.data.objects.get(node_name)
        action = bpy.data.actions.get(action_name)
        if object_ is None or action is None:
            raise RuntimeError(
                f"Missing imported {interpolation} action or target node"
            )
        install_action(object_, action)
        curves = action_curves(action, include_handles=interpolation == "CUBICSPLINE")
        curve_frames = curves.key_frames
        time_values = sorted(
            {frame * scene.render.fps_base / scene.render.fps for frame in curve_frames}
        )
        if len(time_values) != 3:
            raise RuntimeError(
                f"Expected three key times for {interpolation}: {time_values}"
            )
        t0, t1, t2 = time_values
        epsilon = min(1e-6, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0)
        boundary_samples = []
        for offset in dict.fromkeys((1e-5, epsilon)):
            boundary_samples.append((f"before-key-1-{offset:.0e}", t1 - offset))
        if interpolation == "STEP":
            samples = [
                ("key-0", t0),
                *boundary_samples,
                ("key-1", t1),
                ("after-key-1", t1 + epsilon),
                ("key-2", t2),
            ]
        else:
            samples = [
                ("key-0", t0),
                ("midpoint-0-1", (t0 + t1) / 2.0),
                ("key-1", t1),
                ("key-2", t2),
            ]
        sample_rows: list[AnimationSample] = []
        for sample_name, sample_time in samples:
            set_scene_time(scene, sample_time)
            evaluated = object_.evaluated_get(bpy.context.evaluated_depsgraph_get())
            sample_rows.append(
                AnimationSample(
                    sample=sample_name,
                    time_seconds=sample_time,
                    quaternion_wxyz=quaternion_wxyz(evaluated.rotation_quaternion),
                )
            )
        animations.append(
            AnimationObservation(
                interpolation=interpolation,
                action_name=action.name,
                target_object=object_.name,
                key_times_seconds_from_imported_fcurves=time_values,
                fcurves=curves.summary,
                samples=sample_rows,
            )
        )
        object_.animation_data.action = None

    executable = Path(bpy.app.binary_path)
    asset_binary = asset_path.with_suffix(".bin")
    version_directory = executable.parent / f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    importer_root = version_directory / "scripts" / "addons_core" / "io_scene_gltf2"
    importer_paths = (
        Path("blender/imp/animation_node.py"),
        Path("blender/imp/animation_utils.py"),
        Path("blender/imp/node.py"),
    )
    importer_sources = {
        relative.as_posix(): sha256_file(importer_root / relative)
        for relative in importer_paths
        if (importer_root / relative).exists()
    }
    result = BlenderGltfCapture(
        schema_version="1.0",
        evidence_kind="native_blender_gltf_importer_capture",
        transport=GltfTransport(
            name="glTF",
            version="2.0",
            source_coordinates=(
                "glTF specification right-handed Y-up; target axis conversion "
                "measured by explicit translation probes"
            ),
        ),
        loader=BlenderLoader(
            name="Blender glTF importer",
            version=bpy.app.version_string,
            version_tuple=list(bpy.app.version),
            build_hash=text_value(bpy.app.build_hash),
            build_date=text_value(bpy.app.build_date),
            build_platform=text_value(bpy.app.build_platform),
            python_version=sys.version,
            executable=str(executable),
            executable_sha256=sha256_file(executable),
            importer_source_root=str(importer_root),
            importer_source_sha256=importer_sources,
            rotation_component_order=(
                "WXYZ in mathutils Quaternion; outputs labelled explicitly"
            ),
        ),
        provenance=CaptureProvenance(
            gltf_sha256=sha256_file(asset_path),
            binary_sha256=sha256_file(asset_binary),
            exporter_sha256=sha256_file(Path(__file__)),
            exact_invocation=(
                "blender --background --factory-startup --python-exit-code 1 "
                "--python capture_gltf.py -- --asset <rotation_transport.gltf> "
                "--output <native_blender_gltf.json>"
            ),
            operator_arguments=OperatorArguments(filepath=asset_path.name),
            other_operator_properties="Blender 5.1.2 importer defaults",
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
        ),
        scene=SceneSettings(
            fps=scene.render.fps,
            fps_base=scene.render.fps_base,
            frame_start=scene.frame_start,
            frame_end=scene.frame_end,
            unit_settings=SceneUnitSettings(
                system=scene.unit_settings.system,
                scale_length=scene.unit_settings.scale_length,
                system_rotation=scene.unit_settings.system_rotation,
            ),
        ),
        observations=GltfObservations(
            static_transforms=static,
            source_translation_axis_images_in_blender_xyz=axis_images,
            animations=animations,
        ),
    )
    output_path.write_text(
        json.dumps(asdict(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "blender": bpy.app.version_string,
                "animations": len(animations),
                "axis_images": asdict(axis_images),
            }
        )
    )


if __name__ == "__main__":
    main()
