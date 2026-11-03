"""Capture Blender 5.1.2's native glTF importer and animation evaluation."""

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_value(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def matrix_rows(matrix):
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def quaternion_wxyz(quaternion):
    return [float(quaternion.w), float(quaternion.x), float(quaternion.y), float(quaternion.z)]


def object_transform(object_):
    bpy.context.view_layer.update()
    evaluated = object_.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {
        "rotation_mode": object_.rotation_mode,
        "location_xyz": [float(x) for x in evaluated.location],
        "rotation_quaternion_wxyz": quaternion_wxyz(evaluated.rotation_quaternion),
        "scale_xyz": [float(x) for x in evaluated.scale],
        "matrix_basis_rows": matrix_rows(evaluated.matrix_basis),
        "matrix_local_rows": matrix_rows(evaluated.matrix_local),
        "matrix_world_rows": matrix_rows(evaluated.matrix_world),
    }


def action_curves(action):
    rows = []
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                bag = strip.channelbag(slot)
                if bag is None:
                    continue
                for curve in bag.fcurves:
                    rows.append({
                        "slot_identifier": slot.identifier,
                        "data_path": curve.data_path,
                        "array_index": curve.array_index,
                        "keys": [
                            {
                                "frame": float(point.co.x),
                                "value": float(point.co.y),
                                "interpolation": point.interpolation,
                                "handle_left_type": point.handle_left_type,
                                "handle_right_type": point.handle_right_type,
                                "handle_left": [float(point.handle_left.x), float(point.handle_left.y)],
                                "handle_right": [float(point.handle_right.x), float(point.handle_right.y)],
                            }
                            for point in curve.keyframe_points
                        ],
                    })
    return rows


def install_action(object_, action):
    if object_.animation_data is None:
        object_.animation_data_create()
    object_.animation_data.action = action
    if len(action.slots):
        object_.animation_data.action_slot = action.slots[0]


def set_scene_time(scene, seconds):
    # The glTF importer maps t=0 to Blender frame 0, even when scene.frame_start
    # remains 1. Sample the imported action's absolute frame coordinate.
    frame_value = seconds * scene.render.fps / scene.render.fps_base
    whole_frame = math.floor(frame_value)
    scene.frame_set(whole_frame, subframe=frame_value - whole_frame)
    bpy.context.view_layer.update()
    return frame_value


def main():
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
    static = {}
    objects = {}
    for name in names:
        object_ = bpy.data.objects.get(name)
        if object_ is None:
            raise RuntimeError(f"Blender importer did not create object {name!r}")
        objects[name] = object_
        static[name] = object_transform(object_)

    origin = objects["source_axis_origin"].matrix_world.translation.copy()
    axis_images = {}
    for axis in "xyz":
        point = objects[f"source_axis_{axis}"].matrix_world.translation.copy()
        axis_images[axis] = [float(component) for component in point - origin]

    animations = []
    for interpolation, node_name, action_name in (
        ("STEP", "animated_step", "rotation_step"),
        ("LINEAR", "animated_linear", "rotation_linear"),
        ("CUBICSPLINE", "animated_cubic", "rotation_cubic"),
    ):
        object_ = bpy.data.objects.get(node_name)
        action = bpy.data.actions.get(action_name)
        if object_ is None or action is None:
            raise RuntimeError(f"Missing imported {interpolation} action or target node")
        install_action(object_, action)
        curves = action_curves(action)
        time_values = sorted({
            key["frame"] * scene.render.fps_base / scene.render.fps
            for curve in curves
            for key in curve["keys"]
        })
        if len(time_values) != 3:
            raise RuntimeError(f"Expected three key times for {interpolation}: {time_values}")
        t0, t1, t2 = time_values
        epsilon = min(1e-6, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0)
        boundary_samples = []
        for offset in (1e-3, 1e-4, 1e-5, epsilon):
            boundary_samples.append((f"before-key-1-{offset:.0e}", t1 - offset))
        samples = [
            ("key-0", t0),
            ("midpoint-0-1", (t0 + t1) / 2.0),
            *boundary_samples,
            ("before-key-1", t1 - epsilon),
            ("key-1", t1),
            ("after-key-1", t1 + epsilon),
            ("midpoint-1-2", (t1 + t2) / 2.0),
            ("key-2", t2),
        ]
        sample_rows = []
        for sample_name, sample_time in samples:
            frame_value = set_scene_time(scene, sample_time)
            evaluated = object_.evaluated_get(bpy.context.evaluated_depsgraph_get())
            sample_rows.append({
                "sample": sample_name,
                "time_seconds": sample_time,
                "frame_value": frame_value,
                "quaternion_wxyz": quaternion_wxyz(evaluated.rotation_quaternion),
                "quaternion_length": float(evaluated.rotation_quaternion.magnitude),
                "matrix_basis_rows": matrix_rows(evaluated.matrix_basis),
                "matrix_world_rows": matrix_rows(evaluated.matrix_world),
            })
        animations.append({
            "interpolation": interpolation,
            "action_name": action.name,
            "target_object": object_.name,
            "key_times_seconds_from_imported_fcurves": time_values,
            "fcurves": curves,
            "samples": sample_rows,
        })
        object_.animation_data.action = None

    executable = Path(bpy.app.binary_path)
    asset_binary = asset_path.with_suffix(".bin")
    version_directory = executable.parent / f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    importer_root = version_directory / "scripts" / "addons_core" / "io_scene_gltf2"
    importer_sources = {}
    for relative in (
        Path("blender/imp/animation_node.py"),
        Path("blender/imp/animation_utils.py"),
        Path("blender/imp/node.py"),
    ):
        source_path = importer_root / relative
        if source_path.exists():
            importer_sources[relative.as_posix()] = sha256_file(source_path)
    result = {
        "schema_version": "1.0",
        "evidence_kind": "native_blender_gltf_importer_capture",
        "transport": {
            "name": "glTF",
            "version": "2.0",
            "source_coordinates": "glTF specification right-handed Y-up; target axis conversion measured by explicit translation probes",
        },
        "loader": {
            "name": "Blender glTF importer",
            "version": bpy.app.version_string,
            "version_tuple": list(bpy.app.version),
            "build_hash": text_value(bpy.app.build_hash),
            "build_date": text_value(bpy.app.build_date),
            "build_platform": text_value(bpy.app.build_platform),
            "python_version": sys.version,
            "executable": str(executable),
            "executable_sha256": sha256_file(executable),
            "importer_source_root": str(importer_root),
            "importer_source_sha256": importer_sources,
            "rotation_component_order": "WXYZ in mathutils Quaternion; outputs labelled explicitly",
        },
        "provenance": {
            "gltf_sha256": sha256_file(asset_path),
            "binary_sha256": sha256_file(asset_binary),
            "exporter_sha256": sha256_file(Path(__file__)),
            "exact_invocation": "blender --background --factory-startup --python-exit-code 1 --python capture_gltf.py -- --asset <rotation_transport.gltf> --output <native_blender_gltf.json>",
            "operator_arguments": {"filepath": asset_path.name},
            "other_operator_properties": "Blender 5.1.2 importer defaults",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "scene": {
            "fps": scene.render.fps,
            "fps_base": scene.render.fps_base,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "unit_settings": {
                "system": scene.unit_settings.system,
                "scale_length": scene.unit_settings.scale_length,
                "system_rotation": scene.unit_settings.system_rotation,
            },
        },
        "observations": {
            "static_transforms": static,
            "source_translation_axis_images_in_blender_xyz": axis_images,
            "animations": animations,
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "blender": bpy.app.version_string,
        "animations": len(animations),
        "axis_images": axis_images,
    }))


if __name__ == "__main__":
    main()
