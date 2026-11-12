# Native Godot Camera3D node witnesses

This is separate from the quaternion-only capture. Both official Windows
runtimes were executed on PB-MSI on 2026-11-12: Godot 4.7 stable and 4.8-dev1.
The manifests identify the exact build, executable, script, contract and output
SHA-256. Retained JSON is the unchanged native output.

## Scope

The camera is a real `Camera3D` attached to a rotated and translated `Node3D`
inside the SceneTree. Three cases cover identity and both orders of unequal,
noncommuting parent/child rotations. Observations include local/global axes,
`get_camera_transform()` and viewport-center `project_ray_normal()` under the
default perspective projection and zero camera offsets. The center ray is a
camera API observation, not just a relabeled quaternion basis calculation.

The assigned scene frame is right-handed, +X right, +Y up, +Z backward. Camera
forward is -Z. This is an explicit application fixture, not physical sensor
calibration, asset import, XR override, rendered-image, or general projection
qualification. Nonuniform scale, nonzero offsets and animation are excluded.
Versioned camera API reference:
<https://docs.godotengine.org/en/4.7/classes/class_camera3d.html>.

## Reproduce

From the repository root, select the matching official installed executable:

```powershell
$version = '4.7-stable' # or 4.8-dev1
$godot = "C:\Users\Public\Godot\engines\windows\Godot_v${version}_win64.exe\Godot_v${version}_win64_console.exe"
$script = (Resolve-Path experiments/godot/camera3d/capture.gd).Path
$output = Join-Path $env:TEMP "quatops-camera3d-${version}.json"
& $godot --headless --path "experiments/godot/$version" --script $script -- $output
if ($LASTEXITCODE -ne 0) { throw 'Camera capture failed' }
```

Compare with `tests/golden/godot/camera3d/<version>/witnesses.json`; do not
silently replace retained observations. Updated sources or outputs require a
new manifest identity. Replay is independent of executing the runtime:

```shell
python -m pytest tests/experiments/godot/test_camera3d.py
```

The strict target-specific schema is
`scripts/evidence_schema/godot_camera.py`. Replay checks source/provenance
hashes and native camera observations against independent SciPy composition
and point action. Camera evidence must be linked as its own operation/scope
in semantic assurance; it must not expand quaternion-only claims implicitly.
