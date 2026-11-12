# Blender 5.1.2 glTF importer

The Blender embedded-Python capture imports the same reproducible glTF 2.0 asset as
the Three.js loader, then samples scene objects and imported rotation actions.
It records Blender's build and executable hashes, embedded Python version,
importer source hashes, operator arguments, scene FPS, matrices, F-curves, and
the explicit axis-translation probes.

## Reproduce

First [generate the input assets](../../README.md#reproducible-asset). Install
the pinned capture dependencies into a temporary directory using Blender's
Python 3.13 interpreter, then run from the repository root in PowerShell:

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 5.1'
$site = Join-Path $env:TEMP 'quatops-blender-capture-deps'
uv pip install --python "$blender\5.1\python\bin\python.exe" --target $site `
  -r experiments\transports\gltf\2.0.0\blender\5.1.2\requirements-capture.txt
$env:PYTHONPATH = "$PWD;$site"
& "$blender\blender.exe" `
  --background --factory-startup --python-use-system-env --python-exit-code 1 `
  --python 'experiments\transports\gltf\2.0.0\blender\5.1.2\capture_gltf.py' -- `
  --asset 'experiments\transports\gltf\2.0.0\assets\rotation_transport.gltf' `
  --output "$env:TEMP\quatops-blender-gltf-capture.json"
```

The capture uses the replay tests' Pydantic models. The historical exporter and
the matching native rerun are identified in the adjacent provenance file.

The exporter sets the scene to 30 FPS with a unit frame-rate base before import.
Blender 5.1.2 imports keyframe coordinates as `time * FPS`; its scene frame
range still begins at frame 1, so the sampler evaluates the imported action at
absolute frames beginning at frame 0.

## Observed loader behavior

Literal translation probes measured the glTF-to-Blender basis images as
`+X -> +X`, `+Y -> +Z`, and `+Z -> -Y`, a proper rotation with determinant +1.
The parent/child hierarchy and the equivalent local TRS and matrix nodes agree
within Blender's float32 scene evaluation precision (maximum measured world
matrix difference about `6e-8`).

The animation records show important importer-specific differences from the
glTF contract. `STEP` channels use constant F-curves. `LINEAR` quaternion
channels use component-linear F-curves: the retained first-interval midpoint
has quaternion norm about `0.9520`. The glTF
spec says rotational `LINEAR` interpolation should use SLERP and that sampled
rotation quaternions must be normalized before application. The
`CUBICSPLINE` input is captured as Bezier F-curves with automatic handles; the
retained native midpoint differs from the glTF time-scaled Hermite result. Blender's
installed importer source hashes are stored in the capture for this exact
release.

Subframe sampling also exposed a tiny boundary quirk: the sample 1e-5 seconds
before the second key keeps the previous `STEP` value, while the sample 1e-6
seconds before it evaluates to the next key. The retained animation values are
limited to those two boundary outputs and the first-interval midpoint for each
of the other interpolation modes. Authored key values and imported F-curve key
times remain in the source asset and capture. This is a pinned Blender sampler
observation, not a change to the source asset's key time.

The format contract is described by the
[Khronos glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
Blender's versioned importer notes are in the
[Blender 5.1 glTF manual](https://docs.blender.org/manual/en/5.1/addons/import_export/scene_gltf2.html).
