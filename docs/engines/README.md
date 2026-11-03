# Engine quaternion adapters

The QuatOps core stays independent of engine packages. Named adapters in
`quatops.engines` convert Hamilton quaternion storage and caller-declared
frames for Unreal, Unity, Godot, and Blender. Every conversion requires an
explicit `QuaternionFrameMap`; engine names do not imply world axes, model
forward, or active/passive transform meaning.

## Adapter references

| Target API | Native representation | Contract page |
| --- | --- | --- |
| Unreal `TQuat<T>` | Hamilton XYZW; float32 and float64 | [Unreal](unreal.md) |
| Unity `UnityEngine.Quaternion` | Hamilton XYZW; float32 | [Unity](unity.md) |
| Godot `Quaternion` | Hamilton XYZW; float32 | [Godot](godot.md) |
| Blender `mathutils.Quaternion` | Hamilton WXYZ | [Blender](blender.md) |

The [coordinate-mapping page](coordinate-mapping.md) gives the shared basis
math. The [CAP-300 contract](CAP-300.md) explains the release gate and the
[proof matrix](proof-matrix.md) links each selected versioned native witness
set. Capture exporters and rerun instructions are indexed in
[`experiments/README.md`](../../experiments/README.md).

Auxiliary converter-boundary evidence for Cesium, ROS 2/tf2, glTF loaders, and
OpenUSD is indexed in the
[interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md); its fixtures
remain under `tests/golden/` and do not add runtime adapters here.

## Scope

These contracts cover quaternion-to-quaternion coefficient conversion and
native physical-vector action. Euler sequences, interpolation, scene
transforms, geographic placement, and serialized animation need separate
contracts when QuatOps exposes those operations. Asset model-forward axes
remain caller-supplied facts.
