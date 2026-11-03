# OpenXR 1.1.65 discovery

The initial native probe compiles against the Khronos OpenXR SDK 1.1.65
headers. On GCC 15.2.0 / Windows x86-64 it measured `XrQuaternionf` as four
32-bit floats with byte offsets `x=0, y=4, z=8, w=12`, and `XrPosef` as 28
bytes with `orientation` at byte 0 and `position` at byte 16. The capture is
at [`experiments/oracles/openxr/1.1.65`](../../experiments/oracles/openxr/1.1.65/README.md).

This is header ABI evidence only. No OpenXR runtime was configured on the
capture machine, and no `xr*` function was called. Runtime space-location,
tracking, and pose-direction behavior therefore remain pending.

The [OpenXR 1.1.65 `XrQuaternionf` reference](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrQuaternionf.html)
defines a unit quaternion with `x, y, z, w` fields. [`XrPosef`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPosef.html)
contains an orientation and a position in a space; the specification says
the rotation is applied before translation. [`xrLocateSpace`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html)
returns the pose of `space` within `baseSpace`, so interpreting its pose as a
coordinate map from the located space into the base space follows from that
definition. OpenXR does not promise one universal world frame.

The named `VIEW`, `LOCAL`, and `LOCAL_FLOOR` spaces specify +X right, +Y up,
and -Z forward. `STAGE` has its own runtime-defined floor origin and XZ bounds.
The first real runtime capture should use explicit, nontrivial application
spaces and check `xrLocateSpace` in both directions at a fixed valid time.

These conventions are specified in the official [OpenXR 1.1.65 reference-space](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html)
and [space-location](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrSpaceLocation.html)
references. The SDK pin is the [Khronos release-1.1.65 tag](https://github.com/KhronosGroup/OpenXR-SDK/releases/tag/release-1.1.65).
