# OpenXR 1.1.65 ABI and pose-contract discovery

The pinned Khronos SDK 1.1.65 header probe was rebuilt with GCC 15.2.0 on
Windows x86-64. [`header_abi.json`](header_abi.json) is the exact, formatted
native stdout. It records `XrQuaternionf` as four 32-bit floats in XYZW order,
and `XrPosef` as orientation followed by position, with the offsets and sizes
measured by the C++ probe.

The versioned OpenXR specification completes the documented pose contract:
the quaternion is unit length, pose position is in meters, and orientation is
applied before translation. `xrLocateSpace(space, baseSpace, time, ...)`
returns the pose of `space` in `baseSpace`. Interpreting that pose as a
`baseSpace <- space` point mapping follows the pose definition; it is a frame
direction interpretation, not a runtime observation here. Locations are
time-specific, and runtimes may revise them as their reference-space origins
change. `VIEW`, `LOCAL`, and `LOCAL_FLOOR` use +X right, +Y up, -Z forward.
`VIEW` follows the viewer and is not gravity-aligned. `LOCAL` is world-locked
and gravity-aligned, with a runtime-chosen launch or calibrated origin.
`LOCAL_FLOOR` places its origin at an estimated floor level. `STAGE` has a
floor-level origin at the center of a runtime-defined rectangle, uses +Y up,
and aligns XZ to its bounds; support is optional. These are
specification facts, not observations from a particular runtime, and OpenXR
does not define one universal world frame.

No OpenXR runtime is installed or registered in this environment. The probe
does not create an instance or call any `xr*` function, so runtime pose
location, tracking validity, and actual reference-space relations remain
unmeasured. This discovery closes the pinned header and specification scope;
it does not claim runtime validation.

## Reproduce the header capture

Fetch `release-1.1.65` from the SDK repository recorded in `source.json`, then
compile and capture:

```powershell
g++ -std=c++17 -O2 -I C:/path/to/OpenXR-SDK/include `
  capture_header_abi.cpp -o capture_openxr_header.exe
cmd.exe /d /c "capture_openxr_header.exe > header_abi.json"
```

The manifest associates the output with the capture source, executable, and
pinned SDK header hashes. Primary references are the OpenXR 1.1.65
[`XrQuaternionf`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrQuaternionf.html),
[`XrPosef`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPosef.html),
[`xrLocateSpace`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html),
and [`XrReferenceSpaceType`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html)
pages.
