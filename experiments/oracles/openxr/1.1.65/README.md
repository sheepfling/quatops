# OpenXR 1.1.65 discovery

The initial probe compiles against the pinned Khronos OpenXR SDK headers and
records the C ABI layout of `XrQuaternionf` and `XrPosef`. It is explicitly
header evidence, not an OpenXR runtime capture: no runtime is configured on
this machine, and the probe calls no OpenXR functions.

The normative contract already identifies `XrQuaternionf` as a unit quaternion
with `x, y, z, w` fields. `XrPosef` places orientation and position in a
space; it does not itself identify a universal world frame. `xrLocateSpace`
returns the pose of `space` within `baseSpace`, and the `VIEW`, `LOCAL`, and
`LOCAL_FLOOR` spaces define +X right, +Y up, and -Z forward. These documented
semantics are specification evidence, separate from the SDK ABI probe and
from any future runtime measurements.

## Reproduce the header probe

Fetch the `release-1.1.65` tag recorded in `source.json`, then build from this
directory:

```powershell
g++ -std=c++17 -O2 -I C:/path/to/OpenXR-SDK/include `
  capture_header_abi.cpp -o capture_openxr_header.exe
./capture_openxr_header.exe > header_abi.json
```

The next runtime phase needs an installed OpenXR runtime. A runtime capture
should create two application spaces with explicit poses, locate each in the
other at a fixed valid time, and compare returned vectors and orientations in
both directions. Tracking-dependent headset/controller trajectories are not
needed for the first proof.

Primary references: [OpenXR 1.1.65 `XrQuaternionf`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrQuaternionf.html),
[`XrPosef`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPosef.html),
[`XrReferenceSpaceType`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html),
and [`xrLocateSpace`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html).
