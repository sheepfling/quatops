# OpenXR 1.1.65 ABI and pose-contract discovery

The native SDK probe compiles against the pinned Khronos OpenXR 1.1.65
headers. It records `XrQuaternionf` as four float32 fields in XYZW order and
`XrPosef` as orientation followed by position, with measured sizes and byte
offsets in
[`experiments/oracles/openxr/1.1.65`](../../experiments/oracles/openxr/1.1.65/README.md).

The specification defines the quaternion as unit length, pose position in
meters, and rotation before translation. `xrLocateSpace` returns the pose of
`space` in `baseSpace` at the requested time. Reading that pose as a
`baseSpace <- space` mapping for point conversion follows the pose definition;
it is a frame-direction interpretation, not a measurement from this machine's
runtime. Locations are time-specific, and a runtime may revise them as it
updates reference-space origins.

`VIEW`, `LOCAL`, and `LOCAL_FLOOR` use +X right, +Y up, -Z forward. `VIEW`
follows the viewer and is not gravity-aligned. `LOCAL` is world-locked and
gravity-aligned, with an origin the runtime may choose at launch or another
calibrated zero. `LOCAL_FLOOR` is similar but places its origin at an estimated
floor level. `STAGE` has a floor-level origin at the center of a runtime-defined
rectangle, uses +Y up, and aligns XZ to its bounds; its support is optional.
These documented frame facts do not create a universal OpenXR world frame.

The SDK-header ABI capture was compiled and rerun, and its exact output is
retained. No runtime is installed or registered on the capture machine, so no
`xrLocateSpace` call, tracking result, or runtime-specific relation is claimed.
The completed discovery is specification plus header ABI evidence; live
runtime behavior remains a separate gate.

Primary references: OpenXR 1.1.65 [`XrQuaternionf`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrQuaternionf.html),
[`XrPosef`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPosef.html),
[`xrLocateSpace`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html),
and [`XrReferenceSpaceType`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html).
