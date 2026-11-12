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

The typed OpenXR semantic binding retains the `XrSpaceLocation.locationFlags`
bitmask and rejects a location unless
`XR_SPACE_LOCATION_ORIENTATION_VALID_BIT` is set. It reports the
`XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT` separately: a valid orientation
is not necessarily currently tracked. The OpenXR 1.1.65
[`XrSpaceLocationFlagBits` reference](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrSpaceLocationFlagBits.html)
defines those flags. This check does not turn the pinned test runtime into a
vendor tracking or hardware guarantee.

`VIEW`, `LOCAL`, and `LOCAL_FLOOR` use +X right, +Y up, -Z forward. `VIEW`
follows the viewer and is not gravity-aligned. `LOCAL` is world-locked and
gravity-aligned, with an origin the runtime may choose at launch or another
calibrated zero. `LOCAL_FLOOR` is similar but places its origin at an estimated
floor level. `STAGE` has a floor-level origin at the center of a runtime-defined
rectangle, uses +Y up, and aligns XZ to its bounds; its support is optional.
These documented frame facts do not create a universal OpenXR world frame.

The SDK-header ABI capture was compiled and rerun, and its exact output is
retained. A separate native capture ran `xrLocateSpace` against the pinned
Khronos software test runtime for a static three-space LOCAL frame chain. The
selected outputs and SciPy frame-chain replay are documented in the
[versioned capture guide](../../experiments/oracles/openxr/1.1.65/README.md).
This does not establish vendor-runtime, tracking, headset, or live display-time
behavior.

Primary references: OpenXR 1.1.65 [`XrQuaternionf`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrQuaternionf.html),
[`XrPosef`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPosef.html),
[`xrLocateSpace`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html),
and [`XrReferenceSpaceType`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html).
