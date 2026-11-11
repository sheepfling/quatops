# External math and transport oracles

This index is for native runtimes and file/pose transports used as auxiliary
evidence. Their exporters, pins, raw captures and replay tests stay outside
`src/`; none becomes a QuatOps runtime dependency. Read the target page for
what its capture actually establishes.

For a cross-target summary of measured conventions and evidence status, see the
[interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md).

## Game engine captures

Native engine captures are indexed separately because they prove behavior in
the pinned engine runtime. See the [engine adapter index](../engines/README.md)
for Unreal, Unity, Godot, and Blender, with runtime contracts and witnesses
linked from the [engine proof matrix](../engines/proof-matrix.md).

## Captured

- [CesiumJS 1.124.0](cesium.md): local-ENU quaternion conversion and native WGS84 ENU/ECEF placement.
- [OpenCV 4.13.0](opencv.md): Rodrigues vector/matrix converter boundary in float32 and float64.
- [Eigen 5.0.1](eigen.md): native Quaternionf/Quaterniond storage, action, and composition boundary.
- [PROJ 9.5.1](proj.md): supporting WGS84 geodetic/ECEF and topocentric probes.
- [CSPICE via SpiceyPy 8.3.0](spiceypy.md): direct quaternion/matrix converter boundary.
- [MuJoCo 3.17.0](mujoco.md): direct quaternion/matrix converter boundary.
- [ROS 2 Jazzy / tf2 0.36.23](ros2-tf2.md): native named-frame chain,
  forward and reverse lookup, message fields, and QuatOps replay.

## Supporting discovery evidence

- [Sophus 1.24.6](sophus.md): selected native `SO3f` and `SO3d` witnesses
  reviewed against separate SciPy 1.18.1 formula checks; not a named converter.
- [OpenXR 1.1.65](openxr.md): normative pose contract, native SDK header ABI,
  and static frame-chain capture through the pinned Khronos test runtime; no
  vendor runtime or tracking hardware is covered.

## Runtime-dependent evidence still unavailable

OpenXR vendor-runtime `xrLocateSpace` behavior, tracking validity, and
runtime-specific space relations require a configured vendor runtime and
tracking system. Sophus Jacobians and tangent derivatives are outside the
selected `SO3f`/`SO3d` discovery scope, not part of a named converter claim.

See [transport captures](../transports/README.md) for glTF captures and
[converter evidence](../transports/openusd.md) for the OpenUSD stage round trip.
The retained evidence atlas distinguishes captured target behavior, selected
converter witnesses, and separately run SciPy formula checks. Broad standalone
math API sweeps are outside this converter PR; GLTFLoader behavior is covered
by the glTF transport capture.

Do not count two wrappers around the same implementation family as independent
mathematical evidence. Format validation, specification-derived expectations,
and loader execution are separate evidence types.
