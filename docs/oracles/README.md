# External math and transport oracles

This index is for native runtimes and file/pose transports used as auxiliary
evidence. Their exporters, pins, raw captures and replay tests stay outside
`src/`; none becomes a QuatOps runtime dependency. Read the target page for
what its capture actually establishes.

For a cross-target summary of measured conventions and evidence status, see the
[interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md).

## Captured

- [CesiumJS 1.124.0](cesium.md): local heading-pitch-roll plus WGS84 ENU/ECEF placement.
- [Three.js 0.187.0](threejs.md): Node.js math API and Object3D hierarchy.
- [OpenCV 4.13.0](opencv.md): Rodrigues vector/matrix routes, precisions, and Jacobians.
- [PROJ 9.5.1](proj.md): supporting WGS84 geodetic/ECEF and topocentric probes.
- [CSPICE via SpiceyPy 8.3.0](spiceypy.md): pinned toolkit, quaternion,
  Euler, matrix and angular-velocity routines.
- [MuJoCo 3.17.0](mujoco.md): low-level quaternion, rate-frame, integration,
  and tangent-Jacobian APIs.
- [ROS 2 Jazzy / tf2 0.36.23](ros2-tf2.md): native named-frame chain,
  forward and reverse lookup, message fields, and QuatOps replay.

## Still planned

These remain unverified until a pinned native runtime or named loader is
executed and its outputs are frozen:

- Mathematical libraries: Eigen, Sophus.
- Runtime/message contracts: OpenXR.

See [transport captures](../transports/README.md) for glTF and OpenUSD native
captures and converter round trips. Wave 2's delivered OpenCV,
PROJ, VTK, and Trimesh captures ran in the packet's Linux environment; the new
Cesium and expanded Three.js loader captures were run locally on Windows. See
the linked target pages for runtime boundaries and separate SciPy checks.

Do not count two wrappers around the same implementation family as independent
mathematical evidence. Format validation, specification-derived expectations,
and loader execution are separate evidence types.
