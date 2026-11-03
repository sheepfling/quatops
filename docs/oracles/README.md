# External math and transport oracles

This index is for native runtimes and file/pose transports used as auxiliary
evidence. Their exporters, pins, raw captures and replay tests stay outside
`src/`; none becomes a QuatOps runtime dependency. Read the target page for
what its capture actually establishes.

For a cross-target summary of measured conventions and evidence status, see the
[interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md).

## Captured

- [CesiumJS 1.124.0](cesium.md): local-ENU quaternion conversion and native WGS84 ENU/ECEF placement.
- [Three.js 0.187.0](threejs.md): Node.js math API and Object3D hierarchy.
- [OpenCV 4.13.0](opencv.md): Rodrigues vector/matrix converter boundary in float32 and float64.
- [Eigen 5.0.1](eigen.md): native Quaternionf/Quaterniond storage, action, matrix, and composition boundary.
- [PROJ 9.5.1](proj.md): supporting WGS84 geodetic/ECEF and topocentric probes.
- [CSPICE via SpiceyPy 8.3.0](spiceypy.md): pinned toolkit, quaternion,
  Euler, matrix and angular-velocity routines.
- [MuJoCo 3.17.0](mujoco.md): low-level quaternion, rate-frame, integration,
  and tangent-Jacobian APIs.
- [ROS 2 Jazzy / tf2 0.36.23](ros2-tf2.md): native named-frame chain,
  forward and reverse lookup, message fields, and QuatOps replay.

## Discovery started

- [Sophus 1.24.6](sophus.md): native `SO3d` construction, exp/log, actions,
  composition, and perturbation observations; first discovery set is not an
  accepted full contract.
- [OpenXR 1.1.65](openxr.md): native SDK-header ABI only. No OpenXR runtime
  behavior has been captured yet.

## Still planned

These behaviors remain unverified until a suitable runtime is available and
executed:

- OpenXR `xrLocateSpace`, reference-space relations, and native pose transfer.
- Broader Sophus scalar, Jacobian, and derivative contracts.

See [transport captures](../transports/README.md) for glTF captures and
[converter evidence](../transports/openusd.md) for the OpenUSD stage round trip.
The retained evidence atlas distinguishes
captured target behavior, selected converter witnesses, and separately run
SciPy formula checks. Broader successful discovery sweeps are not shipped as
permanent goldens.

Do not count two wrappers around the same implementation family as independent
mathematical evidence. Format validation, specification-derived expectations,
and loader execution are separate evidence types.
