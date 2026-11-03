# Quaternion interoperability evidence atlas

This page is the at-a-glance inventory of rotation conventions and auxiliary
captures collected for QuatOps. It links to the versioned engine, oracle, and
transport reports, and separates measured runtime behavior from independent
formula checks. The detailed reports and frozen fixtures remain the source of
record for individual observations.

## QuatOps interchange contract

The current canonical boundary is active Hamilton WXYZ in a declared
right-handed frame, rotating column vectors, with unit quaternion inputs.
Angles used to construct or inspect rotations are radians. A bare quaternion
does not identify its world or body frame, so conversions carry explicit
source-to-destination mappings. Algebra, coefficient order, coordinate frames,
and native API behavior are separate conversion layers.

For a shared orthogonal coordinate map `B`,
`R_target = B R_source B^T` and the active Hamilton vector part maps as
`v_target = det(B) B v_source`, with the scalar coefficient unchanged. For
different world and body maps, use
`R_target = B_world R_source B_body^T`; a quaternion result requires matching
determinants. See [coordinate mapping](engines/coordinate-mapping.md) and the
[CAP-300 converter matrix](engines/CAP-300.md).

## Pinned engine conventions

The following maps all originate from the explicitly declared canonical probe
frame `+X right, +Y forward, +Z up` (right-handed). They are evidence for the
listed runtime and frame pair; they are not universal engine or asset defaults.

| Target/build | Native quaternion storage | Captured Euler/API facts | Canonical probe map | Converter proof |
| --- | --- | --- | --- | --- |
| [Unreal 5.7.4](engines/unreal.md) | Hamilton XYZW | `FRotator` Pitch/Yaw/Roll degrees; native active model `Rz(yaw) Ry(-pitch) Rx(-roll)`; native world `+X forward, +Y right, +Z up`; FQuat product applies RHS first | `("y", "x", "z")`, determinant −1 | Canonical ↔ native and native vector action pass for float32/float64 |
| [Unity 6000.5.2f1](engines/unity.md) | Hamilton XYZW | `UnityEngine.Quaternion.Euler` takes degrees and applies Z, X, Y; Mathematics Euler constructors take radians and names specify application order | `("x", "z", "y")`, determinant −1 | Canonical ↔ native and native vector action pass; transfer to Godot 4.7 passes |
| [Godot 4.7 stable](engines/godot.md) | Hamilton XYZW | `Quaternion.from_euler` radians, intrinsic YXZ; `Basis` exposes six named orders; world forward is −Z | `("x", "z", "-y")`, determinant +1 | Canonical ↔ native and native vector action pass; transfer from Unity passes |
| Godot 4.8-dev1 | Hamilton XYZW | Same captured proof profile as 4.7; this is a development snapshot | `("x", "z", "-y")`, determinant +1 | Pass for the captured snapshot |
| [Blender 5.1.2](engines/blender.md) | Hamilton WXYZ | Radian Euler inputs, six orders; uppercase labels match SciPy lowercase extrinsic sequences; `euler_compat` improves trajectory continuity | Identity for this fixture’s assigned probe frame only | Canonical ↔ native and native vector action pass in the assigned frame |

The dedicated [engine proof matrix](engines/proof-matrix.md) compares native
capture categories. The [CAP-300 tests](../tests/test_canonical_interoperability.py)
also check native-created Euler quaternions in the reverse direction,
noncommuting composition, and the explicit Unity-to-Godot map
`diag(1, 1, -1)`. These tests replay frozen runtime outputs; they do not launch
the engines.

## External math and coordinate oracles

These runs provide independent convention or operation evidence. A captured
oracle does not become a QuatOps target adapter until bidirectional conversion
and native physical-vector equivalence are tested.

| Oracle/build | Captured convention or behavior | Evidence boundary |
| --- | --- | --- |
| [CesiumJS 1.124.0](oracles/cesium.md) | Local HPR uses `Rz(-heading) Ry(-pitch) Rx(roll)`; WGS84 ENU is kept separate from ECEF/model placement | Node native math/placement capture plus SciPy/geodesy checks; no importer or QuatOps adapter |
| [Three.js 0.187.0](oracles/threejs.md) | Quaternion XYZW; uppercase Euler orders match SciPy intrinsic orders; `Matrix4.elements` is column-major | Pinned Node native API, hierarchy, and loader capture; no general application frame adapter |
| [OpenCV 4.13.0](oracles/opencv.md) | Rodrigues rotation-vector/matrix APIs captured in binary32 and binary64 with Jacobians | Native Python binding capture and independent SciPy/finite-difference checks; not quaternion evidence |
| [PROJ 9.5.1](oracles/proj.md) | WGS84 geodetic/ECEF and topocentric frame probes | Native pyproj/PROJ coordinates; supporting geographic frame evidence, not a rotation library |
| [CSPICE / SpiceyPy 8.3.0](oracles/spiceypy.md) | Scalar-first quaternion routines; `qxq` matches Hamilton product; `eul2m` observed as transpose of the tested SciPy active intrinsic matrix; `qdq2av` depends on derivative multiplication side | Pinned CSPICE runtime plus formula checks; no universal aerospace frame direction inferred |
| [MuJoCo 3.17.0](oracles/mujoco.md) | WXYZ; active column-vector matrix; local-rate integration right-multiplies; world-rate derivative left-multiplies; Euler case mapping is API-specific | Native Windows wheel plus SciPy, explicit Hamilton, and finite-difference replay |
| Eigen, Sophus, ROS 2/tf2, OpenXR | No pinned capture in this workspace | Planned; no convention claim accepted |

## Rotation transports and loaders

File-format semantics, validator output, and named-loader behavior are tracked
separately. Loader results below are tied to the listed versions and test
assets.

| Transport / loader | Captured conventions | Evidence status |
| --- | --- | --- |
| [glTF 2.0](transports/gltf.md), [Three.js GLTFLoader 0.187.0](oracles/threejs.md) | TRS and matrix hierarchy; `STEP`; rotational `LINEAR` uses shortest SLERP; `CUBICSPLINE` uses time-scaled Hermite quaternion components followed by normalization | Twelve Wave 2 assets validated; 112 Three.js static/animation observations compared to an independent evaluator |
| glTF through [Blender 5.1.2](transports/gltf.md) | This importer maps source axes `X→X, Y→Z, Z→−Y`; captured LINEAR/CUBICSPLINE and STEP subframe behaviors differ from the glTF equations in specific probes | Native importer observations; frame result is asset/loader evidence, not Blender’s universal model-forward convention |
| glTF through [VTK 9.6.2 and Trimesh 4.11.1](transports/gltf.md) | VTK static and animation samples; Trimesh static hierarchy samples | Native library captures from the packet’s Linux environment, replayed from frozen outputs on Windows |
| Khronos glTF Validator 2.0.0-dev.3.10 | Structural/spec validation for original and twelve Wave 2 assets | All recorded reports have zero errors, warnings, infos, or hints; this is not a rotation-action oracle |
| [OpenUSD 26.11](transports/openusd.md) | `Gf` real-first quaternion coefficients; `Gf.Rotation` uses degrees; captured matrix/vector actions are row-vector; `xformOpOrder`, parent/local multiplication, and `resetXformStack` are recorded | Pinned OpenUSD runtime and saved stage; SciPy/matrix replay; no importer or QuatOps frame adapter |

## Proof layers and evidence labels

Use these layers to decide what a capture proves:

| Layer | Question answered | Current evidence |
| --- | --- | --- |
| Algebra | Hamilton or JPL? Product, conjugate, inverse, active/passive interpretation? | QuatOps scalar/JPL oracle suites; native engine/oracle product captures where listed |
| Storage | WXYZ or XYZW? Matrix row/column layout? Serialization order? | Named engine adapters; native fixture metadata; Three.js and glTF records; OpenUSD `Gf` records |
| Frames | Which coordinates map to which, and are world/body maps the same? | Explicit `SignedAxisMap`/`QuaternionFrameMap`; per-capture frame descriptions; Cesium ENU/ECEF and PROJ support |
| Native action | Does the native API rotate basis and mixed vectors as predicted? | Unreal, Unity, Godot, Blender CAP-300 captures; target-specific oracle observations |
| Composition | Which operand acts first? How do hierarchy and transform stacks compose? | Engine native product/transform captures; Cesium model-to-ECEF; Three.js Object3D; OpenUSD stage |
| Interpolation | Which path, sign policy, and animation sampler are used? | Native SLERP captures; glTF STEP/LINEAR/CUBICSPLINE loader probes; Blender Euler trajectories |
| Numerical behavior | Precision, normalization fallbacks, branch edges, Jacobians? | Versioned engine/oracle fixtures, split by dtype and native API; never generalized across versions without evidence |
| End-to-end conversion | Do both directions preserve a physical orientation through the target? | CAP-300 pass for Unreal, Unity, Godot 4.7/4.8-dev1, Blender in declared probe frames; pending for external oracles/transports |

### What remains unproven

- Asset-specific model-forward axes and importer settings.
- QuatOps bidirectional adapters and native-vector acceptance for Cesium,
  Three.js, OpenCV, PROJ, CSPICE, MuJoCo, and OpenUSD.
- End-to-end glTF exporter/importer parity across all loaders, compression
  extensions, animation mixers, and applications.
- ROS 2/tf2, OpenXR, Eigen, and Sophus pinned captures.
- Nonuniform-scale shear, renderer/GPU behavior, and animation runtime policies
  beyond the recorded probes.

## Where to inspect the evidence

- [Engine API facts](engines/README.md) and [native proof matrix](engines/proof-matrix.md).
- [Canonical converter acceptance](engines/CAP-300.md) and
  [SciPy/native verification separation](engines/verification.md).
- [External oracle index](oracles/README.md) and [transport index](transports/README.md).
- Auxiliary capture projects and fixture locations are linked from each target
  page; all capture code and data remain outside `src/`.
