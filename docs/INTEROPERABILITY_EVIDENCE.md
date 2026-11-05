# Quaternion interoperability evidence atlas

This atlas separates shared mathematics, target-specific contracts, native
physical-action evidence, and transport behavior. A native convention capture
does not by itself validate a QuatOps adapter.

## Canonical converter contract

Canonical values are active Hamilton WXYZ unit quaternions in an explicit
right-handed source frame, acting on column vectors. A caller supplies the
source and destination world/body maps. For one shared orthogonal basis map
`B`, `R_T = B R_C Bᵀ` and the Hamilton vector part transforms as
`v_T = det(B) B v_C`. Storage order and algebra remain separate from frame
conversion. Different world and body maps use
`R_T = B_world R_C B_bodyᵀ` and require equal determinant parity to yield a
proper rotation.

These are the captured, version-specific quaternion rules for the declared
canonical frame `+X right, +Y forward, +Z up`:

| Target build | Native interface/order | Declared map | Canonical `(w,x,y,z)` to native values |
| --- | --- | --- | --- |
| [Unreal 5.7.4](engines/unreal.md) | `TQuat<T>` / XYZW; float32 and float64 | `(+Y,+X,+Z)`, determinant −1 | `(-y,-x,-z,w)` |
| [Unity 6000.5.2f1](engines/unity.md) | `UnityEngine.Quaternion` / XYZW; float32 | `(+X,+Z,+Y)`, determinant −1 | `(-x,-z,-y,w)` |
| [Godot 4.7 stable and 4.8-dev1](engines/godot.md) | `Quaternion` / XYZW; float32 | `(+X,+Z,-Y)`, determinant +1 | `(x,z,-y,w)` |
| [Blender 5.1.2](engines/blender.md) | `mathutils.Quaternion` / WXYZ | Identity in the assigned probe frame | `(w,x,y,z)` |

These maps are specific to the source and target frames recorded in each
versioned [contract and witness set](engines/proof-matrix.md). They are not
universal engine world-frame or asset-import defaults.

## Retained evidence

Each game-engine witness set retains eight canonical inputs: identity, three
positive quarter-turns, two unequal mixed rotations, one antipode, and a
nonprincipal half-turn. For each, it stores the exact native quaternion and
native images of +X, +Y, +Z and an asymmetric probe. Both orders of the
noncommuting mixed product are included. The selected values are preserved in
[`tests/oracles`](../tests/oracles/) with engine/build identity, native
precision, explicit frame contract, capture-source/input hashes, and the hash
of the exact native capture output.

The native capture programs call each engine's vector action independently of
QuatOps. The compact selector retains those values without recomputing them.
Godot 4.7 and 4.8 have separate version contracts and capture hashes; their
identical selected native values share one immutable witness file.
The package tests then compare the physical action with an independently built
SciPy basis matrix. SciPy formula checks and execution in pinned engine
runtimes are reported separately.

The shared tests generate all 48 signed-axis maps in memory, including both
determinant parities, noncommuting composition, inverse conversion, and
distinct world/body corrections. Passing generated outputs are not stored as
native goldens.

The retained native outputs under [`tests/golden`](../tests/golden/) are the
selected witnesses for accepted converter claims. They preserve native build
details and the SHA-256 digest of the capture run. Broad standalone math API
sweeps for Euler conversion, integration, derivatives, and perturbations are
outside this converter evidence set. OpenUSD evidence is limited to its
converter stage proof.

The capture exporters and retained witnesses remain auxiliary evidence outside
`src/`. The common field-level converter profiles under `src/quatops/converters`
map explicitly labelled orientation values into the declared target layouts;
they do not perform target SDK object construction or complete asset, message,
geographic, or scene serialization. Separate native proofs compare canonical
values and frame mappings against Cesium, Three.js GLTFLoader, OpenUSD, ROS
2/tf2, CSPICE direct quaternion APIs, MuJoCo direct quaternion APIs, OpenCV
Rodrigues, Eigen's native C++ quaternion APIs, and the OpenXR test-runtime
contract.

The [converter guide](converters.md) indexes every profile with its acceptance
state. OpenXR has a declared field converter but no vendor-runtime acceptance;
Sophus remains a mathematical oracle, and PROJ remains frame-transform
evidence. The generated [converter registry](converter-registry.md) is the
single target/API/version/layout/state index.

Run the converter acceptance set with:

```shell
uv run --extra dev python -m pytest -q \
  tests/quatops/converters \
  tests/quatops/engines \
  tests/experiments/engines \
  tests/experiments/converters
```

## Converter-boundary evidence and remaining scope

| Target | Captured converter proof | Remaining scope not claimed |
| --- | --- | --- |
| [CesiumJS 1.124.0](oracles/cesium.md) | QuatOps local-ENU orientations placed at selected WGS84 origins with native ENU-to-ECEF transforms, point actions, and inverse recovery; 5 selected witnesses | Other local frames, asset axes, rendering, and time-dependent Earth orientation |
| [glTF 2.0 / Three.js GLTFLoader 0.187.0](transports/gltf.md) | QuatOps-generated static `.gltf` hierarchy loaded by the pinned native loader; local/world actions, hierarchy composition, inverse, and file hashes | Animation export/import is outside this converter proof; other loaders retain separate behavior evidence |
| glTF 2.0 loader observations through Blender 5.1.2, VTK, and Trimesh | Native loader captures for authored assets | No QuatOps-generated round-trip acceptance claim for these loader routes |
| Khronos glTF Validator 2.0.0-dev.3.10 | Native file/spec validation | Validation only; not a rotation-action oracle |
| [OpenUSD 26.11](transports/openusd.md) | QuatOps-generated USDA reopened by pinned OpenUSD; orient ops, parent/child composition, reset stack, and native point/direction actions | Time-sampled ops, scale/shear decomposition, renderer/importer behavior, and asset axes |
| [ROS 2 Jazzy / tf2 0.36.23](oracles/ros2-tf2.md) | Native `BufferCore` three-frame chain, both lookup directions, transformed points, `TransformStamped` fields, and `fromMsg` round trip | DDS/CDR wire transport, live network, time interpolation/extrapolation, and dynamic broadcasters |
| [SpiceyPy 8.3.0 / CSPICE_N0067](oracles/spiceypy.md) | Eight selected canonical inputs, native `q2m`/`mxv` actions, `m2q` round trip, native `axisar` matrix to `m2q` quaternion route, and both `qxq` product orders | `eul2m`/Euler frame direction, named SPICE frames, kernels, and spacecraft frame semantics |
| [MuJoCo 3.17.0](oracles/mujoco.md) | Eight selected canonical inputs, native matrix/vector actions, `mju_mat2Quat` round trip, native axis-angle construction, and both `mju_mulQuat` product orders | Model, body, joint, camera, robot link, simulation, and viewer frame assignments |
| [OpenCV 4.13.0](oracles/opencv.md) | Native `cv2.Rodrigues` plus `cv2.gemm` basis/probe actions, reverse rotvec conversion, noncommuting matrix composition; float32 and float64 | C++ `cv::Quat`, camera/world frames, and Jacobian adapters |
| [Eigen 5.0.1](oracles/eigen.md) | Native C++ `Quaternionf`/`Quaterniond` WXYZ construction, XYZW coeffs, vector action, matrix conversion, `AngleAxis`, and both product orders | Application-specific frames, Sophus tangent semantics, serialization, and interpolation |
| PROJ 9.5.1 | 11 native WGS84 probes, second Windows run, analytic ENU/ECEF basis and SciPy matrix review | Supporting frame evidence only; no PROJ quaternion or Cesium runtime claim |
| [Sophus 1.24.6](oracles/sophus.md) | Selected native `SO3f` and `SO3d` witnesses for constructor normalization/order, antipodes, exp/log, action, inverse, noncommuting products, and perturbation side; separate SciPy 1.18.1 checks pass | Supporting math evidence only; no named converter, Jacobian, or tangent-derivative claim |
| [OpenXR 1.1.65 / Khronos test runtime](oracles/openxr.md) | Reviewed specification pose/frame contract, exact SDK header ABI, and a native static-space `xrLocateSpace` chain using the pinned test runtime | Static test-runtime relation and pose action captured; no vendor runtime, tracking, or hardware behavior claimed |

These are auxiliary proofs and fixtures, not new runtime APIs under `src/`.
Native library/runtime execution, SDK-header ABI capture, and SciPy formula
checks are recorded as separate evidence kinds. Python replay tests separately
use QuatOps and SciPy formulas to check accepted captured actions. Retained
fixtures contain selected native witnesses and distinct known discrepancies;
passing broad sweeps are not included in the product evidence set.

The compact converter suites are grouped by target under
`tests/experiments/converters/`: ROS 2/tf2, Cesium, glTF, OpenUSD, SpiceyPy,
MuJoCo, OpenCV, and Eigen. Separate loader observations for Blender, VTK, and
Trimesh and format validation by Khronos glTF Validator remain discovery
evidence without converter acceptance claims. Accepted converter manifests and selected outputs
live under `tests/golden/`; selected Sophus discovery witnesses and the
OpenXR SDK and test-runtime captures remain under `experiments/oracles/`.
Sophus raw sweeps remain temporary. The OpenXR capture exercises the Khronos
test runtime only; it does not represent a vendor runtime or tracked hardware.
Neither Sophus nor OpenXR is a QuatOps converter acceptance claim.

## Evidence policy

- Contracts specify API, algebra, storage, action, frames, conversion and
  inverse, precision/input domain, and exclusions.
- Native witnesses store only selected exact values and native basis actions.
- Numeric goldens store round-trippable decimal values and native dtype; exact
  IEEE bits remain only where signed zero or nonfinite values require them.
- Capture exporters and environment/build records make selected values
  reproducible; hashes identify inputs but never replace the values.
- Random and broad boundary tests stay generated. Promote a minimized native
  rerun to permanent evidence only when it reveals a distinct contract or
  regression.
- Euler, interpolation, scene transforms, geographic placement, and messages
  receive separate contracts only when QuatOps exposes those features.
