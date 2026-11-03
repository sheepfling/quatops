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
of the original discovery capture.

The native capture programs call each engine's vector action independently of
QuatOps. The compact selector retains those values without recomputing them.
The package tests then compare the physical action with an independently built
SciPy basis matrix. SciPy formula checks and execution in pinned engine
runtimes are reported separately.

The shared tests generate all 48 signed-axis maps in memory, including both
determinant parities, noncommuting composition, inverse conversion, and
distinct world/body corrections. Passing generated outputs are not stored as
native goldens.

The Three.js, MuJoCo, CSPICE/SpiceyPy, and OpenUSD files under
[`tests/golden`](../tests/golden/) now retain small operation-specific witness
sets. Their provenance preserves native build details and the SHA-256 digest of
the full discovery capture used to select them. Capture exporters remain
available for replay. These witnesses are math or transform-stack oracles;
they do not imply production adapters under `src/`. Separate auxiliary
converter proofs now exercise QuatOps-generated values through native Cesium,
Three.js GLTFLoader, OpenUSD, and ROS 2/tf2 boundaries.

Run the converter acceptance set with:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_canonical_interoperability.py tests\test_engine_basis_oracle.py tests\test_engine_proof_profiles.py tests\test_ros_tf2_converter.py tests\test_cesium_converter.py tests\test_gltf_converter_roundtrip.py tests\test_openusd_converter_roundtrip.py
```

## Converter-boundary evidence and remaining scope

| Target | Captured converter proof | Remaining scope not claimed |
| --- | --- | --- |
| [CesiumJS 1.124.0](oracles/cesium.md) | QuatOps local-ENU orientations placed at selected WGS84 origins with native ENU-to-ECEF transforms, point actions, and inverse recovery; 9 selected witnesses | Other local frames, asset axes, rendering, and time-dependent Earth orientation |
| [glTF 2.0 / Three.js GLTFLoader 0.187.0](transports/gltf.md) | QuatOps-generated static `.gltf` hierarchy loaded by the pinned native loader; local/world actions, hierarchy composition, inverse, and file hashes | Animation export/import is outside this converter proof; other loaders retain separate behavior evidence |
| [OpenUSD 26.11](transports/openusd.md) | QuatOps-generated USDA reopened by pinned OpenUSD; orient ops, parent/child composition, reset stack, and native point/direction actions | Time-sampled ops, scale/shear decomposition, renderer/importer behavior, and asset axes |
| [ROS 2 Jazzy / tf2 0.36.23](oracles/ros2-tf2.md) | Native `BufferCore` three-frame chain, both lookup directions, transformed points, `TransformStamped` fields, and `fromMsg` round trip | DDS/CDR wire transport, live network, time interpolation/extrapolation, and dynamic broadcasters |
| CSPICE, MuJoCo, OpenCV, PROJ | Independent operation-specific runtime/math captures | No named QuatOps converter is accepted |
| OpenXR, Eigen, Sophus | No pinned capture in this workspace | Not required for the current converter scope |

These are auxiliary proofs and fixtures, not new runtime APIs under `src/`.
The native side was executed in each pinned target runtime; Python replay tests
separately use QuatOps and SciPy formulas to check the captured actions.
Cesium's existing 171-observation discovery capture, glTF's broader multi-loader
captures, and OpenCV/PROJ sweeps remain available as discovery evidence; the
converter acceptance fixtures retain only selected observations.

The compact converter suites are `tests/test_ros_tf2_converter.py`,
`tests/test_cesium_converter.py`, `tests/test_gltf_converter_roundtrip.py`,
and `tests/test_openusd_converter_roundtrip.py`. Their native manifests and
exact selected outputs live under `tests/golden/`; passing sweep outputs are
not copied into those witness sets.

## Evidence policy

- Contracts specify API, algebra, storage, action, frames, conversion and
  inverse, precision/input domain, and exclusions.
- Native witnesses store only selected exact values and native basis actions.
- Capture exporters and environment/build records make selected values
  reproducible; hashes identify inputs but never replace the values.
- Random and broad boundary tests stay generated. Promote a minimized native
  rerun to permanent evidence only when it reveals a distinct contract or
  regression.
- Euler, interpolation, scene transforms, geographic placement, and messages
  receive separate contracts only when QuatOps exposes those features.
