# QuatOps frame-semantics audit

Snapshot reviewed: `138b730db2a6512b5adde2b031e9f0fa00a870be` from the uploaded `quatops-evidence-review-138b730.zip`.
Audit date: 2026-11-06.

## Conclusion

The concern is justified, but it is not evidence that the quaternion mathematics or all captures are
invalid. The bundle contains (1) genuine documented API conventions, (2) application/fixture frame
assignments, and (3) native tests that establish numerical action under those assignments. Those categories
must not be promoted into one another.

The production gateway does not appear to invent a target basis from the target name: it explicitly takes
source and target parent/child frames. The strongest concrete problem found is the glTF contract's
`asset_frame` label: it gives the camera-like +X-right/+Y-up/-Z-forward convention, while glTF distinguishes
asset front (+Z, with -X to the asset's right) from camera viewing direction (-Z, with +X to the camera's
right). The current fixture is a valid arbitrary declared-basis transform test, not proof of glTF
asset-front conversion.

The appropriate repair is primarily to tighten claims and identify frame roles, not to discard captures or
globally change every glTF quaternion swizzle.

## Scope and checks actually performed

- Read the bundle's review map, manifest, convention/semantic code, converter gateway and profile layer,
  target contracts, selected capture generators, replay tests, and target documentation.
- Verified every one of the 305 repository files listed by the bundle manifest against both its byte size
  and SHA-256. No mismatches.
- Independently replayed the two retained static glTF observations with NumPy and SciPy 1.17.0, without
  importing QuatOps. Maximum absolute differences: local basis actions 2.22e-16; composed world basis action
  5.55e-16; composed point action 1.78e-15.
- Confirmed the accepted static glTF asset contains two nodes, zero meshes, and zero cameras.
- Checked relevant primary documentation, especially Khronos glTF sections 3.4 and 3.10.2, Godot
  model-export conventions, OpenUSD stage/camera distinctions, REP-103, and Cesium ENU transforms.
- Did not relaunch native engines, reproduce their capture runs, or run the full QuatOps test suite. This is
  a selected-code/contract audit and a focused frozen-output replay, not a release-wide execution audit.
  Manifest hashes establish internal integrity, not independent authenticity of the original capture.

Machine-readable results are in `quatops_frame_audit_checks_138b730.json`.

## Finding 1 — glTF conflates a declared camera-like probe basis with asset-front semantics

### Evidence in the snapshot

- `experiments/converters/gltf/2.0.0/create_asset.py:30–33` explicitly chooses `SignedAxisMap(("x", "z",
  "-y"))` and calls it a test-frame declaration.
- The same file, lines 64–74, records +X right, +Y up, +Z backward, and forward -Z.
- `tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/contract.json:13–20` puts that declaration
  under `asset_frame` and uses the same map.
- `tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py:24–26,49–50,60–63,114–168` fixes that
  same map in the comparison and checks the native loader's numerical action against it.
- The asset construction at `create_asset.py:99–119` creates two transform-only nodes, not a camera or an oriented mesh.

### External distinction

Khronos glTF 2.0 section 3.4 identifies asset front with +Z, up with +Y, and the asset's right with -X.
Section 3.10.2 instead specifies camera-local right +X, up +Y, and viewing direction -Z. Those are different
roles in the same format. [S1]

The existing fixture's numbers can be correct, and the file can be valid, without validating the semantic
claim that an asset's front is -Z. The generator already narrows its intent to a chosen probe frame; the
contract's `asset_frame` terminology is the misleading part.

### Required correction

Retain the immutable observations. Describe the old case as a caller-assigned camera-like probe basis, not a
discovery of glTF asset axes. Do not silently rewrite the old capture's physical meaning or replace its hash
without a new provenance revision.

When an actual asset-front conversion is offered, define its source/target body roles independently from
world roles. Add one small role-specific witness: an oriented asymmetric object or explicit front/right/up
landmarks, with expected role directions anchored to the specification. A camera witness should instead
inspect the camera's documented forward direction.

### Map illustration, not a universal replacement

From an RFU source coordinate basis:

```text
camera-like right/up/back target: (x, z, -y)
glTF asset right/front/up identification: (-x, z, y)
```

The first sends source forward `(0,1,0)` to -Z; the second sends it to +Z. Their relative map is
`diag(-1,1,-1)`, a half-turn about Y. Both are proper orthogonal maps. Which belongs on the parent side or
child side is an application-role decision. Do not apply the second to every `node.rotation` merely because
glTF has an asset-front convention.

## Finding 2 — neutral numeric interfaces have explicitly assigned probe frames

The following contracts correctly acknowledge that their RFU frame is a fixture assignment:

- Eigen: `tests/golden/oracles/eigen/5.0.1/converter/contract.json`,
  `target_boundary.probe_frame_assignment` explicitly says there is no selected application/world frame.
- MuJoCo: `tests/golden/oracles/mujoco/3.17.0/converter/contract.json`, the same field excludes
  model/simulator semantics.
- OpenCV: `tests/golden/oracles/opencv/4.13.0/converter/contract.json` excludes camera/world interpretation
  for its Rodrigues matrix boundary.
- CSPICE: `tests/golden/oracles/spiceypy/8.3.0/converter/contract.json:23–50` distinguishes its numeric
  `q2m`/`qxq` interface from named SPICE frames and mission data.
- Sophus: `docs/oracles/sophus.md:3–6` expressly identifies RFU as a probe, not a universal Sophus frame.
- Blender: `tests/oracles/blender/5.1.2/contract.json:18–41` assigns the probe frame and excludes
  imported-object forward axes. `experiments/blender/5.1.2/capture.py:114–147` calls quaternion/vector
  arithmetic directly.

This is legitimate test setup, not a discovered native frame. Eigen's quaternion constructor and coefficient
order are real interface facts [S2]; no application-axis selection follows from that constructor. CSPICE's
documented quaternion convention is real [S3]; the names of spacecraft and reference frames are a separate
input to other APIs. OpenCV Rodrigues conversion is not equivalent to accepting arbitrary camera-pose
semantics: `solvePnP` explicitly produces object/world-to-camera transforms [S4].

Keep these numeric boundaries accepted where their evidence supports that, but label their application
frames as caller-supplied/unspecified. Do not present them as independent discoveries that six ecosystems
use RFU.

## Finding 3 — world convention, object convention, camera convention, and a specific frame are not interchangeable

- Unreal and Unity document world-coordinate directions; those are not fabricated. [S5, S6] A
  TQuat/Quaternion value or imported mesh still does not independently supply calibrated body axes or a
  mounting correction.
- Godot explicitly distinguishes camera forward -Z from oriented-asset front +Z and provides model-front
  options. [S7] Its existing `same declared signed-axis map` world/body fixture must not be generalized to
  every character mesh.
- Three.js exposes a generic quaternion type and a local `Object3D.quaternion`; Camera overrides the
  ordinary object direction operation because its viewing direction is -Z. [S8, S9] A target label
  `threejs.quaternion` does not identify which consumer role is intended.
- OpenUSD lets stages select Y-up or Z-up; camera axes have their own fixed convention independent of stage
  up-axis. [S10, S11] `GfQuatd` by itself supplies neither a stage nor an asset-forward identity.

The OpenUSD stage in this bundle explicitly declares Y-up and a chosen right/up/back probe.
`experiments/converters/openusd/26.11/create_stage.py:40–41,69–83` and
`tests/golden/transports/openusd/26.11/converter/contract.json:9–20` support that scoped test. Keep the
assignment; do not infer that a generic Gf quaternion uniquely selects it or that `upAxis=Y` calibrates
every model's front.

## Finding 4 — ROS and Cesium provide real semantic anchors, but only at specific APIs

### ROS

REP-103 documents body FLU, optical RDF, and geographic ENU conventions, while explicitly allowing justified
documented exceptions. [S12] Those conventions do not determine the pose of a camera mounted on a particular
robot.

`tests/golden/oracles/ros2/0.36.23/contract.json` says its world/base/optical maps are fixture declarations,
not inferred from frame IDs. `docs/oracles/ros2-tf2.md:23–38` preserves that distinction. Keep the native
BufferCore frame-direction and message-field evidence. Do not say a bare quaternion message or the string
`sensor_optical` establishes the physical mounting.

### Cesium

`Transforms.eastNorthUpToFixedFrame` really specifies local east, north, and ellipsoid-normal axes at an
origin and ellipsoid. [S13] That is stronger semantic evidence than an arbitrary numeric probe.
`Cesium.Quaternion` itself is only a four-component rotation type. [S14]

`tests/golden/transports/cesium/1.124.0/contract.json:6–12` pairs east/right and north/forward.
`docs/oracles/cesium.md:5–11` does say this is an assigned canonical frame. Preserve that qualification:
right=east and forward=north is the fixture's alignment, not a general identity between RFU and ENU.

## Finding 5 — the production architecture is mostly doing the right thing

`src/quatops/converters/gateway.py:34–73` derives frame maps from four explicit `CoordinateFrame` objects,
not from the target name. `src/quatops/converters/targets.py` handles representation/layout and acceptance
metadata; it does not contain a per-target physical-axis default. `src/quatops/semantics/frames.py:54–61`
explicitly states that the numeric basis is the source of truth and axis labels are descriptive.

These are important protections. No blanket rewrite of the conversion formulas is justified by this audit. A
caller can still supply an incorrect physical interpretation; numeric validation cannot determine where a
real object's nose or a sensor's calibrated axis points.

The `CANONICAL_WORLD_RFU` and `CANONICAL_BODY_RFU` presets in `semantics/conventions.py:101–116` are QuatOps
design choices. They must remain named opt-in conventions, not the implied meaning of every Hamilton WXYZ
quaternion. Algebra and coefficient order do not select north, up, or an object's front.

## Why the current numerical checks cannot discover physical labels

For an orthogonal coordinate map B, let

```text
q = (w, v)
q_B = (w, det(B) B v)
```

Then

```text
R(q_B) B p = B R(q) p.
```

This identity holds for any orthogonal B. A correct native Hamilton implementation will agree for both of
the role maps in Finding 1 when inputs and comparison vectors are mapped consistently.

Therefore these tests establish:

> Given this declared map, the converter and native API agree numerically.

They do not independently establish:

> This declared map is the target's physical world/body/asset convention.

This is not necessarily circular arithmetic: the native execution and SciPy calculation can be independent
and genuinely useful. It is a shared semantic assumption. Additional random quaternions cannot resolve it.

Likewise, q_parent_from_child is a useful directed coordinate mapping. Prefer that explicit meaning over
declaring an entire library active or passive. The numerical rotation operator may be used to rotate a
vector or to re-express components between frames; the consuming API identifies which transformation is
required.

## Recommended small correction plan

### 1. Separate acceptance of the numeric boundary from authority of the frame claim

Use two independent pieces of metadata, not one broad accepted flag:

```text
numeric_boundary: verified against <capture>
frame_meaning: target_documented | application_declared | fixture_only | unspecified
```

For a frame claim, retain its role, authoritative source section (when applicable), and caller-supplied
alignment/mounting assumptions. This can be a few fields in existing contracts, not a new framework.

Examples:

```text
Eigen constructor: numeric verified; application frame unspecified.
Blender mathutils: numeric verified; RFU fixture-only.
Cesium ENU transform: origin-specific axes target-documented; RFU-to-ENU alignment application-declared.
glTF current hierarchy: numeric/static-loader verified; camera-like probe fixture-only; asset-front claim not established.
```

### 2. Repair the glTF wording first

Keep the existing raw values and scoped static-loader proof. Rename/classify `asset_frame` as the selected
probe frame and document why its -Z forward is camera-like. Add a distinct spec-backed role test only when
advertising glTF asset-front conversion. Never substitute a new basis into old goldens without recording a
new capture/contract version.

### 3. Make shared world/body mapping visibly conditional

A test using the same B on both sides is legitimate. It does not imply that an arbitrary model and its scene
share that semantic convention. Use the existing two-map API when roles differ; no new quaternion
mathematics is needed.

### 4. Retain tiny semantic witnesses, not more sweeps

For an advertised physical role, require one independently anchored example such as a target-native camera
direction, a model-front marker defined by the format, a known geographic ENU landmark, or a supplied sensor
extrinsic. The expected semantic axis must not be obtained by applying the converter under test to both the
object and its expected answer.

For a generic math interface with no physical-axis promise, no semantic witness is required. State that its
frames are caller-supplied and stop there.

### 5. Tighten release language

Prefer:

> Verified quaternion/rotation representation boundaries with explicit application frame mappings and
> separately documented target-specific frame contracts.

Avoid a headline that implies all accepted profiles discovered a unique physical coordinate frame belonging
to their library.

## What not to do

Do not discard valid native observations, invent new defaults to replace ambiguous old ones, require every
math library to have a world frame, recapture thousands of rotations, or globally conjugate/swizzle all
glTF/USD values. The corrective action is claim and role specificity. A format's mathematical encoding and a
model's physical axes are different contracts.

## Primary sources consulted

- **S1:** Khronos glTF 2.0, sections 3.4, 3.5.3, and 3.10.2: <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>
- **S2:** Eigen 5 Quaternion reference: <https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1Quaternion.html>
- **S3:** NAIF CSPICE q2m: <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/q2m_c.html>
- **S4:** OpenCV 4.13 solvePnP coordinate mapping: <https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html>
- **S5:** Unreal 5.7 coordinate system and spaces:
  <https://dev.epicgames.com/documentation/en-us/unreal-engine/coordinate-system-and-spaces-in-unreal-engine?application_version=5.7>
- **S6:** Unity 6000.5 quaternion conventions:
  <https://docs.unity.com/en-us/engine/6000.5/manual/scripting/programming-math/unity-engine-math/class-quaternion>
- **S7:** Godot model export and camera/asset directions:
  <https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_3d_scenes/model_export_considerations.html>
- **S8:** Three.js Quaternion: <https://threejs.org/docs/pages/Quaternion.html>
- **S9:** Three.js Camera and Object3D: <https://threejs.org/docs/pages/Camera.html> and
  <https://threejs.org/docs/pages/Object3D.html>
- **S10:** OpenUSD stage up-axis: <https://openusd.org/release/api/group___usd_geom_up_axis__group.html>
- **S11:** OpenUSD camera: <https://openusd.org/release/api/class_usd_geom_camera.html>
- **S12:** ROS REP-103: <https://reps.openrobotics.org/rep-0103/>
- **S13:** Cesium transforms and ENU: <https://cesium.com/learn/cesiumjs/ref-doc/Transforms.html>
- **S14:** Cesium Quaternion: <https://cesium.com/learn/cesiumjs/ref-doc/Quaternion.html>

Official documentation establishes semantics at the named API/role. It does not recertify the old pinned
binary by itself. The existing native witnesses remain evidence only for their recorded executions.
