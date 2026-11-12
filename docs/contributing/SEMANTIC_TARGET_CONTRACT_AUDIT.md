# Semantic target and consumer contract audit

**Discovery snapshot:** 2026-11-11
**Scope:** registered physical frame roles, registered game-engine and consumer
profiles, and the evidence needed to justify a role-to-consumer contract.

## Current enforcement

The findings below preserve the discovery that motivated hardening. The
executable registry now uses `SemanticEvidenceClaim` links naming the profile,
role, capture version, operation, scope, and witness IDs. Repository assurance
validation checks those links against target-specific validated and normalized
captures, including provenance. The generated registry reports only those
operations/scopes. LOCAL is the sole native OpenXR space claim; VIEW and
LOCAL_FLOOR remain specification-backed. glTF loader evidence is assigned to
static node fields/hierarchy, and Godot probe action remains separate from
Camera3D execution. No unrelated native evidence needs recapture for these
claim corrections. See the [current registry](../converter-registry.md).

## Implementation status — 2026-11-12

The accepted converter and engine schemas now constrain interpretation
selectors including target identity/version, axis meanings and directions,
coefficient order, handedness, signed-axis maps, actions, composition order,
entry points, and transform operations. Cesium, glTF, and OpenUSD contracts use
resolved target fields rather than combined storage/frame prose. Their
serializers compare declared frames and axis maps with the shared canonical
corpus, and validate mapped probes, translations, and quaternion coefficients.
Native engine and direct-library basis/probe actions are checked against their
captured quaternion coefficients; OpenCV matrix actions and Rodrigues
roundtrips are checked against the declared rotations. Scene outputs retain
orientation-field and unit-scale information. Blender's probe-axis labels are
explicit fields in its semantic contract. Native observations remain unchanged;
contract digests are updated in the retained manifests. Formula, conversion,
and scope explanations remain prose.

## Findings at a glance

- `src/quatops/experimental/semantics/roles.py` declares 11 semantic roles:
  ten target- or standard-defined roles and one generic QuatOps
  parent-from-child contract.
- `docs/converter-registry.md` lists 16 orientation-converter profiles: 14
  accepted profiles, OpenXR as declared evidence, and Sophus as a supporting
  math oracle. The registry separately lists five frame-provider, format-
  validator, and loader-observation targets.
- The accepted engine/converter pipeline already validates native JSON
  against target-specific Pydantic models, runs target semantic serializers,
  and normalizes a closed union of concrete semantic products. JSON is the
  capture interchange format, not the accepted semantic type. OpenXR, Sophus,
  and PROJ supporting evidence use separate validators and remain outside that
  accepted-converter normalizer.
- The role-to-evidence coverage was initially weaker than the capture-to-
  normalizer link. Typed `SemanticEvidenceClaim` links now bind each role to its
  profile, operation, scope, witness IDs, and capture version. The remaining
  OpenXR VIEW/LOCAL_FLOOR and glTF asset-front limits are explicit status
  boundaries, not claims supported by unrelated runtime witnesses.

This is a repository discovery, not a new native capture or a test run.

## 1. Semantic physical-role inventory

“Authority” states where the frame meaning comes from. It does not establish
that a particular project, asset, robot, camera, or runtime instance is
physically aligned to that meaning. That alignment remains an explicit input.

| Role ID | Declared meaning | Role authority and required binding | Consumer profile and current native evidence | What the evidence does not establish |
| --- | --- | --- | --- | --- |
| `unreal.world-axes` | Unreal Editor world: +X forward, +Y right, +Z up; left-handed. | Unreal 5.7 documentation; `fixed_axes`, `scene_alignment`. | `unreal.fquat`; Unreal 5.7.4 native quaternion action. | Project-specific world overrides, a scene's alignment to another physical frame, or asset/model axes. |
| `unity.world-axes` | Unity world: +X right, +Y up, +Z forward; left-handed. | Unity 6000.5 documentation; `fixed_axes`, `scene_alignment`. | `unity.quaternion`; Unity 6000.5.2f1 native quaternion action. | Project scene alignment, imported-model axes, or geographic alignment. |
| `godot.camera-axes` | Camera3D local sensor frame: +X right, +Y up, +Z back (camera forward is −Z); right-handed. | Godot 4.7 documentation; `fixed_axes`, `sensor_mount`. | `godot.quaternion`; Godot 4.7 and 4.8-dev1 native quaternion action in the captured probe frame. | A Camera3D node action, arbitrary Node3D world alignment, asset-front, or camera mounting/calibration. The evidence catalog explicitly limits this to a numeric probe. |
| `ros.rep103.body-flu` | Conventional ROS body frame: +X forward, +Y left, +Z up; right-handed. | REP-103; `fixed_axes`, `body_alignment`. | `ros2.geometry-msgs-quaternion`; ROS 2/tf2 0.36.23 named-frame chain and message roundtrip. | Whether a particular robot conforms to REP-103 or its body frame is physically calibrated. |
| `ros.rep103.optical` | ROS camera optical frame: +X right, +Y down, +Z forward; right-handed. | REP-103 suffix-frame convention; `fixed_axes`, `sensor_mount`. | Same ROS 2/tf2 package as body-FLU. | Camera-to-body extrinsics, sensor installation, or compliance of a particular ROS graph. |
| `cesium.enu-to-fixed` | Local x=east, y=north, z=ellipsoid normal at the supplied origin, mapped into the selected fixed frame. | Cesium 1.124.0 API; `geographic_enu`; typed origin and ellipsoid required. | `cesium.quaternion`; role-specific native ENU/fixed actions at explicit geographic inputs. | Body-forward alignment, inertial-to-fixed conversion, time-dependent Earth orientation, or pose translation. |
| `gltf.asset-front` | glTF oriented-asset convention: −X right, +Y up, +Z forward; right-handed. | glTF 2.0 specification; `fixed_axes`, `model_alignment`. | `gltf.rotation`; the linked Three.js loader capture proves static field/loader behavior. | Whether a particular mesh's geometry was authored facing +Z. The linked artifact says it contains no oriented-mesh verification. |
| `gltf.node-local` | A glTF node transform is local to its parent; `node.rotation` is a unit XYZW quaternion. | glTF 2.0 specification; `relational` contract. | `gltf.rotation` and Three.js GLTFLoader 0.187.0; a static hierarchy fixture exercises local/world actions and composition. | General exporter/importer behavior, animation curves, arbitrary assets, or application model alignment. |
| `openxr.reference-space` | LOCAL and LOCAL_FLOOR: +X right, +Y up, −Z forward; runtime-defined origin. | OpenXR 1.1.65 specification; `fixed_axes`, `space_alignment`; runtime/session/space/time context required. | `openxr.xrquaternionf`; Khronos test-runtime capture currently exercises LOCAL static spaces only. Adapter conformance is not demonstrated. | LOCAL_FLOOR runtime behavior, vendor tracking, geodetic alignment, or a universal origin. |
| `openxr.view-space` | VIEW sensor frame: +X right, +Y up, −Z forward; viewer-relative and not gravity-aligned. | OpenXR 1.1.65 specification; `fixed_axes`, `space_alignment`; runtime/session/space/time context required. | `openxr.xrquaternionf` is declared, but its linked runtime artifact exercises LOCAL, not VIEW. | Any native VIEW-space relation or vendor/hardware tracking behavior. The current LOCAL witness is not evidence for VIEW. |
| `quatops.parent-child-orientation` | Active `q_parent_from_child`: maps child coordinates into parent coordinates. | Application-declared frame pair; both frame definitions and their alignment are supplied by the caller. | Common QuatOps semantic value; no target-specific profile or native evidence is assigned. | Automatic physical frame discovery, physical calibration, or target behavior. |

Primary source claims and exclusions are summarized in the
[semantic source audit](../semantic-source-audit.md). The executable contracts
and their typed runtime contexts live in
[`roles.py`](../../src/quatops/experimental/semantics/roles.py) and
[`bindings.py`](../../src/quatops/experimental/semantics/bindings.py).

## 2. Game-engine and consumer profile inventory

These are the 16 registered orientation-converter profiles. Multiple profile
IDs can target one product because they exercise different native boundaries.
“Accepted” refers only to the registry's captured scope; it does not grant a
physical frame meaning to a representation-only API.

| Consumer area | Profile ID | Native API and version | Registry state | Proof boundary |
| --- | --- | --- | --- | --- |
| Game engine | `unreal.fquat` | Unreal `TQuat<T>`, 5.7.4 | Accepted | Native quaternion action; float32 and float64. |
| Game engine | `unity.quaternion` | Unity `UnityEngine.Quaternion`, 6000.5.2f1 | Accepted | Native quaternion action; float32. |
| Game engine | `godot.quaternion` | Godot `Quaternion`, 4.7 / 4.8-dev1 | Accepted | Native quaternion action for declared probe maps; float32. |
| DCC / scene consumer | `blender.mathutils.quaternion` | Blender `mathutils.Quaternion`, 5.1.2 | Accepted | WXYZ field/action boundary in an assigned probe frame; no universal Blender or mesh axes. |
| Geographic consumer | `cesium.quaternion` | CesiumJS `Quaternion`, 1.124.0 | Accepted | Quaternion boundary; ENU/ECEF placement needs origin-specific frame context. |
| Scene format | `gltf.rotation` | glTF 2.0 `node.rotation` | Accepted | Static field conversion under the pinned loader proof. |
| Scene consumer | `threejs.quaternion` | Three.js `Quaternion`, 0.187.0 | Accepted | Quaternion field boundary; hierarchy serialization is a separate operation. |
| Scene consumer | `openusd.gfquatd` | OpenUSD `GfQuatd` orient, 26.11 | Accepted | Static orient-op field conversion. |
| Robotics / messages | `ros2.geometry-msgs-quaternion` | ROS 2 Jazzy / `geometry_msgs` 5.3.8 | Accepted | Quaternion field; frame IDs, time, and translation remain explicit. |
| Aerospace math API | `spiceypy.cspice-q` | SpiceyPy 8.3.0 / CSPICE N0067 | Accepted | Direct native quaternion boundary in the captured probe frame. |
| Simulation | `mujoco.quaternion` | MuJoCo `mju` quaternion APIs, 3.17.0 | Accepted | Direct API; model, body, joint, and camera frames remain caller supplied. |
| Vision math API | `opencv.rodrigues` | OpenCV Rodrigues, 4.13.0 | Accepted | Rotation-vector boundary in radians; camera extrinsics are outside scope. |
| General math API | `eigen.quaternion-constructor` | Eigen `Quaternion(w,x,y,z)`, 5.0.1 | Accepted | Constructor order boundary. |
| General math API | `eigen.quaternion-coeffs` | Eigen `Quaternion::coeffs()`, 5.0.1 | Accepted | Coefficient accessor order boundary. |
| XR API | `openxr.xrquaternionf` | OpenXR `XrQuaternionf` in `XrPosef`, 1.1.65 | Declared; not native-accepted | Specification plus Khronos test runtime; no vendor-runtime or QuatOps adapter acceptance. |
| Math oracle | `sophus.so3-params` | Sophus `SO3::params()`, 1.24.6 | Supporting oracle only | Mathematical representation evidence; not an accepted application-frame converter. |

Five additional registry entries are not orientation-converter profiles:
PROJ 9.5.1 (frame provider), Khronos glTF Validator (format validator), and
Blender 5.1.2, VTK 9.6.2, and Trimesh 4.11.1 glTF loader observations. They
must remain labeled as frame, validation, or loader evidence; none silently
becomes a quaternion-action acceptance claim. See the generated
[converter registry](../converter-registry.md) and
[interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md).

## 3. What a role-to-consumer contract must prove

A complete contract needs separate evidence for each of these claims. A single
quaternion golden cannot prove all of them.

| Claim layer | Information required | Evidence that can support it |
| --- | --- | --- |
| Semantic authority | Role identifier; authority kind; exact standard/product version and section; positive-axis meanings; handedness; role (world, body, sensor, or relation); scope and exclusions. | Pinned primary source and a reviewed claim record. For application-defined roles, the application coordinate contract. |
| Native representation | Exact API/class/constructor/accessor; product and version; Hamilton/JPL algebra; coefficient and storage order; active/passive action; vector and matrix convention; precision; normalization and sign behavior; multiplication order; inverse route. | Target-specific capture executed through the named API, with exact API/runtime identity. Field-order observations alone prove only representation. |
| Frame relation | Source and destination frame identities; directed transform meaning; positive-axis semantics for each frame; separate parent/world and child/body maps; determinant parity; any model-forward or sensor-local declaration. | Typed frame contract plus an independently derived mapping. For different parent and child bases, check `R_target = B_parent R_source B_child⁻¹`; a single shared-axis quaternion rule is valid only when the same map applies to both sides. |
| Contextual binding | The selected role instance and alignment source: application declaration, calibration, survey, geographic context, or runtime relation. Require context for the operation being claimed. | Typed `FrameAlignment` and role-specific context. Cesium ENU placement needs origin and ellipsoid; runtime OpenXR observations need their space/session/sample context. Directed ROS orientation needs frame identities, direction, and basis relationship. Time-dependent transforms additionally need sample context; full poses need translation; physical sensor alignment needs instance-specific extrinsics or calibration evidence. Timeless quaternion conversion does not require unrelated timestamps or translation. |
| Numeric/native behavior | Identity; unequal mixed-axis rotations; antipode; nonprincipal half-turn; basis-vector actions; asymmetric probe; both orders of a noncommuting product; inverse/roundtrip. Add target-specific constructors, hierarchy, message or space operations when those are part of the claim. | Native retained outputs and an independently constructed oracle. Report SciPy/formula checks separately from a pinned native runtime. |
| Consumer integration | Actual node/object/asset/message/space to which the orientation is attached; parent chain and local/world operation; loader/import settings; physical model/sensor alignment when claimed. | A minimal reproducible consumer fixture and native consumer execution. For mesh front, use asymmetric authored geometry or an authoritative asset declaration; a quaternion field test does not establish mesh semantics. |
| Reproducibility | Capture source and revision/hash; input and semantic-config hashes; binary/runtime/library versions and platform; output hash; selected witness IDs; tolerance by precision; rerun instructions. | Provenance manifest tied to the exact semantic config and output. Hashes identify artifacts but do not replace the captured values. |

The layers should remain separate in both types and reports. In the existing
R/S/B/N shorthand, R is representation evidence, S is semantic source
authority, B is explicit physical binding, and N is native execution evidence.
None upgrades another. A caller-declared or calibrated binding can be checked
for structural consistency, but only external evidence can establish that it
matches a real installation.

## 4. Evidence typing audit

The accepted engine/converter capture flow currently follows this boundary:

```text
target-native capture JSON
  -> recursive JSON value validation
  -> strict target-specific Pydantic capture/contract/input models
  -> target semantic serializer
  -> concrete member of SemanticSerialization
  -> shared normalizer
```

The converter loader selects a target-specific capture, contract, and input
model and rejects unknown fields. The semantic serializer registry is keyed by
the validated contract model type. The common output is a closed union whose
contract and witness family are linked. The native capture process therefore
does not need to import the repository normalizer; it exports evidence and its
experiment-owned config, while the repository validates and normalizes them.
See [semantic evidence serialization](EVIDENCE_SEMANTIC_SERIALIZATION.md),
[`converter_evidence.py`](../../scripts/converter_evidence.py), and
[`crystallized.py`](../../scripts/evidence_schema/crystallized.py).

The JSON boundary is not an untyped semantic object. Target-specific Pydantic
models reject unknown fields before serialization. The former typing gap
between semantic-role declarations and evidence artifacts is closed: links now
name claims and validation checks them against target evidence. Future role
links should continue to name the exact authority, representation, action,
binding, and instance-alignment claims they support.

Interpretation selectors now use target-specific literals or small unions,
including coefficient order, action, handedness, signed-axis maps, and
composition order. Engine API/version identities, precision profiles and
tolerance keys, native entry points, and converter identity details are checked
against their captures and manifests. Cesium, glTF, and OpenUSD storage/frame
meanings are validated as target-specific literals, and serializers no longer
recover them with regex parsing. Blender's probe-axis labels are explicit
fields in its semantic contract; the retained native witness is unchanged,
and its contract digest is updated in the retained manifest.

Explanatory formulas, conversion rules, and scope statements remain prose. They
describe the contract without selecting a serializer branch. Dynamic JSON
dictionaries in manifests and target metadata remain appropriate for extensible
provenance; values that change semantic interpretation belong in target-specific
validated fields before serialization.

## 5. Earlier findings and current disposition

### Finding 1 - evidence was not scoped to semantic claims

Resolved: `validate_semantic_assurance()` checks typed claim links against the
validated and normalized target evidence, including the named operation, scope,
capture version, and witness IDs. The generated registry reports the supported
claim scope.

### Finding 2 - OpenXR VIEW and LOCAL_FLOOR do not have native runtime witnesses

Resolved as a scope correction: the native test-runtime claim is LOCAL only.
VIEW and LOCAL_FLOOR remain specification-backed. A future claim about their
native runtime behavior requires matching captures and is outside this target
typing goal.

### Finding 3 - glTF asset-front was linked to a proof that excludes mesh-front

Resolved as a scope correction: the Three.js loader artifact supports static
node-local fields and hierarchy behavior. The glTF standard remains the
authority for the asset-front convention; no arbitrary mesh-front behavior is
claimed.

### Finding 4 - Godot camera axes are not a native Camera3D action claim

Current boundary: the Godot evidence proves quaternion action in the declared
numeric probe frame. Camera3D axes remain source-backed, and no native camera
node action is claimed. A Camera3D fixture is an optional evidence expansion,
not a blocker for the accepted probe-frame contract.

### Completion status

The current accepted target contracts type their interpretation selectors
before serialization and cross-check capture profiles, input coefficients,
target frames, units, identity, and the retained native vector actions in the
engine, direct-quaternion, and OpenCV profiles against those declarations. The
role-claim corrections are reflected in the generated
registry. No native recapture is needed for the closed findings above. Future
adapters must follow the same typed-contract rule; additional OpenXR, Godot, or
asset-specific captures remain optional until those stronger claims are
requested.

## 6. PB-MSI audit checkpoint - 2026-11-12

Inspected Git HEAD and live PR #9 head: `00f030dca5a9a20289141a5a9f5044ecf6c576c9`.
The checkout contains pre-existing uncommitted target-contract and assurance
hardening. This checkpoint validates that **working tree**, not a claim that
its changes have reached PR #9. The preceding implementation/disposition
sections describe local work. No native runtime was launched for this audit.

On PB-MSI, Python 3.15 source-tree evidence verification passed: 53 retained
JSON documents, eight converter packages, five engine-version packages, and
92 normalized witness records. The generated registry passed its freshness
and semantic-assurance checks. Godot versions share a witness file; package
and witness counts must not be read as independent runtime executions.

The existing `.venv` focused semantic, context-identity, and canonical engine
replay tests passed: **75 tests**. These are replay/contract checks, not new
Unreal, Unity, Godot, or other native captures. Full CI, Linux verification,
and installed-wheel checks were not rerun for this documentation audit.

### Traceable normalized-package matrix

The 16-profile inventory above includes two Eigen interfaces backed by one
capture, and glTF/Three.js interfaces backed by one capture. OpenXR and Sophus
remain outside this accepted normalizer. This table covers all 13 validated
engine-version/converter packages and links their exact contract, provenance,
and retained observations. Serializer paths are relative to
`scripts/evidence_schema/semantic_serializers/`; all outputs pass through the
existing `normalizer.normalize_capture`, not a parallel contract system.

| Target/version | Serializer; normalized product | Frames and authority | Order/action/composition | Witnesses and provenance |
| --- | --- | --- | --- | --- |
| Unreal Engine 5.7.4 | `engines.py`; `EngineQuaternionSerialization` | forward, right, up; left-handed; application_declared | xyzw; active; both noncommuting product orders | 14; [contract](../../experiments/unreal/5.7.4/semantic_contract.json), [provenance](../../tests/oracles/unreal/5.7.4/contract.json), [witnesses](../../tests/oracles/unreal/5.7.4/witnesses.json) |
| Unity Editor 6000.5.2f1 | `engines.py`; `EngineQuaternionSerialization` | right, up, forward; left-handed; application_declared | xyzw; active; both noncommuting product orders | 7; [contract](../../experiments/unity/6000.5.2f1/semantic_contract.json), [provenance](../../tests/oracles/unity/6000.5.2f1/contract.json), [witnesses](../../tests/oracles/unity/6000.5.2f1/witnesses.json) |
| Godot Engine 4.7-stable | `engines.py`; `EngineQuaternionSerialization` | right, up, backward; right-handed; application_declared | xyzw; active; both noncommuting product orders | 7; [contract](../../experiments/godot/semantic_contract.json), [provenance](../../tests/oracles/godot/4.7-stable/contract.json), [witnesses](../../tests/oracles/godot/shared/witnesses.json) |
| Godot Engine 4.8-dev1 | `engines.py`; `EngineQuaternionSerialization` | right, up, backward; right-handed; application_declared | xyzw; active; both noncommuting product orders | 7; [contract](../../experiments/godot/semantic_contract.json), [provenance](../../tests/oracles/godot/4.8-dev1/contract.json), [witnesses](../../tests/oracles/godot/shared/witnesses.json) |
| Blender 5.1.2 | `engines.py`; `EngineQuaternionSerialization` | right, forward, up; right-handed; fixture_only | wxyz; active; both noncommuting product orders | 7; [contract](../../experiments/blender/5.1.2/semantic_contract.json), [provenance](../../tests/oracles/blender/5.1.2/contract.json), [witnesses](../../tests/oracles/blender/5.1.2/witnesses.json) |
| Eigen 5.0.1 | `eigen.py`; `NativeQuaternionApiSerialization` | right, forward, up; right-handed; fixture_only | wxyz; storage xyzw; active; both noncommuting product orders | 16; [contract](../../experiments/converters/eigen/5.0.1/semantic_contract.json), [provenance](../../tests/golden/oracles/eigen/5.0.1/converter/manifest.json), [witnesses](../../tests/golden/oracles/eigen/5.0.1/converter/witnesses.json) |
| MuJoCo 3.17.0 | `direct_quaternion.py`; `NativeQuaternionApiSerialization` | right, forward, up; right-handed; fixture_only | wxyz; active; both noncommuting product orders | 8; [contract](../../experiments/converters/mujoco/3.17.0/semantic_contract.json), [provenance](../../tests/golden/oracles/mujoco/3.17.0/converter/manifest.json), [witnesses](../../tests/golden/oracles/mujoco/3.17.0/converter/witnesses.json) |
| OpenCV 4.13.0 | `opencv.py`; `RotationVectorSerialization` | right, forward, up; right-handed; fixture_only | radians; active; both noncommuting product orders | 14; [contract](../../experiments/converters/opencv/4.13.0/semantic_contract.json), [provenance](../../tests/golden/oracles/opencv/4.13.0/converter/manifest.json), [witnesses](../../tests/golden/oracles/opencv/4.13.0/converter/witnesses.json) |
| ROS 2 tf2 0.36.23 | `ros2.py`; `FrameChainSerialization` | world/base/sensor: explicit fixture-only maps; no mounting inferred | xyzw; target-from-source; two-edge chain and inverse | 1; [contract](../../experiments/converters/ros2/0.36.23/semantic_contract.json), [provenance](../../tests/golden/oracles/ros2/0.36.23/manifest.json), [witnesses](../../tests/golden/oracles/ros2/0.36.23/witnesses.json) |
| SpiceyPy / CSPICE 8.3.0 | `direct_quaternion.py`; `NativeQuaternionApiSerialization` | right, forward, up; right-handed; fixture_only | wxyz; active; both noncommuting product orders | 8; [contract](../../experiments/converters/spiceypy/8.3.0/semantic_contract.json), [provenance](../../tests/golden/oracles/spiceypy/8.3.0/converter/manifest.json), [witnesses](../../tests/golden/oracles/spiceypy/8.3.0/converter/witnesses.json) |
| CesiumJS 1.124.0 | `cesium.py`; `GeographicSerialization` | east, north, ellipsoid normal; right-handed; target_documented; local alignment application-declared | xyzw; ECEF-from-ENU * ENU-from-model | 5; [contract](../../experiments/converters/cesium/1.124.0/semantic_contract.json), [provenance](../../tests/golden/transports/cesium/1.124.0/manifest.json), [witnesses](../../tests/golden/transports/cesium/1.124.0/witnesses.json) |
| Three.js GLTFLoader 0.187.0 | `gltf.py`; `SceneTransportSerialization` | right, up, backward; right-handed; fixture_only | xyzw; node local to parent; world = parent world * child local; column-vector action | 2; [contract](../../experiments/converters/gltf/2.0.0/semantic_contract.json), [provenance](../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json), [witnesses](../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json) |
| OpenUSD 26.11 | `openusd.py`; `SceneTransportSerialization` | right, up, backward; right-handed; application_declared | wxyz; OpenUSD Gf row vectors; child world row matrix = child local row matrix * parent world row matrix | 3; [contract](../../experiments/converters/openusd/26.11/semantic_contract.json), [provenance](../../tests/golden/transports/openusd/26.11/converter/manifest.json), [witnesses](../../tests/golden/transports/openusd/26.11/converter/witnesses.json) |

### Snapshot identities

The following SHA-256 values identify the local contract and witness bytes
used above. Provenance validation also checks the existing retained producer
and input identities; these hashes are a checkpoint, not new native evidence.

- Unreal Engine 5.7.4:
  contract `b0979fe872e9374ed607b541b6535ee492cd13c0999dc00f07928ad42a2f9af5`;
  witnesses `9579b62fbee2956c13e52d71544a139e77c9aae5b42feabe60602dffc35112c2`.
- Unity Editor 6000.5.2f1:
  contract `c1a2d84eb3e91a63f19ce776ad59b0f3b810ca31ce82ec59df2be4ec2e096b9e`;
  witnesses `cd7a440865636e1a6e9705cb63958c2575d6a39621e0f1270e007fca38f9fe9e`.
- Godot Engine 4.7-stable:
  contract `81c3c5f2e435af6607bb460736368e32fb89813e50ccbb2e0ac367647744ce99`;
  witnesses `96c0dbc6fb0c3a650bba748b7bf536c25534bdbc553b12f55a151ad38f143747`.
- Godot Engine 4.8-dev1:
  contract `81c3c5f2e435af6607bb460736368e32fb89813e50ccbb2e0ac367647744ce99`;
  witnesses `96c0dbc6fb0c3a650bba748b7bf536c25534bdbc553b12f55a151ad38f143747`.
- Blender 5.1.2:
  contract `f5d892b74874d8dbd89001ca3c325bf7baf6f0e23a6342691026d2e4d9fa1320`;
  witnesses `e8f9a4080c98f9484d4f6895361457bd2070ffb99415b4c1afe17643a812f763`.
- Eigen 5.0.1:
  contract `a6332d04f6c84bee1902b5b875ef0cc9afdb6a5d3b425fa910d7fb6c08117e8f`;
  witnesses `5a2a1a59364b8a78e8b463e4d6295296dd0435565d81be59ff16d78f4d75d974`.
- MuJoCo 3.17.0:
  contract `9a9e115858020bd870cd4c3b081a582e0f10a7f14e639b8747f6ab097a6d4be9`;
  witnesses `0c5eb585275c19930d6b0ee2a1ebe92cfd65dff0b66776f52b9f85bfda0e3485`.
- OpenCV 4.13.0:
  contract `d198b111f44f8fecad526d73cf6507a8099e6c04aac1bba798297872136ea1f9`;
  witnesses `9a42d672f7a5b4fecf26c308659ed04a7118076ccaa6e72dd250f0a78abc6df7`.
- ROS 2 tf2 0.36.23:
  contract `4fef1703f0d2a982c2087d058b0703272f3e1cc5608673ea9890f0655c42ce48`;
  witnesses `a6e2569b79295b44c3598decc70efba233f1ddd20e420c22c0673ee3aee64920`.
- SpiceyPy / CSPICE 8.3.0:
  contract `75b58bb87dca0d97767d2ff638115c9d4537f769b44576dd2a2de5b4b8b1e45d`;
  witnesses `ea498c1a3b326bb76531aa94004ecaf66bc15098111ff2b6669a479f0d9dbe60`.
- CesiumJS 1.124.0:
  contract `b99e29264dea4f59f8dc958e16b09fa32773cc38cc1e4c466f6f821dc28be644`;
  witnesses `cf56f2ed41685d79fb08f3d04ef7e36818e5fb920286758aa813d85746613ed2`.
- Three.js GLTFLoader 0.187.0:
  contract `7e93e0ca67cfb010ee33193e50fd0277d92a10e956decfee2ceefe087b6efefb`;
  witnesses `691bb98ad41404e7e90cc8713dad55ec38f78e62e9d3181feb7d0a5b1d7d82f1`.
- OpenUSD 26.11:
  contract `1386b60889ddc2480e1f9c43a61c036f47b04121a3bc8fd7f203b4614af6077d`;
  witnesses `1ccf08a682e1283c7f86894accadd1eba0a46e12a63bf280282305407c80125e`.

### Qualification priorities and remaining gaps

1. Highest value now: review and land the existing scoped-claim and typed
   interpretation hardening, with full exact-candidate CI. PR-only readers
   currently cannot rely on these uncommitted corrections. Retain negative
   tests for mismatched version, operation, scope, witness IDs and frame map.
2. Physical interoperability still requires an explicit consumer binding.
   The cheapest meaningful expansion is an asymmetric scene-object fixture
   with declared parent/body bases and known forward/up markers, observing
   local and world action. This is more informative than more quaternion
   coefficient samples. It must not imply calibration of arbitrary assets.
3. Add Godot Camera3D, glTF oriented-mesh front, or OpenXR VIEW/LOCAL_FLOOR
   witnesses only when qualifying those stronger claims. Existing numeric
   action/LOCAL/static hierarchy captures cannot be promoted to cover them.
4. PROJ, glTF Validator, Blender importer, VTK and Trimesh remain the five
   separately registered supporting targets. Their retained frame/format/
   loader observations are not accepted converter proofs. Preserve that
   separation and the documented Blender animation discrepancies.

No new unsupported mathematical claim or replay regression was established
by this bounded audit. Native reproductions, real physical alignments and
excluded operations remain unassessed here. Changes of interpretation alone
do not require recapture when the unchanged raw observations support them;
changed native calls, input assets, runtime, observed fields or physical
assignments do require a new capture/provenance revision.

## 7. Native follow-up: scoped gaps closed, not universal alignment

Earlier checkpoints describe the state before this follow-up. New supplemental
captures exercise previously absent native operations, without upgrading old
numeric-only artifacts or granting broader adapter acceptance.

| Claim | Actual native execution | Established boundary | Still excluded |
| --- | --- | --- | --- |
| Camera3D node action | Official Godot 4.7-stable and 4.8-dev1 | Local/global/camera axes, perspective center ray, translation, both asymmetric parent-child orders in assigned frames | Calibration, XR overrides, rendered images, nonzero offsets, arbitrary assets |
| Oriented mesh observation | Three.js GLTFLoader 0.187.0 | Actual asymmetric geometry with author-declared front/right/up, transformed through a static hierarchy | Arbitrary mesh-front inference, physical alignment, animation, other loaders |
| VIEW and LOCAL_FLOOR space relations | Pinned Khronos OpenXR test runtime, API 1.1.65 | Same-reference-type static offset chains and inverse relations | Head tracking, measured floor height, cross-type relations, vendor hardware |

Target-specific validators `godot_camera.load_camera_capture`,
`oriented_mesh.validate_oriented_mesh`, and `openxr_spaces.validate_space_capture`
under `scripts/evidence_schema/` feed the existing `scripts.semantic_claims`
pipeline. Exact role, version, operation, artifact and witness links are in the
current generated registry. These are supplemental evidence, not new accepted
converter-normalizer variants. Reproduction instructions accompany each
experiment; provenance records identify native source/runtime/input/output.

All new artifacts report `adapter_conformance_checked=False`: executing a native
object API does not prove a QuatOps consumer used it correctly. Explicit
frame-map replay remains separately reportable. Physical alignment still comes
from a declared application or independent calibration, not product names.

Candidate verification exposed stale Cesium target-string validation, rejection
of the direct glTF capture's known point-operation label, and historical source
hashes pointing at edited code. Corrections preserve structured identity and
known API distinctions. Historical producer bytes are retained in the existing
provenance archive, rather than relabeling edited code as a past native run.
Native witness values were not changed to satisfy these repairs.

Digest integrity alone cannot establish semantics. Negative tests must reject
altered input meaning or forged observations even after hashes are refreshed.
Final combined verification and Linux checks must run on the exact candidate
before publication. Earlier pass counts are checkpoints, not release claims.
