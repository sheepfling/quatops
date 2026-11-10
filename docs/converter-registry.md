# Converter and evidence-target registry

This generated view comes from the experimental converter declarations.
Representation evidence and semantic authority are recorded separately.
Regenerate it with `python -m scripts.render_converter_registry --repository-root .`.

## Target profiles

### `unreal.fquat`

- Target/API: Unreal: TQuat<T>
- Version: 5.7.4
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Native quaternion action; float32 and float64.

### `unity.quaternion`

- Target/API: Unity: UnityEngine.Quaternion
- Version: 6000.5.2f1
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Native quaternion action; float32.

### `godot.quaternion`

- Target/API: Godot: Quaternion
- Version: 4.7 / 4.8-dev1
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Native quaternion action for the captured frame maps; float32.

### `blender.mathutils.quaternion`

- Target/API: Blender: mathutils.Quaternion
- Version: 5.1.2
- Role: orientation_converter
- Representation/order: quaternion / wxyz
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Quaternion field conversion; mesh axes and importer behavior remain separate.

### `cesium.quaternion`

- Target/API: CesiumJS: Quaternion
- Version: 1.124.0
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Quaternion field conversion; ENU/ECEF placement requires an origin-specific frame map.

### `gltf.rotation`

- Target/API: glTF: node.rotation
- Version: 2.0
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Quaternion field conversion for the accepted static Three.js loader proof.

### `threejs.quaternion`

- Target/API: Three.js: Quaternion
- Version: 0.187.0
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Quaternion field conversion; scene hierarchy serialization remains a transport operation.

### `openusd.gfquatd`

- Target/API: OpenUSD: GfQuatd orient
- Version: 26.11
- Role: orientation_converter
- Representation/order: quaternion / wxyz
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Static orient-op quaternion field conversion.

### `ros2.geometry-msgs-quaternion`

- Target/API: ROS 2: geometry_msgs/Quaternion
- Version: Jazzy / 5.3.8
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Quaternion message field; frame IDs, timestamps, and translation remain explicit.

### `spiceypy.cspice-q`

- Target/API: SpiceyPy / CSPICE: q2m / m2q quaternion
- Version: 8.3.0 / N0067
- Role: orientation_converter
- Representation/order: quaternion / wxyz
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Direct numeric quaternion boundary in the captured probe frame.

### `mujoco.quaternion`

- Target/API: MuJoCo: mju quaternion APIs
- Version: 3.17.0
- Role: orientation_converter
- Representation/order: quaternion / wxyz
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Direct quaternion API; model and body frames remain caller supplied.

### `opencv.rodrigues`

- Target/API: OpenCV: Rodrigues rotation vector
- Version: 4.13.0
- Role: orientation_converter
- Representation/order: rotation_vector / —
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Rotation-vector boundary in radians; camera extrinsics are outside scope.

### `eigen.quaternion-constructor`

- Target/API: Eigen: Quaternion(w,x,y,z)
- Version: 5.0.1
- Role: orientation_converter
- Representation/order: quaternion / wxyz
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Constructor coefficient boundary; application frames remain caller supplied.

### `eigen.quaternion-coeffs`

- Target/API: Eigen: Quaternion::coeffs()
- Version: 5.0.1
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_verified
- Semantic authority: representation_only
- Evidence state: accepted_converter
- Scope: Coefficient accessor boundary; application frames remain caller supplied.

### `openxr.xrquaternionf`

- Target/API: OpenXR: XrQuaternionf in XrPosef
- Version: 1.1.65
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: spec_and_test_runtime
- Semantic authority: representation_only
- Evidence state: declared_not_native_accepted
- Scope: Spec and Khronos test-runtime contract only; no vendor-runtime or hardware acceptance.

### `sophus.so3-params`

- Target/API: Sophus: SO3::params()
- Version: 1.24.6
- Role: orientation_converter
- Representation/order: quaternion / xyzw
- Representation evidence: native_supporting
- Semantic authority: representation_only
- Evidence state: supporting_oracle_only
- Scope: Captured SO(3) representation; not an accepted application-frame converter.

### `proj.9.5.1`

- Target/API: PROJ: WGS84 coordinate transforms
- Version: 9.5.1
- Role: frame_provider
- Representation/order: —
- Representation evidence: —
- Semantic authority: —
- Evidence state: supporting_frame_evidence_only
- Scope: Coordinate-frame evidence; PROJ exposes no quaternion boundary here.

### `khronos.gltf-validator`

- Target/API: Khronos glTF Validator: glTF structural validation
- Version: 2.0.0-dev.3.10
- Role: format_validator
- Representation/order: —
- Representation evidence: —
- Semantic authority: —
- Evidence state: format_validation_only
- Scope: Checks asset validity; does not provide a rotation-action oracle.

### `blender.gltf-import`

- Target/API: Blender glTF importer: glTF scene and animation import
- Version: 5.1.2
- Role: loader_observation
- Representation/order: —
- Representation evidence: —
- Semantic authority: —
- Evidence state: loader_observation_only
- Scope: Importer behavior evidence; distinct from mathutils.Quaternion conversion.

### `vtk.gltf-loader`

- Target/API: VTK glTF loader: glTF scene loading
- Version: 9.6.2
- Role: loader_observation
- Representation/order: —
- Representation evidence: —
- Semantic authority: —
- Evidence state: loader_observation_only
- Scope: Loader evidence only; no QuatOps-generated converter acceptance claim.

### `trimesh.gltf-loader`

- Target/API: Trimesh glTF loader: glTF scene loading
- Version: 4.11.1
- Role: loader_observation
- Representation/order: —
- Representation evidence: —
- Semantic authority: —
- Evidence state: loader_observation_only
- Scope: Loader evidence only; no QuatOps-generated converter acceptance claim.

## Role-specific semantic contracts

These declarations preserve source authority, role scope, axes, and binding context.
The cited primary documents are collected in the
[semantic source audit](semantic-source-audit.md).

### `unreal.world-axes`

- Authority: target_defined
- Frame role: world
- Description: Unreal world axes
- Target family: Unreal
- Source: target_documentation; 5.7 / Coordinate System; World Space
- Axes/handedness: ('forward', 'right', 'up'); left
- Binding context: fixed_axes; alignment=scene_alignment
- Claim: Unreal Editor axes are +X forward, +Y right, +Z up; the editor coordinate system is
  left-handed.
- Scope: Unreal Editor world-axis role for the documented 5.7 coordinate-space contract.
- Exclusions: project-specific coordinate-space overrides; asset or mesh local-forward axis;
  geographic alignment

### `unity.world-axes`

- Authority: target_defined
- Frame role: world
- Description: Unity world axes
- Target family: Unity
- Source: target_documentation; 6000.5 / Coordinate handedness
- Axes/handedness: ('right', 'up', 'forward'); left
- Binding context: fixed_axes; alignment=scene_alignment
- Claim: Unity's documented left-handed coordinate system uses +X right, +Y up, +Z forward.
- Scope: Unity world-coordinate convention documented for the 6000.5 quaternion API.
- Exclusions: imported model local axes; project-specific scene alignment; geographic alignment

### `godot.camera-axes`

- Authority: target_defined
- Frame role: sensor
- Description: Godot Camera3D local axes
- Target family: Godot
- Source: target_documentation; 4.7 / 3D asset direction conventions
- Axes/handedness: ('right', 'up', 'back'); right
- Binding context: fixed_axes; alignment=sensor_mount
- Claim: Godot 4.7 documents right-handed 3D coordinates with +Y up and -Z as the camera-forward
  direction.
- Scope: Godot Camera3D local sensor-axis role, interpreted with the engine's documented 3D
  coordinates.
- Exclusions: arbitrary Node3D world alignment; asset-forward conventions; camera mount calibration

### `ros.rep103.body-flu`

- Authority: standard_defined
- Frame role: body
- Description: ROS body FLU frame
- Target family: ROS 2
- Source: formal_standard; REP-103 / Axis Orientation
- Axes/handedness: ('forward', 'left', 'up'); right
- Binding context: fixed_axes; alignment=body_alignment
- Claim: REP-103 recommends right-handed body frames with +X forward, +Y left, +Z up; it permits
  documented exceptions.
- Scope: ROS conventional body FLU frame under REP-103, not a rule that every ROS frame complies.
- Exclusions: per-robot body-frame compliance; sensor mount calibration; nonconforming application
  frames

### `ros.rep103.optical`

- Authority: standard_defined
- Frame role: sensor
- Description: ROS camera optical frame
- Target family: ROS 2
- Source: formal_standard; REP-103 / Suffix Frames
- Axes/handedness: ('right', 'down', 'forward'); right
- Binding context: fixed_axes; alignment=sensor_mount
- Claim: REP-103 defines camera optical axes as +Z forward, +X right, +Y down.
- Scope: ROS camera optical frame convention under REP-103 suffix-frame guidance.
- Exclusions: camera-to-body extrinsics; sensor mounting calibration; nonconforming application
  frames

### `cesium.enu-to-fixed`

- Authority: target_defined
- Frame role: world
- Description: Cesium local ENU frame
- Target family: CesiumJS
- Source: target_documentation; 1.124.0 / Transforms.eastNorthUpToFixedFrame
- Axes/handedness: —
- Binding context: geographic_enu
- Claim: Cesium defines local x=east, y=north, z=ellipsoid-surface normal at the supplied origin,
  mapped to the selected ellipsoid's fixed frame.
- Scope: Cesium local ENU-to-ellipsoid-fixed basis at an explicit origin and ellipsoid.
- Exclusions: body-forward alignment; inertial-to-fixed conversion; time-dependent Earth
  orientation; position or pose translation

### `gltf.asset-front`

- Authority: standard_defined
- Frame role: body
- Description: glTF oriented asset frame
- Target family: glTF
- Source: formal_standard; 2.0 / 3.4 Coordinate System and Units
- Axes/handedness: ('left', 'up', 'forward'); right
- Binding context: fixed_axes; alignment=model_alignment
- Claim: glTF 2.0 defines +Y up, +Z forward, and -X right, and says the front of an oriented asset
  faces +Z.
- Scope: glTF oriented-asset coordinate convention and static node rotation field.
- Exclusions: proof that arbitrary mesh geometry was authored facing +Z; application model
  alignment; animation interpolation

### `gltf.node-local`

- Authority: standard_defined
- Frame role: generic
- Description: glTF node-local to parent relation
- Target family: glTF
- Source: formal_standard; 2.0 / 3.5.2 Nodes and Hierarchy; 3.5.3 Transformations; 5.25.6
  node.rotation
- Axes/handedness: —
- Binding context: relational
- Claim: glTF node transforms are local to the node's parent; the node rotation is a unit quaternion
  stored XYZW.
- Scope: Static glTF node-local rotation field and hierarchy behavior in the pinned loader fixture.
- Exclusions: general scene-export/import semantics; animation curves; application model alignment

### `openxr.reference-space`

- Authority: standard_defined
- Frame role: world
- Description: OpenXR LOCAL or LOCAL_FLOOR reference-space axes
- Target family: OpenXR
- Source: formal_standard; 1.1.65 / XR_REFERENCE_SPACE_TYPE_LOCAL;
  XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR
- Axes/handedness: ('right', 'up', 'back'); right
- Binding context: fixed_axes; alignment=space_alignment
- Claim: OpenXR 1.1 LOCAL and LOCAL_FLOOR use +X right, +Y up, -Z forward; their origins are
  runtime-defined and can be revised.
- Scope: Named LOCAL or LOCAL_FLOOR reference-space orientation at the supplied runtime, base-space,
  and XrTime context.
- Exclusions: geodetic alignment; vendor-runtime tracking quality; STAGE or arbitrary action spaces;
  validity without orientation-valid flag

### `openxr.view-space`

- Authority: standard_defined
- Frame role: sensor
- Description: OpenXR VIEW reference-space axes
- Target family: OpenXR
- Source: formal_standard; 1.1.65 / XR_REFERENCE_SPACE_TYPE_VIEW
- Axes/handedness: ('right', 'up', 'back'); right
- Binding context: fixed_axes; alignment=space_alignment
- Claim: OpenXR 1.1 VIEW uses +X right, +Y up, -Z forward and tracks the viewer without gravity
  alignment.
- Scope: Named VIEW-space orientation at the supplied runtime, base-space, and XrTime context.
- Exclusions: gravity-aligned orientation; vendor-runtime tracking quality; invalid orientation
  locations

### `quatops.parent-child-orientation`

- Authority: application_declared
- Frame role: generic
- Description: Directed parent-from-child orientation
- Target family: —
- Source: application_contract; docs/semantics.md
- Axes/handedness: —
- Binding context: application_frames
- Claim: QuatOps defines q_parent_from_child as the active rotation mapping child coordinates into
  parent coordinates.
- Scope: User-declared parent and child coordinate frames for a physical orientation.
- Exclusions: automatic discovery of frame axes; physical calibration; target-specific native
  behavior

## Semantic assurance matrix

R identifies representation evidence. S records the declared role source. B is a
caller- or runtime-supplied physical alignment. N states the native proof scope.
Adapter conformance is reported separately; none of these proves physical
asset or sensor alignment.

### `unreal.world-axes`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: numeric_action: unreal.native.5.7.4 (adapter conformance replayed)
- Scope: Unreal Editor world-axis role for the documented 5.7 coordinate-space contract.
- Exclusions: project-specific coordinate-space overrides; asset or mesh local-forward axis;
  geographic alignment
- Adapter conformance: demonstrated
- Evidence: [unreal.native.5.7.4](../tests/oracles/unreal/5.7.4/witnesses.json)

### `unity.world-axes`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: numeric_action: unity.native.6000.5.2f1 (adapter conformance replayed)
- Scope: Unity world-coordinate convention documented for the 6000.5 quaternion API.
- Exclusions: imported model local axes; project-specific scene alignment; geographic alignment
- Adapter conformance: demonstrated
- Evidence: [unity.native.6000.5.2f1](../tests/oracles/unity/6000.5.2f1/witnesses.json)

### `godot.camera-axes`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: numeric_action: godot.native.4.7 (adapter conformance replayed);
  numeric_action: godot.native.4.8-dev1 (adapter conformance replayed)
- Scope: Godot Camera3D local sensor-axis role, interpreted with the engine's documented 3D
  coordinates.
- Exclusions: arbitrary Node3D world alignment; asset-forward conventions; camera mount calibration
- Adapter conformance: demonstrated
- Evidence: [godot.native.4.7](../tests/oracles/godot/shared/witnesses.json);
  [godot.native.4.8-dev1](../tests/oracles/godot/shared/witnesses.json)

### `ros.rep103.body-flu`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: runtime_space_relation: ros2.tf2.0.36.23 (adapter conformance replayed)
- Scope: ROS conventional body FLU frame under REP-103, not a rule that every ROS frame complies.
- Exclusions: per-robot body-frame compliance; sensor mount calibration; nonconforming application
  frames
- Adapter conformance: demonstrated
- Evidence: [ros2.tf2.0.36.23](../tests/golden/oracles/ros2/0.36.23/witnesses.json)

### `ros.rep103.optical`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: runtime_space_relation: ros2.tf2.0.36.23 (adapter conformance replayed)
- Scope: ROS camera optical frame convention under REP-103 suffix-frame guidance.
- Exclusions: camera-to-body extrinsics; sensor mounting calibration; nonconforming application
  frames
- Adapter conformance: demonstrated
- Evidence: [ros2.tf2.0.36.23](../tests/golden/oracles/ros2/0.36.23/witnesses.json)

### `cesium.enu-to-fixed`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable contextual resolver; typed origin and ellipsoid required; body
  alignment remains caller supplied
- N — native action: role_specific_action: cesium.native.1.124.0 (adapter conformance replayed)
- Scope: Cesium local ENU-to-ellipsoid-fixed basis at an explicit origin and ellipsoid.
- Exclusions: body-forward alignment; inertial-to-fixed conversion; time-dependent Earth
  orientation; position or pose translation
- Adapter conformance: demonstrated
- Evidence: [cesium.native.1.124.0](../tests/golden/transports/cesium/1.124.0/witnesses.json)

### `gltf.asset-front`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: numeric_action: gltf.threejs.loader.0.187.0 (adapter conformance replayed)
- Scope: glTF oriented-asset coordinate convention and static node rotation field.
- Exclusions: proof that arbitrary mesh geometry was authored facing +Z; application model
  alignment; animation interpolation
- Adapter conformance: demonstrated
- Evidence:
  [gltf.threejs.loader.0.187.0](../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json)

### `gltf.node-local`

- R — representation: R verified (native representation)
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: declared contract only; no executable semantic binder
- N — native action: numeric_action: gltf.threejs.loader.0.187.0 (adapter conformance replayed)
- Scope: Static glTF node-local rotation field and hierarchy behavior in the pinned loader fixture.
- Exclusions: general scene-export/import semantics; animation curves; application model alignment
- Adapter conformance: demonstrated
- Evidence:
  [gltf.threejs.loader.0.187.0](../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json)

### `openxr.reference-space`

- R — representation: R specification/test-runtime only
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: runtime_space_relation: openxr.test-runtime.1.1.65 (native runtime only;
  adapter conformance not demonstrated)
- Scope: Named LOCAL or LOCAL_FLOOR reference-space orientation at the supplied runtime, base-space,
  and XrTime context.
- Exclusions: geodetic alignment; vendor-runtime tracking quality; STAGE or arbitrary action spaces;
  validity without orientation-valid flag
- Adapter conformance: not demonstrated
- Evidence:
  [openxr.test-runtime.1.1.65](../experiments/oracles/openxr/1.1.65/witnesses/test_runtime_frame_chain.json)

### `openxr.view-space`

- R — representation: R specification/test-runtime only
- S — role source: pinned source and scoped claim recorded; see the semantic source audit
- B — physical alignment: executable binder; explicit alignment required; structure checked,
  physical truth supplied by caller/runtime
- N — native action: runtime_space_relation: openxr.test-runtime.1.1.65 (native runtime only;
  adapter conformance not demonstrated)
- Scope: Named VIEW-space orientation at the supplied runtime, base-space, and XrTime context.
- Exclusions: gravity-aligned orientation; vendor-runtime tracking quality; invalid orientation
  locations
- Adapter conformance: not demonstrated
- Evidence:
  [openxr.test-runtime.1.1.65](../experiments/oracles/openxr/1.1.65/witnesses/test_runtime_frame_chain.json)

### `quatops.parent-child-orientation`

- R — representation: not applicable
- S — role source: application contract; see the semantic source audit
- B — physical alignment: declared contract only; no executable semantic binder
- N — native action: not demonstrated
- Scope: User-declared parent and child coordinate frames for a physical orientation.
- Exclusions: automatic discovery of frame axes; physical calibration; target-specific native
  behavior
- Adapter conformance: not demonstrated
- Evidence: —

Representation-only profiles remain coefficient/action converters over
caller-supplied frame definitions. A target name never supplies physical axes.
