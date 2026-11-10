# Converter and evidence-target registry

This generated view comes from the experimental converter declarations.
Representation evidence and semantic authority are recorded separately.
Regenerate it with `python -m scripts.render_converter_registry --repository-root .`.

## Target profiles

| Profile | Target, API, and version | Role and representation | Evidence, authority, and state | Scope |
| --- | --- | --- | --- | --- |
| `unreal.fquat` | Unreal: TQuat&lt;T&gt;; 5.7.4 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Native quaternion action; float32 and float64. |
| `unity.quaternion` | Unity: UnityEngine.Quaternion; 6000.5.2f1 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Native quaternion action; float32. |
| `godot.quaternion` | Godot: Quaternion; 4.7 / 4.8-dev1 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Native quaternion action for the captured frame maps; float32. |
| `blender.mathutils.quaternion` | Blender: mathutils.Quaternion; 5.1.2 | orientation_converter; quaternion / wxyz | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Quaternion field conversion; mesh axes and importer behavior remain separate. |
| `cesium.quaternion` | CesiumJS: Quaternion; 1.124.0 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Quaternion field conversion; ENU/ECEF placement requires an origin-specific frame map. |
| `gltf.rotation` | glTF: node.rotation; 2.0 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Quaternion field conversion for the accepted static Three.js loader proof. |
| `threejs.quaternion` | Three.js: Quaternion; 0.187.0 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Quaternion field conversion; scene hierarchy serialization remains a transport operation. |
| `openusd.gfquatd` | OpenUSD: GfQuatd orient; 26.11 | orientation_converter; quaternion / wxyz | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Static orient-op quaternion field conversion. |
| `ros2.geometry-msgs-quaternion` | ROS 2: geometry_msgs/Quaternion; Jazzy / 5.3.8 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Quaternion message field; frame IDs, timestamps, and translation remain explicit. |
| `spiceypy.cspice-q` | SpiceyPy / CSPICE: q2m / m2q quaternion; 8.3.0 / N0067 | orientation_converter; quaternion / wxyz | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Direct numeric quaternion boundary in the captured probe frame. |
| `mujoco.quaternion` | MuJoCo: mju quaternion APIs; 3.17.0 | orientation_converter; quaternion / wxyz | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Direct quaternion API; model and body frames remain caller supplied. |
| `opencv.rodrigues` | OpenCV: Rodrigues rotation vector; 4.13.0 | orientation_converter; rotation_vector / — | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Rotation-vector boundary in radians; camera extrinsics are outside scope. |
| `eigen.quaternion-constructor` | Eigen: Quaternion(w,x,y,z); 5.0.1 | orientation_converter; quaternion / wxyz | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Constructor coefficient boundary; application frames remain caller supplied. |
| `eigen.quaternion-coeffs` | Eigen: Quaternion::coeffs(); 5.0.1 | orientation_converter; quaternion / xyzw | evidence=native_verified; semantic authority=representation_only; state=accepted_converter | Coefficient accessor boundary; application frames remain caller supplied. |
| `openxr.xrquaternionf` | OpenXR: XrQuaternionf in XrPosef; 1.1.65 | orientation_converter; quaternion / xyzw | evidence=spec_and_test_runtime; semantic authority=representation_only; state=declared_not_native_accepted | Spec and Khronos test-runtime contract only; no vendor-runtime or hardware acceptance. |
| `sophus.so3-params` | Sophus: SO3::params(); 1.24.6 | orientation_converter; quaternion / xyzw | evidence=native_supporting; semantic authority=representation_only; state=supporting_oracle_only | Captured SO(3) representation; not an accepted application-frame converter. |
| `proj.9.5.1` | PROJ: WGS84 coordinate transforms; 9.5.1 | frame_provider; — | evidence=—; semantic authority=—; state=supporting_frame_evidence_only | Coordinate-frame evidence; PROJ exposes no quaternion boundary here. |
| `khronos.gltf-validator` | Khronos glTF Validator: glTF structural validation; 2.0.0-dev.3.10 | format_validator; — | evidence=—; semantic authority=—; state=format_validation_only | Checks asset validity; does not provide a rotation-action oracle. |
| `blender.gltf-import` | Blender glTF importer: glTF scene and animation import; 5.1.2 | loader_observation; — | evidence=—; semantic authority=—; state=loader_observation_only | Importer behavior evidence; distinct from mathutils.Quaternion conversion. |
| `vtk.gltf-loader` | VTK glTF loader: glTF scene loading; 9.6.2 | loader_observation; — | evidence=—; semantic authority=—; state=loader_observation_only | Loader evidence only; no QuatOps-generated converter acceptance claim. |
| `trimesh.gltf-loader` | Trimesh glTF loader: glTF scene loading; 4.11.1 | loader_observation; — | evidence=—; semantic authority=—; state=loader_observation_only | Loader evidence only; no QuatOps-generated converter acceptance claim. |

## Semantic assurance matrix

R identifies representation evidence. S records the declared role source. B is a
caller- or runtime-supplied physical alignment. N states the native proof scope.
Adapter conformance is reported separately; these claims do not establish physical
asset or sensor alignment. Primary sources are indexed in the
[semantic source audit](semantic-source-audit.md).

### `unreal.world-axes`

- Declaration: Unreal world axes; authority=target_defined; frame role=world; target=Unreal
- Source: target_documentation; 5.7 / Coordinate System; World Space
- Axes and binding: ('forward', 'right', 'up'); left; fixed_axes; alignment=scene_alignment
- Claim: Unreal Editor axes are +X forward, +Y right, +Z up; the editor coordinate system is
  left-handed.
- Scope: Unreal Editor world-axis role for the documented 5.7 coordinate-space contract.
- Exclusions: project-specific coordinate-space overrides; asset or mesh local-forward axis;
  geographic alignment
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: numeric_action: unreal.native.5.7.4 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence: [unreal.native.5.7.4](../tests/oracles/unreal/5.7.4/witnesses.json)

### `unity.world-axes`

- Declaration: Unity world axes; authority=target_defined; frame role=world; target=Unity
- Source: target_documentation; 6000.5 / Coordinate handedness
- Axes and binding: ('right', 'up', 'forward'); left; fixed_axes; alignment=scene_alignment
- Claim: Unity's documented left-handed coordinate system uses +X right, +Y up, +Z forward.
- Scope: Unity world-coordinate convention documented for the 6000.5 quaternion API.
- Exclusions: imported model local axes; project-specific scene alignment; geographic alignment
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: numeric_action: unity.native.6000.5.2f1 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence: [unity.native.6000.5.2f1](../tests/oracles/unity/6000.5.2f1/witnesses.json)

### `godot.camera-axes`

- Declaration: Godot Camera3D local axes; authority=target_defined; frame role=sensor; target=Godot
- Source: target_documentation; 4.7 / 3D asset direction conventions
- Axes and binding: ('right', 'up', 'back'); right; fixed_axes; alignment=sensor_mount
- Claim: Godot 4.7 documents right-handed 3D coordinates with +Y up and -Z as the camera-forward
  direction.
- Scope: Godot Camera3D local sensor-axis role, interpreted with the engine's documented 3D
  coordinates.
- Exclusions: arbitrary Node3D world alignment; asset-forward conventions; camera mount calibration
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: numeric_action: godot.native.4.7 (adapter conformance replayed);
  numeric_action: godot.native.4.8-dev1 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence: [godot.native.4.7](../tests/oracles/godot/shared/witnesses.json);
  [godot.native.4.8-dev1](../tests/oracles/godot/shared/witnesses.json)

### `ros.rep103.body-flu`

- Declaration: ROS body FLU frame; authority=standard_defined; frame role=body; target=ROS 2
- Source: formal_standard; REP-103 / Axis Orientation
- Axes and binding: ('forward', 'left', 'up'); right; fixed_axes; alignment=body_alignment
- Claim: REP-103 recommends right-handed body frames with +X forward, +Y left, +Z up; it permits
  documented exceptions.
- Scope: ROS conventional body FLU frame under REP-103, not a rule that every ROS frame complies.
- Exclusions: per-robot body-frame compliance; sensor mount calibration; nonconforming application
  frames
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: runtime_space_relation: ros2.tf2.0.36.23 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence: [ros2.tf2.0.36.23](../tests/golden/oracles/ros2/0.36.23/witnesses.json)

### `ros.rep103.optical`

- Declaration: ROS camera optical frame; authority=standard_defined; frame role=sensor; target=ROS 2
- Source: formal_standard; REP-103 / Suffix Frames
- Axes and binding: ('right', 'down', 'forward'); right; fixed_axes; alignment=sensor_mount
- Claim: REP-103 defines camera optical axes as +Z forward, +X right, +Y down.
- Scope: ROS camera optical frame convention under REP-103 suffix-frame guidance.
- Exclusions: camera-to-body extrinsics; sensor mounting calibration; nonconforming application
  frames
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: runtime_space_relation: ros2.tf2.0.36.23 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence: [ros2.tf2.0.36.23](../tests/golden/oracles/ros2/0.36.23/witnesses.json)

### `cesium.enu-to-fixed`

- Declaration: Cesium local ENU frame; authority=target_defined; frame role=world; target=CesiumJS
- Source: target_documentation; 1.124.0 / Transforms.eastNorthUpToFixedFrame
- Axes and binding: —; geographic_enu
- Claim: Cesium defines local x=east, y=north, z=ellipsoid-surface normal at the supplied origin,
  mapped to the selected ellipsoid's fixed frame.
- Scope: Cesium local ENU-to-ellipsoid-fixed basis at an explicit origin and ellipsoid.
- Exclusions: body-forward alignment; inertial-to-fixed conversion; time-dependent Earth
  orientation; position or pose translation
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable contextual resolver; typed origin and ellipsoid required; body alignment remains caller
  supplied; N: role_specific_action: cesium.native.1.124.0 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence: [cesium.native.1.124.0](../tests/golden/transports/cesium/1.124.0/witnesses.json)

### `gltf.asset-front`

- Declaration: glTF oriented asset frame; authority=standard_defined; frame role=body; target=glTF
- Source: formal_standard; 2.0 / 3.4 Coordinate System and Units
- Axes and binding: ('left', 'up', 'forward'); right; fixed_axes; alignment=model_alignment
- Claim: glTF 2.0 defines +Y up, +Z forward, and -X right, and says the front of an oriented asset
  faces +Z.
- Scope: glTF oriented-asset coordinate convention and static node rotation field.
- Exclusions: proof that arbitrary mesh geometry was authored facing +Z; application model
  alignment; animation interpolation
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: numeric_action: gltf.threejs.loader.0.187.0 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence:
  [gltf.threejs.loader.0.187.0](../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json)

### `gltf.node-local`

- Declaration: glTF node-local to parent relation; authority=standard_defined; frame role=generic;
  target=glTF
- Source: formal_standard; 2.0 / 3.5.2 Nodes and Hierarchy; 3.5.3 Transformations; 5.25.6
  node.rotation
- Axes and binding: —; relational
- Claim: glTF node transforms are local to the node's parent; the node rotation is a unit quaternion
  stored XYZW.
- Scope: Static glTF node-local rotation field and hierarchy behavior in the pinned loader fixture.
- Exclusions: general scene-export/import semantics; animation curves; application model alignment
- Assurance: R: R verified (native representation); S: pinned source and scoped claim recorded; B:
  declared contract only; no executable semantic binder; N: numeric_action:
  gltf.threejs.loader.0.187.0 (adapter conformance replayed)
- Adapter conformance: demonstrated
- Evidence:
  [gltf.threejs.loader.0.187.0](../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json)

### `openxr.reference-space`

- Declaration: OpenXR LOCAL or LOCAL_FLOOR reference-space axes; authority=standard_defined; frame
  role=world; target=OpenXR
- Source: formal_standard; 1.1.65 / XR_REFERENCE_SPACE_TYPE_LOCAL;
  XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR
- Axes and binding: ('right', 'up', 'back'); right; fixed_axes; alignment=space_alignment
- Claim: OpenXR 1.1 LOCAL and LOCAL_FLOOR use +X right, +Y up, -Z forward; their origins are
  runtime-defined and can be revised.
- Scope: Named LOCAL or LOCAL_FLOOR reference-space orientation at the supplied runtime, base-space,
  and XrTime context.
- Exclusions: geodetic alignment; vendor-runtime tracking quality; STAGE or arbitrary action spaces;
  validity without orientation-valid flag
- Assurance: R: R specification/test-runtime only; S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: runtime_space_relation: openxr.test-runtime.1.1.65 (native runtime only;
  adapter conformance not demonstrated)
- Adapter conformance: not demonstrated
- Evidence:
  [openxr.test-runtime.1.1.65](../experiments/oracles/openxr/1.1.65/witnesses/test_runtime_frame_chain.json)

### `openxr.view-space`

- Declaration: OpenXR VIEW reference-space axes; authority=standard_defined; frame role=sensor;
  target=OpenXR
- Source: formal_standard; 1.1.65 / XR_REFERENCE_SPACE_TYPE_VIEW
- Axes and binding: ('right', 'up', 'back'); right; fixed_axes; alignment=space_alignment
- Claim: OpenXR 1.1 VIEW uses +X right, +Y up, -Z forward and tracks the viewer without gravity
  alignment.
- Scope: Named VIEW-space orientation at the supplied runtime, base-space, and XrTime context.
- Exclusions: gravity-aligned orientation; vendor-runtime tracking quality; invalid orientation
  locations
- Assurance: R: R specification/test-runtime only; S: pinned source and scoped claim recorded; B:
  executable binder; explicit alignment required; structure checked, physical truth supplied by
  caller/runtime; N: runtime_space_relation: openxr.test-runtime.1.1.65 (native runtime only;
  adapter conformance not demonstrated)
- Adapter conformance: not demonstrated
- Evidence:
  [openxr.test-runtime.1.1.65](../experiments/oracles/openxr/1.1.65/witnesses/test_runtime_frame_chain.json)

### `quatops.parent-child-orientation`

- Declaration: Directed parent-from-child orientation; authority=application_declared; frame
  role=generic; target=—
- Source: application_contract; docs/semantics.md
- Axes and binding: —; application_frames
- Claim: QuatOps defines q_parent_from_child as the active rotation mapping child coordinates into
  parent coordinates.
- Scope: User-declared parent and child coordinate frames for a physical orientation.
- Exclusions: automatic discovery of frame axes; physical calibration; target-specific native
  behavior
- Assurance: R: not applicable; S: application contract; B: declared contract only; no executable
  semantic binder; N: not demonstrated
- Adapter conformance: not demonstrated
- Evidence: —

Representation-only profiles remain coefficient/action converters over
caller-supplied frame definitions. A target name never supplies physical axes.
