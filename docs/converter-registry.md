# Converter and evidence-target registry

This generated view comes from `quatops.experimental.converters.CONVERTERS` and
`TARGET_DECLARATIONS`. Edit the registry definitions, then regenerate
this page with `python -m scripts.render_converter_registry --repository-root .`.

These APIs are experimental. Evidence acceptance is not API stability;
see [the experimental policy](experimental.md).

| ID | Target API | Version | Role | Representation / order | Evidence state | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| unreal.fquat | Unreal: TQuat<T> | 5.7.4 | orientation_converter | quaternion / xyzw | accepted_converter | Native quaternion action; float32 and float64. |
| unity.quaternion | Unity: UnityEngine.Quaternion | 6000.5.2f1 | orientation_converter | quaternion / xyzw | accepted_converter | Native quaternion action; float32. |
| godot.quaternion | Godot: Quaternion | 4.7 / 4.8-dev1 | orientation_converter | quaternion / xyzw | accepted_converter | Native quaternion action for the captured frame maps; float32. |
| blender.mathutils.quaternion | Blender: mathutils.Quaternion | 5.1.2 | orientation_converter | quaternion / wxyz | accepted_converter | Quaternion field conversion; mesh axes and importer behavior remain separate. |
| cesium.quaternion | CesiumJS: Quaternion | 1.124.0 | orientation_converter | quaternion / xyzw | accepted_converter | Quaternion field conversion; ENU/ECEF placement requires an origin-specific frame map. |
| gltf.rotation | glTF: node.rotation | 2.0 | orientation_converter | quaternion / xyzw | accepted_converter | Quaternion field conversion for the accepted static Three.js loader proof. |
| threejs.quaternion | Three.js: Quaternion | 0.187.0 | orientation_converter | quaternion / xyzw | accepted_converter | Quaternion field conversion; scene hierarchy serialization remains a transport operation. |
| openusd.gfquatd | OpenUSD: GfQuatd orient | 26.11 | orientation_converter | quaternion / wxyz | accepted_converter | Static orient-op quaternion field conversion. |
| ros2.geometry-msgs-quaternion | ROS 2: geometry_msgs/Quaternion | Jazzy / 5.3.8 | orientation_converter | quaternion / xyzw | accepted_converter | Quaternion message field; frame IDs, timestamps, and translation remain explicit. |
| spiceypy.cspice-q | SpiceyPy / CSPICE: q2m / m2q quaternion | 8.3.0 / N0067 | orientation_converter | quaternion / wxyz | accepted_converter | Direct numeric quaternion boundary in the captured probe frame. |
| mujoco.quaternion | MuJoCo: mju quaternion APIs | 3.17.0 | orientation_converter | quaternion / wxyz | accepted_converter | Direct quaternion API; model and body frames remain caller supplied. |
| opencv.rodrigues | OpenCV: Rodrigues rotation vector | 4.13.0 | orientation_converter | rotation_vector / — | accepted_converter | Rotation-vector boundary in radians; camera extrinsics are outside scope. |
| eigen.quaternion-constructor | Eigen: Quaternion(w,x,y,z) | 5.0.1 | orientation_converter | quaternion / wxyz | accepted_converter | Constructor coefficient boundary; application frames remain caller supplied. |
| eigen.quaternion-coeffs | Eigen: Quaternion::coeffs() | 5.0.1 | orientation_converter | quaternion / xyzw | accepted_converter | Coefficient accessor boundary; application frames remain caller supplied. |
| openxr.xrquaternionf | OpenXR: XrQuaternionf in XrPosef | 1.1.65 | orientation_converter | quaternion / xyzw | declared_not_native_accepted | Spec and Khronos test-runtime contract only; no vendor-runtime or hardware acceptance. |
| sophus.so3-params | Sophus: SO3::params() | 1.24.6 | orientation_converter | quaternion / xyzw | supporting_oracle_only | Captured SO(3) representation; not an accepted application-frame converter. |
| proj.9.5.1 | PROJ: WGS84 coordinate transforms | 9.5.1 | frame_provider | — | supporting_frame_evidence_only | Coordinate-frame evidence; PROJ exposes no quaternion boundary here. |
| khronos.gltf-validator | Khronos glTF Validator: glTF structural validation | 2.0.0-dev.3.10 | format_validator | — | format_validation_only | Checks asset validity; does not provide a rotation-action oracle. |
| blender.gltf-import | Blender glTF importer: glTF scene and animation import | 5.1.2 | loader_observation | — | loader_observation_only | Importer behavior evidence; distinct from mathutils.Quaternion conversion. |
| vtk.gltf-loader | VTK glTF loader: glTF scene loading | 9.6.2 | loader_observation | — | loader_observation_only | Loader evidence only; no QuatOps-generated converter acceptance claim. |
| trimesh.gltf-loader | Trimesh glTF loader: glTF scene loading | 4.11.1 | loader_observation | — | loader_observation_only | Loader evidence only; no QuatOps-generated converter acceptance claim. |
