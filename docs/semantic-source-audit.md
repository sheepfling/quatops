# Semantic source audit

This document records a human review of each registered semantic role against
its cited primary source. It validates the published convention claim only.
It does not prove that a particular application's scene, mesh, robot, or sensor
is physically aligned to that convention; that remains an explicit alignment
input with caller/runtime provenance.

| Role | Source-backed claim confirmed | Boundary retained in QuatOps |
| --- | --- | --- |
| `unreal.world-axes` | Unreal Editor 5.7 uses a left-handed coordinate system; +X forward, +Y right, +Z up. | Editor/world scope only; local spaces, project overrides, and model-forward are separate. [Epic coordinate systems](https://dev.epicgames.com/documentation/en-us/unreal-engine/coordinate-system-and-spaces-in-unreal-engine?application_version=5.7) |
| `unity.world-axes` | Unity 6000.5 documents left-handed axes: +X right, +Y up, +Z forward. | World convention does not establish imported-model axes or application scene alignment. [Unity Quaternion documentation](https://docs.unity.com/en-us/engine/6000.5/manual/scripting/programming-math/unity-engine-math/class-quaternion) |
| `godot.camera-axes` | Godot 4.7 is right-handed, +Y up; camera forward is -Z, so camera right is +X and back is +Z. | This camera/sensor role is distinct from Godot oriented assets, whose front is +Z; camera mounting remains external. [Godot model export conventions](https://docs.godotengine.org/en/4.7/tutorials/assets_pipeline/importing_3d_scenes/model_export_considerations.html) |
| `ros.rep103.body-flu` | REP-103 recommends right-handed body frames with +X forward, +Y left, +Z up. | The REP allows documented exceptions; a ROS application is not assumed to conform. [Pinned REP-103](https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md) |
| `ros.rep103.optical` | REP-103 camera optical suffix frames use +Z forward, +X right, +Y down. | This names an optical frame convention; it does not prove camera-to-body extrinsics or mounting. [Pinned REP-103](https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md) |
| `cesium.enu-to-fixed` | Cesium 1.124 resolves local East-North-Up axes into the selected fixed frame at a supplied origin and ellipsoid. | The basis depends on geographic context. Body-forward alignment, inertial conversion, and time-dependent Earth orientation are excluded. [Cesium Transforms](https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/Transforms.html#eastNorthUpToFixedFrame) |
| `gltf.asset-front` | glTF 2.0 defines +Y up, +Z forward, -X right, with the front of an oriented asset facing +Z. | The specification cannot prove arbitrary mesh geometry was authored with that front direction. [glTF 2.0 §3.4](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) |
| `gltf.node-local` | Node hierarchy semantics are in §3.5.2, transformations in §3.5.3, and the XYZW node.rotation quaternion property in §5.25.6. | The retained loader fixture checks static hierarchy behavior, not arbitrary exports, meshes, or animation. [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) |
| `openxr.reference-space` | OpenXR 1.1 LOCAL and LOCAL_FLOOR use +X right, +Y up, -Z forward; LOCAL is gravity-aligned and LOCAL_FLOOR places its origin at an estimated floor. | Origin and relation are runtime/session/time specific. The software test runtime is not vendor hardware proof. [OpenXR reference spaces](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html) |
| `openxr.view-space` | OpenXR VIEW uses +X right, +Y up, -Z forward and represents the primary viewer's view origin. | Runtime validity/tracking flags and the actual XrSpace relation are required; no physical room alignment is inferred. [OpenXR reference spaces](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html) |
| `quatops.parent-child-orientation` | This is an application-defined QuatOps contract: an active orientation relates explicitly named parent and child coordinate frames. | QuatOps requires both frame definitions and does not assign universal forward/up axes to this generic role. |

The axis claims above were checked against the referenced documents. The
implementation also carries each role's scope/exclusions in
`quatops.experimental.semantics.roles`; the generated registry reports native evidence
scope separately from source authority and application alignment.

## Review findings and decisions

The frame-semantics audit of snapshot `138b730` distinguished native numeric
behavior, documented frame conventions, and application-assigned alignment.
It found that the glTF transform fixture used a camera-like probe basis; that
does not establish the separate glTF asset-front convention. The fixture is
therefore described as a declared probe, while the asset-front role and its
source are recorded separately above. Neutral quaternion APIs likewise use
fixture-assigned frames; native vector action cannot determine physical axis
labels. Engine world axes do not establish object or camera axes. ROS named
frames and Cesium ENU have documented meanings only within their respective
API and context. Parent and child mappings remain explicit, including for
shared-basis conversion.

The audit replayed frozen glTF values using SciPy but did not launch native
engines or run the full suite. Its
[historical research notes](https://github.com/sheepfling/quatops/blob/a5510f68ea8d4d8ebea9725492265ac8d773de9a/quatops_frame_semantics_audit_138b730.md)
refer to that earlier snapshot; this page and the target contracts describe
the current scope.
