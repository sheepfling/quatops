# Frame-aware target converters

QuatOps' converter front uses one canonical value contract and keeps the
target's storage details in a declared profile:

- Canonical coefficients are active Hamilton WXYZ quaternions.
- An orientation is labelled `q_parent_from_child` and acts as
  `v_parent = R(q) @ v_child`.
- Source and destination coordinate frames, positive-axis meanings, and
  handedness are explicit.
- Conversion preserves quaternion magnitude and sign. It does not normalize,
  canonicalize, infer a model-forward axis, or choose an engine default.

The scalar API uses only the Python standard library. NumPy batch methods load
NumPy only when called.

Frame labels also make transform chains composable: a
`world <- base` orientation can compose with `base <- sensor`, and the
intermediate label is checked before multiplying. Hamilton product order is
`lhs * rhs`, with the right-hand orientation acting first.

## Declare the frames, then choose the target profile

`CoordinateFrame.basis_to_reference` contains the frame's positive X/Y/Z axes
as columns expressed in a named, right-handed reference frame. A
`FrameCorrespondence` derives the component map from two such declarations.
Use a separate correspondence for the parent/world and child/body sides.
Their determinant parities must match for the result to remain a proper
rotation. If `A_source` and `A_destination` contain the frame basis vectors as
columns in the same reference, the derived component map is
`B_destination_from_source = A_destination.T @ A_source`.

```python
from quatops.converters import (
    CoordinateFrame,
    FrameCorrespondence,
    FramedOrientation,
    SemanticQuaternionMap,
    get_converter,
)
from quatops.engines import OrthogonalBasisMap, SignedAxisMap

# Both canonical reference bases are explicitly right-handed.
canonical_world = CoordinateFrame(
    "canonical.world",
    "physical-world",
    ("right", "forward", "up"),
    SignedAxisMap(("x", "y", "z")),
    "right",
)
canonical_body = CoordinateFrame(
    "canonical.body",
    "physical-body",
    ("right", "forward", "up"),
    SignedAxisMap(("x", "y", "z")),
    "right",
)

# This target basis maps canonical +X,+Y,+Z to target +X,+Z,+Y.
# The same declared map applies to world and body coordinates here.
target_basis = OrthogonalBasisMap(((1, 0, 0), (0, 0, 1), (0, 1, 0)))
target_world = CoordinateFrame(
    "unity.world",
    "physical-world",
    ("right", "up", "forward"),
    target_basis,
    "left",
)
target_body = CoordinateFrame(
    "unity.body",
    "physical-body",
    ("right", "up", "forward"),
    target_basis,
    "left",
)
mapping = SemanticQuaternionMap(
    world=FrameCorrespondence(canonical_world, target_world),
    body=FrameCorrespondence(canonical_body, target_body),
)

source = FramedOrientation(
    quaternion_wxyz=(0.8, 0.2, -0.4, 0.4),
    from_frame="canonical.body",
    to_frame="canonical.world",
)
unity = get_converter("unity.quaternion")
native = unity.to_target(source, mapping)  # target XYZW, frame labels retained
recovered = unity.from_target(native, mapping)
```

The example's axes describe that particular application mapping. Replace them
with the actual world and model/body frame declarations for the data being
converted. `OrthogonalBasisMap` supports arbitrary orthogonal rotations and
reflections; `SignedAxisMap` is the efficient signed-permutation form.

## Standard ECEF-from-body orientation

For applications that use an Earth-fixed world, QuatOps provides a named
semantic value, `ECEFBodyOrientation`, for `q_ECEF_from_body`. Its parent frame
is the right-handed ECEF frame (+X at the equator/prime meridian, +Y at the
equator/90° east, +Z toward the north pole). Its child frame is the explicit
right-handed QuatOps body frame (+X right, +Y forward, +Z up). The value wraps
the standard Hamilton WXYZ quaternion; it does not include position or a
geodetic origin.

The conversion path is explicit: semantic `ECEFBodyOrientation` → labelled
Hamilton WXYZ → target profile. This example deliberately declares target
world and body axes aligned to the two canonical frames. For a scene with a
different world basis or model-forward axis, declare those maps from the
actual scene and asset frames.

```python
from quatops.converters import (
    CANONICAL_BODY_FRAME,
    ECEF_FRAME,
    CoordinateFrame,
    ECEFBodyOrientation,
    FrameCorrespondence,
    SemanticQuaternionMap,
    get_converter,
)
from quatops.engines import SignedAxisMap

identity = SignedAxisMap(("x", "y", "z"))
target_world = CoordinateFrame(
    "application.world",
    ECEF_FRAME.reference,
    ECEF_FRAME.axes,
    identity,
)
target_body = CoordinateFrame(
    "application.body",
    CANONICAL_BODY_FRAME.reference,
    CANONICAL_BODY_FRAME.axes,
    identity,
)
frame_map = SemanticQuaternionMap(
    FrameCorrespondence(ECEF_FRAME, target_world),
    FrameCorrespondence(CANONICAL_BODY_FRAME, target_body),
)

physical = ECEFBodyOrientation.from_wxyz((0.8, 0.2, -0.4, 0.4))
canonical_hamilton = physical.to_framed_hamilton()
unity = get_converter("unity.quaternion")
native = unity.to_target(canonical_hamilton, frame_map)
restored = ECEFBodyOrientation.from_framed_hamilton(
    unity.from_target(native, frame_map)
)
```

Use `FramedOrientation` directly when the parent frame is a local scene,
robot, camera, or another explicitly named reference rather than ECEF. The
same target profiles handle both forms after the semantic value is expressed
as a labelled Hamilton orientation.

## Batch conversion

Pass a labelled WXYZ batch to the same target profile. `component_axis` defaults
to the last axis, and the output retains that axis position and target frame
labels.

```python
import numpy as np
from quatops.converters import FramedOrientationBatch

keys = FramedOrientationBatch(
    values=np.asarray([source.quaternion_wxyz, source.quaternion_wxyz]),
    from_frame="canonical.body",
    to_frame="canonical.world",
)
native_keys = unity.to_target_array(keys, mapping)
source_keys = unity.from_target_array(native_keys, mapping)
```

For a signed-axis map, quaternion batch conversion is component selection and
sign changes and preserves the input dtype. General world/body corrections and
rotation-vector boundaries use float64 Hamilton NumPy kernels. No target
package is imported by QuatOps.

## Declared target boundaries and evidence status

`quatops.converters.CONVERTERS` contains immutable, typed profiles. The same
`to_target`, `from_target`, `to_target_array`, and `from_target_array`
operations are available for every quaternion or rotation-vector profile.
`TARGET_DECLARATIONS` also records related evidence targets that do not expose
a quaternion converter.

| Profile group | Declared boundary | Evidence state and scope |
| --- | --- | --- |
| Unreal, Unity, Godot, Blender | `unreal.fquat`, `unity.quaternion`, `godot.quaternion`, `blender.mathutils.quaternion` | Accepted for the captured versions and frame contracts. Blender's quaternion profile is distinct from its glTF importer observations. |
| CesiumJS, glTF, Three.js, OpenUSD, ROS 2/tf2 | `cesium.quaternion`, `gltf.rotation`, `threejs.quaternion`, `openusd.gfquatd`, `ros2.geometry-msgs-quaternion` | Accepted within the captured ENU/ECEF, static asset, transform-stack, and static frame-chain scopes. Translation, serialization, and dynamic runtime behavior remain separate operations. |
| SpiceyPy/CSPICE, MuJoCo, Eigen | `spiceypy.cspice-q`, `mujoco.quaternion`, `eigen.quaternion-constructor`, `eigen.quaternion-coeffs` | Accepted for the captured numeric probe frames and operations. Application-specific frames remain caller-declared. |
| OpenCV | `opencv.rodrigues` | Accepted for the captured radians-based vector/matrix boundary. Camera extrinsics remain separate. Rotation-vector conversion preserves the physical rotation, not a quaternion's chosen sign. |
| OpenXR | `openxr.xrquaternionf` | Declared from the 1.1.65 contract and Khronos test runtime. No vendor runtime or hardware behavior is claimed. |
| Sophus | `sophus.so3-params` | Supporting mathematical oracle only; no application-frame converter acceptance claim. |

PROJ is indexed as frame-transform evidence, the Khronos validator as format
validation, and Blender/VTK/Trimesh loader captures as loader observations.
Those roles are explicit in `TARGET_DECLARATIONS`; none is silently presented
as a quaternion converter.

These profiles convert orientation values and retain frame identity. They do
not implement every target's object construction, serialized asset, pose
translation, time sampling, or runtime lifecycle. Consult the
[interoperability evidence atlas](INTEROPERABILITY_EVIDENCE.md) for accepted
scope and exclusions, and the [shared coordinate mapping reference](engines/coordinate-mapping.md)
for the basis equations.
