# Unity

Unity exposes more than one Euler API. Keep `UnityEngine.Quaternion` separate
from `Unity.Mathematics.quaternion`; their units and order controls differ.

## `UnityEngine.Quaternion`

The constructor and named components use XYZW order. Unity's world frame is
left-handed: +X right, +Y up, and +Z forward. `Quaternion.Euler(x, y, z)`
accepts degrees and applies extrinsic rotations around Z, then X, then Y, so
the captured active matrix is `R = Ry(y) Rx(x) Rz(z)`. The fixture labels its
quaternions as active Hamilton rotations, keeping algebra, coefficient order,
and coordinate handedness as separate facts.

The native fixture captures `Quaternion.Euler`, `Quaternion.eulerAngles`,
rotated basis axes, and `Matrix4x4.Rotate` on Unity Editor 6000.5.2f1. Across
fifteen identity, principal-axis, mixed, near/exact gimbal, four X-axis
singularity-neighbor, over-90-degree, and signed-180-degree cases, QuatOps'
ZXY Euler constructor matches the native quaternion orientation, basis images,
and matrix. The round-trip residual from
`Quaternion.Euler(q.eulerAngles)` is recorded separately; the near-gimbal
sample has about 0.0961 degrees of residual in this build. The readback angles
can use a different Euler representation and should not be treated as stable
stored state.

## `Unity.Mathematics.quaternion`

Unity Mathematics names its six constructors by **application order**: for
example, `EulerXYZ(x, y, z)` applies X, then Y, then Z. For active Hamilton
rotations this is the product `qz(z) * qy(y) * qx(x)`. It therefore maps to
QuatOps' lowercase extrinsic sequence `"xyz"`, with the axis values passed in
sequence order. It does not map to uppercase intrinsic `"XYZ"`; the two differ
for noncommuting rotations. The pinned capture's `(10°, 20°, 30°)` diagnostic
checks this against SciPy and records both interpretations.

The same mapping rule applies to all six names:

| Unity Mathematics call and XYZ-valued input | QuatOps extrinsic sequence | QuatOps sequence-ordered angles |
| --- | --- | --- |
| `EulerXYZ(x, y, z)` | `"xyz"` | `(x, y, z)` |
| `EulerXZY(x, y, z)` | `"xzy"` | `(x, z, y)` |
| `EulerYXZ(x, y, z)` | `"yxz"` | `(y, x, z)` |
| `EulerYZX(x, y, z)` | `"yzx"` | `(y, z, x)` |
| `EulerZXY(x, y, z)` | `"zxy"` | `(z, x, y)` |
| `EulerZYX(x, y, z)` | `"zyx"` | `(z, y, x)` |

Unity Mathematics takes radians. Unity Editor 6000.5.2f1 resolves the
project's declared `com.unity.mathematics` 1.4.0 dependency as a built-in
module. The native fixture captures all six constructors over 37 shared input
triples (222 order/input pairs): 29 fixed mixed/boundary cases and eight
pseudo-random triples generated with `System.Random` seed `0x514F5053`. The
fixed inputs include cases around each middle-axis singularity and angles
outside the canonical range. The capture also records the six corresponding
`math.Euler...` inverse readbacks and checks their reconstructed rotations.
Readback values may use a different Euler branch; the proof compares rotation,
not equality to the input angles.

## Native basics and the explicit source frame

The fixture also captures native Unity and Unity.Mathematics cross products,
noncommuting quaternion multiplication, and a parent/child `Transform` using
unit scale. The runtime assertions verify the product action and that the
child's world position and rotation match local-then-parent composition.

The conversion example declares its source frame as right-handed (+X right,
+Y forward, +Z up) and maps it to Unity's documented world frame as
`SignedAxisMap(("x", "z", "y"))`, with determinant -1. The capture checks the
mapped quaternion action for each principal source-axis rotation. This map is
for the declared source and Unity world frames; it does not determine an
imported asset's local model-forward axis.

## Extended conversion proofs

The capture obtains quaternions from nine native matrices through both
`Matrix4x4.rotation` and `Unity.Mathematics.quaternion(float3x3)`. Axis-angle
construction uses `Quaternion.AngleAxis` and Mathematics `AxisAngle`; the
UnityEngine `ToAngleAxis` readback is sampled for both `q` and `-q`. Unity does
not expose quaternion log/exp in these captured APIs, so the rotation-vector
proof derives a principal vector through `AngleAxis`/`ToAngleAxis` and checks
its orientation. Native `Slerp`, `SlerpUnclamped`, and `math.slerp` samples
cover ordinary, long-path endpoint signs, antipodal, nearly identical,
near-half-turn, and extrapolated inputs. `Slerp` clamps its weight;
`SlerpUnclamped` and Mathematics `math.slerp` extrapolate. The captured public
APIs have no full-path interpolation option. Normalization records zero,
below/above-threshold, and nonunit inputs separately for UnityEngine and
Mathematics; `Quaternion.Normalize` returns identity for zero, while
`math.normalize` returns non-finite components.

The same capture samples `Quaternion.Lerp`, `LerpUnclamped`, and
`math.nlerp` over ordinary, endpoint-sign, antipodal, nearly identical, and
extrapolated inputs. The tests compare QuatOps `nlerp` to the native
unclamped routes and clamp the amount explicitly for `Quaternion.Lerp`.

The versioned fixture is
[`native_rotation.json`](../../tests/golden/unity/6000.5.2f1/native_rotation.json).
The C# project and rerun instructions are in
[`experiments/unity/6000.5.2f1`](../../experiments/unity/6000.5.2f1/README.md).
QuatOps-side checks compare native values across all 23 scalar and NumPy
representations; SciPy checks remain separate from the Unity runtime capture.
See the [proof matrix](proof-matrix.md) for parity across engines.

## Trajectory sampling

`Quaternion.Slerp(a, b, t)` expects unit endpoints and clamps `t` to `[0, 1]`;
`SlerpUnclamped` leaves `t` unclamped. Keep trajectory sampling weights in
`[0, 1]` when using `Slerp`, and use the unclamped method only when
extrapolation is intended.

```csharp
Quaternion rotation = Quaternion.Slerp(engineKeys[i], engineKeys[i + 1], alpha);
```

The native fixture exercises the available Unity shortest-path interpolation
routes but does not validate an imported model. For keyframe conversion and
sampling guidance, see the [trajectory guide](trajectory.md).

## Primary references

- [Unity 6000.5 `Quaternion` coordinate frame and Euler order](https://docs.unity.com/en-us/engine/6000.5/manual/scripting/programming-math/unity-engine-math/class-quaternion)
- [Unity 6000.5 `Quaternion.Euler`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unityengine/quaternion/euler)
- [Unity 6000.5 `Quaternion.eulerAngles`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unityengine/quaternion/eulerangles)
- [Unity 6000.5 `Quaternion.Slerp`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unityengine/quaternion/slerp)
- [Unity 6000.5 `Quaternion.SlerpUnclamped`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unityengine/quaternion/slerpunclamped)
- [Unity 6000.5 `Quaternion.Lerp`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unityengine/quaternion/lerp)
- [Unity 6000.5 `Quaternion.LerpUnclamped`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unityengine/quaternion/lerpunclamped)
- [Unity 6000.5 `Unity.Mathematics.quaternion`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unity/mathematics/quaternion)
- [Unity 6000.5 `Unity.Mathematics.math`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unity/mathematics/math)
- [Unity 6000.5 `math.nlerp`](https://docs.unity.com/en-us/engine/6000.5/script-reference/unity/mathematics/math/nlerp)
- [Unity Mathematics quaternion implementation](https://github.com/Unity-Technologies/Unity.Mathematics/blob/master/src/Unity.Mathematics/quaternion.cs)

For the shared basis-change math, see
[coordinate mappings](coordinate-mapping.md).
