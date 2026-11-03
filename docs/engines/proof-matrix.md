# Native engine proof matrix

This matrix tracks what the pinned native captures actually execute. “Parity”
means the four engines have the same core proof categories and comparable edge
inputs; it does not mean that their APIs expose identical operations.

| Proof category | Unreal 5.7.4 | Unity 6000.5.2f1 | Godot 4.7 stable and 4.8-dev1 | Blender 5.1.2 |
| --- | --- | --- | --- |
| Euler construction and readback | `FRotator::Quaternion`, `TRotator::MakeFromEuler`, `TQuat::MakeFromEuler`, `FQuat::Rotator/Euler`; fifteen samples include four singularity neighbors | `Quaternion.Euler` (ZXY); all six `Unity.Mathematics.quaternion.Euler...` constructors and matching `math.Euler...` readbacks over 37 triples each (29 fixed, 8 seeded); UnityEngine samples include four X-axis singularity neighbors | `Quaternion.from_euler` and `get_euler(YXZ)` baseline over fifteen samples; all six `Basis.from_euler`, `Basis.get_euler`, and `Quaternion.get_euler` routes over 67 shared inputs per order, including float32 neighbors around ±90° on every axis | `Euler.to_quaternion`, `Euler.to_matrix`, `Quaternion.to_euler`, and `Matrix.to_euler`; 87 inputs per order across all six orders, including 64 seeded triples and float32-adjacent gimbal probes. Native order labels match SciPy lowercase extrinsic sequences. |
| Matrix to quaternion | `TQuat(TMatrix)` in float32 and float64 | `Matrix4x4.rotation` and `Unity.Mathematics.quaternion(float3x3)` | `Basis.get_rotation_quaternion` and `Quaternion(Basis)` for ten valid matrices; three near-collinear inputs are native-only diagnostics | `Matrix.to_quaternion` for eight proper rotations, including near-half-turn and negative-trace matrices |
| Axis-angle | Native constructor and axis/angle readback for `q` and `-q` | `Quaternion.AngleAxis` / `ToAngleAxis` and Mathematics `AxisAngle`; readback is through UnityEngine | Native `Quaternion(axis, angle)`, `get_axis`, and `get_angle`, including `q` and `-q` | `Quaternion(axis, angle)` and `to_axis_angle`, with sign-pair branches |
| Rotation vector | Native constructor and `ToRotationVector`, float32 and float64 | Derived from native `AngleAxis` / `ToAngleAxis`; neither captured API exposes quaternion log/exp | Native `Quaternion.exp` / `log`; captures the full-vector convention, tiny input, 180° and antipodal branches | Exponential-map constructor and `to_exponential_map`, including tiny, half-turn, and greater-than-180° inputs |
| Shortest-path SLERP | Native `FQuat::Slerp` | `Quaternion.Slerp`, `SlerpUnclamped`, and Mathematics `math.slerp` | Native `Quaternion.slerp` | `Quaternion.slerp` across ordinary, near-identical, antipodal, near-half-turn, and greater-than-180° endpoints |
| Full-path SLERP | Native `FQuat::SlerpFullPath` | No full-path option exposed by the captured public quaternion APIs | Native `Quaternion.slerpni` | No full-path option exposed by `mathutils.Quaternion.slerp` |
| Normalization edges | `GetNormalized()` with zero, below/above `1e-8` squared tolerance, and nonunit input; float32 and float64 | `Quaternion.Normalize` and `math.normalize`; zero is identity for UnityEngine and non-finite for Mathematics | `Quaternion.normalized`; zero input produces non-finite components in this build | `Quaternion.normalized` for zero, tiny, unit, sign-reversed, and nonunit input; zero returns a half-turn around +X |
| Basis/vector action | `RotateVector` and `ToMatrix`; separate native semantics assertions | Quaternion-vector action and `Matrix4x4.Rotate` | Quaternion-vector action and `Basis` columns/rows | `Quaternion @ Vector`, `Matrix @ Vector`, and object `matrix_world` |
| Cross product | Native positive-axis pairs and reversals, float32 and float64 | `Vector3.Cross` and `math.cross` for positive-axis pairs and reversals | Native positive-axis pairs and reversals | Native positive-axis pairs and reversals |
| Composition and source-frame map | Separate `FQuat` and `FTransform` order probes; explicit right-handed source frame | UnityEngine and Mathematics quaternion products, parent/child `Transform`, explicit right-handed source frame | Quaternion and `Transform3D` order probes; explicit right-handed source frame | `q1 @ q2` application order, `rotation_difference`, and an explicitly declared identity source-frame map |

Blender's supplemental animation probes include ordinary and `euler_compat`
readbacks over positive and negative 180° trajectories and middle-axis gimbal
paths. Its swing-twist probe records X/Y/Z decomposition and reconstruction
order. These categories are engine-specific additions; they are not required
of the other three APIs.

Every fixture records a `proof_profile` so future versions can report the same
categories while naming their actual API route. The JSON lives at
[`tests/golden/unreal/5.7.4/native_rotation.json`](../../tests/golden/unreal/5.7.4/native_rotation.json),
[`tests/golden/unity/6000.5.2f1/native_rotation.json`](../../tests/golden/unity/6000.5.2f1/native_rotation.json),
both Godot fixtures:
[`4.7 stable`](../../tests/golden/godot/4.7-stable/native_rotation.json) and
[`4.8-dev1`](../../tests/golden/godot/4.8-dev1/native_rotation.json), plus
[`Blender 5.1.2`](../../tests/golden/blender/5.1.2/native_rotation.json).
Godot's additional from-to vector-constructor cases are a supplemental
Godot-specific probe, not one of the shared proof categories.
The independent native capture projects and rerun commands are indexed from
[`experiments/README.md`](../../experiments/README.md).

The conversion checks compare the frozen native values with all 23 QuatOps
scalar and NumPy representations. SciPy matrix/orientation checks are separate
Python evidence and do not run inside an engine. The native captures also do
not establish an imported asset's local axes, cover transform shear, or certify
other engine releases. See each [engine page](README.md) for its precise frame,
API, and version facts.

The separate [CAP-300 converter matrix](CAP-300.md) tracks canonical to native,
native to canonical, native physical-vector equivalence, noncommuting
composition, and Unity-to-Godot transfer. Those checks use native actions
captured from the pinned runtimes; the SciPy reference remains separate.
