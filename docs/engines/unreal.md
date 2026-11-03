# Unreal Engine

This page covers Epic's public `FRotator` and `FQuat` API and records a native
runtime capture from Unreal Engine 5.7.4. That capture pins the behavior of the
captured build; integrations using another Unreal release should capture and
check that release separately.

## Euler API

`FRotator` stores `Pitch`, `Yaw`, and `Roll` in degrees. Epic describes its
intrinsic rotation sequence as Yaw, then Pitch, then Roll. The Blueprint
`Make Rotator` node exposes pins as X = Roll, Y = Pitch, Z = Yaw, so the pin
order differs from the C++ field listing. Do not pass an XYZ vector to either
surface without explicitly reordering it.

The attached research bundle models these values against a right-handed,
active matrix convention as
`R = Rz(yaw) Ry(-pitch) Rx(-roll)`. The native UE 5.7.4 cases in the
[fixture](../../tests/golden/unreal/5.7.4/native_rotation.json) match this
model for both captured precisions. Treat it as evidence for the pinned
`FRotator`/`FQuat` build and tested cases; another release should be captured
separately.

## Coordinate and composition conventions

`FQuat` exposes XYZW components and takes X, Y, Z, W in its component
constructor. Treat that as API value order; do not infer a serialization or
binary memory layout from it.

Unreal documents its native coordinate system as left-handed, with X forward,
Y right, and Z up. That describes the world basis; it does not make the
quaternion algebra JPL. Keep basis conversion separate from the choice of
Hamilton or JPL representation in QuatOps.

The native capture confirms the conventional numeric cross-product identity
`e_x × e_y = e_z`, with the reversed inputs producing `-e_z`. Unreal's world
frame is left-handed because its physical axis labels are +X forward, +Y right,
and +Z up. The cross-product values and the world-frame handedness are separate
facts.

For an explicit source-to-Unreal example, the supplemental experiment declares
a right-handed source frame with +X right, +Y forward, and +Z up. Against
Unreal's documented world frame, that gives `B = (y, x, z)` with determinant
−1. The native `FQuat` probes confirm the converted principal rotations and
their rotated basis images. This is one declared mapping, not a default for
every source frame or imported asset.

Quaternion and transform composition orders differ. The native capture
confirms that `FQuat A * B` applies B first and then A. For a point `p`,
`(FTransform A * B).TransformPosition(p)` instead matches
`B.TransformPosition(A.TransformPosition(p))`: A acts first, then B. The
experiment uses unit scale so rotation and translation determine the result;
it does not cover shear from nonuniform scale.

`FQuat::Slerp` corrects endpoint alignment and returns a normalized result.
`SlerpFullPath` preserves endpoint signs and does not check for the shortest
distance, so choose signs deliberately. The native capture freezes both
normalized methods for ordinary interpolation, a deliberate 270° long path,
antipodal endpoints, nearly identical rotations, near-half-turn separation,
and extrapolation. The captured full-path interpolation returns identity for
the interior antipodal samples after Unreal normalizes its degenerate blend.
The same path is
nonunique in QuatOps, where `shortest=False` raises `ValueError`.
For a sampled trajectory, convert and normalize keyframes once, then sample
between adjacent keys with the intended interpolation method.

```cpp
const FQuat rotation = FQuat::Slerp(engineKeys[i], engineKeys[i + 1], alpha);
```

## Native capture

[`native_rotation.json`](../../tests/golden/unreal/5.7.4/native_rotation.json)
contains raw results captured by Unreal Engine 5.7.4, changelist 51494982,
branch `++UE5+Release-5.7`, on WindowsEditor x64. The C++ Automation Test
exercised `FRotator3f/FQuat4f` and `FRotator3d/FQuat4d` using named Pitch, Yaw,
and Roll fields across fifteen Euler inputs. It records quaternion
coefficients in XYZW order, both native Euler readbacks, rotated positive
basis vectors, `TQuat::ToMatrix`, and multiplication coefficients with sign
bits. Each case also captures `TRotator::MakeFromEuler` and
`TQuat::MakeFromEuler` using the explicit `(Roll, Pitch, Yaw)` Euler vector and
checks both APIs against the named-field Rotator conversion. The capture source
and rerun steps are in
[`experiments/unreal/5.7.4`](../../experiments/unreal/5.7.4/README.md).

Unreal's captured `ToMatrix` values are in row-vector layout: each matrix row
matches the `RotateVector` image of one positive basis vector. They therefore
match the transpose of SciPy's usual column-vector rotation matrix. This
transpose changes matrix layout only; do not apply a handedness sign flip to
it. Apply a signed-axis basis map only when converting between the explicitly
declared source and target frames. The Python commandlet probe was run first,
but its reflected Rotator wrapper quantized the input angles to float32. The
C++ capture supplies the native precision-specific results used by the
fixture.

The four added pitch probes are 89.91°, 89.93°, 90.07°, and 90.09°. In this
build, the middle two read back through the singular branch as exactly 90°;
89.91° reads back near 89.91°, and 90.09° returns near 89.91° with yaw and
roll at 180°. The test records the float32 and float64 outputs separately and
allows the expected orientation loss in these threshold-neighbor readbacks.

The `extended_conversions` section captures `TQuat(TMatrix)` for identity,
principal quarter and half turns, a mixed-axis rotation, and a negative-trace
rotation. It also records a slightly perturbed matrix, a reflection, and a
zero-axis matrix as native-only diagnostics; those are not treated as proper
rotation inputs by QuatOps. Axis-angle and rotation-vector constructors and
readbacks cover zero, tiny, mixed, exact-π, and just-over-π values. Unreal's
`ToAxisAndAngle` is recorded for both `q` and `-q`; the angle branches retain
their native values instead of being canonicalized to a shortest angle. For a
tiny vector part, Unreal's axis readback falls back to +X. The axis-angle
constructor assumes a unit axis, and the fixtures use unit axes.
At the `1e-7` tiny-angle probe, float32 `ToRotationVector(-q)` returns the tiny
negative input vector. In float64, the pinned build's `Log()` path instead
returns a vector with magnitude about 5.8473 radians, which does not reconstruct
the original rotation. The fixture and tests preserve this precision-specific
engine result as a numerical branch anomaly.

The capture also records `GetNormalized()` for identity, zero, below- and
above-threshold, and nonunit inputs. Unreal compares squared magnitude with
`UE_SMALL_NUMBER` (`1e-8`) and returns identity below the threshold. QuatOps
rejects an exactly zero quaternion and normalizes finite nonzero inputs,
including values below Unreal's threshold. The fixture keeps this compatibility
difference visible.

[`test_unreal_engine_golden.py`](../../tests/test_unreal_engine_golden.py)
compares Euler-to-quaternion orientation, all three axes and matrices,
the two `MakeFromEuler` APIs, quaternion-to-Euler readbacks by represented
orientation, and multiplication coefficients across all 23 QuatOps
representations. It separately pins the
gimbal-lock, four singularity-threshold neighbors, pitch-over-90, and
signed-180-degree readback branches for both precisions.
[`test_unreal_extended_conversions.py`](../../tests/test_unreal_extended_conversions.py)
checks matrix-to-quaternion, axis-angle, rotation-vector, normalized SLERP and
full-path SLERP samples, and normalization behavior across all 23
representations. Explicit formulas pin the `q`/`-q` axis-angle and rotation-
vector branches, including the float64 tiny-`-q` anomaly. Its full-path
antipodal assertion records the intentional QuatOps/Unreal difference.
[`test_unreal_native_semantics.py`](../../tests/test_unreal_native_semantics.py)
compares the declared right-handed source mapping with the captured Unreal
quaternion rotations across those representations, and pins the cross-product
and composition experiments. These checks do not validate an imported asset's
model axes.

## Primary references

- [Unreal 5.7 TRotator API](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TRotator?application_version=5.7)
- [Unreal 5.7 Make Rotator Blueprint API](https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Math/Rotator/MakeRotator?application_version=5.7)
- [Unreal 5.7 coordinate system and spaces](https://dev.epicgames.com/documentation/en-us/unreal-engine/coordinate-system-and-spaces-in-unreal-engine?application_version=5.7)
- [Unreal 5.7 TQuat API](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TQuat?application_version=5.7)
- [Unreal 5.7 TTransform API](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TTransform?application_version=5.7)
- [Unreal 5.7 TQuat::SlerpFullPath](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Core/TQuat/SlerpFullPath?application_version=5.7)

For the research bundle's component-basis mapping, see
[coordinate mappings](coordinate-mapping.md). For keyframe conversion and
sampling, see the [trajectory guide](trajectory.md).
