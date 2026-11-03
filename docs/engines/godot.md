# Godot

This page records the Godot 4 quaternion and basis behavior exercised in the
native capture. Keep `Quaternion`, `Basis`, and `Transform3D` APIs distinct;
their Euler-order controls and composition behavior are checked separately.

## Quaternion and basis APIs

`Quaternion(x, y, z, w)` uses XYZW order. `Quaternion.from_euler(Vector3(x,
y, z))` takes radians and uses the intrinsic YXZ order. `Basis.from_euler`
takes radians and accepts each of the six Tait-Bryan order constants. The
fixture labels its quaternions as active Hamilton rotations. It records the
default `Quaternion.from_euler` route, order-specific `Basis.from_euler`,
`Basis.get_euler` and `Quaternion.get_euler` readbacks, basis rows/columns/
vector images, and both matrix-to-quaternion constructors.

Both the installed official Godot 4.7 stable build and 4.8-dev1 build were run
as headless native captures; their runtime identities and hashes are recorded
in the versioned fixtures. The `Quaternion.from_euler` baseline covers fifteen
identity, principal-axis, mixed, singular-neighbor, over-90-degree, and
signed-180-degree cases. The all-order `Basis` experiment uses 67 shared
inputs for each of XYZ, XZY, YXZ, YZX, ZXY, and ZYX (402 captures per build).
Inputs include positive and negative 90-degree neighborhoods, float32-adjacent
values around both signs of 90 degrees on every axis, values outside the
canonical range, and eight deterministic seeded samples. Both readback APIs
are roundtripped in each order.

QuatOps' intrinsic Euler routes match the native basis rotations and SciPy's
intrinsic-order matrix formula for both pinned builds. The default YXZ
near-gimbal `Quaternion.get_euler` then `from_euler` reconstruction has about
0.00161 degrees of residual in the 4.8-dev1 build. These are float32 runtime
results for the two captured versions.

`Node3D.rotation_order` controls local rotation order and defaults to YXZ;
`Node3D.global_rotation` uses YXZ. Treat these node properties separately
from the explicit order arguments accepted by `Basis` and `Quaternion` APIs.

## Native basics and the explicit source frame

Godot's world frame in this fixture is right-handed: +X right, +Y up, +Z
backward, with forward along -Z. Native assertions capture cross-product
results, noncommuting quaternion multiplication, and `Transform3D` composition
with unit scale. The product action and the transform's resulting rotation,
translation, and probe position are compared against the right operand applied
first.

The conversion example explicitly declares a right-handed source frame (+X
right, +Y forward, +Z up) and maps it to Godot as
`SignedAxisMap(("x", "z", "-y"))`, with determinant +1. The capture checks the
mapped quaternion action for each principal source-axis rotation. This map
describes the declared source and Godot world frames; it does not determine an
imported asset's local model-forward axis.

## Extended conversion proofs

The native script captures both `Basis.get_rotation_quaternion` and
`Quaternion(Basis)` for ten valid matrices, including a near-180-degree
negative-trace case. Axis-angle construction and `get_axis`/`get_angle`
readbacks are sampled for `q` and `-q`. The separate `Quaternion.exp`/`log`
probe records identity, tiny, 90-degree, 180-degree, mixed, and antipodal
branches: Godot's `exp` accepts the full rotation vector, while `log` returns
that full vector on the tested unit rotations. Its `q`/`-q` log branches retain
the coefficient sign (for example, the negative 90-degree quaternion reports
a -270-degree vector); QuatOps `to_rotvec` uses a principal branch, so those
raw vectors are compared with the captured Godot values and reconstructed by
orientation rather than requiring identical branch choice. A float32 tiny
vector rounds to identity in Godot.

The fixture also has three explicitly diagnostic near-collinear bases with
determinants `1e-2`, `1e-4`, and `1e-6`. Both native conversion routes return
identity for these inputs in Godot 4.7 stable and 4.8-dev1. The direct
`Quaternion(Basis)` constructor emits Godot's orthonormality error; these
non-orthonormal inputs are not sent through QuatOps' valid-rotation matrix
conversion.

`Quaternion(from, to)` captures parallel, orthogonal, near-parallel,
near-antiparallel, exact antiparallel, and nonunit-vector inputs. The 1e-5
near-parallel probe returns identity within the capture tolerance. The
1e-4 near-antiparallel probe misses the target by 1e-4; the 1e-2 probe
resolves to within about 9.51e-6. Exact antiparallel rotations are not unique,
so the fixture records their native axis and vector action without prescribing
an axis choice.

Native `slerp` and `slerpni` are sampled at ordinary, long-path, antipodal,
near-identical, near-half-turn, and extrapolated inputs. The near-identical
`slerpni` midpoint differs from the direct QuatOps/SciPy formula by about
0.0005 radians in this float32 build. Normalization covers zero,
below/above-threshold, and nonunit inputs; zero produces non-finite components,
recorded as JSON `null` with a false finite flag.

The versioned fixtures are
[`Godot 4.7 stable`](../../tests/golden/godot/4.7-stable/native_rotation.json)
and [`Godot 4.8-dev1`](../../tests/golden/godot/4.8-dev1/native_rotation.json).
Their GDScript projects and rerun instructions are in
[`experiments/godot`](../../experiments/godot/README.md).
QuatOps-side checks compare native values across all 23 scalar and NumPy
representations; SciPy checks remain separate from the Godot runtime capture.
See the [proof matrix](proof-matrix.md) for parity across engines.

## Trajectory sampling

`Quaternion.slerp` requires normalized endpoints and applies Godot's default
path check. `slerpni` omits that path check; use it only when the input signs
intentionally encode a different path. For ordinary keyframed motion, convert
the keys once and sample adjacent keys with `slerp`.

```gdscript
var rotation: Quaternion = engine_keys[i].slerp(engine_keys[i + 1], alpha)
```

The native fixture exercises both Godot interpolation routes but does not
validate an imported model. For keyframe conversion and sampling guidance, see
the [trajectory guide](trajectory.md).

## Primary references

- [Godot 4.5 `Quaternion`](https://docs.godotengine.org/en/4.5/classes/class_quaternion.html)
- [Godot 4.5 `Basis`](https://docs.godotengine.org/en/4.5/classes/class_basis.html)
- [Godot 4.5 `Node3D`](https://docs.godotengine.org/en/4.5/classes/class_node3d.html)
- [Godot 4.5 `Transform3D`](https://docs.godotengine.org/en/4.5/classes/class_transform3d.html)
- [Captured Godot commit: `quaternion.cpp`](https://github.com/godotengine/godot/blob/ebbf577a041d1ab8d7824b6f90e7d0575461ca45/core/math/quaternion.cpp)
- [Captured Godot commit: `basis.cpp`](https://github.com/godotengine/godot/blob/ebbf577a041d1ab8d7824b6f90e7d0575461ca45/core/math/basis.cpp)
- [Captured Godot commit: `transform_3d.cpp`](https://github.com/godotengine/godot/blob/ebbf577a041d1ab8d7824b6f90e7d0575461ca45/core/math/transform_3d.cpp)

For the shared basis-change math, see
[coordinate mappings](coordinate-mapping.md).
