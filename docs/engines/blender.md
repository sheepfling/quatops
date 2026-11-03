# Blender

This page records the Blender 5.1.2 `mathutils` and scene-object behavior in
the pinned native capture. The fixture is versioned under
[`tests/golden/blender/5.1.2`](../../tests/golden/blender/5.1.2/native_rotation.json);
the headless script and rerun instructions are in
[`experiments/blender/5.1.2`](../../experiments/blender/5.1.2/README.md).

## Quaternion, Euler, and matrix behavior

The `mathutils.Quaternion` constructor and iteration use WXYZ order. The
fixture keeps those native values and also writes XYZW arrays for QuatOps and
SciPy comparisons. `Quaternion @ Vector` is captured as an active vector
rotation. `mathutils.Matrix` values are recorded by rows and tested as matrices
acting on column vectors.

Blender exposes all six Tait-Bryan `Euler` orders and accepts radians. We
compared native `Euler.to_quaternion`, `Euler.to_matrix`, both quaternion and
matrix Euler readbacks, and matrix-to-quaternion conversion for 87 inputs per
order. Each order includes fifteen fixed cases, 64 deterministic seeded
triples, offsets around both signs of the middle-axis gimbal angle, and
float32-adjacent angle values. Noncommuting probes establish the sequence
mapping: Blender's uppercase labels match SciPy's lowercase **extrinsic**
sequences when the axis-labelled XYZ values are supplied in the named order.
The uppercase intrinsic SciPy candidates are checked separately and diverge
on the noncommuting inputs. QuatOps tests use the captured mapping across all
23 scalar and NumPy representations.

The runtime reports float32-rounded `mathutils.Euler` and `Quaternion` values
in the precision probe, although Python exposes them as `float` values. Euler
readbacks are checked by their reconstructed orientation rather than requiring
an identical canonical branch.

## Animation continuity

`Quaternion.to_euler(order, euler_compat)` is captured as a trajectory API.
For every order, the fixture records ordinary and compatibility readbacks
across both positive and negative 180-degree crossings and across a
middle-axis gimbal neighborhood. On the 180-degree paths, the ordinary
readback's largest component jump is about 358 degrees; the compatibility
route's largest is about 5 degrees. Near gimbal lock, compatibility does not
remove every component change. These captures provide reference behavior for
a possible reference-aware Euler conversion; they do not establish a
canonicalization contract for QuatOps.

## Extended quaternion behavior

The native capture includes eight matrix-to-quaternion probes, including
near-half-turn and negative-trace rotations. Axis-angle reads are recorded for
both `q` and `-q`. Those values preserve a sign-sensitive branch: for example,
the negative identity quaternion reports a `2π` angle, while an input above
180 degrees can return a readback angle above π. Rotation-vector construction
uses the `Quaternion` exponential-map constructor, and `to_exponential_map()`
returns the principal represented vector; a 220-degree Z input reads back as
approximately -140 degrees around Z.

`Quaternion.slerp` is sampled for ordinary, near-identical, antipodal,
near-half-turn, and greater-than-180-degree endpoints. The public method has no
full-path option. For an antipodal `q`/`-q` pair the captured interpolation
stays at the start orientation. `Quaternion.rotation_difference` matches
`start.inverted() @ end` on a noncommuting pair.

`Quaternion.to_swing_twist` is captured for X, Y, and Z twist axes. For the
mixed rotation probes, Blender's returned swing and twist reconstruct the
input as `swing @ twist`; reversing the product does not reproduce the same
orientation. The fixture includes a pure twist and an orthogonal half-turn to
show cases where the two product orders happen to agree.

Blender 5.1.2's `Quaternion.normalized()` returns a 180-degree X rotation for
the zero quaternion. That native fallback is recorded as engine behavior and
is not used as the QuatOps zero-normalization contract.

## Scene object and frame probes

The capture assigns one empty object's local rotation in `XYZ` Euler mode and
then quaternion mode, recording `rotation_mode`, `rotation_euler`,
`rotation_quaternion`, and `matrix_world`. Both routes produce the expected
orientation for this unit-scale, unparented object.

The source frame for the conversion probe is explicitly declared right-handed
with +X right, +Y forward, and +Z up. The target probe uses the same numeric
axes, so its captured signed-axis map is the identity with determinant +1.
This assignment applies only to the probe; it does not infer a model's local
forward axis from Blender or from an imported asset. Cross products and
principal-axis quaternion actions are also captured directly.

The capture is headless and uses Blender's factory-startup file. It does not
test an imported asset, animation F-curves, transform shear, or other Blender
versions. The captured build hash is `ec6e62d40fa9` and the embedded Python is
3.13.9. SciPy formula checks and QuatOps comparisons run after Blender exits.

## Primary references

- [Blender 5.1 `mathutils`](https://docs.blender.org/api/5.1/mathutils.html)
- [Blender 5.1 `bpy.types.Object`](https://docs.blender.org/api/5.1/bpy.types.Object.html)

For shared basis-change math, see
[coordinate mappings](coordinate-mapping.md). For version coverage across all
four engines, see the [proof matrix](proof-matrix.md).
