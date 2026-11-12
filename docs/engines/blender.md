# Blender 5.1.2 quaternion adapter contract

This capture validates Blender's `mathutils.Quaternion` boundary in the
explicit identity probe frame assigned by the fixture. It does not establish
universal Blender world axes or an imported asset's local forward direction.

## Contract

- API exercised: `mathutils.Quaternion((w,x,y,z))` and
  `mathutils.Quaternion @ mathutils.Vector`.
- Native storage: Hamilton WXYZ, double-precision Python values in the pinned
  Blender 5.1.2 build.
- Canonical frame: right-handed, +X right, +Y forward, +Z up.
- Declared coordinate map: identity, with determinant +1, for this explicitly
  assigned probe frame. World and body use this same map.
- Rotation action: active native quaternion-vector action.
- Input domain: unit canonical quaternion. The QuatOps adapter preserves
  magnitude and does not normalize.

For canonical Hamilton `q_C=(w,x,y,z)`, the adapter emits the same WXYZ values
`q_B=(w,x,y,z)`. The inverse preserves that order and those values.

The native witnesses include identity, two unequal mixed rotations, an
antipode, a nonprincipal half-turn, both
noncommuting product orders, and the native action on each basis vector. The
selected goldens and provenance are in
[`tests/oracles/blender/5.1.2`](../../tests/oracles/blender/5.1.2/contract.json).
The native capture script and rerun instructions are in
[`experiments/blender/5.1.2`](../../experiments/blender/5.1.2/README.md).

This contract covers quaternion conversion only. Euler branches, animation,
scene import, and asset-axis corrections require separate contracts if QuatOps
adds those adapter features.
