# Godot quaternion adapter contract

The retained contracts cover Godot 4.7 stable and the captured 4.8-dev1 build.
The frame map is explicit and does not define a model-forward default for
imported assets.

## Contract

- API exercised: `Quaternion(x,y,z,w)` and `Quaternion * Vector3`.
- Native storage: Hamilton XYZW, float32.
- Canonical frame: right-handed, +X right, +Y forward, +Z up.
- Declared coordinate map: native components are ordered from canonical axes
  `(+X,+Z,-Y)`, with determinant +1. World and body use this same map.
- Rotation action: active native quaternion vector rotation.
- Input domain: unit canonical quaternion. The QuatOps adapter preserves
  magnitude and does not normalize.

For canonical Hamilton `q_C=(w,x,y,z)`, the adapter emits
`q_G=(x,z,-y,w)` in native XYZW order. Its inverse is
`q_C=(W,X,-Z,Y)` from native `(X,Y,Z,W)`.

Each captured version retains the selected native outputs for identity, two
unequal mixed rotations, an antipode, a nonprincipal
half-turn, both orders of their noncommuting product, and the three native basis
actions. The versioned contracts and witnesses are
[Godot 4.7](../../tests/oracles/godot/4.7-stable/contract.json) and
[Godot 4.8-dev1](../../tests/oracles/godot/4.8-dev1/contract.json). Capture
projects and rerun instructions are under
[`experiments/godot`](../../experiments/godot/README.md).

These contracts validate quaternion coefficient and physical-action transfer.
They do not claim an Euler adapter, animation behavior, imported-scene parity,
or a universal asset axis.

The registered `godot.camera-axes` role uses Godot 4.7's
[3D asset direction guidance](https://docs.godotengine.org/en/4.7/tutorials/assets_pipeline/importing_3d_scenes/model_export_considerations.html):
right-handed, +Y up, and camera forward along -Z. The same page distinguishes
camera forward from the oriented-asset convention, whose front points along
+Z. QuatOps keeps those as different roles; neither is inferred for an
arbitrary imported object.
