# Unity 6000.5.2f1 quaternion adapter contract

This is the captured QuatOps converter contract for the pinned Unity Editor
build. Coordinate meaning is supplied by the declared frame map; Unity or an
asset name alone does not choose a source frame or model-forward axis.

## Contract

- API exercised: `UnityEngine.Quaternion(x,y,z,w)` and quaternion-vector
  multiplication (`Quaternion * Vector3`).
- Native storage: Hamilton XYZW, float32.
- Canonical frame: right-handed, +X right, +Y forward, +Z up.
- Declared coordinate map: native components are ordered from canonical axes
  `(+X,+Z,+Y)`, with determinant −1. World and body use this same map.
- Rotation action: active native quaternion vector rotation.
- Input domain: unit canonical quaternion. The QuatOps adapter preserves
  magnitude and does not normalize.

For canonical Hamilton `q_C=(w,x,y,z)`, the adapter emits
`q_Unity=(-x,-z,-y,w)` in native XYZW order. Its inverse is
`q_C=(W,-X,-Z,-Y)` from native `(X,Y,Z,W)`.

Native witness values include identity, three positive quarter-turns, two
unequal mixed rotations, an antipode, a nonprincipal half-turn, both orders of
their noncommuting product, and the native action on each basis vector. The
selected witnesses and provenance are in
[`tests/oracles/unity/6000.5.2f1`](../../tests/oracles/unity/6000.5.2f1/contract.json).
The pinned capture project and rerun instructions are in
[`experiments/unity/6000.5.2f1`](../../experiments/unity/6000.5.2f1/README.md).

This is quaternion conversion evidence. It does not claim an Euler adapter,
animation behavior, imported-scene parity, or a universal asset axis.

The registered `unity.world-axes` role follows Unity 6000.5's
[Quaternion API convention page](https://docs.unity.com/en-us/engine/6000.5/manual/scripting/programming-math/unity-engine-math/class-quaternion),
which states that Unity uses a left-handed coordinate system with +X right,
+Y up, and +Z forward. This describes the engine convention; it does not
establish an imported model's local axes or a project's physical alignment.
