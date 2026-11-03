# Unreal Engine 5.7.4 quaternion adapter contract

This page specifies the captured `FQuat` boundary for the explicit QuatOps
source frame. It does not select defaults for other Unreal versions, assets, or
imported model-forward axes.

## Contract

- API exercised: `UE::Math::TQuat<T>(X,Y,Z,W)` and `TQuat::RotateVector`.
- Native storage: Hamilton XYZW; `FQuat4f` and `FQuat4d` were captured.
- Canonical frame: right-handed, +X right, +Y forward, +Z up.
- Declared coordinate map: native components are ordered from canonical axes
  `(+Y,+X,+Z)`, with determinant −1. World and body use this same map.
- Rotation action: active native quaternion vector rotation.
- Input domain: unit canonical quaternion. The QuatOps adapter preserves
  magnitude and does not normalize.

For canonical Hamilton `q_C=(w,x,y,z)`, the captured adapter rule is
`q_U=(-y,-x,-z,w)` in native XYZW order. Its inverse is
`q_C=(W,-Y,-X,-Z)` from native `(X,Y,Z,W)`.

The rule is accepted against the native quaternion values and native images of
the three basis vectors and a mixed probe. The retained, selected witnesses and
build/exporter provenance are in
[`tests/oracles/unreal/5.7.4`](../../tests/oracles/unreal/5.7.4/contract.json).
The capture project and exact rerun commands are documented in
[`experiments/unreal/5.7.4`](../../experiments/unreal/5.7.4/README.md).

This contract covers quaternion coefficient and physical-action conversion. It
does not claim an Euler adapter, transform/asset import behavior, or a default
model axis.
