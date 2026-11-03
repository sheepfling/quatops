# Coordinate-basis mappings

Keep a basis change separate from Euler order, active/passive choice, and
Hamilton/JPL algebra. This page records the signed-axis mappings in the
supplied research bundle; it does not claim that every asset import uses these
same bases.

## General formula

For a Hamilton quaternion, let `B` map vector components from source
coordinates to destination coordinates, so `v_D = B v_S`. If the same
orthogonal basis change applies to both the input and output vector coordinates
of a rotation, then

```text
R_D = B R_S B^T
q_D.xyz = det(B) B q_S.xyz
q_D.w = q_S.w
```

Here quaternions are written in XYZW order. The quaternion formula holds for
orthogonal `B`, including a reflection. It changes the coordinate components
of the same physical rotation; it does not switch Hamilton to JPL algebra.

## Mappings from the research bundle

| Source → destination | Vector components `v_D` | Quaternion components `q_D` (XYZW) | `SignedAxisMap` |
| --- | --- | --- | --- |
| Unity → Godot | `(x, y, -z)` | `(-qx, -qy, qz, qw)` | `SignedAxisMap(("x", "y", "-z"))` |
| Unity → Unreal | `(z, x, y)` | `(qz, qx, qy, qw)` | `SignedAxisMap(("z", "x", "y"))` |
| Unreal → Godot | `(y, z, -x)` | `(-qy, -qz, qx, qw)` | `SignedAxisMap(("y", "z", "-x"))` |

These are pure orientation-basis mappings under the bundle's stated basis
assumptions. For differing world and body bases, the rotation matrix relation
is `R_D = B_world R_S B_body^-1`; handle translation, scale, mesh-forward
correction, and import settings separately. `B_world` and `B_body` must have
the same determinant for `R_D` to remain a proper rotation representable by a
unit quaternion. A signed-axis map should not be applied to a mesh just because
its engine uses a familiar handedness.
Do not treat these as universal engine constants: choose a map for the actual
source axes and target scene/model axes.

The conversion math is covered by the
[SciPy basis oracle](../../tests/test_engine_basis_oracle.py). Native-runtime
comparison is still needed for the engine and asset pipeline actually
targeted; see [issue #4](https://github.com/sheepfling/quatops/issues/4) and
[issue #7](https://github.com/sheepfling/quatops/issues/7).
