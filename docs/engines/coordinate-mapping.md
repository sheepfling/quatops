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
The cross product is an axial vector under the same basis change:
`(B u) × (B v) = det(B) B (u × v)`. This is the same determinant factor that
appears on the quaternion vector part. Native cross-product signs have been
captured separately for [Unreal](unreal.md), [Unity](unity.md), [Godot](godot.md),
and [Blender](blender.md); check them for any other target API because its
coordinate conventions may differ.

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

`quatops.engines.SignedAxisMap.matrix` exposes the row-major matrix for a
caller-declared signed permutation. `.inverse()` returns its transpose and
reverses vector or quaternion conversion. For a general orthogonal basis,
`quatops.experimental.engines.OrthogonalBasisMap` validates and stores the row-major 3x3
map. Use
`quatops.experimental.engines.change_rotation_basis(rotation, world_map, body_map)` for the
two-map matrix expression; omitting `body_map` uses the same map on both sides.
For quaternion-valued two-frame changes, compile a
`quatops.experimental.engines.QuaternionFrameMap` once and pass it to the named target
boundary (`to_unreal`/`from_unreal`, `to_unity`/`from_unity`,
`to_godot`/`from_godot`, or `to_blender`/`from_blender`). Unreal, Unity, and
Godot boundaries use Hamilton XYZW storage; Blender uses Hamilton WXYZ. These
functions only apply the supplied map and storage conversion; they do not
select world axes, model-forward axes, or an active/passive interpretation.
`change_rotation_basis` returns a matrix even if the maps have different
determinants, because such a result is useful to inspect but is not a proper
quaternion orientation. The scalar/frame primitive checks cover all 48 signed
maps and all 48-by-48 world/body map pairs in
[`test_engine_basis_oracle.py`](../../tests/quatops/experimental/engines/test_engine_basis_oracle.py).

## Different parent and child maps

For `R_target = B_parent R_source B_child^-1`, an identity source orientation
becomes `B_parent B_child^-1`. When the two maps differ this can be nonidentity:
the same aligned physical frames can use different coordinate bases. For
example, `B_parent = diag(-1, 1, 1)` and `B_child = diag(1, -1, 1)` yield
`diag(-1, -1, 1)`, a half turn, from a source identity. Their determinants match.

Composition needs a matching intermediate frame. In a connected frame chain,

```text
R'_AB R'_BC = (B_A R_AB B_B^-1) (B_B R_BC B_C^-1)
             = B_A (R_AB R_BC) B_C^-1
```

An arbitrary two-sided converter applied to both factors does not generally
preserve their product, because its child and parent maps need not cancel.
Use converters for A-from-B, B-from-C, and A-from-C with the same declared B
frame in the first two. `PhysicalOrientation.compose()` checks that shared
frame; gateway tests independently check the matrix action and cancellation.

For semantic frame IDs, axis descriptions, and common physical conventions,
use `quatops.experimental.semantics`. The common [`quatops.experimental.converters` gateway](../converters.md)
derives a map from the declared parent and child bases, then converts the
canonical Hamilton orientation to the selected target profile.

The native Unity, Godot, and Blender examples use separately declared source
frames; their specific maps are recorded on the [Unity](unity.md),
[Godot](godot.md), and [Blender](blender.md) pages rather than added here as
universal engine-pair mappings.

The reported SciPy results and their limits are summarized in
[verification](../contributing/VERIFICATION.md). The versioned native game-engine and
converter-boundary comparisons are indexed in the
[proof matrix](proof-matrix.md) and
[interoperability atlas](../INTEROPERABILITY_EVIDENCE.md). Runtime comparison
is still needed for any other target API or asset pipeline.
