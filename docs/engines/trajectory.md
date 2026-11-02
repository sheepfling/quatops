# Convert quaternion trajectories for an engine

The target is a piecewise rotation trajectory that has the same physical path
after moving from a QuatOps quaternion to an engine's coordinate basis. Convert
the keyframes once at the boundary; keep runtime sampling in the engine's
quaternion type. Euler angles are for authoring or display, not an intermediate
trajectory format.

## Required input contract

Before converting, identify all of these for the source data:

1. Quaternion coefficient order (`wxyz` or `xyzw`).
2. Algebra and action: Hamilton or JPL, and active rotation or passive frame
   transform.
3. The source frame axes and how source world/body axes map to the engine's
   world/body axes.
4. Keyframe timestamps and whether the authored path means shortest rotation
   between keys or a deliberate longer turn.

“Standard quaternion” specifies too little to choose a unique engine mapping.
QuatOps documents active, right-handed rotation math, but the caller still has
to identify what each source axis means (for example, which way is forward).
For a passive transform, invert the quaternion before treating it as an active
rotation. If the source is JPL, use the JPL API consistently or explicitly map
it to the Hamilton rotation before applying a Hamilton formula; coefficient
reordering alone does not change algebra conventions. The pairwise maps on the
[coordinate page](coordinate-mapping.md) are only valid for the source and
destination frames stated there.

## One-time conversion, then native SLERP

For Hamilton quaternions in a common source/destination basis, let `B` map
source vector coordinates into destination coordinates. The same rotation in
the destination basis is

```text
R_D = B R_S B^T
q_D.xyz = det(B) B q_S.xyz
q_D.w = q_S.w
```

For signed axis permutations, this is only component selection and sign
changes. No Euler conversion, 3x3 matrix multiply, or runtime basis multiply is
needed. A full source-frame to engine-frame conversion can also include a
separate body/model basis; see the coordinate page before reducing it to a
single component map.

Compile the axis mapping once, then apply it at import time:

```python
from quatops.engines import SignedAxisMap

# Destination components are (source x, source y, -source z).
to_engine = SignedAxisMap(("x", "y", "-z"))
engine_keys = tuple(
    to_engine.map_hamilton_xyzw(q) for q in source_keys_xyzw
)
```

For NumPy trajectories, pass the complete key array to
`quatops.engines.numpy.map_hamilton_xyzw`; it maps all keys in one vectorized
call and preserves the selected component axis. The map also has explicit
Hamilton/JPL and `wxyz`/`xyzw` entry points. JPL entry points convert directly
to Hamilton XYZW coefficients for the target engine, without an intermediate
JPL-to-Hamilton allocation.

If source and destination already use the same basis, quaternion rotation
convention, and `xyzw` component order, the keyframe conversion is the identity.
If only the source uses `wxyz`, reorder to the engine API's `xyzw` constructor
order once. This describes the API values, not a promise about a library's
binary memory layout.

The quaternion map is an orthogonal map of the unit 4-sphere. It preserves
norms, coefficient dot products, and therefore shortest-path selection. For
the same endpoint sign choice and interpolation parameter `u`,

```text
map(slerp(q0, q1, u)) = slerp(map(q0), map(q1), u)
```

So the low-overhead workflow is:

1. Normalize the source keys once if they are not already unit quaternions.
2. For shortest-path interpolation, run the selected source backend's
   `make_sign_continuous` once over the keys. Do this before mapping so the
   backend's coefficient order and convention remain explicit. Skip it when
   authored signs intentionally encode a longer path.
3. Convert the source keyframes into the target engine's component order and
   basis once when loading or importing the trajectory. A signed permutation
   preserves the norm, so it requires no second normalization.
4. Preserve the timestamps. For each sample time `t` in `[t_i, t_(i+1)]`,
   compute `u = (t - t_i) / (t_(i+1) - t_i)`.
5. Call the engine's quaternion SLERP for keys `i` and `i+1` with `u`.

The engine path then performs no Euler round trip and no basis conversion per
frame. With linearly advancing time inside one segment, SLERP has constant
angular speed on that segment. Angular velocity can still change at keyframe
boundaries; a spherical cubic method such as SQUAD can provide continuous
angular velocity when its tangent controls are chosen for that purpose.

QuatOps `make_sign_continuous` can prepare a key sequence in the selected
source backend. QuatOps `slerp` also normalizes each pair and flips the second
endpoint by default. Coordinate conversion preserves the dot products, so
shortest-path decisions are preserved by the map. At an exact 180-degree relative
rotation, two equally short paths exist; use an authored sign or an extra key
to make the intended path clear. If the trajectory intentionally spins beyond
the shortest path between samples, add intermediate keys or use a full-path
interpolator with deliberate quaternion signs.

## Engine interpolation notes

| Engine API | Sampling note |
| --- | --- |
| Unreal `FQuat::Slerp` | Corrects alignment and returns normalized output. `SlerpFullPath` skips shortest-distance correction. |
| Unity `Quaternion.Slerp` | Unit endpoints; clamps `u` to `[0, 1]`. Use `SlerpUnclamped` for intentional extrapolation. |
| Godot `Quaternion.slerp` | Both endpoints must be normalized; uses Godot's default path check. `slerpni` omits that check. |

Use `u` within `[0, 1]` for trajectory segments. When implementing an adapter,
keep a SciPy matrix oracle for the conversion math, then compare the selected
engine's native SLERP separately for precision and edge-case behavior,
especially near-identical endpoints and half-turns. See the per-engine pages
for versioned source references.
