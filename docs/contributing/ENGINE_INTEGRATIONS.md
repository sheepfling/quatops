# Engine conversion development guide

This guide is for developers adding or integrating game-library quaternion
conversions. The package-level contract is in [`../API.md`](../API.md); this
directory owns the engine and trajectory references.

## Define the conversion before coding

Record these facts for the actual data and API being connected:

| Fact | Required detail |
| --- | --- |
| Source frame | Right-handed axes and what each positive axis means |
| Quaternion action | Active rotation or passive frame transform |
| Quaternion representation | Hamilton or JPL, and `wxyz` or `xyzw` coefficient order |
| Target frame | Engine world axes and the target model/asset axes |
| Engine API | Exact class, methods, precision, and supported engine version |
| Trajectory path | Shortest path or authored full turn, with timestamps |

Handedness describes the coordinate basis; it does not decide Hamilton versus
JPL algebra. A passive transform must be inverted before applying the active
rotation formula. Reordering four coefficients does not convert Hamilton to
JPL. The [trajectory guide](trajectory.md) explains these boundaries.

## Implement the lowest-cost boundary

For a signed axis permutation, construct `SignedAxisMap` once and map the
trajectory keys when importing or loading them. The map uses precomputed
selectors and sign changes; it does not normalize, change quaternion signs, or
call an engine. For NumPy batches, map the complete key array in one call.
Keep the per-sample hot path in the destination engine's native quaternion
type and SLERP implementation. Do not route an interpolation trajectory
through Euler angles.

This removes matrix conversion and basis mapping from every sample. It does
not claim that Python calls, output objects, copying the input array, or engine
precision have literally zero cost; measure those costs in the actual
integration before making a performance claim.

Use the signed-axis helper only when the source-to-destination basis is an
orthogonal signed permutation. If world and body/model bases differ, derive
`R_D = B_world R_S B_body^-1` explicitly and verify that the result remains a
proper rotation. Keep translation, scale, asset import settings, and mesh
correction outside the quaternion-only map.

## Establish mathematical correctness

The [SciPy oracle](../../tests/test_engine_basis_oracle.py) independently
constructs the basis matrix from the declared axes, obtains source rotation
matrices from SciPy, and compares the result with `B R B^T`. It covers all 48
signed axis permutations, Hamilton and JPL source values, both coefficient
orders, NumPy layout and dtype behavior, and sampled SLERP/map commutation.
The expected matrix does not call the conversion implementation under test.

The identity behind trajectory conversion is
`map(slerp(q0, q1, u)) = slerp(map(q0), map(q1), u)` when both sides use the
same endpoint sign choice and path policy. A basis map preserves quaternion
norms and dot products, so it preserves shortest-path selection. A half-turn
has two equally short paths; the authored signs or an extra key must make the
choice explicit.

Run the focused and complete suites when changing implementation code:

```sh
.venv/bin/pytest -q tests/test_engine_basis_oracle.py
.venv/bin/pytest -q
```

The SciPy basis oracle and upstream Euler fixture suites cover the shared math
separately from the native engine captures. The latest complete QuatOps run,
including the Unreal, Unity, Godot, and Blender fixture suites, is recorded in
[verification](verification.md) with its interpreter and dependency versions.

## Establish engine-runtime correctness

SciPy proves the SO(3) formula for the declared basis. It cannot prove that a
particular engine release implements the documented API exactly as expected,
or that an imported model uses the assumed local axes. Before claiming an
engine integration is runtime-verified:

1. Pin the engine and API/package versions in the integration's test setup.
2. Exercise identity and positive quarter-turn rotations about each source
   axis, then a nontrivial mixed-axis rotation.
3. Compare the engine's rotated basis vectors or rotation matrix with the
   independently computed `B R B^T`, not only quaternion component values.
4. Check constructor order, quaternion multiplication order, active/passive
   action, and float precision through the exact target API.
5. Sample native SLERP at `u = 0`, `0.5`, and `1`, plus near-identical and
   near-half-turn endpoints; compare the represented rotations with SciPy.
6. Exercise one real imported asset and confirm its model-forward and up axes
   after import. Record importer settings separately from the world-basis map.

Keep this runtime evidence distinct from the SciPy oracle evidence. This
checkout records versioned native captures for Unreal 5.7.4 Rotator/FQuat,
Unity 6000.5.2f1, Godot 4.7 stable and 4.8-dev1, and Blender 5.1.2. Their
shared proof categories include Euler branch samples, matrix conversion,
axis-angle, rotation vectors, native shortest-path SLERP, normalization edges,
basis action, composition, and explicit source-frame probes. Unreal and Godot
also expose a captured full-path route; the selected Unity and Blender APIs do
not. Unity rotation vectors are reconstructed through axis-angle APIs. Both
Godot captures cover all six `Basis` Euler orders; Blender adds all six
`mathutils.Euler` orders, animation continuity trajectories, and swing-twist
decomposition. Supplementary Godot from-to behavior is recorded on the Godot
page. These fixtures do not test imported assets, so they establish pinned
conversion/API behavior rather than complete engine integration validation.
See the [proof matrix](proof-matrix.md) and
[verification](verification.md).
