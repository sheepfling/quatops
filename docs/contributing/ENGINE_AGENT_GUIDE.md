# Coding-agent guide for engine rotations

Read this guide when changing `quatops.engines`, engine-specific references,
or quaternion trajectory conversion. For implementation workflow, use the
[developer guide](DEVELOPMENT.md). For concrete API details, use the separate
[Unreal](unreal.md), [Unity](unity.md), [Godot](godot.md), and
[Blender](blender.md) pages.

## Required reasoning

- Treat the source as an explicit right-handed frame. State what its positive
  axes mean and obtain the source-to-target signed-axis map from those meanings.
- Track active versus passive transforms. Invert a passive transform before
  using an active-rotation formula.
- Track Hamilton versus JPL algebra separately from coordinate handedness.
  Track `wxyz` versus `xyzw` separately from both.
- Treat engine world axes and an asset's model-forward axis as separate input
  facts. Engine name alone does not select a universal conversion.
- For an active Hamilton rotation and one shared orthogonal basis change `B`,
  use `R_D = B R_S B^T` and `q_D.xyz = det(B) B q_S.xyz`, with `q_D.w = q_S.w`.
  For different world and body bases, use the two-basis matrix expression in
  [coordinate mapping](coordinate-mapping.md); do not force it into a single
  axis map unless the derivation permits it.
- Map keyframes once, preserve timestamps and deliberate quaternion signs, and
  use the engine's native SLERP at sample time. Do not convert trajectories
  through Euler angles.

## Required evidence and reporting

- Add or update deterministic SciPy matrix-oracle cases for conversion math.
  Construct the expected basis independently from the code under test.
- Check interpolation separately: compare mapped native/library samples with
  the SciPy trajectory under the same endpoint-sign and shortest/full-path
  policy.
- Keep runtime validation separate. Only say an engine API is runtime-tested
  when a pinned native engine or DCC runtime was actually exercised.
- State the tested interpreter and dependency versions. Report missing or
  skipped runtime checks plainly; do not treat the attached research bundle's
  report as a QuatOps test run.
- Keep engine-specific facts on that engine's page, include versioned primary
  documentation links, and put shared math on `coordinate-mapping.md`.
- Preserve the low-cost path: no per-sample basis conversion, no implicit
  normalization or sign changes in the boundary map, and no mandatory engine
  or NumPy dependency in the scalar core.

## Current status to preserve

The current SciPy tests cover all 48 signed axis maps and the mapped SLERP
identity. Native proof captures are versioned under
`experiments/<engine>/<version>/` and
`tests/golden/<engine>/<version>/native_rotation.json` for Unreal 5.7.4, Unity
Editor 6000.5.2f1, Godot 4.7 stable and 4.8-dev1, and Blender 5.1.2. Each
capture records Euler construction/readback, matrix-to-quaternion, axis-angle,
rotation-vector, shortest SLERP, normalization edges, basis/vector action, and
composition. The frame map is explicit per fixture. Unreal and Godot also
capture a full-path/no-shortest-path interpolation API; the captured Unity and
Blender public APIs do not expose that mode. Unity rotation vectors are derived
through native axis-angle calls, Godot's six-order `Basis` probes and `exp`/`log`
route are checked against explicit conventions, and Blender adds all six Euler
orders, `euler_compat` trajectories, and swing-twist decomposition. The Godot
4.8 capture is a development snapshot. None of these fixtures validates an
imported asset or proves a specific asset's local forward axis.
Consult the [proof matrix](proof-matrix.md) and
[verification note](verification.md) before making an accuracy claim.
