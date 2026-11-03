# Coding-agent guide for engine rotations

Read this guide when changing `quatops.engines`, engine-specific references,
or quaternion trajectory conversion. For implementation workflow, use the
[developer guide](ENGINE_INTEGRATIONS.md). For concrete API details, use the
separate [Unreal](../engines/unreal.md), [Unity](../engines/unity.md), and
[Godot](../engines/godot.md) pages.

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
  [coordinate mapping](../engines/coordinate-mapping.md); do not force it into a single
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
  when a pinned Unreal, Unity, or Godot runtime was actually exercised.
- State the tested interpreter and dependency versions. Report missing or
  skipped runtime checks plainly; do not treat the attached research bundle's
  report as a QuatOps test run.
- Keep engine-specific facts on that engine's page, include versioned primary
  documentation links, and put shared math on `coordinate-mapping.md`.
- Preserve the low-cost path: no per-sample basis conversion, no implicit
  normalization or sign changes in the boundary map, and no mandatory engine
  or NumPy dependency in the scalar core.

## Current status to preserve

The SciPy tests cover the basis conversion and mapped SLERP identity. They do
not run Unreal, Unity, or Godot, validate an asset import, or prove a specific
asset's local forward axis. See issues
[#4](https://github.com/sheepfling/quatops/issues/4) and
[#7](https://github.com/sheepfling/quatops/issues/7) for native-runtime work.
