# Native engine proof captures

These projects and scripts are auxiliary evidence for the coordinate and
rotation conventions used by QuatOps. They stay outside `src/` and never run
as package runtime dependencies.

## Version layout

Keep each pinned engine build in its own folder:

```text
experiments/<engine>/<version>/capture project and scripts
tests/golden/<engine>/<version>/native_rotation.json
```

The engine folder contains an index; the version folder owns its exact project,
capture command, runtime/build identity, and proof-profile details. A new engine
release gets a new sibling version folder and fixture. Do not replace an older
version's capture in place.

Every native fixture records the engine-reported version and a `proof_profile`
for Euler conversion/readback, matrix conversion, axis-angle, rotation vectors,
shortest/full-path SLERP, normalization, basis action, and composition/frame
mapping. `not exposed` means the pinned public API has no corresponding route;
it does not imply equivalent behavior. Read each engine page for the facts and
limits specific to that runtime, and [proof-matrix.md](../docs/engines/proof-matrix.md)
for coverage parity.

Converter acceptance captures share the fixed inputs in
[`interoperability/v1`](interoperability/README.md). Their canonical-to-native,
native-to-canonical, vector-action, and cross-engine checks are tracked in
[CAP-300](../docs/engines/CAP-300.md), separately from each engine's math proof
profile.

## Pinned captures

- [Unreal Engine](unreal/README.md): [5.7.4](unreal/5.7.4/README.md)
- [Unity Editor](unity/README.md): [6000.5.2f1](unity/6000.5.2f1/README.md)
- [Godot Engine](godot/README.md): [4.7 stable](godot/4.7-stable/README.md),
  [4.8-dev1](godot/4.8-dev1/README.md)
- [Blender](blender/README.md): [5.1.2](blender/5.1.2/README.md)

Native runtime captures and the later SciPy/QuatOps comparisons are separate
evidence steps. Each version guide gives both commands.

## Additional mathematical oracles and transports

These targets use a separate version layout because mathematical packages and
asset/pose formats have different evidence boundaries:

```text
experiments/oracles/<library>/<version>/
experiments/transports/<format>/<spec-version>/<loader>/<version>/
tests/golden/oracles/<library>/<version>/
tests/golden/transports/<format>/<spec-version>/<loader>/<version>/
```

- [Oracle index](oracles/README.md): CesiumJS 1.124.0, Three.js 0.187.0,
  OpenCV 4.13.0, PROJ 9.5.1, CSPICE through SpiceyPy 8.3.0, and MuJoCo 3.17.0.
- [Transport index](transports/README.md): glTF 2.0 with Three.js, Blender,
  VTK, and Trimesh importer captures, plus OpenUSD 26.11 stage evaluation.

The additive [Wave 2 capture package](quatops_capture_wave2/LOCAL_REPLAY.md)
retains the supplied source archive, shared corpus, native library runs,
local Cesium and Three.js captures, Khronos validation, and separate SciPy
replay. Its target/version pages under `experiments/oracles/` and
`experiments/transports/` point to the promoted frozen outputs.
