# Retained native evidence

## Game-engine converter witnesses

The compact, versioned game-engine converter evidence lives in
[`tests/oracles`](../oracles/), with one contract and selected native witness
set per build. Full discovery output is temporary; the selector at
[`scripts/retain_engine_witnesses.py`](../../scripts/retain_engine_witnesses.py)
copies the chosen native values without recomputing them and records hashes for
the capture source, original capture, and shared input.

## Operation-specific native oracles

The Three.js, MuJoCo, CSPICE/SpiceyPy, and OpenUSD oracle fixtures retain small
operation-specific witness sets with native build provenance and the digest of
the discovery capture used to select them. Their broad passing sweeps are not
stored as goldens.

## Converter-boundary witnesses

Separate compact fixtures preserve the native side of these auxiliary proofs:

- [ROS 2 tf2 0.36.23](oracles/ros2/0.36.23/manifest.json): two directions of
  the named three-frame chain, message fields, and native `fromMsg` result.
- [CesiumJS 1.124.0](transports/cesium/1.124.0/manifest.json): five local
  orientations placed in selected WGS84 ENU/ECEF frames.
- [glTF 2.0 / Three.js 0.187.0](transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json):
  a QuatOps-generated static hierarchy loaded by native `GLTFLoader`.
- [OpenUSD 26.11](transports/openusd/26.11/converter/manifest.json): a
  QuatOps-generated stage reopened and evaluated by native OpenUSD.
- [SpiceyPy 8.3.0 / CSPICE_N0067](oracles/spiceypy/8.3.0/converter/manifest.json):
  selected direct-quaternion matrix/vector actions, matrix round trip, the
  native `axisar` matrix/`m2q` route, and noncommuting products.
- [MuJoCo 3.17.0](oracles/mujoco/3.17.0/converter/manifest.json): selected
  direct-quaternion matrix/vector actions, matrix round trip, native axis-angle
  construction, and noncommuting products.
- [OpenCV 4.13.0](oracles/opencv/4.13.0/converter/manifest.json): native
  Rodrigues vector/matrix conversion, matrix actions, and composition in two
  scalar precisions.
- [Eigen 5.0.1](oracles/eigen/5.0.1/converter/manifest.json): native C++
  quaternion storage, vector action, matrix conversion, and composition.

The converter replay tests are linked from the
[interoperability atlas](../../docs/INTEROPERABILITY_EVIDENCE.md). These
fixtures are auxiliary evidence; they do not add runtime dependencies or
named transport adapters under `src/`.

The shared tooling validates each package's contract, manifest, optional input,
and witness JSON, then exposes a normalized in-memory envelope with a target
identity and observation groups. Existing witness files may encode groups as a
keyed object or as a record array; the envelope labels both consistently while
retaining each target-specific JSON payload value tree. It never rewrites
captured JSON. Run `python scripts/validate_evidence_json.py` for syntax and
JSON-value validation across maintained captures, then
`python scripts/validate_converter_evidence.py` for package identity and
envelope validation.

## Retention boundary

This directory contains selected native observations, their contracts, and
the provenance needed to identify the target runs. Broad successful sweeps
are generated or temporary. A minimized native counterexample is retained
when it protects a distinct contract or a real regression. Native captures,
specification-derived expectations, and SciPy formula checks remain separate
evidence types.
