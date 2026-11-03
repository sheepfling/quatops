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
- [CesiumJS 1.124.0](transports/cesium/1.124.0/manifest.json): nine local
  orientations placed in selected WGS84 ENU/ECEF frames.
- [glTF 2.0 / Three.js 0.187.0](transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json):
  a QuatOps-generated static hierarchy loaded by native `GLTFLoader`.
- [OpenUSD 26.11](transports/openusd/26.11/converter/manifest.json): a
  QuatOps-generated stage reopened and evaluated by native OpenUSD.

The converter replay tests are linked from the
[interoperability atlas](../../docs/INTEROPERABILITY_EVIDENCE.md). These
fixtures are auxiliary evidence; they do not add runtime dependencies or
named transport adapters under `src/`.

## Discovery captures

The broad Cesium and multi-loader glTF discovery captures remain available
alongside their compact converter witnesses. OpenCV and PROJ still have broad
observation sets awaiting a retention review. A captured convention, a shared
mathematics check, and a converter-boundary proof are distinct claims.
