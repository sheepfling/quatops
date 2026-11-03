# Integration into QuatOps

No QuatOps repository was inspected or changed during this wave. Preserve current project naming and existing engine fixture paths; this packet does not require restructuring them.

## Suggested import boundaries

- Keep exporters and raw runs under experiments/oracles, with source-specific run folders.
- Keep authored format inputs and the frozen corpus separate from expected outputs.
- Promote reviewed operation subsets into tests/golden only after native identity and comparator audits.
- Normal CI replays frozen observations without downloading engines. Native recapture is explicit maintenance.
- Keep Cesium local orientation, geographic frame placement and glTF model import as separate adapters. Only the first two have exporters in this packet.

## Adapters to prove

| Consumer | Raw convention to preserve | Comparison adapter |
|---|---|---|
| ROS tf2 | XYZW, fixed-axis RPY, explicit target/source frame IDs | Convert component order/algebra only; compare action and both frame-chain directions |
| Cesium local | XYZW, heading about -Z, pitch about -Y, roll about +X | Prove signed angle/product mapping; do not insert a globe transform |
| Cesium geographic | Native ENU/ECEF transforms, origin in metres | Compare origin, local basis, HPR rotation, point and direction separately |
| OpenUSD Gf | Real+imaginary quaternion, native row-action matrices | Transpose only for a declared column-action reference; preserve native op order |
| glTF | XYZW buffers, column-major matrices, local TRS and scene hierarchy | Evaluate formats and consumer outputs explicitly, with absolute sample time and loop policy |
| OpenCV Rodrigues | Native rotation vectors, matrices and 3x9/9x3 Jacobians | Compare shape/flattening correctly; do not confuse matrix derivative and SO(3) Jacobian |
| PROJ | Geographic/ECEF/ENU point coordinates | Supporting frame evidence, not a quaternion implementation or Cesium native capture |

Quaternion orientation comparison allows one global sign; general algebra comparison does not. Interpolation retains sign/path information. Invalid native inputs are separately classified, not part of ordinary unit-rotation requirements.

## Additional comparisons to add after target execution

ROS: matrix/action versus QuatOps, world_from_sensor composition, reverse lookup, message field preservation. Do not label this DDS/CDR serialization.

Cesium: zero/local principal HPR and mixed inputs; quaternion versus matrix action; geographic ENU orientation versus PROJ away from poles; native pole fallback recorded separately; inverse point mapping. ICRF/EOP and imported model axes remain outside scope.

OpenUSD: f32/f64 outputs, order-labelled Euler probes, matrix construction/extraction, reset and inverse operations, animation time samples. Up-axis metadata alone should not be treated as an implicit scene-coordinate conversion.

glTF: add Three.js and Khronos reports; compare sample matrices, not merely keyframe quaternions. Preserve input buffer scalar precision and cubic tangent signs. For negative-scale or skinned assets, create a separate scope; these inputs use unit scale and simple meshes.

OpenCV: retain 26 observed binary64 inverse-boundary diagnostics as versioned behavior. QuatOps forward precision must not be relaxed to emulate them. C++ quaternion transcendental capture is a separate missing work item.

## Portability and provenance

Raw manifests preserve original absolute path strings. The packet tests resolve the suffix after quatops_capture_wave2 against the extracted root for replay. External binary paths are historical provenance, not files bundled for redistribution. SHA256SUMS covers the portable bundle contents; runtime_inventory.json supplements the per-run manifests with shared helper/reference-script hashes.

The original manifest field independent_input_count includes accounted-but-unrun IDs and separate glTF/GLB encodings. Use the corrected coverage interpretation in CAPTURE_LOG.md; do not turn 399 observations or 179 tests into a claim of 399 independent orientations.

## Deliberate non-goals

No imported asset/model-forward assumptions, renderer image equivalence, scene scaling/shear decomposition, live ROS timing, hardware tracking, full format conformance, all versions, all architecture bits, or complete native Euler-branch emulation. No new promise of QuatOps support until its tests are wired and executed.
