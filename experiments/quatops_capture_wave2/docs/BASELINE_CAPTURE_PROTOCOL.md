# Capture protocol

## 1. Evidence types and independence

Use four evidence labels:

| Label | Meaning | Not sufficient for |
|---|---|---|
| `native_runtime` | Values emitted by the named product/library binary | Claiming another product/version behaves identically |
| `upstream_static` | Literal expected values copied from upstream tests, with permalink and license | Claiming we executed those tests |
| `derived_reference` | Independent mathematical implementation or analytic invariants | Claiming native engine behavior |
| `spec_contract` | Requirements of a format/API standard | Pretending that a standard is an executable numerical oracle |

`not_executed` is reserved for templates and plans. Never admit it to the golden-data directory.

Independence is per operation. Sophus depends on Eigen; SpiceyPy invokes CSPICE; a ROS wrapper and the underlying tf2 operation are not two independent implementations. Record `oracle_family` and upstream dependencies. Do not count QuatOps tuple, dataclass and NumPy paths as independent native sources. Copying engine formulas into an exporter produces a derived reference, not a native capture.

**Exporters must not import QuatOps, calculate expected results with SciPy, canonicalize output quaternion signs, or silently normalize/reorder raw native output.** Inputs can be generated mathematically; expected outputs must come from the named native entry point. Postprocessing belongs in a separate, hashed comparison stage.

## 2. Dataset layout

Proposed repository organization:

```text
experiments/oracles/
  shared/orientation_inputs_v1.json
  shared/capture-manifest.schema.json
  <oracle>/README.md
  <oracle>/capture.*
  <oracle>/dependency-lock-or-build-config
  <oracle>/runs/<version>/<platform>/<run-id>/
    manifest.json
    observations.jsonl
    stdout.log
    stderr.log
    checksums.sha256
    native-inputs.json
    assets/                         # only for interchange captures

tests/golden/oracles/<oracle>/<version>/
  fixtures.json
  manifest.json
  comparison-policy.json

tests/oracles/test_<oracle>_<operation>.py
docs/oracles/<oracle>.md
```

Existing engine paths need not move. Add an index and consistent metadata first; avoid unnecessary churn.

## 3. Run metadata

Capture the product/API name, exact version, release channel, source commit or changelist when available, binary/library hash, wrapper version, resolved package graph, OS/architecture, CPU/runtime, compiler, optimization/fast-math/SIMD settings, and scalar type.

Record **input precision, arithmetic precision, output precision and serialization precision separately**. A Python float does not prove that the native operation ran in double precision. Include actual post-binding input coefficients. For Unreal, preserve the distinction between its reflected Python input behavior and its C++ float/double captures. For Godot, distinguish GDScript float from `real_t` in quaternion/vector components.

Also record UTC timestamp, local timezone if useful, exporter source hash and commit, full command argument vector, corpus hash, selected operation profiles, return code, runtime assertions, and SHA-256 of each artifact. Do not invent a source commit for a vendor binary that exposes only a version; record what is unavailable and hash the binary when feasible.

Native captures must be performed with known, clean dependency resolution. Do not download `latest` during the test. For JavaScript, retain the lockfile and package integrity; for Python retain installed versions and actual native library identity; for C++ retain compiler output and resolved dependencies.

## 4. Per-observation record

Each record identifies a `case_id`, `operation_id`, exact native entry point, units, coefficient order, matrix algebra convention, argument labels, actual native input values and output values.

Possible outcomes: `ok`, `rejected`, `nonfinite`, `unsupported`, `process_failed`, `not_run`. A rejected/unsupported case is not a numerical pass. Store the diagnostic and the reason for not running any case. Potential assertions/crashes belong in isolated processes, never in the valid-input baseline.

For diagnostic replay, record IEEE-754 bit patterns for numeric arrays in an explicitly stated format (e.g. hex representation of each binary32 value, not unspecified host-endian memory). Preserve signed zero. JSON numbers alone are insufficient for distinguishing every bit-level behavior. Nonfinite values must be tagged strings/objects plus bits; do not write nonstandard JSON NaN or Infinity literals. Preserve decimal displays in addition to bits for readable reviews.

Do not claim that a widened binary32 output became binary64 computation. Bits describe the actual native scalar type, not the JSON parser type.

## 5. Shared operation profiles

| Profile | Minimum native observations |
|---|---|
| ALGEBRA | Raw ordered multiplication for general nonunit coefficients; reversed product; conjugate; norm; inverse when supported |
| ACTION | `q` applied to +X,+Y,+Z and a mixed vector; inverse action; composed action in both orders |
| EULER | All supported orders, default API separately, forward quaternion, native inverse angles, rebuilt native quaternion, basis action |
| MATRIX | Native quaternion-to-matrix and matrix-to-quaternion; indexed rows/columns; applied vectors; separate independently supplied matrices |
| AXIS_ROTVEC | Axis-angle and rotation-vector input/output, units, antipodal signs, branches at zero and pi |
| INTERPOLATION | Native methods separately; signs, norms, rotated axes, time, clamping and extrapolation; ambiguity classified |
| DIFFERENTIAL | Derivative, integration, group log/exp, perturbation-side and frame conventions, Jacobians plus finite-difference probes |
| FRAME | Declared source/target axes and map; orientation and translation in parent/child tests; input/output frame names |
| TRAJECTORY | Sequential keys, times, raw sign choices, canonical and continuity-aware Euler readbacks; no silently regenerated sign path |
| FORMAT | Raw asset/message, schema/version, node or frame identity, interpreted rotation, parent transforms, interpolation rules |

Capture only profiles relevant to the target's declared scope. A quaternion-only library does not owe an asset-import test.

## 6. Input corpus design

The bundled `orientation_inputs_v1.json` contains inputs, not expected outputs. Its sections cover general and unit quaternions, Euler-axis triples, all six order-labelled middle-angle boundary sweeps, axis-angle inputs, exact independent matrices, interpolation pairs, integration samples, signed axis maps, vector probes and trajectories.

Freeze the actual random values. The seed documents generation, but cross-language RNG agreement is not assumed. Keep two random populations separate: axis-angle/Euler stress inputs and normalized-Gaussian quaternion inputs. Do not call uniform Euler sampling a uniform SO(3) distribution.

### Required boundary details

- Nonzero unequal mixed angles; mixed signs; angles beyond one revolution.
- Both signs of each principal half/quarter-turn, and independent negative-trace matrices.
- Middle-angle singularities for each order; sample both sides, not only one rounded 90-degree value.
- Binary32 and binary64 neighboring representable inputs in **the native API unit**. Degrees-to-radians conversion can collapse a boundary sweep; record post-conversion values as well.
- Zero/tiny/pi/just-below-pi/just-above-pi rotation vectors and antipodal quaternions.
- Distinguish `q` versus `-q` (the same rotation, opposite points on S3) from orientations separated by 180 degrees in SO(3).
- SLERP endpoint equality, antipodal endpoints, negative dot products, near-identical inputs, near-pi separation, and t outside [0,1] only where the native contract permits it.
- Nonidentity starting orientations and angular velocities not aligned with their axes. Identity-only integration cannot distinguish left from right increments.
- Nonunit/zero/nonfinite quaternion cases are contract probes, not valid unit-rotation requirements.
- Basis maps include proper and improper signed permutations. Mapping a quaternion's vector part under a reflected coordinate change requires the axial-vector rule; a transpose is not a handedness conversion.

“Exactly 90 degrees” and a binary approximation to pi/2 are not bit-identical inputs. Store both intended mathematical landmark and actual native scalar.

## 7. Comparison policies

### A. General algebra

Compare coefficients with their signs. Do not allow global sign equivalence for multiplication, inverse, algebraic exp/log, or trigonometric functions. Apply the explicit Hamilton/JPL adapter first, if appropriate.

### B. Rotations

Check native norm/validity before normalizing a comparison copy. For normalized quaternions q and r, a stable sign-invariant angular residual is

`4 * atan2(min(norm(q-r), norm(q+r)), max(norm(q-r), norm(q+r)))`.

It returns radians in [0, pi]. Retain raw norms and also compare all three transformed basis vectors and the matrix. This avoids `acos(dot)` loss of resolution near zero and detects matrix-action mistakes. Never repair a bad quaternion by normalizing the only recorded copy.

### C. Euler extraction

Separate three assertions: native forward conversion; orientation represented by QuatOps inverse conversion; exact native branch/wrapping compatibility. The third is opt-in and version-pinned. Capture native inverse->forward residual independently. The user-reported Unity ~0.0961-degree and Godot ~0.00161-degree residuals are observations to reproduce and investigate, not global QuatOps tolerances. Verify the metric and native input bits before interpreting them as engine errors.

### D. Interpolation and continuity

Orientation agreement at each sample is necessary but does not establish lift/path compatibility. Retain quaternion signs and neighboring samples when a selected long path or continuous sign sequence is part of the API contract. At exactly antipodal endpoints the full-path geodesic is not unique. Record the native fallback; do not demand one universal midpoint.

Euler continuity is reference-dependent and not unique. Test the returned orientation, jump sizes under a stated cost, and native outputs separately. Quaternions cannot recover unobserved full revolutions.

### E. Differentials

Never compare a 3x3 SO(3) tangent Jacobian to a 4x3 quaternion coefficient derivative or a 9x3 matrix derivative. Record output/input dimensions, flattening order, perturbation side, frame, and angle/half-angle definition. Check native derivatives against central finite differences at several step sizes, away from branch cuts. Retain convergence curves; do not select one step that hides a discrepancy.

### F. Tolerances

Use per-operation, per-precision comparison policies. Suggested diagnostic starting points away from singularities: unit orientation angle 1e-6 radians for binary32, 1e-12 for binary64; matrix/action absolute differences approximately 1e-6 and 1e-12. These are initial investigation thresholds, not promises or permission to waive a failure. Calibrate against observed native output resolution, stable independent references and conditioning. Algebra requires both relative and absolute tolerances, with explicit scales. Jacobians require conditioning-aware thresholds and step convergence.

Do not loosen the whole suite for one native Euler branch. Classify a documented behavior separately or define an explicitly narrower compatibility scope. Exact-bit replay is limited to the same executable/backend/settings.

## 8. Coordinate and frame contract

A source-to-destination map B should be stored explicitly alongside axis labels. For the same orthogonal basis change on both object and world coordinates, compare `R_dst = B R_src B^T`. For Hamilton coefficients, the corresponding vector-part mapping is `v_dst = det(B) B v_src`, with scalar unchanged.

When world and object mappings differ, use `R_dst = B_world R_src inverse(B_body)` and analyze whether it is a proper rotation; not every transform can be represented by a quaternion. Keep asset-model forward, camera optical conventions, active/passive frame direction, matrix row/column action and quaternion storage independent.

## 9. Review and promotion gates

A capture task progresses through `planned -> exporter_ready -> native_captured -> audited -> accepted_for_scope`.

`reported_native_partial` records existing user-reported work until inspection. It must not silently become `audited`.

Acceptance requires: exact build and corpus identification; archived raw output and exporter; explicit conventions; all mandatory valid cases accounted for; a second reviewer or independent verification of argument binding; no expected values generated by QuatOps; per-operation comparator policies; documented limitations; tests on every supported QuatOps representation that exposes the operation; and no unexplained discrepancies.

Record count must be the number of native input-operation observations, independent input counts separately, and QuatOps replay counts separately. Missing or unsupported cases stay visible.

Normal CI replays frozen goldens without installing engines. Native recapture is an explicit maintenance action for a new version or expanded scope. Never overwrite an old version's dataset or auto-bless a changed output because the test failed.
