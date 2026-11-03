# Independent mathematical library captures

Native library captures and format/transport checks live here, separate from
`src/`. Each target owns its pinned version, exporter, exact invocation, and
runtime identity. Frozen observations are promoted to
`tests/golden/oracles/<oracle>/<version>/`; tests compare them with QuatOps and
independent formulas only after the target process exits.

## Shared inputs

The shared input corpus and capture envelope are documented in
[`shared/README.md`](shared/README.md).

[`shared/orientation_inputs_v1.json`](shared/orientation_inputs_v1.json) is an
input-only retained witness set. SHA-256:
`eaf7605b3298ec9dbdf7838f43f2da61d90af5351bb15e455d3eb37c6c6b5446`.
It contains 5 axis-angle inputs, 7 Euler boundary cases, one mixed Euler input,
9 quaternion cases, 3 multiplication pairs, 7 interpolation pairs, 4 matrix
inputs, 4 integration inputs, and 3 short trajectory examples. The 48 signed
axis maps are generated in the shared property tests instead of duplicated in
the capture corpus. The broad discovery source digest is retained in the
corpus metadata; passing sweep outputs were discarded.

The corpus freezes actual values and IEEE-754 input bits where relevant. A
runtime adapter must preserve those source values, record any unit conversion
and post-binding values, and map each order label using native observations.
Do not recreate the corpus's random generator in another language.

## Captures

Pyright stubs for the small MuJoCo and SpiceyPy native API surfaces used by the
captures live in [`stubs/`](stubs/). They describe calls for static analysis;
only the pinned capture runs establish native behavior.

- [Three.js](threejs/README.md): Node math capture, no renderer or GPU.
- [CesiumJS 1.124.0](../converters/cesium/1.124.0/README.md): local quaternion and WGS84 ENU/ECEF converter capture.
- [CSPICE via SpiceyPy](spiceypy/README.md): scalar-first algebra, Euler and
  frame-matrix direction, axis-angle, and quaternion derivatives.
- [MuJoCo](mujoco/README.md): WXYZ algebra, Euler order labels, rotation
  differences, angular-rate frames, integration, and Jacobians.
- [OpenCV 4.13.0](../converters/opencv/4.13.0/README.md): Rodrigues vector/matrix converter boundary.
- [PROJ 9.5.1](proj/9.5.1/README.md): WGS84 geodetic, ECEF, and ENU probes.
- [Sophus 1.24.6](sophus/1.24.6/README.md): native SO3d quaternion, exp/log,
  composition, inverse, and perturbation discovery.
- [OpenXR SDK 1.1.65](openxr/1.1.65/README.md): compile-time header ABI
  capture only; no runtime was available.
- Additional mathematical libraries will get separate versioned target
  indexes as their native runtimes are captured. File formats and loaders live
  under [`experiments/transports`](../transports/README.md).

The retained fixtures keep selected native witnesses and distinct known
discrepancies. Passing discovery sweeps are generated or temporary.

Keep native runtime output, spec-derived expectations, and formula checks
labelled as different evidence. A schema validator or QuatOps backend replay
does not count as a native capture.
