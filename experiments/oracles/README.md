# Independent mathematical library captures

Native library captures and format/transport checks live here, separate from
`src/`. Each target owns its pinned version, exporter, exact invocation, and
runtime identity. Frozen observations are promoted to
`tests/golden/oracles/<oracle>/<version>/`; tests compare them with QuatOps and
independent formulas only after the target process exits.

## Shared inputs

[`shared/orientation_inputs_v1.json`](shared/orientation_inputs_v1.json) is an
input-only corpus. SHA-256:
`07e4d809f7eb75ad6e5454febf2b725b0b8fb389536743dd7196566d4640135f`.
It includes 89 Euler inputs, 540 Euler boundary probes, 60 axis-angle cases,
140 quaternion algebra inputs, 3 multiplication pairs, 7 interpolation pairs,
11 independent rotation matrices, 48 integration inputs, 48 signed-axis maps,
and 3 trajectories. It contains no native expected outputs.

The corpus freezes actual values and IEEE-754 input bits where relevant. A
runtime adapter must preserve those source values, record any unit conversion
and post-binding values, and map each order label using native observations.
Do not recreate the corpus's random generator in another language.

## Captures

- [Three.js](threejs/README.md): Node math capture, no renderer or GPU.
- [CSPICE via SpiceyPy](spiceypy/README.md): scalar-first algebra, Euler and
  frame-matrix direction, axis-angle, and quaternion derivatives.
- [MuJoCo](mujoco/README.md): WXYZ algebra, Euler order labels, rotation
  differences, angular-rate frames, integration, and Jacobians.
- Additional mathematical libraries will get separate versioned target
  indexes as their native runtimes are captured. File formats and loaders live
  under [`experiments/transports`](../transports/README.md).

Keep native runtime output, spec-derived expectations, and formula checks
labelled as different evidence. A schema validator or QuatOps backend replay
does not count as a native capture.
