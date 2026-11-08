# Changelog

## Unreleased

Frame-aware interoperability is experimental, not a release candidate.
The alpha source fallback remains unchanged; this change does not publish a release.

- Places the new runtime APIs under `quatops.experimental` and preserves the
  earlier PR import paths as deprecated compatibility facades.
- Keeps native captures and evidence intact. Evidence acceptance describes
  captured operations, not API stability.
- Adds a shared physical-orientation converter with explicit parent and child
  frame definitions across the registered target profiles.
- Retains full target frame definitions and NumPy component axes in conversion
  results, so reverse conversion can reject mismatched interpretations.
- Publishes evidence acceptance states and keeps OpenXR and Sophus claims
  scoped to their declared/test-oracle evidence.
- Clarifies sign preservation, unit tolerance, precision, composition, and
  installed-package behavior.

## 0.1.0a0

Initial alpha release of QuatOps.

- Adds explicit Hamilton and JPL APIs for tuple, `NamedTuple`, frozen
  dataclass, and NumPy representations.
- Adds trailing-axis and component-first NumPy layouts in both `wxyz` and
  `xyzw` coefficient orders.
- Adds quaternion algebra and analytic functions, rotation conversions,
  interpolation, SO(3) matrices and Jacobians, attitude kinematics, alignment,
  estimation, random attitudes, and homogeneous transforms.
- Adds seeded independent-oracle coverage with NumPy quaternion, SciPy, and
  separately derived mathematical references.
- Documents and benchmarks the public APIs and runnable experiments.

The source fallback is `0.1.0a0`. A release tag and publication are separate
actions; this changelog is not proof of publication. This alpha may change API
details before 0.1.0.
