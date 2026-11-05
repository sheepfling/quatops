# Changelog

## 0.1.0rc1

Release candidate for the first frame-aware quaternion interoperability
release.

The intended release tag is `v0.1.0rc1`; the source fallback version matches.

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

At the 0.1.0a0 release, the commit was tagged `v0.1.0a0` and that release's
source fallback was `0.1.0a0`. The alpha allowed API details to change before
the 0.1.0 release.
