# Sophus 1.24.6 SO(3) discovery

Pinned native captures cover both `Sophus::SO3f` and `Sophus::SO3d`, compiled
against Sophus 1.24.6 and Eigen 5.0.1 with GCC 15.2.0 on Windows x86-64. The
capture assigns +X right, +Y forward, +Z up as its right-handed probe frame;
that is not a universal Sophus frame.

The selected observations in
[`experiments/oracles/sophus/1.24.6`](../../experiments/oracles/sophus/1.24.6/README.md)
cover constructor normalization and order, XYZW `SO3::params()`, antipodal
signs and equivalent action, basis-vector actions, and both noncommuting
product orders. The [version manifest](../../experiments/oracles/sophus/1.24.6/manifest.json)
references the shared, content-addressed native values for each scalar profile.
The experimental `sophus.so3-params` profile is supporting-only, not an
accepted application-frame converter.

Separate SciPy 1.18.1 comparisons passed against the original native captures: float64
maximum absolute error `1.78e-15` at `2e-14` tolerance; float32 maximum
`9.5e-7` at `2e-6`. SciPy is derived-reference evidence and did not execute
QuatOps. Logarithms, inverse, perturbation sides, principal-π branch behavior,
Jacobians, and tangent derivatives are outside this selected scope.

Primary sources are the [Sophus 1.24.6 release](https://github.com/strasdat/Sophus/releases/tag/1.24.6),
pinned [`so3.hpp`](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp),
and [Eigen 5.0.1's Quaternion API](https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1Quaternion.html).
