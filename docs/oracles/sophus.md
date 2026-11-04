# Sophus 1.24.6 SO(3) discovery

Pinned native captures cover both `Sophus::SO3f` and `Sophus::SO3d`, compiled
against Sophus 1.24.6 and Eigen 5.0.1 with GCC 15.2.0 on Windows x86-64. The
capture assigns +X right, +Y forward, +Z up as its right-handed probe frame;
that is not a universal Sophus frame.

The selected observations in
[`experiments/oracles/sophus/1.24.6`](../../experiments/oracles/sophus/1.24.6/README.md)
cover constructor normalization and order, XYZW `SO3::params()`, antipodal
signs and equivalent action, basis-vector actions, `exp`/`log` through the
principal-π branch, inverse, both noncommuting product orders, and left/right
perturbation products. The profile-specific native values are in
[`float32 witnesses`](../../experiments/oracles/sophus/1.24.6/witnesses/float32.json)
and [`float64 witnesses`](../../experiments/oracles/sophus/1.24.6/witnesses/float64.json).
This is supporting SO(3) evidence, not a named QuatOps converter.

Separate SciPy 1.18.1 comparisons passed against the native captures: float64
maximum absolute error `1.78e-15` at `2e-14` tolerance; float32 maximum
`9.5e-7` at `2e-6`. SciPy is derived-reference evidence and did not execute
QuatOps. Jacobians and tangent derivatives remain outside this selected scope.

Primary sources are the [Sophus 1.24.6 release](https://github.com/strasdat/Sophus/releases/tag/1.24.6),
pinned [`so3.hpp`](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp),
and [Eigen 5.0.1's Quaternion API](https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1Quaternion.html).
