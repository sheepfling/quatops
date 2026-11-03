# Sophus 1.24.6 SO(3) discovery

The first native capture builds Sophus 1.24.6 against Eigen 5.0.1 with GCC
15.2.0 on Windows x86-64. Its assigned right-handed probe frame is +X right,
+Y forward, +Z up; this is the capture's test frame, not a frame assigned by
Sophus.

The discovery set records `SO3d` construction from WXYZ `Eigen::Quaterniond`
inputs, native coefficient access, vector action, matrices, `exp`/`log`,
inverse, both noncommuting product orders, and left/right perturbation
products. It includes direct quaternion logs for a non-unit input, an
antipodal pair, a non-principal half-turn, a near-π case, and an input above π.
The raw capture and pin are in
[`experiments/oracles/sophus/1.24.6`](../../experiments/oracles/sophus/1.24.6/README.md).

The observed constructor normalizes the non-unit input, while `params()` and
Eigen quaternion coefficients use XYZW order. The antipodal inputs preserve
opposite stored quaternion signs but produce the same native matrix and vector
action. The `log(exp(omega))` output wraps the 3.5-radian input to a principal
rotation vector. These are observations from this pinned `SO3d` run, not a
complete Sophus compatibility contract.

The separate SciPy `Rotation` comparison covers construction, vector action,
storage, exp/log, composition, inverse, and both perturbation products. Its
largest recorded matrix difference is `2.22e-16`; vector action differs by at
most `1.78e-15`. This is derived-reference evidence, not a QuatOps execution.
The capture remains a discovery result: float32, Jacobians, tangent
derivatives, and broader edge cases have not been accepted.

The [Sophus 1.24.6 release](https://github.com/strasdat/Sophus/releases/tag/1.24.6)
and pinned [`so3.hpp`](https://github.com/strasdat/Sophus/blob/1.24.6/sophus/so3.hpp)
are the primary API references. The `SO3` source documents unit-quaternion
storage, normalization on quaternion construction, the double cover, and the
principal logarithm branch.
