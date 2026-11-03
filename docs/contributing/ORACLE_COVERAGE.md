# Independent oracle coverage

## Status

The coverage plan below is complete for the public operations and layouts
listed here. The checked items record completed coverage; use the coverage map
to find the tests. Add a new item when the public API or its edge contracts
expand.

## Coverage goals

Check every public quaternion and attitude operation against an independent
reference or a separately derived mathematical invariant across the 15 scalar
modules and eight NumPy layouts (coefficient axis last or at axis 0, both `wxyz`/`xyzw`
orders and Hamilton/JPL conventions). NumPy batches must also be checked against their scalar samples
and retain the documented batch shapes.

An independent check must not use the same QuatOps operation as its expected
value. Use `numpy-quaternion` for generic algebra and rotation conversions,
SciPy for SO(3) reference behavior, and explicit formulas or separately
implemented identities where neither package supplies the operation.

## Coverage groups

- [x] Generic algebra Monte Carlo: add/subtract/negate/scale/conjugate,
  multiply/divide and left/right quotients, inverse, exp/log, expm1/log1p,
  log-base conversions, integer and real powers, the `pow_real` alias,
  quaternion power, square root, commutator,
  dot/norm/norm-squared/normalize and vector-part extraction.
- [x] Rotation-vector and matrix Monte Carlo: from/to rotation vector,
  quaternion rotation of vectors, and from/to rotation matrices, including
  non-unit coefficients.
- [x] Analytic quaternion functions: sine/cosine, tangent, hyperbolic and
  inverse functions against complex-slice formulas; scalar-part extraction.
- [x] Coefficient and convention conversions: `wxyz`/`xyzw`, Hamilton/JPL,
  canonicalization, sign alignment/continuity, and rotation distance/equality.
- [x] Axis-angle, MRP, Euler, and two-vector conversions against NumPy
  quaternion, SciPy, explicit axis-angle formulas, and frozen Unity.Mathematics
  and Godot Euler golden fixtures, including singular and antipodal cases.
- [x] Engine-boundary signed-axis conversion and trajectory commutation:
  all 48 signed maps are checked against SciPy, and the full suite passes with
  the focused NumPy layout and SLERP commutation cases included.
- [x] Interpolation and local attitude updates: SLERP, NLERP, SQUAD, boxplus,
  and boxminus against independent composition/interpolation references;
  explicitly exercise sign-preserving long arcs and SLERP/NLERP extrapolation.
- [x] Kinematics: constant-rate integration, sampled left/midpoint integration,
  and body/world quaternion derivatives against independent integration or
  finite-difference references.
- [x] Matrix representations: left/right multiplication identities, skew
  matrices, SO(3) Jacobians and inverses, adjoint, and homogeneous transforms.
- [x] Alignment and statistics: weighted Wahba alignment, Markley and intrinsic
  means, and body/world covariance against SciPy or separately computed
  optimization/eigenvalue references.
- [x] Random attitudes: reproducibility, unit norm, uniform-rotation moments,
  sign policy, and scalar/array generator behavior.
- [x] Edge and batch contracts: zero, near-zero, half-turn, branch-cut,
  non-finite, invalid-shape, broadcasting, empty batch, and component-axis
  behavior for each operation family.

## Follow-up API audit

- [x] Audit all public operation names exported by `quatops.scalar`
  against the oracle suites; add direct Monte Carlo assertions for `pow_real`
  rather than relying on its relationship to `power`.
- [x] Check optional interpolation path controls against direct S3 coefficient
  formulas, including `shortest=False` and extrapolation on both ends of the
  interpolation interval.
- [x] Exercise positive, zero, and negative integer powers against
  `numpy-quaternion`, separately from real exponents, and a huge integer power
  against the exact identity invariant.
- [x] Exercise estimator defaults without explicit weights and compare
  unweighted means, covariance, and Wahba alignment with SciPy references.
- [x] Exercise accepted and rejected custom `atol` values for rotation and
  homogeneous matrix conversion across every representation.
- [x] Parameterize scalar input annotations with a `SupportsFloat`-bounded
  `TypeVar`; verify `numpy.float32` inputs and Python-float outputs across all
  15 scalar modules at runtime and with static consumers.

## Acceptance

For each group, keep deterministic seeds, include at least one nontrivial
randomized sample set and the documented boundary cases, and run the checks
through every applicable public representation. Record the oracle and any
branch-policy exception next to the tests. Run the full suite with the local
`.venv` before marking a group complete.

## Coverage map and branch policies

- Generic algebra and rotation-vector/matrix Monte Carlo:
  [`test_numpy_quaternion_oracle.py`](../../tests/test_numpy_quaternion_oracle.py).
- Analytic functions and scalar/vector extraction:
  [`test_analytic_oracle.py`](../../tests/test_analytic_oracle.py).
- Coefficient/convention conversion, axis-angle, Euler, MRP, vector alignment,
  and rotation-vector aliases:
  [`test_rotation_conversion_oracle.py`](../../tests/test_rotation_conversion_oracle.py).
- Engine signed-axis maps and basis-converted SLERP trajectories:
  [`test_engine_basis_oracle.py`](../../tests/test_engine_basis_oracle.py).
- Frozen upstream Unity.Mathematics/Godot Euler-to-quaternion and matrix values:
  [`test_engine_euler_golden.py`](../../tests/test_engine_euler_golden.py) with
  source-attributed values in [`engine_euler_fixtures.json`](../../tests/golden/engine_euler_fixtures.json).
- SLERP, NLERP, SQUAD, boxplus/boxminus, quaternion derivatives, and angular
  velocity integration:
  [`test_manifold_kinematics_oracle.py`](../../tests/test_manifold_kinematics_oracle.py).
- Multiplication matrices, skew, Jacobians, adjoint, and homogeneous matrices:
  [`test_matrix_oracle.py`](../../tests/test_matrix_oracle.py).
- Weighted alignment, means, covariance, and swing/twist:
  [`test_estimation_oracle.py`](../../tests/test_estimation_oracle.py).
- Seeded random attitudes and stream parity:
  [`test_random_oracle.py`](../../tests/test_random_oracle.py).
- Singular inputs, branch edges, invalid shapes, and empty NumPy layouts:
  [`test_oracle_edge_contracts.py`](../../tests/test_oracle_edge_contracts.py),
  [`test_numpy_layouts.py`](../../tests/test_numpy_layouts.py), and
  [`test_numpy_component_axes.py`](../../tests/test_numpy_component_axes.py).
- Accuracy and failure examples for algebra, analytic functions, and attitude
  operations: [`test_algebra_numeric_contract.py`](../../tests/test_algebra_numeric_contract.py),
  [`test_analytic_numeric_contract.py`](../../tests/test_analytic_numeric_contract.py), and
  [`test_attitude_numeric_contract.py`](../../tests/test_attitude_numeric_contract.py).
  These examples do not establish a uniform error bound for all inputs.

The logarithm of a negative real uses the positive first coefficient axis in
both Hamilton and JPL algebraic representations. At this singular input, a
physical JPL-to-Hamilton mapping of numpy-quaternion selects the opposite vector
sign; the branch test therefore uses the independent complex-slice formula and
records the coefficient-policy exception explicitly.

Use the [contributor verification guide](VERIFICATION.md) for commands to run
the checks in the current checkout. Coverage status here describes test
structure; it does not publish machine-specific pass counts or timing results.
