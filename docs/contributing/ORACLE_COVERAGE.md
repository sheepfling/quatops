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
- [x] Unreal 5.7.4 native `FRotator`/`FQuat` fixtures: float32 and float64
  Euler conversion (including both `MakeFromEuler` APIs), matrix-to-quaternion,
  axis-angle and rotation-vector
  construction/readbacks, shortest/full-path SLERP, normalization threshold
  behavior, basis axes, matrices, multiplication coefficients, and explicit
  branch/wrapping contracts across all 23 public representations. Native-only
  diagnostics record nonrotation matrix behavior without treating those inputs
  as valid QuatOps matrices.
- [x] Unity 6000.5.2f1 native `UnityEngine` and Unity.Mathematics fixtures:
  all six Mathematics Euler constructors/readbacks across 37 input triples
  each (29 fixed and eight seeded), UnityEngine Euler/readback, matrix and
  axis-angle conversions, SLERP/LERP/NLERP and normalization edges, basis
  actions, cross products, quaternion/Transform composition, and an explicit
  source-frame map across
  all 23 public representations.
- [x] Godot 4.7-stable and 4.8-dev1 native `Quaternion`, `Basis`, and
  `Transform3D` fixtures: all six `Basis` Euler construction/readback orders
  over 67 shared inputs each, matrix and axis-angle conversion (plus three
  native-only near-degenerate basis diagnostics), rotation `exp`/`log`
  branches, from-to vector behavior, SLERP, normalization edges,
  cross products, quaternion/transform composition, and an explicit source
  frame map across all 23 public representations. The 4.8-dev1 result is a
  development snapshot.
- [x] Blender 5.1.2 native `mathutils` and scene-object fixtures: all six Euler
  orders over 87 inputs each (64 seeded plus fixed and gimbal-neighbor cases),
  SciPy order mapping, Euler continuity across both 180-degree signs,
  matrix/axis-angle/exponential-map routes, shortest SLERP, rotation
  difference, swing-twist about X/Y/Z, normalization edges, active vector
  action, composition, and explicit source-frame probes across all 23 public
  representations.
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

- [x] Audit all 86 public operation names exported by `quatops.scalar`
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
branch-policy exception next to the tests. Run the full suite with the
project-supported Python development environment before marking a group
complete.

## Coverage map and branch policies

- Generic algebra and rotation-vector/matrix Monte Carlo:
  [`test_numpy_quaternion_oracle.py`](../tests/test_numpy_quaternion_oracle.py).
- Analytic functions and scalar/vector extraction:
  [`test_analytic_oracle.py`](../tests/test_analytic_oracle.py).
- Coefficient/convention conversion, axis-angle, Euler, MRP, vector alignment,
  and rotation-vector aliases:
  [`test_rotation_conversion_oracle.py`](../tests/test_rotation_conversion_oracle.py).
- Engine signed-axis maps and basis-converted SLERP trajectories:
  [`test_engine_basis_oracle.py`](../tests/test_engine_basis_oracle.py).
- Frozen upstream Unity.Mathematics/Godot Euler-to-quaternion and matrix values:
  [`test_engine_euler_golden.py`](../tests/test_engine_euler_golden.py) with
  source-attributed values in [`engine_euler_fixtures.json`](../tests/golden/engine_euler_fixtures.json).
- Native Unreal 5.7.4 `FRotator`/`FQuat` float32/float64 captures:
  [`test_unreal_engine_golden.py`](../tests/test_unreal_engine_golden.py),
  [`native_rotation.json`](../tests/golden/unreal/5.7.4/native_rotation.json),
  and the [`capture procedure`](../experiments/unreal/5.7.4/README.md).
- Unreal 5.7.4 extended conversion goldens and explicit branch differences:
  [`test_unreal_extended_conversions.py`](../tests/test_unreal_extended_conversions.py).
- Unreal 5.7.4 native coordinate basics and an explicitly declared
  right-handed source mapping:
  [`test_unreal_native_semantics.py`](../tests/test_unreal_native_semantics.py)
  and the `native_semantics` section of the same fixture.
- Unity Editor 6000.5.2f1 native conversion/basic behavior:
  [`test_unity_native_fixtures.py`](../tests/test_unity_native_fixtures.py),
  [`test_native_engine_extended_conversions.py`](../tests/test_native_engine_extended_conversions.py),
  [`native_rotation.json`](../tests/golden/unity/6000.5.2f1/native_rotation.json),
  and the [`capture procedure`](../experiments/unity/6000.5.2f1/README.md).
- Godot 4.7-stable and 4.8-dev1 native conversion/basic behavior:
  [`test_godot_native_fixtures.py`](../tests/test_godot_native_fixtures.py),
  [`test_native_engine_extended_conversions.py`](../tests/test_native_engine_extended_conversions.py),
  [`4.7 fixture`](../tests/golden/godot/4.7-stable/native_rotation.json),
  [`4.8 fixture`](../tests/golden/godot/4.8-dev1/native_rotation.json), and
  their [`capture procedures`](../experiments/godot/README.md).
- Blender 5.1.2 native `mathutils` conversions, Euler continuity, and
  swing-twist behavior:
  [`test_blender_native_fixtures.py`](../tests/test_blender_native_fixtures.py),
  [`native_rotation.json`](../tests/golden/blender/5.1.2/native_rotation.json),
  and the [`capture procedure`](../experiments/blender/5.1.2/README.md).
- Shared native proof-category parity and future versioning:
  [`proof-matrix.md`](engines/proof-matrix.md),
  [`test_engine_proof_profiles.py`](../tests/test_engine_proof_profiles.py),
  and the [`versioned capture layout`](../experiments/README.md).
- SLERP, NLERP, SQUAD, boxplus/boxminus, quaternion derivatives, and angular
  velocity integration:
  [`test_manifold_kinematics_oracle.py`](../tests/test_manifold_kinematics_oracle.py).
- Multiplication matrices, skew, Jacobians, adjoint, and homogeneous matrices:
  [`test_matrix_oracle.py`](../tests/test_matrix_oracle.py).
- Weighted alignment, means, covariance, and swing/twist:
  [`test_estimation_oracle.py`](../tests/test_estimation_oracle.py).
- Seeded random attitudes and stream parity:
  [`test_random_oracle.py`](../tests/test_random_oracle.py).
- Singular inputs, branch edges, invalid shapes, and empty NumPy layouts:
  [`test_oracle_edge_contracts.py`](../tests/test_oracle_edge_contracts.py),
  [`test_numpy_layouts.py`](../tests/test_numpy_layouts.py), and
  [`test_numpy_component_axes.py`](../tests/test_numpy_component_axes.py).

The logarithm of a negative real uses the positive first coefficient axis in
both Hamilton and JPL algebraic representations. At this singular input, a
physical JPL-to-Hamilton mapping of numpy-quaternion selects the opposite vector
sign; the branch test therefore uses the independent complex-slice formula and
records the coefficient-policy exception explicitly.

Recorded full-suite evidence is summarized in
[`VERIFICATION.md`](VERIFICATION.md). The completed suite used the repository
`.venv` and exercised the full scalar and NumPy representation matrix.
