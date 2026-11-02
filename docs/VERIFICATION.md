# Verification

This is a contributor guide. Package users do not need the development tools
or checks described here.

NumPy is an optional runtime dependency. SciPy is a development dependency and
serves as an independent reference for rotation behavior. Public NumPy modules
select both convention and storage order, for example
`quatops.numpy.hamilton.xyzw`; importing `quatops.numpy` alone imports no array
operations or NumPy dependency.

## Set up the development environment

From the repository root, create the local environment and install the declared
development dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/mypy tests/typing_tests/foundational_consumer.py \
    tests/typing_tests/consumer.py tests/typing_tests/float32_consumer.py
.venv/bin/ruff check --select F,E9 src tests experiments
```

The `dev` extra installs NumPy, numpy-quaternion, SciPy, pytest, SciPy stubs,
Mypy, and Ruff. Pytest's `pythonpath` setting
loads the checkout's `src` package directly. The repository `.venv` is ignored
by Git and is the expected environment for Python checks.

## Current source-suite run

The current checkout passed its full suite on CPython 3.9.6 with NumPy 1.26.4,
numpy-quaternion 2023.0.4, SciPy 1.13.1, and pytest 8.4.2:

**2,155 passed, 1 skipped** in 7.77 seconds on 2026-11-02. The skip is the
expected `longdouble` range check on a platform where `longdouble` has no wider range
than `float64`. The engine-coordinate SciPy oracle is included in this run.
The full `.[dev]` extra did not resolve on Python 3.9 because its
`numpy-quaternion>=2024.0.3` requirement has no compatible release; the run
used the environment's preinstalled numpy-quaternion 2023.0.4. The checkout was
installed editable without changing its dependencies.

An earlier local run on another checkout passed in the environment listed
below. It is historical evidence, not an automatic claim about later source
changes; rerun the setup commands above to verify the checkout you are using.

The full default source suite passed in the repository environment on CPython
3.15.2 with NumPy 2.5.4, numpy-quaternion 2024.0.3, SciPy 1.18.1, and pytest
9.2.0:

**1,325 passed, 1 skipped; 111 opt-in benchmark tests were deselected.** The
skip is the extended-range check because this platform's `longdouble` has no
wider range than `float64`. SciPy-backed and numpy-quaternion reference tests
ran in this environment.

The three documented Mypy consumer files also passed with no issues. The
float32 consumer checks scalar inputs and explicit Python-float outputs across
the tuple, NamedTuple, dataclass, Hamilton, JPL, and compatibility-alias APIs.

The foundational-operation tests cover scalar tuples, NamedTuple and dataclass
APIs in both Hamilton and JPL conventions. NumPy checks cover both coefficient
orders and both component-axis layouts, including broadcasted inputs, matrix
conversion, vector alignment, and dot products with extreme magnitudes.

The prioritized feature tests cover algebraic powers and roots, hyperbolic
functions, interpolation, SO(3) Jacobians, skew matrices, and angular-velocity
integration across scalar, NamedTuple, dataclass, all NumPy convention and
coefficient-order modules, and the four NumPy adapters with components on axis 0. The
component-axis tests compare adapter results with the corresponding trailing-
axis kernels, including broadcast, empty, and ambiguous square shapes.

The extended-attitude tests cover all six Tait-Bryan sequences in both fixed
and body-axis forms, including gimbal lock; inverse algebraic functions and
small-argument paths; quaternion-valued powers; local attitude increments;
derivatives against finite differences; sampled integration; MRP and
swing/twist factorization; weighted Markley and intrinsic means; body/world
covariance; random-sample second moments; homogeneous transforms; and
weighted Wahba alignment against SciPy. They exercise all 15 scalar-facing
modules and all eight NumPy layouts, with the coefficient axis last or at axis 0 across
both conventions and coefficient orders. NumPy checks include operations with different component widths,
broadcasted batches, and outputs whose component and matrix axes move.

Seeded Monte Carlo checks compare 21 generic algebra operations across all 15
scalar-facing modules and all eight NumPy layouts against numpy-quaternion. They
use 192 scalar samples and 1,024 array samples per backend. Positive, zero, and
negative integer powers are checked separately against numpy-quaternion.
Rotation-vector conversion, vector action, and matrix conversion are
independently checked against numpy-quaternion for the same sample counts,
including scaled non-unit inputs.

## Independent oracle coverage

`tests/test_analytic_oracle.py` checks all twelve sine, cosine, tangent,
hyperbolic, and inverse analytic functions against complex-slice `cmath`
formulas. It also checks scalar/vector extraction and the negative-real
logarithm branch on every representation. For `log(-r)`, the documented
algebraic coefficient policy selects the positive first vector axis in both
Hamilton and JPL; `numpy-quaternion`'s physical JPL mapping selects the
opposite sign at this zero-vector branch, so the oracle uses the explicit
complex-slice formula there.

`tests/test_rotation_conversion_oracle.py` checks coefficient order and
Hamilton/JPL conversion, canonicalization and sign policies, rotation distance,
axis-angle and two-vector conversions, all twelve Euler axis patterns in both
extrinsic and intrinsic forms (24 sequence strings), MRP, and the rotation-vector
aliases against SciPy or direct coefficient rules. Its samples include non-unit
inputs, angles beyond a full turn, antipodal vector pairs, and MRP's principal
range.

`tests/test_engine_basis_oracle.py` checks engine-boundary signed-axis maps for
Hamilton and JPL inputs in both coefficient orders against SciPy rotation
matrices, across all 48 signed permutations. It also checks the SLERP/map
commutation property and NumPy component-axis preservation. The focused run in
this checkout passed 107 cases; it does not test an engine runtime.

`tests/test_engine_euler_golden.py` checks seven frozen, source-attributed
Unity.Mathematics and Godot Euler fixtures across 23 scalar and NumPy
representations (161 case/backend combinations). Expected quaternion and
matrix values are stored in `tests/golden/engine_euler_fixtures.json`; these
are upstream static fixtures, not local engine captures.

`tests/test_manifold_kinematics_oracle.py` compares SLERP and SQUAD with SciPy
`Slerp`, NLERP with a direct normalized coefficient blend, local updates with
SciPy rotation composition, quaternion derivatives with numpy-quaternion
products, and constant and sampled body/world integration with composed SciPy
increments. Sampled integration covers both left-hold and midpoint methods.
SLERP and NLERP also check sign-preserving long arcs (`shortest=False`) and
extrapolation before and after the interval against direct S3 coefficient
formulas.

`tests/test_matrix_oracle.py` checks left/right multiplication matrices using
numpy-quaternion basis products, SO(3) Jacobians and inverses against explicit
closed forms, and adjoint/homogeneous transforms against SciPy. The matrix
oracles include both component-axis layouts and both quaternion coefficient
orders.

`tests/test_estimation_oracle.py` checks weighted and default unweighted Markley
means against SciPy, intrinsic means against an independently solved SciPy
tangent residual, covariance against SciPy rotation-log samples, weighted and
unweighted Wahba alignment against `Rotation.align_vectors`, and swing/twist
factors against a separate SciPy factorization. `tests/test_random_oracle.py`
checks seed compatibility between
scalar and NumPy generation, unit length, S3 first and second moments, and empty
NumPy batch shapes. These oracle tests run over all 15 scalar-facing modules
and all eight NumPy layouts.

`tests/test_oracle_edge_contracts.py`, together with the existing NumPy
layout/component-axis and numeric-extreme tests, checks zero and singular
inputs, half-turn and near-zero rotation limits, invalid shapes, non-finite
inputs, broadcasting, empty batches, custom matrix-validation tolerances, and
component-axis outputs. The adapter fix for arrays with components on axis 0 also ensures inverse
analytic functions return their quaternion component axis first.
Coefficient-order adapters for sampled-rate integration, rotation
means/covariance, and vector alignment are checked against the same independent
SciPy references as the trailing-axis backends.

Scalar input typing is checked by `tests/typing_tests/float32_consumer.py` with
Mypy across all 15 scalar-facing modules. Runtime checks in
`tests/test_scalar_float32_inputs.py` compare float32 inputs with independent
numpy-quaternion, complex-slice, and SciPy references. Decimal cases exercise
the `SupportsFloat` fallback outside the built-in and NumPy real-number paths.
Scalar functions return built-in Python floats as documented.

## Hamilton/JPL performance check

`experiments/convention_benchmark.py` compares matched operations on equivalent
Hamilton and JPL inputs. The scalar workloads include arithmetic, normalization,
rotation, analytic functions, interpolation, distance, matrices, conversions,
estimation, and integration across tuple, NamedTuple, and dataclass APIs. Batched NumPy
workloads compare multiply and rotate on 10,000 XYZW quaternions. Timings
include input validation and result construction, and are local sanity checks,
not a general performance guarantee.

The recorded run found broadly similar scalar performance between Hamilton
and JPL, with operation-specific winners. NumPy multiplication and rotation
were comparable between conventions. `tests/test_namespaces.py` checks direct
function identity for convention-neutral JPL kernels and patches
Hamilton/conversion paths while exercising basic and extended JPL scalar
operations.

Reproducible scalar and batch comparisons live in `tests/benchmarks/`. Run
`.venv/bin/python experiments/one_off_benchmark.py` to compare hot and cold
Hamilton/JPL one-off workloads. Add `--include-vectorized` to include array
workloads. The report labels each workload's winner; `--json PATH` records the
medians and environment for that run.

Run all grouped vectorized comparisons and save the raw results with:

```sh
.venv/bin/python -m pytest tests/benchmarks -m benchmark --benchmark-only --benchmark-group-by=group --benchmark-columns=median --benchmark-json=/tmp/quatops-benchmarks.json
```

The benchmark groups check correctness before timing and warm optional Numba
kernels. The recorded run used CPython 3.15.2, NumPy 2.5.4, SciPy 1.18.1,
numpy-quaternion 2024.0.3, and Numba 0.68.1. The run reports a winner for each
one-off case and shows vectorized parity for the tested conversion and rotation
workloads. Some prebuilt SciPy and raw numpy-quaternion routes remain faster;
the JSON report keeps each per-case result visible.

Numba remains optional and lazy. Install the `performance` extra to enable the
compiled multiplication, vector rotation, matrix, Euler, rotation-vector,
axis-angle, integration, interpolation, MRP, analytic-function and power,
skew/Jacobian, sequence-continuity, batch-estimation, and coefficient paths.
Most kernels serve batches of 128 or more; sampled integration uses trajectory
count times interval count. The pure NumPy fallbacks remain available when
Numba is absent or a kernel cannot compile.

## Additional compiled NumPy coverage

Inspect Numba dispatcher coverage and confirm that every registered kernel
compiled a nopython signature:

```sh
.venv/bin/python experiments/numba_coverage.py
```

The report warms the public batch paths for Hamilton/JPL and WXYZ/XYZW, then
lists each registered `njit` signature or GUFunc loop signature as `compiled`,
`registered` (loaded but not compiled), or `missing`. The pytest suite runs the
same check, so an installed Numba backend with an uncompiled registered kernel
fails validation. This measures dispatcher compilation coverage; it does not
measure which branches or source lines inside the compiled kernels ran.
The recorded CPython 3.15.2 / Numba 0.68.1 run compiled all 32 registered
dispatchers.

Run the grouped interpolation, MRP, integration, and coefficient-operation
comparisons with:

```sh
.venv/bin/python -m pytest tests/benchmarks/test_numpy_fast_coverage.py -m benchmark --benchmark-only --benchmark-group-by=group --benchmark-columns=median
```

The preflight checks correctness and compares compiled, fallback, and external
implementations. In the recorded run, compiled interpolation, MRP, and sampled
integration paths outperformed their scalar or NumPy fallbacks. Coefficient
operations had mixed rankings: normalization, inverse, and dot were
competitive with checked numpy-quaternion, while norm remained slower. Use the
saved benchmark output for exact medians and the current winner per operation.

The remaining compiled paths have their own opt-in comparisons:

```sh
.venv/bin/python -m pytest tests/benchmarks/test_numpy_remaining_numba.py -m benchmark --benchmark-only --benchmark-group-by=group --benchmark-columns=median
```

The recorded run showed faster compiled axis-angle, analytic-function, power,
estimator, and Jacobian paths than their NumPy or scalar fallbacks. QuatOps
`exp` and `log` were comparable to raw and checked numpy-quaternion for the
tested batch. These are qualitative results from one machine; the benchmark
output contains exact medians and environment metadata for each rerun.

Other batch APIs reuse the compiled primitives: SQUAD composes SLERP, local
attitude updates compose rotation-vector conversion and multiplication,
quaternion derivatives use the multiplication kernel, and homogeneous
transforms and SO(3) adjoints use the matrix kernels. Their correctness is
covered in `tests/test_numba_compatibility.py`; this benchmark file reports
direct timings for the added batch paths above.

## Package and numerical scope

`pyproject.toml` defines the package metadata and optional dependencies.
`setuptools-scm` supplies the version from Git metadata and is a build-only
dependency. NumPy and SciPy are not runtime requirements for scalar APIs.

The implementation does not claim correctly rounded results for arbitrary
float64 extremes. Rotation-cancellation and subnormal conversion-rounding
limits are documented in the [API contract](API.md) and the
[numerical investigation](ROTATION_NUMERICS.md).
