# Verification

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
```

The `dev` extra installs NumPy, numpy-quaternion, SciPy, pytest, SciPy stubs,
and Mypy. Pytest's `pythonpath` setting loads the checkout's `src` package
directly. The repository `.venv` is ignored by Git and is the expected
environment for Python checks.

## Latest source-suite run

The full suite passed in the repository environment on CPython 3.15.2 with
NumPy 2.5.4, numpy-quaternion 2024.0.3, SciPy 1.18.1, and pytest 9.2.0:

**1,335 passed, 6 skipped.** Five skips are the expected Hamilton cases for a
golden test of explicit JPL-to-Hamilton conversion; the remaining skip is the
extended-range check because this platform's `longdouble` has no wider range
than `float64`. SciPy-backed and numpy-quaternion reference tests ran in this
environment.

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
coefficient-order modules, and the four component-first NumPy adapters. The
component-axis tests compare adapter results with the corresponding trailing-
axis kernels, including broadcast, empty, and ambiguous square shapes.

The extended-attitude tests cover all six Tait-Bryan sequences in both fixed
and body-axis forms, including gimbal lock; inverse algebraic functions and
small-argument paths; quaternion-valued powers; local attitude increments;
derivatives against finite differences; sampled integration; MRP and
swing/twist factorization; weighted Markley and intrinsic means; body/world
covariance; random-sample second moments; homogeneous transforms; and
weighted Wahba alignment against SciPy. They exercise all 15 scalar-facing
modules and all eight trailing/component-first NumPy convention and storage
layouts. NumPy checks include operations with different component widths,
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
axis-angle and two-vector conversions, all twelve extrinsic/intrinsic Euler
sequences, MRP, and the rotation-vector aliases against SciPy or direct
coefficient rules. Its samples include non-unit inputs, angles beyond a full
turn, antipodal vector pairs, and MRP's principal range.

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
component-axis outputs. The component-first adapter fix also ensures inverse
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

On the repository machine, an interleaved run with CPython 3.15.2 and NumPy
2.5.4 put nearly all matched scalar workloads within about 4% between
conventions. JPL dataclass inversion and JPL matrix-to-quaternion conversion
were about 11% and 5–7% faster, respectively. NumPy multiply and rotate were
within measurement resolution. `tests/test_namespaces.py` checks direct
function identity for convention-neutral JPL kernels and patches
Hamilton/conversion paths while exercising basic and extended JPL scalar
operations.

## Package and numerical scope

`pyproject.toml` defines the package metadata and optional dependencies.
`setuptools-scm` supplies the version from Git metadata and is a build-only
dependency. NumPy and SciPy are not runtime requirements for scalar APIs.

The implementation does not claim correctly rounded results for arbitrary
float64 extremes. Rotation-cancellation and subnormal conversion-rounding
limits are documented in the [API contract](API.md) and the
[numerical investigation](ROTATION_NUMERICS.md).
