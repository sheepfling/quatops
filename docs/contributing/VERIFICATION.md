# Contributor verification

This page is for development checks. Package users can use the
[README](../../README.md) and [API contract](../API.md) without installing the
development tools.

## Set up the development environment

Create and activate a virtual environment, then install the verification
dependencies from the repository root:

```shell
python -m pip install -e ".[dev,typecheck]" build uv==0.11.28
```

Use the activated environment for the commands below.

## Run routine checks

Run the full local checks, including the CI suite, type checking, the capture
lock, distribution builds, and isolated installed-wheel consumers:

```shell
python -m scripts.verify
```

Use `python -m scripts.verify --help` for the individual evidence, engine,
interoperability, benchmark, CI, type-check, and lock scopes. The CI matrix
runs the same `ci` scope; its package and Pyright jobs run separately. The
runner discovers its repository automatically; `--repository-root` can select
an explicit checkout.

Run only the shared packaging checks with `python -m scripts.verify package`.
Local verification and the hosted package job use that same command. It builds
the wheel and source distribution in a temporary directory, checks the wheel
for runtime caches, and installs that exact wheel into a fresh virtual
environment. The isolated interpreter runs outside the checkout. It checks
scalar conversion and semantic context copying before NumPy is installed,
then checks a float32 component-first batch after installing NumPy.

Pytest's default configuration skips opt-in performance cases. The routine
suite covers API behavior, conventions, numerical edge cases, scalar and NumPy
surfaces, and independent SciPy and numpy-quaternion references. The detailed
test map is in [oracle coverage](ORACLE_COVERAGE.md). The evidence validators
parse every maintained capture JSON without rewriting native values, then
check cross-file identity and a normalized Pydantic view for converter
packages. Raw target-specific observations remain intact in their fixtures.

CI runs the declared Python compatibility matrix. Check the workflow results
for the current branch when confirming a change; this page intentionally does
not preserve old pass counts or environment snapshots.
The pinned Alphus quality review is local-only, not a hosted CI workflow.
Install its optional tools in a separate Python 3.15 environment, then run from
the repository root:

```shell
python -m pip install "alphus @ https://github.com/sheepfling/alphus/archive/77b191b8bdf1fc4cfb50c6f3e4518be297f3d3ee.zip" -r requirements-dev-experimental.txt
python -m alphus --root . --config quality-gates.toml --run --profile review --output plain
```

The normal CI, type-check, package, and evidence-replay jobs remain unchanged.

Before pushing, run the checks in a Linux Docker container as well:

```shell
python -m scripts.verify linux
```

The command requires a clean committed checkout, bundles `HEAD`, and clones it
inside the container so Git applies Linux file modes and line-ending rules.
It does not replace the CI Python-version matrix or native captures in their
pinned target runtimes. Run additional change-specific checks as needed.

The installed scalar consumer also checks a forward-vector action, distinct
calibration cache keys, and rejection of an old value under a new binding.
The optional live [Three.js object example](../engines/threejs.md) observes
native forward/up markers and an explicit float32 assignment separately from
the standard Python checks. Evidence-bundle replay remains a separate hosted
job.

## Run opt-in benchmarks

Install the declared development dependencies, then run the benchmark suite:

```shell
python -m scripts.verify benchmarks
```

To measure reusable versus one-shot scalar and 10k-item orientation conversion
calls specifically, run:

```shell
python -m scripts.verify converter-benchmark
```

The benchmark cases check correctness before timing and warm optional Numba
kernels where applicable. The full suite writes its JSON report to the system
temporary directory. Results describe only the machine and software versions
used for that run. Keep raw output with the run artifact; do not treat timings
or winner rankings in a checked-in document as durable performance claims.
See [the experiment guide](../../experiments/README.md) for narrower commands.

## Inspect optional Numba coverage

Numba is an optional accelerator. To inspect the public API route catalog and
the signatures actually compiled in the current environment, run:

```shell
python experiments/numba_coverage.py
```

The report distinguishes direct Numba paths, compositions that reuse compiled
kernels, and ordinary NumPy vectorization. The compatibility tests also check
that registered dispatchers compile when Numba is installed. Correctness tests
remain runnable without Numba through the NumPy fallbacks.

## Interoperability evidence

See the [interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md) for
the consolidated engine, oracle, transport, and native-capture coverage matrix.
