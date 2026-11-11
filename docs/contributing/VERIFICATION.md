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
lock, and package build:

```shell
python -m scripts.verify
```

Use `python -m scripts.verify --help` for the individual evidence, engine,
interoperability, benchmark, CI, type-check, and lock scopes. The CI matrix
runs the same `ci` scope; its package and Pyright jobs run separately. The
runner discovers its repository automatically; `--repository-root` can select
an explicit checkout.

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
The pinned Alphus quality review runs on pushes and pull requests as a separate
workflow.

Before pushing, run the checks in a Linux Docker container as well:

```shell
python -m scripts.verify linux
```

The command requires a clean committed checkout, bundles `HEAD`, and clones it
inside the container so Git applies Linux file modes and line-ending rules.
It does not replace the CI Python-version matrix or native captures in their
pinned target runtimes. Run additional change-specific checks as needed.

The package job also builds both distribution formats, installs the wheel into
an isolated target directory, checks scalar conversion without optional
dependencies, then installs NumPy and checks a float32 batch round trip. Wheel
contents are expected to exclude local Numba cache files.

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
