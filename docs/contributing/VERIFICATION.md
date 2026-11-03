# Contributor verification

This page is for development checks. Package users can use the
[README](../../README.md) and [API contract](../API.md) without installing the
development tools.

## Set up the development environment

From the repository root, create the local environment and install the declared
development dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,typecheck]'
```

Use the repository `.venv` for all Python commands. Do not substitute a global
interpreter when a dependency is missing.

## Run routine checks

These are the same checks used by CI:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy tests/typing_tests/foundational_consumer.py \
    tests/typing_tests/consumer.py tests/typing_tests/float32_consumer.py
.venv/bin/pyright --pythonpath .venv/bin/python
.venv/bin/ruff check src tests experiments
```

Pytest's default configuration skips opt-in performance cases. The routine
suite covers API behavior, conventions, numerical edge cases, scalar and NumPy
surfaces, and independent SciPy and numpy-quaternion references. The detailed
test map is in [oracle coverage](ORACLE_COVERAGE.md).

CI runs the declared Python compatibility matrix. Check the workflow results
for the current branch when confirming a change; this page intentionally does
not preserve old pass counts or environment snapshots.

## Run opt-in benchmarks

Install the declared development dependencies, then run the benchmark suite:

```sh
.venv/bin/python -m pytest tests/benchmarks -m benchmark --benchmark-only \
    --benchmark-group-by=group --benchmark-columns=median \
    --benchmark-json=/tmp/quatops-benchmarks.json
```

The benchmark cases check correctness before timing and warm optional Numba
kernels where applicable. Results describe only the machine and software
versions used for that run. Keep raw output with the run artifact; do not treat
timings or winner rankings in a checked-in document as durable performance
claims. See [the experiment guide](../../experiments/README.md) for narrower
commands.

## Inspect optional Numba coverage

Numba is an optional accelerator. To inspect the public API route catalog and
the signatures actually compiled in the current environment, run:

```sh
.venv/bin/python experiments/numba_coverage.py
```

The report distinguishes direct Numba paths, compositions that reuse compiled
kernels, and ordinary NumPy vectorization. The compatibility tests also check
that registered dispatchers compile when Numba is installed. Correctness tests
remain runnable without Numba through the NumPy fallbacks.
