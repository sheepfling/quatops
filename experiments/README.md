# Experiments and benchmarks

These scripts and opt-in tests are contributor tools. They are not part of the
runtime package. For the user-facing documentation, start with the
[project README](../README.md); for development checks, see
[contributor verification](../docs/contributing/VERIFICATION.md).

Run from the repository root with the local `.venv`. List the experiment
commands with:

```sh
.venv/bin/python experiments/run.py --list
```

Install optional experiment dependencies when needed:

```sh
.venv/bin/python -m pip install -e '.[experiments]'
```

## Script commands

| Command | Purpose | Requirements |
| --- | --- | --- |
| `check-homogeneous` | Compare rotation prototypes with exact rational cases | NumPy |
| `homogeneous-scalar` | Run a scalar rotation prototype example | None |
| `homogeneous-numpy` | Run a batched NumPy rotation prototype example | NumPy |
| `homogeneous-benchmark` | Time experimental and current rotation kernels | NumPy |
| `rotation-benchmark` | Compare rotation formulas with Fraction and Decimal references | NumPy, SciPy |
| `rotation_accuracy_comparison.py` | Compare componentwise accuracy and benchmark certified candidates | NumPy |
| `numpy-layout-benchmark` | Compare NumPy convention and layout paths | NumPy |
| `convention-benchmark` | Compare matched Hamilton/JPL operations across representations | NumPy |
| `one_off_benchmark.py` | Compare scalar hot/cold paths and optional vectorized backends | `dev` extra |
| `numba_coverage.py` | Report public API routes and compiled dispatcher signatures | Numba optional |

The homogeneous rotation programs are research prototypes. Their compensated
coefficients improve some cancellation cases, but scaling, underflow and the
final dot product still lose components on the adversarial reference set. They
are not production implementations.

`rotation_accuracy_comparison.py` compares those prototypes, a small-integer
fast path with exact fallback, and the explicit `accurate=True` path. The
integer candidate has a zero-error bound on its documented input domain and
uses the exact path elsewhere. It remains an experiment because that domain
is narrow; no general inexpensive error detector has been established. The
comparison checks accuracy before timing either candidate and saves run
details outside the repository:

```sh
PYTHONPATH=src .venv/bin/python experiments/rotation_accuracy_comparison.py \
    --json /tmp/quatops-rotation-accuracy.json
```

## Run performance benchmarks

The one-off runner checks outputs, labels each run's winner, and can save that
run's metadata and results to a local JSON file:

```sh
.venv/bin/python experiments/one_off_benchmark.py --json /tmp/quatops-one-off.json
.venv/bin/python experiments/one_off_benchmark.py --include-vectorized \
    --json /tmp/quatops-performance.json
```

For grouped NumPy and engine comparisons:

```sh
.venv/bin/python -m pytest tests/benchmarks -m benchmark --benchmark-only \
    --benchmark-group-by=group --benchmark-columns=median \
    --benchmark-json=/tmp/quatops-benchmarks.json
```

Benchmark groups verify outputs before timing. Optional Numba kernels are
prewarmed where applicable so measurements focus on steady-state calls. The
groups compare scalar one-off representations, NumPy batches, SciPy, and
numpy-quaternion where an equivalent operation is available. The signed-axis
engine benchmark compares scalar, NumPy fallback, and optional Numba routes.

Treat every report as evidence for that exact run only. Preserve detailed
results as local or CI artifacts with their environment metadata; do not copy
medians or winner tables into long-lived documentation. Timing thresholds are
not part of the benchmark tests.

## Optional Numba paths

Install the performance extra to enable Numba acceleration:

```sh
.venv/bin/python -m pip install -e '.[performance]'
```

Large NumPy batches use optional kernels for multiplication, vector rotation,
matrix and conversion operations, interpolation, integration, estimation,
selected coefficient operations, random sampling, and signed-axis game-engine
mapping. Small arrays and unavailable or failed optional kernels use NumPy
fallbacks. The full route catalog is generated from the current public exports
by `experiments/numba_coverage.py`; no timing result is implied by a compiled
signature.
