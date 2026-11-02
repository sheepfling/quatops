# QuatOps experiments

These contributor and research tools are optional for package users.

Run commands from the repository root with Python 3.9 or newer. The launcher
adds the checkout's `src/` directory automatically, so you do not need to set
`PYTHONPATH`. In this checkout, use `.venv/bin/python` so optional dependencies
come from the repository environment.

```sh
.venv/bin/python experiments/run.py --list
.venv/bin/python experiments/run.py check-homogeneous
.venv/bin/python experiments/run.py rotation-benchmark
.venv/bin/python experiments/one_off_benchmark.py --json /tmp/quatops-one-off.json
.venv/bin/python experiments/one_off_benchmark.py --include-vectorized --json /tmp/quatops-performance.json
```

For the optional NumPy and SciPy experiments, install the dependencies once
from the checkout:

```sh
.venv/bin/python -m pip install -e '.[experiments]'
```

| Command | Purpose | Needs |
| --- | --- | --- |
| `check-homogeneous` | Compare experimental rotation prototypes with exact rational cases | NumPy |
| `homogeneous-scalar` | Run one scalar prototype example | None |
| `homogeneous-numpy` | Run one batched NumPy prototype example | NumPy |
| `homogeneous-benchmark` | Time prototype and current scalar/NumPy rotation kernels | NumPy |
| `rotation-benchmark` | Compare current rotations with Fraction and Decimal references | NumPy, SciPy |
| `numpy-layout-benchmark` | Compare the four NumPy convention/layout paths | NumPy |
| `convention-benchmark` | Compare matched Hamilton/JPL operations across tuple, NamedTuple, dataclass, and NumPy paths | NumPy |
| `one_off_benchmark.py` | Compare hot/cold Hamilton/JPL representations; optionally add 10k NumPy comparisons with SciPy and numpy-quaternion; print the winner per workload | Full `dev` extra |

`check-homogeneous` accepts optional module names after the command. For
example, `.venv/bin/python experiments/run.py check-homogeneous homogeneous_scalar`
runs only the scalar prototype. Benchmarks are local, workload-specific
measurements; they do not establish general performance or accuracy guarantees.
The convention benchmark reports five interleaved median timings and the JPL to
Hamilton ratio. Scalar timings include validation and the selected output
representation; NumPy timings use 10,000 equivalent XYZW quaternions per call.
See the investigation documents under `docs/` for the cases and limitations
behind each experiment.

The one-off runner reports median time per call, labels the winner, and marks
backends within 1% of the fastest median as co-winners. `--include-vectorized`
adds the 10k multiply and rotation comparisons against SciPy and
numpy-quaternion. The runner records Python, NumPy, numpy-quaternion, SciPy,
pytest, platform, and measurement settings; `--json PATH` saves the report for
later comparison. Adjust `--min-rounds` and `--max-time` to trade run time for
more measurement samples.

To compare scalar and 10,000-item vector workloads with SciPy and
`numpy-quaternion`, use:

```sh
.venv/bin/python -m pytest -m benchmark --benchmark-only tests/benchmarks
```

The signed-axis game-engine benchmark compares the scalar oracle, the NumPy
fallback, and the optional Numba batch mapper. It checks exact outputs before
timing each route.

Large NumPy batches use optional Numba kernels for multiplication, vector
rotation, matrix/rotation-vector/axis-angle conversion, interpolation, MRP,
analytic functions and powers, integration, SO(3) Jacobians, sequence
continuity, batch estimators, coefficient operations, and signed-axis
game-engine mapping. Install them with
`.venv/bin/python -m pip install -e '.[performance]'`. Most paths switch at 128
items; sampled integration uses trajectory count times interval count. The
first eligible call compiles each kernel; Numba caches them for later runs.
Without Numba, QuatOps uses its NumPy fallbacks, and small arrays stay on those
paths. If an installed Numba fails to import or compile a kernel, QuatOps logs
the failure, disables that accelerator, and continues with NumPy. Benchmark
preflight calls compile the kernels before timing, so the results show
steady-state performance. Pytest prints the NumPy, numpy-quaternion, SciPy, and
Numba versions. Run `experiments/numba_coverage.py` to inspect actual compiled
signatures, or `tests/benchmarks/test_numpy_remaining_numba.py` for the
axis-angle, analytic, power, Jacobian, sequence, and estimator comparisons.

CI exercises the Numba kernels on every advertised Python version, from 3.9
through 3.15. `requires-python` remains open-ended at `>=3.9`. The performance
extra selects a Numba minimum compatible with each interpreter: 0.60 for
Python 3.9, 0.61 for 3.10–3.13, 0.63 for 3.14, and 0.68 for 3.15 or newer.
Newer interpreters can resolve a later compatible Numba release; Numba remains
optional for users who want only the pure NumPy implementation.
The Python 3.9 development extra also selects NumPy below 2 and
numpy-quaternion 2022.4.4, and omits `scipy-stubs`, to keep the oracle and type
check dependencies installable on that interpreter.
