# QuatOps experiments

Run commands from the repository root with Python 3.9 or newer. The launcher
adds the checkout's `src/` directory automatically, so you do not need to set
`PYTHONPATH`. In this checkout, use `.venv/bin/python` so optional dependencies
come from the repository environment.

```sh
.venv/bin/python experiments/run.py --list
.venv/bin/python experiments/run.py check-homogeneous
.venv/bin/python experiments/run.py rotation-benchmark
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

`check-homogeneous` accepts optional module names after the command. For
example, `.venv/bin/python experiments/run.py check-homogeneous homogeneous_scalar`
runs only the scalar prototype. Benchmarks are local, workload-specific
measurements; they do not establish general performance or accuracy guarantees.
The convention benchmark reports five interleaved median timings and the JPL to
Hamilton ratio. Scalar timings include validation and the selected output
representation; NumPy timings use 10,000 equivalent XYZW quaternions per call.
See the investigation documents under `docs/` for the cases and limitations
behind each experiment.
