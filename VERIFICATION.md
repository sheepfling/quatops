# Verification: QuatOps local rename

This is a provisional local candidate, not a release. The prior `plainquat`
wheel, sdist, source archive, manifest and verification report are preserved
in `experiments/plainquat-pre-rename/`; earlier milestone snapshots remain
alongside it. The distribution is named `quatops` and the import root is
`quatops`. The wheel does not contain a `plainquat` import shim.

The current wheel and sdist were built offline from a temporary source copy:

```sh
mkdir -p /tmp/quatops-rename-build/source /tmp/quatops-rename-build/dist
cp README.md API.md PROPOSED_ROTATION_REPRESENTATIONS.md LICENSE pyproject.toml setup.cfg MANIFEST.in /tmp/quatops-rename-build/source/
cp -R src tests experiments /tmp/quatops-rename-build/source/
PYTHONDONTWRITEBYTECODE=1 python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir --wheel-dir /tmp/quatops-rename-build/dist /tmp/quatops-rename-build/source
cd /tmp/quatops-rename-build/source
PYTHONDONTWRITEBYTECODE=1 python3 -c 'from setuptools.build_meta import build_sdist; print(build_sdist("/tmp/quatops-rename-build/dist"))'
```

CPython 3.9.6, setuptools 58.0.4 and wheel 0.37.0 were the existing build
tools. Setuptools warned that author and URL metadata are absent. The wheel
contains `quatops/py.typed`, no `plainquat/` package, and both convention-first
and backend-first modules. Its RECORD hashes and all package-source bytes were
verified; the extracted sdist's 64 package-source, test and documentation files
were compared to the workspace. NumPy is optional; SciPy is development-only.

- Wheel SHA-256: `1730ce69eb4a9689747bc923244c00126c75d135967755a3b666840d9cae5c63`
- Sdist SHA-256: `1f23adcbce12c19b8598de084cba26ea7e8aeff41cffc56d0838dff7b095aab0`

## Source and installed-wheel suites

The source suite used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python -m unittest discover -s tests
```

Result: **96 passed, 1 skipped** out of 97. The skip is the extended-range
test: this platform's `longdouble` has no wider range than float64.

Each wheel was installed in a separate virtual environment, and tests ran from
`/tmp`, outside the checkout. The Python 3.9 array environment uses existing
system NumPy; the Python 3.12 oracle run adds the approved SciPy environment's
site-packages through `PYTHONPATH`. No dependency was installed for these checks.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/quatops-rename-scalar/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/quatops-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/quatops-rename-array/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/quatops-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/quatops-rename-oracle/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/quatops-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/quatops-rename-scalar/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 /tmp/quatops-rename-array/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /tmp/quatops-rename-oracle/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
```

- Python 3.9.6 without NumPy or SciPy: **57 passed, 40 dependency skips**.
  Confirmed NumPy and `plainquat` were not importable in that environment.
- Python 3.9.6 / NumPy 2.0.2: **89 passed, 8 skipped**. SciPy was absent;
  the longdouble extended-range test also skipped.
- Python 3.12.14 / NumPy 2.3.5 / SciPy 1.17.0: **96 passed, 1 skipped**.

The existing numerical, convention, layout, shape, non-mutation and SciPy
checks remain. Added tests assert backend-first imports are the same function
or class objects as the convention-first implementations, including all four
NumPy convention/layout choices, and that scalar imports do not load NumPy.
There is no per-call dispatch or conversion in these aliases.

## Installed sdist

A clean Python 3.9.6 environment installed the sdist offline. The existing
wheel build dependency was visible only during installation; NumPy and
`plainquat` were not importable afterward. Extracted tests passed **57 tests
with 40 dependency skips** out of 97:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m venv /tmp/quatops-rename-sdist
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages /tmp/quatops-rename-sdist/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-build-isolation --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/quatops-0.1.0.tar.gz
tar -xzf /Users/rick/Documents/Codex/2026-10-29/task-8/dist/quatops-0.1.0.tar.gz -C /tmp/quatops-rename-build/
PYTHONDONTWRITEBYTECODE=1 /tmp/quatops-rename-sdist/bin/python -m unittest discover -s /tmp/quatops-rename-build/quatops-0.1.0/tests
```

## Lint, typing and sampled hot paths

Ruff 0.17.3 and mypy 2.4.1 passed on source; the latter ignores unresolved
optional imports in this source-only check:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-7/gallery-env/bin/ruff check --no-cache src tests experiments
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/quatops-mypy --ignore-missing-imports src/quatops
```

From `/tmp/quatops-rename-typing`, containing copies of `tests/typing` fixtures,
the following checked the installed wheel and NumPy 2.3.5. Mypy 2.4.1,
Pyright 1.1.414 and ty 0.0.95 passed both valid scalar and NumPy consumers;
each rejected `invalid.py` when checked separately.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/quatops-rename-mypy --python-executable /tmp/quatops-rename-oracle/bin/python consumer.py numpy_consumer.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/pyright --pythonpath /tmp/quatops-rename-oracle/bin/python consumer.py numpy_consumer.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/ty check --python /tmp/quatops-rename-oracle consumer.py numpy_consumer.py
```

A local sample on Python 3.12.14, NumPy 2.3.5, macOS ARM64 used preconstructed
C-contiguous arrays, 10,000 pairs per batch and five repeats. Best rotate times
were 0.766 ms Hamilton wxyz, 0.725 ms Hamilton xyzw, 0.759 ms JPL wxyz and
0.722 ms JPL xyzw. This sample includes validation and allocation and is not a
universal speed claim. Backend-first aliases are the identical function objects,
so they do not add a wrapper to these calls. Reproduce the sample with:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python experiments/numpy_layout_benchmark.py
```

## Limits

Existing extreme componentwise rotation-cancellation and subnormal conversion
rounding limits remain; neither convention nor layout is promised correctly
rounded for arbitrary float64 extremes. Tuple, NamedTuple and dataclass
operations retain one native layout per convention. Operator overloads, SLERP,
matrix conversions and a common runtime convention registry are not implemented.
The earlier Git initialization denial was not retried.
