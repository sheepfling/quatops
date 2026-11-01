# Verification: four explicit NumPy layouts

This is a provisional local candidate, not a release. The preceding JPL
milestone wheel, sdist, manifest, and report are preserved in
`experiments/jpl-milestone/`; earlier milestones remain alongside it. The
current wheel and sdist were built offline from a temporary source copy:

```sh
mkdir -p /tmp/plainquat-layout-build/source /tmp/plainquat-layout-build/dist
cp README.md API.md PROPOSED_ROTATION_REPRESENTATIONS.md LICENSE pyproject.toml setup.cfg MANIFEST.in /tmp/plainquat-layout-build/source/
cp -R src tests experiments /tmp/plainquat-layout-build/source/
PYTHONDONTWRITEBYTECODE=1 python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir --wheel-dir /tmp/plainquat-layout-build/dist /tmp/plainquat-layout-build/source
cd /tmp/plainquat-layout-build/source
PYTHONDONTWRITEBYTECODE=1 python3 -c 'from setuptools.build_meta import build_sdist; print(build_sdist("/tmp/plainquat-layout-build/dist"))'
```

After adjusting the NumPy type-check fixture, the sdist was rebuilt from that
temporary copy and retested. The wheel's package-source bytes match the working
source, and its RECORD hashes, lengths, and `py.typed` marker were verified.
The extracted final sdist's 51 package-source, test and documentation files
were compared with the working files. The builder used Python 3.9.6,
setuptools 58.0.4, and wheel 0.37.0, and warned that author and URL metadata are
missing. SciPy is a development extra, not a runtime dependency.

- Wheel SHA-256: `e3002527095ec2ed7597454c74b744b1a564b74c7b4dbd0202111016bb597b65`
- Sdist SHA-256: `6d4f33c52969e0c87746c3f0211dbf48d50457ecf16d4d8e22215a5e89a1b1d1`

## Source and installed-wheel suites

The source run used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python -m unittest discover -s tests
```

Result: **92 passed, 1 skipped** out of 93. The skip is the extended-range
test: this platform's `longdouble` has no wider range than float64.

Each wheel was installed in a separate virtual environment. Tests ran from
`/tmp`, outside the checkout. The 3.9 array environment uses existing system
NumPy; the 3.12 oracle run adds the approved SciPy environment's site-packages
through `PYTHONPATH`. No dependency was installed for these checks.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-layout-scalar/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-layout-array/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-layout-oracle/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-layout-scalar/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-layout-array/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /tmp/plainquat-layout-oracle/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
```

- Python 3.9.6 without NumPy or SciPy: **54 passed, 39 dependency skips**.
  Both scalar convention families imported without importing NumPy.
- Python 3.9.6 / NumPy 2.0.2: **85 passed, 8 skipped**; SciPy was absent and
  the extended-range test skipped.
- Python 3.12.14 / NumPy 2.3.5 / SciPy 1.17.0: **92 passed, 1 skipped**.

The shared NumPy tests cover all four convention/layout modules, known basis
products, active rotations and noncommuting compositions, cross-convention and
cross-order conversion, inverse, q/-q, half-turns, axis-angle and RotVec,
broadcast and empty batches, validation, read-only inputs, and error state.
The SciPy oracle explicitly maps the chosen convention and layout. Hot-path
checks guard against calling other layouts' multiply/rotate or conversion
helpers and check that multiply/rotate have no convention or layout selector.

## Installed sdist

A clean Python 3.9.6 virtual environment installed the final sdist offline;
the existing wheel build dependency was visible only during installation.
With that path removed, NumPy was not importable, and the extracted suite
passed **54 tests with 39 dependency skips** out of 93.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m venv /tmp/plainquat-layout-sdist
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages /tmp/plainquat-layout-sdist/bin/python -m pip install --force-reinstall --no-index --no-deps --no-cache-dir --no-build-isolation --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0.tar.gz
tar -xzf /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0.tar.gz -C /tmp/plainquat-layout-build/
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-layout-sdist/bin/python -m unittest discover -s /tmp/plainquat-layout-build/plainquat-0.1.0/tests
```

## Lint, typing and sampled hot paths

Ruff 0.17.3 and mypy 2.4.1 passed on source; the latter ignores unresolved
optional imports in this source-only check:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-7/gallery-env/bin/ruff check --no-cache src tests experiments
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/plainquat-mypy-layout --ignore-missing-imports src/plainquat
```

From `/tmp/plainquat-layout-typing`, containing copies of the consumer fixtures,
the following checked the installed wheel and NumPy 2.3.5. Each passed the
scalar and NumPy consumers and rejected `invalid.py` when checked separately.
Pyright was 1.1.414 and ty was 0.0.95.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/plainquat-layout-mypy --python-executable /tmp/plainquat-layout-oracle/bin/python consumer.py numpy_consumer.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/pyright --pythonpath /tmp/plainquat-layout-oracle/bin/python consumer.py numpy_consumer.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/ty check --python /tmp/plainquat-layout-oracle consumer.py numpy_consumer.py
```

A local timing sample on Python 3.12.14, NumPy 2.3.5, macOS ARM64 used
preconstructed C-contiguous arrays, 10,000 pairs per batch, and five repeats.
Best times in milliseconds per batch:

| Module | multiply | rotate |
| --- | ---: | ---: |
| Hamilton wxyz | 0.152 | 0.668 |
| Hamilton xyzw | 0.143 | 0.654 |
| JPL wxyz | 0.143 | 0.651 |
| JPL xyzw | 0.143 | 0.649 |

These observations apply to the sampled workload and include validation and
output allocation; they are not universal speed claims. Reproduce with:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python experiments/numpy_layout_benchmark.py
```

## Limits

The documented extreme componentwise rotation-cancellation and subnormal
conversion-rounding examples remain; neither convention nor layout is promised
correctly rounded for arbitrary float64 extremes. Scalar, NamedTuple and
dataclass operations have one native layout per convention; alternate native
scalar hot paths are not implemented. The earlier Git initialization denial
was not retried.
