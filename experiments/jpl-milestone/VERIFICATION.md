# Verification: explicit Hamilton and JPL backends

This is a provisional local candidate, not a release. The previous canonical
zero milestone wheel, sdist, manifest and verification report are preserved in
`experiments/canonical-zero-milestone/`; earlier snapshots remain alongside it.
Both current artifacts were built offline from a temporary source copy rather
than generating metadata in the production source tree:

```sh
mkdir -p /tmp/plainquat-jpl-build/source /tmp/plainquat-jpl-build/dist
cp README.md API.md PROPOSED_ROTATION_REPRESENTATIONS.md LICENSE pyproject.toml setup.cfg MANIFEST.in /tmp/plainquat-jpl-build/source/
cp -R src tests experiments /tmp/plainquat-jpl-build/source/
PYTHONDONTWRITEBYTECODE=1 python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir --wheel-dir /tmp/plainquat-jpl-build/dist /tmp/plainquat-jpl-build/source
cd /tmp/plainquat-jpl-build/source
PYTHONDONTWRITEBYTECODE=1 python3 -c 'from setuptools.build_meta import build_sdist; print(build_sdist("/tmp/plainquat-jpl-build/dist"))'
```

CPython 3.9.6, setuptools 58.0.4, wheel 0.37.0 were the existing build tools.
The sdist builder warned that author and URL metadata are absent. The wheel
contains `py.typed` and both convention families; its RECORD hashes and lengths
were verified. The extracted sdist's package source, tests and documentation
were checked against the working files. SciPy is a development extra, not a
runtime dependency.

- Wheel SHA-256: `a5ed6a57691addbad86e51989ce85996badf8ab49bc013715f00741ad2afda70`
- Sdist SHA-256: `423f7945cf84b78bd0445dffdf8b5343a4104945a0563966232414b850c2775d`

## Source and installed-wheel suites

The source run used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python -m unittest discover -s tests
```

Result: **85 passed, 1 skipped** out of 86. The skip is the extended-range test:
this platform's NumPy `longdouble` has no wider range than float64.

Each wheel was installed in a separate virtual environment, and each test
command ran from `/tmp`, outside the checkout. The 3.9 array environment uses
existing system NumPy; the 3.12 oracle run adds the existing approved
SciPy environment's site-packages through `PYTHONPATH`. No dependency was
installed for these checks.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-jpl-scalar/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-jpl-array/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-jpl-oracle/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-jpl-scalar/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-jpl-array/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /tmp/plainquat-jpl-oracle/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
```

- Python 3.9.6 without NumPy or SciPy: **54 passed, 32 dependency skips**.
  Both scalar convention families imported without importing NumPy.
- Python 3.9.6 / NumPy 2.0.2: **79 passed, 7 skipped**. SciPy was absent;
  the longdouble extended-range test also skipped.
- Python 3.12.14 / NumPy 2.3.5 / SciPy 1.17.0: **85 passed, 1 skipped**.

The suites cover both products and explicit physical convention conversion;
noncommuting composition; active rotation, inverse, identity, half turns,
`q` versus `-q`; representation and axis-angle/RotVec parity; batch shapes,
validation, non-mutation and NumPy error settings. SciPy comparisons explicitly
translate JPL to SciPy's Hamilton scalar-last convention. Separate tests check
that JPL multiplication and rotation do not call Hamilton hot paths or the
explicit conversion helpers.

## Installed sdist

A clean Python 3.9.6 virtual environment installed the sdist offline; the
existing wheel build dependency was made visible only while installing. With
that path removed, NumPy was not importable, both conventions imported from
the installed package, and the extracted test suite passed **54 tests with 32
dependency skips** out of 86.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m venv /tmp/plainquat-jpl-sdist
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages /tmp/plainquat-jpl-sdist/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-build-isolation --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0.tar.gz
tar -xzf /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0.tar.gz -C /tmp/plainquat-jpl-build/
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-jpl-sdist/bin/python -m unittest discover -s /tmp/plainquat-jpl-build/plainquat-0.1.0/tests
```

## Lint, typing and sampled hot paths

Ruff 0.17.3 and mypy 2.4.1 passed on source; the latter ignores unresolved
optional imports in this source-only check:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-7/gallery-env/bin/ruff check --no-cache src tests experiments
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/plainquat-mypy-jpl --ignore-missing-imports src/plainquat
```

From `/tmp/plainquat-jpl-typing`, containing copies of the consumer fixtures,
the following checked the installed wheel. Each passed the valid consumer and
rejected `invalid.py` when substituted, confirming the installed hints were read.
Pyright was 1.1.414 and ty was 0.0.95.

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/plainquat-jpl-mypy --python-executable /tmp/plainquat-jpl-oracle/bin/python consumer.py
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/pyright --pythonpath /tmp/plainquat-jpl-oracle/bin/python consumer.py
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/ty check --python /tmp/plainquat-jpl-oracle consumer.py
```

A local timing sample on Python 3.12.14, NumPy 2.3.5, macOS ARM64, used
preconstructed inputs and five repeats. Best tuple multiplication was 2.269 µs
Hamilton and 2.250 µs JPL; tuple rotation was 3.834 µs and 3.805 µs. For
10,000 pairs, NumPy multiplication was 0.142 ms for each, and rotation was
0.647 ms Hamilton and 0.649 ms JPL. These are workload-specific observations,
not universal speed claims. The full script includes NamedTuple and dataclass:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python experiments/convention_benchmark.py
```

## Limits

The documented extreme componentwise rotation-cancellation and subnormal
conversion-rounding examples remain; neither convention is promised correctly
rounded for arbitrary float64 extremes. Canonicalization and large-angle limits
are in `API.md`. Frozen slotted dataclass pickle round-tripping remains outside
the documented contract. The earlier Git initialization denial was not retried.
