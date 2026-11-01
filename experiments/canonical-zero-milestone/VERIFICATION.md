# Verification of the local conversion milestone

This report covers the provisional local candidate, not a release. The prior
conversion milestone wheel and its verification report and source manifest are
preserved in `experiments/conversion-milestone/`. The earlier wheel remains in
`experiments/prior-wheel/`. The current wheel was built without network access
or dependency installation:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir --wheel-dir /tmp/plainquat-wheel-zero .
```

It contains `plainquat/py.typed`; SciPy is a development extra, not a runtime
requirement. SHA-256 of `dist/plainquat-0.1.0-py3-none-any.whl`:
`7131255f3d7131baaf872f2d2e68242054638a8a5726acab8766d4e23ca2a556`.

## Installed-wheel suites

Each test command below was run from `/tmp`, outside the source checkout. Each
virtual environment received the wheel using `pip install --no-index --no-deps
--no-cache-dir --no-compile`. The 3.9 array environment uses existing system
NumPy; the 3.12 oracle run adds the existing approved development environment's
site-packages to `PYTHONPATH`. No dependency was installed for these checks.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-scalar/bin/python -m pip install --force-reinstall --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-array/bin/python -m pip install --force-reinstall --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-oracle/bin/python -m pip install --force-reinstall --no-index --no-deps --no-cache-dir --no-compile /Users/rick/Documents/Codex/2026-10-29/task-8/dist/plainquat-0.1.0-py3-none-any.whl

PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-scalar/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-array/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /tmp/plainquat-conversions-oracle/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
```

- CPython 3.9.6, scalar-only environment: 42 passed, 27 NumPy/SciPy-dependent
  skips out of 69. Confirmed NumPy is not importable there.
- CPython 3.9.6, NumPy 2.0.2: 63 passed, 6 skipped out of 69. SciPy is not
  available there; the longdouble extended-range test also skips on this host.
- CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0: 68 passed, 1 skipped out of 69.
  The skip is the extended-range test because longdouble has no larger range
  than float64 on this platform. SciPy conversion and rotation references ran.

The source suite also passed 68 tests with one extended-range skip:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/bin/python -m unittest discover -s tests
```

The suite separates exact Fraction/Decimal edge checks from SciPy rotation
comparisons. It covers norm return type; the NumPy coefficient-order correction;
identity, quarter turns, exact and adjacent pi, canonical positive zeros, subnormal and large
inputs; invalid values, shapes, zero axes and quaternions, and norm overflow;
array views, empty batches, non-mutation and NumPy error-state preservation.

## Source distribution

The sdist was built without network access from a temporary source copy, so
building it did not generate metadata in the production source tree. `MANIFEST.in`
includes the API, proposal and experiment documentation and tests in the archive.
Setuptools 58.0.4 and wheel 0.37.0 were the existing build tools; the build
warned that author and URL metadata are absent from this provisional candidate.
SHA-256 of `dist/plainquat-0.1.0.tar.gz`:
`64049a9ce6c6cbdccdedf919638f69421f9b856ede616645b300bdec3f2235be`.

```sh
# From the project directory, create the temporary build copy:
mkdir -p /tmp/plainquat-sdist-final/source /tmp/plainquat-sdist-final/dist
cp README.md API.md PROPOSED_ROTATION_REPRESENTATIONS.md LICENSE pyproject.toml setup.cfg MANIFEST.in /tmp/plainquat-sdist-final/source/
cp -R src tests experiments /tmp/plainquat-sdist-final/source/

# Run from /tmp/plainquat-sdist-final/source:
PYTHONDONTWRITEBYTECODE=1 python3 -c 'from setuptools.build_meta import build_sdist; print(build_sdist("/tmp/plainquat-sdist-final/dist"))'

# Create the isolated runtime, then supply an existing wheel build dependency
# only during the offline install:
PYTHONDONTWRITEBYTECODE=1 python3 -m venv /tmp/plainquat-sdist-final-venv

# Existing wheel build dependency supplied only for this offline install:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/site-packages /tmp/plainquat-sdist-final-venv/bin/python -m pip install --no-index --no-deps --no-cache-dir --no-build-isolation --no-compile /tmp/plainquat-sdist-final/dist/plainquat-0.1.0.tar.gz

# Extract tests, then run with PYTHONPATH unset, outside the project checkout:
tar -xzf /tmp/plainquat-sdist-final/dist/plainquat-0.1.0.tar.gz -C /tmp/plainquat-sdist-final/
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-sdist-final-venv/bin/python -m unittest discover -s /tmp/plainquat-sdist-final/plainquat-0.1.0/tests
```

The Python 3.9.6 installed-sdist test passed 42 tests with 27 NumPy/SciPy skips
out of 69; NumPy was not importable in that runtime. Scalar, NamedTuple and
dataclass conversion smoke checks included both positive-zero regressions.
The package sources, tests and documentation in the archive match the final
working source; setuptools rewrites equivalent `setup.cfg` formatting and
adds an `egg_info` section when building an sdist. The wheel's RECORD hashes,
lengths and `py.typed` marker were checked separately.

## Lint and typing

Ruff 0.17.3 and mypy 2.4.1 ran on source; `--ignore-missing-imports` is used
for the optional NumPy import in this source-only check:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-7/gallery-env/bin/ruff check --no-cache src tests experiments
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/plainquat-mypy-cache --ignore-missing-imports src/plainquat
```

The following were run from `/tmp/plainquat-conversions-typing`, containing
copies of `tests/typing/consumer.py` and `tests/typing/invalid.py`. The target
interpreter had the new wheel installed. The positive consumer passed; replacing
`consumer.py` with `invalid.py` caused each checker to reject the invalid call,
confirming it read the package's installed type information.

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/mypy --cache-dir /tmp/plainquat-conversions-mypy --python-executable /tmp/plainquat-conversions-oracle/bin/python consumer.py
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/pyright --pythonpath /tmp/plainquat-conversions-oracle/bin/python consumer.py
PYTHONDONTWRITEBYTECODE=1 /Users/rick/Documents/Codex/2026-10-29/task-3/snakemd-typing-env/bin/ty check --python /tmp/plainquat-conversions-oracle consumer.py
```

Checker versions: mypy 2.4.1, Pyright 1.1.414, ty 0.0.95. The manual frozen
slots used for dataclass values are covered on Python 3.9; pickle round-tripping
of those values is not part of the documented contract and remains unsupported.

## Limits

The quaternion rotation kernel still has the documented extreme cancellation
case in `API.md`; the experimental homogeneous kernel was not adopted. Conversion
round trips are limited by float64 representability, huge-angle trigonometric
argument reduction, and a possible one-ULP scalar/NumPy norm difference at the
pi boundary. For minimum subnormal `u`, the quaternion vector components of
`from_rotvec((5*u,5*u,0))` currently round to `3*u`; exact results round to
`2*u`, as documented in `API.md`. Git initialization was previously denied and was not retried.
