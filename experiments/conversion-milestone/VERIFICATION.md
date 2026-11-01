# Verification of the local conversion milestone

This report covers the provisional local candidate, not a release. The previous
wheel is preserved at `experiments/prior-wheel/plainquat-0.1.0-py3-none-any.whl`.
The current wheel was built without network access or dependency installation:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir --wheel-dir /tmp/plainquat-wheel-conversions .
```

It contains `plainquat/py.typed`; SciPy is a development extra, not a runtime
requirement. SHA-256 of `dist/plainquat-0.1.0-py3-none-any.whl`:
`caa0d474ed878992421751a43c751850731014b0016e43629d48634d9e7b5bcd`.

## Installed-wheel suites

Each test command below was run from `/tmp`, outside the source checkout. Each
virtual environment received the wheel using `pip install --no-index --no-deps
--no-cache-dir --no-compile`. The 3.9 array environment uses existing system
NumPy; the 3.12 oracle run adds the existing approved development environment's
site-packages to `PYTHONPATH`. No dependency was installed for these checks.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-scalar/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 /tmp/plainquat-conversions-array/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/rick/Documents/Codex/2026-10-29/task-7/quaternion-env/lib/python3.12/site-packages /tmp/plainquat-conversions-oracle/bin/python -m unittest discover -s /Users/rick/Documents/Codex/2026-10-29/task-8/tests
```

- CPython 3.9.6, scalar-only environment: 40 passed, 26 NumPy/SciPy-dependent
  skips out of 66. Confirmed NumPy is not importable there.
- CPython 3.9.6, NumPy 2.0.2: 60 passed, 6 skipped out of 66. SciPy is not
  available there; the longdouble extended-range test also skips on this host.
- CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0: 65 passed, 1 skipped out of 66.
  The skip is the extended-range test because longdouble has no larger range
  than float64 on this platform. SciPy conversion and rotation references ran.

The suite separates exact Fraction/Decimal edge checks from SciPy rotation
comparisons. It covers norm return type; the NumPy coefficient-order correction;
identity, quarter turns, exact and adjacent pi, signed zeros, subnormal and large
inputs; invalid values, shapes, zero axes and quaternions, and norm overflow;
array views, empty batches, non-mutation and NumPy error-state preservation.

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
pi boundary. Git initialization was previously denied and was not retried.
