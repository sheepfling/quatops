# QuatOps

## Quaternion math, with fewer convention surprises

QuatOps brings quaternion algebra and 3D rotation tools together in a small,
typed toolkit. Its goals are to make conventions visible, keep the basic Python
path lightweight, and provide a consistent set of operations across plain
Python and NumPy representations. Choose a fully specified module: there is no
convention flag, automatic dispatch, or required quaternion wrapper. The
tuple-based Python backends use only the standard library; NumPy is optional.
SciPy is used only for development checks.

## When to use QuatOps

QuatOps selects convention and representation at import time, keeping those
choices visible in code and avoiding per-call convention dispatch or
convention conversion.

Choose a different tool when its data model better fits the job:

| Use | Best fit |
| --- | --- |
| QuatOps | Explicit Hamilton/JPL behavior, plain Python tuple support, or direct access to quaternion algebra alongside rotation tools. |
| [SciPy `Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) | A rotation-focused object with conversions among quaternions, matrices, rotation vectors, Euler angles, and other rotation representations. |
| [`numpy-quaternion`](https://quaternion.readthedocs.io/en/latest/) | A quaternion dtype integrated with NumPy arrays and NumPy-style quaternion arithmetic. |

QuatOps covers general quaternion algebra alongside rotation operations. These
tools overlap; choose based on the conventions, value model, and ecosystem
that fit your code.

**Terminology:** a *tuple backend* stores quaternion values as Python tuples;
NamedTuple, dataclass, and NumPy backends name other value representations.
The *scalar coefficient* is the real component `w`. `wxyz` (`scalar-first`)
and `xyzw` (`scalar-last`) describe where that coefficient appears, not the
value representation. The top-level `quatops.scalar` module is the historical
Hamilton tuple backend; its quaternion coefficients are stored as `wxyz`.

## Choose an API

Choose the convention to match the system that provides or consumes your
quaternions. Hamilton uses the positive vector cross term; JPL uses the
negative one. Both represent active, right-handed rotations. `wxyz` and
`xyzw` describe coefficient order; they do **not** select Hamilton or JPL.
Use a JPL module's `from_hamilton` or `to_hamilton` when converting
conventions. Reordering components alone does not convert a quaternion.

| Representation | Pick it when |
| --- | --- |
| Plain Python tuple | You want ordinary tuple values, no third-party runtime dependency, and minimal representation overhead. |
| `NamedTuple` | You want named immutable fields and tuple-style unpacking. |
| Frozen slotted dataclass | You want named immutable attributes and do not need tuple indexing. |
| NumPy array | You want vectorized operations over arrays or batches and already use NumPy. |

Use these fully specified module paths for new code:

| Representation | Hamilton | JPL |
| --- | --- | --- |
| Plain Python tuple | `quatops.python.hamilton` | `quatops.python.jpl` |
| `NamedTuple` | `quatops.types.hamilton_namedtuple` | `quatops.types.jpl_namedtuple` |
| Frozen slotted dataclass | `quatops.types.hamilton_dataclass` | `quatops.types.jpl_dataclass` |
| NumPy, `wxyz` coefficient order | `quatops.numpy.hamilton.wxyz` | `quatops.numpy.jpl.wxyz` |
| NumPy, `xyzw` coefficient order | `quatops.numpy.hamilton.xyzw` | `quatops.numpy.jpl.xyzw` |

For NumPy, choose `wxyz` or `xyzw` to match the coefficient order expected by
the surrounding code. The standard modules put the four-value quaternion
component axis last (`(..., 4)`). Choose a `components_first` module only when
your arrays need that axis at axis 0 (`(4, ...)`). For example,
`quatops.numpy.jpl.components_first.xyzw` accepts `(4, ...)` quaternions whose
four values are ordered `x, y, z, w`. Coefficient order and component-axis
placement are independent choices.

For new code, import a fully specified module and call operations through that
module. This keeps the convention visible at each call site. If the source
system does not state its convention, verify its multiplication rule; do not
guess from `wxyz` or `xyzw`:

```python
from quatops.numpy.jpl import xyzw as jpl_xyzw

q = jpl_xyzw.from_rotvec((0.0, 0.0, 1.0))
```

Plain tuples and NumPy arrays carry no convention tag. Keep operations
qualified through the selected module, and explicitly convert values before
passing them to another convention.

The package retains some alias and historical import paths, including
`quatops.scalar` and convention-first facades such as
`quatops.jpl.numpy_xyzw`. They point to the same convention-specific kernels
and add no runtime dispatch or convention conversion. Use the paths in the
table above for new code. The full storage rules, operation catalog,
validation behavior, and numerical limits are in the
[API contract](docs/API.md).

## Example

```python
from math import pi
from quatops.python import hamilton

q = hamilton.from_axis_angle((0, 0, 1, pi / 2))
rotated = hamilton.rotate(q, (1, 0, 0))
```

For NumPy arrays, import a complete layout module, for example:

```python
from quatops.numpy.jpl import xyzw

q = xyzw.from_rotvec((0, 0, pi / 2))
rotated = xyzw.rotate(q, (1, 0, 0))
```

For arrays with the coefficient axis at axis 0, use the matching
`components_first` module.

## Install and verify

`python -m pip install .` installs the dependency-free scalar backends. Use `python -m pip install '.[numpy]'` for NumPy operations. For development, create and populate the repository-local environment, then run its test and static-check tools:

```sh
# Create the environment once, if .venv does not already exist.
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/mypy \
    tests/typing_tests/foundational_consumer.py \
    tests/typing_tests/consumer.py \
    tests/typing_tests/float32_consumer.py
.venv/bin/ruff check --select F,E9 src tests experiments
```

The `dev` extra includes NumPy, numpy-quaternion, SciPy (the numerical
reference dependency), pytest, Mypy, Ruff, and SciPy stubs.
Pytest is configured to import directly from `src`, so no `PYTHONPATH` setting
is needed. Use the repository-local `.venv` so the oracle tests and static
checks run with the declared development dependencies.

For workload-specific timings, use the
[experiment suite](experiments/README.md). Treat those results as local
measurements, not general performance guarantees.

See the [documentation map](docs/README.md) for the API reference, verification
record, experiment guide, and historical research notes. The old
[axis-angle and rotation-vector proposal](docs/PROPOSED_ROTATION_REPRESENTATIONS.md)
is a historical design record; the implemented behavior is in the API contract.

To run a local experiment without setting `PYTHONPATH`, see the
[experiment guide](experiments/README.md) and use
`.venv/bin/python experiments/run.py --list`.
