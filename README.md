# QuatOps

## Reuse verified quaternion math with conventions chosen up front

QuatOps exists so applications and coding agents can use quaternion math
without reimplementing it just to keep a lightweight, plain-Python path. The
canonical `quatops.python.hamilton` and `quatops.python.jpl` backends operate on
ordinary four-tuples: no quaternion wrapper class, no third-party runtime
dependency, and no per-call convention dispatch. Their public operations are
covered by seeded tests against independent references (including
`numpy-quaternion` and SciPy) and separately derived invariants; see the
[oracle coverage](docs/ORACLE_COVERAGE.md).

If an operation you need is missing, add it to QuatOps with oracle coverage
rather than copying an unverified quaternion kernel into an application.

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

The package retains alternative import paths for compatibility. The
[API contract](docs/API.md#alternative-import-paths) lists them; they do not
auto-detect or convert a value's convention. Use the canonical paths in the
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

## Install

`python -m pip install .` installs the dependency-free scalar backends. Use
`python -m pip install '.[numpy]'` for NumPy operations.

Contributor setup, verification, performance timing, and research experiments
are collected in the [documentation map](docs/README.md).
