# QuatOps

QuatOps is a small quaternion library with explicit convention and representation modules. Choose a fully specified module; there is no convention flag, automatic dispatch, or required quaternion wrapper. The scalar backends use only the Python standard library. NumPy is optional, and SciPy is used only for development checks.

## Choose an API

| Representation | Hamilton | JPL |
| --- | --- | --- |
| Plain Python tuples | `quatops.python.hamilton` | `quatops.python.jpl` |
| `NamedTuple` | `quatops.types.hamilton_namedtuple` | `quatops.types.jpl_namedtuple` |
| Frozen dataclass | `quatops.types.hamilton_dataclass` | `quatops.types.jpl_dataclass` |
| NumPy, scalar-first | `quatops.numpy.hamilton.wxyz` | `quatops.numpy.jpl.wxyz` |
| NumPy, scalar-last | `quatops.numpy.hamilton.xyzw` | `quatops.numpy.jpl.xyzw` |

The convention is part of the module: Hamilton multiplication uses the positive vector cross term; JPL uses the negative one. Both rotate actively and right-handedly. In either convention, `multiply(q, r)` applies `r` first and then `q`. NumPy module paths include both convention and component order; importing `quatops.numpy` alone selects no operations.

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

## Install and verify

`python -m pip install .` installs the dependency-free scalar backends. Use `python -m pip install '.[numpy]'` for NumPy operations. `python -m pip install '.[dev]'` adds NumPy and SciPy for development checks. Run the test suite with `PYTHONPATH=src python -m unittest discover -s tests -v`.

The full signatures, storage rules, validation behavior, conversions, and numerical limits are in [the API contract](docs/API.md). [Verification details](docs/VERIFICATION.md) list the tested environments and commands. The initial [axis-angle and rotation-vector proposal](docs/PROPOSED_ROTATION_REPRESENTATIONS.md) records the choices that shaped the current API.
