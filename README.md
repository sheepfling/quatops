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

The convention is part of the module: Hamilton multiplication uses the positive vector cross term; JPL uses the negative one. Both rotate actively and right-handedly. In either convention, `multiply(lhs, rhs)` applies `rhs` first and then `lhs`. The APIs also provide component-wise `add`, `subtract`, `negate`, real `scale`, `inverse`, and right-quotient `divide(lhs, rhs) = multiply(lhs, inverse(rhs))`. NumPy module paths include both convention and component order; importing `quatops.numpy` alone selects no operations.

The NumPy modules above use a trailing component axis, such as `(..., 4)` for
quaternions. For arrays with components on axis zero, use the matching
`quatops.numpy.{hamilton,jpl}.components_first.{wxyz,xyzw}` module. The names
`wxyz` and `xyzw` describe coefficient order; `components_first` describes
array-axis placement. Equivalent convention-first aliases are available under
`quatops.{hamilton,jpl}.numpy_{wxyz,xyzw}_components_first`.

Each operation module also provides convention-neutral algebraic functions `exp`, `log`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, inverse trig and hyperbolic functions, stable `expm1`/`log1p`, logarithms to other bases, real and quaternion-valued powers, plus coefficient `dot` and `norm_squared`. Rotation tools include left/right division, rotation distance and equivalence, sign handling, matrix and two-vector conversions, Euler angles, MRP, SLERP/NLERP/SQUAD, local `boxplus`/`boxminus`, angular-velocity derivatives and sampled integration, Wahba alignment, rotation means and covariance, random attitudes, swing/twist, homogeneous transforms, and SO(3) Jacobians. For physical SO(3) conversion, use `from_rotation_vector` and `to_rotation_vector` (also available as `from_rotvec` and `to_rotvec`).

All scalar, named-field, and NumPy operation modules provide `to_rotation_matrix(q)`, `to_left_matrix(q)`, and `to_right_matrix(q)`. Rotation matrices are active 3×3 transforms; the multiplication matrices act on quaternion coefficients. The separate `to_homogeneous_matrix(q, translation)` creates a 4×4 rigid transform for points and `from_homogeneous_matrix` extracts its rotation. Trailing-axis NumPy modules return `(..., 3, 3)` or `(..., 4, 4)` matrices. Component-first modules accept `(4, ...)` quaternions and return `(3, 3, ...)` or `(4, 4, ...)` matrices.

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

For component-first arrays, select the matching explicit module, such as
`quatops.numpy.jpl.components_first.xyzw`; it accepts quaternions shaped
`(4, ...)` and vectors shaped `(3, ...)`.

## Install and verify

`python -m pip install .` installs the dependency-free scalar backends. Use `python -m pip install '.[numpy]'` for NumPy operations. For development, create and populate the repository-local environment, then run its test and static-check tools:

```sh
# Create the environment once, if .venv does not already exist.
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The `dev` extra includes NumPy, SciPy (the numerical reference dependency),
pytest, and SciPy stubs. Pytest is configured to import directly from `src`, so
no `PYTHONPATH` setting is needed. Use the repository-local `.venv` for checks
so the SciPy reference tests run in the same environment.

The full signatures, storage rules, validation behavior, conversions, and numerical limits are in [the API contract](docs/API.md). [Verification details](docs/VERIFICATION.md) list the tested environments and commands. The initial [axis-angle and rotation-vector proposal](docs/PROPOSED_ROTATION_REPRESENTATIONS.md) records the choices that shaped the current API.

To run a local experiment without setting `PYTHONPATH`, see the [experiment guide](experiments/README.md) and use `python3 experiments/run.py --list`.
