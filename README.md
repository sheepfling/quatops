# QuatOps

A small, provisional quaternion library with explicit convention and
representation modules. Choose the module that matches your data: there is no
convention flag or automatic backend dispatch. The scalar representations have
no third-party runtime dependencies. This is a local candidate for review, not
a published package or a performance claim.

This unpublished candidate was previously named `plainquat`. Its prior source
and distribution artifacts are preserved in `experiments/plainquat-pre-rename/`.
The distribution and import root are now `quatops`; an old `plainquat` import
is not provided by this package.

## Choose a module

QuatOps exposes backend-first imports alongside the convention-first modules.
Choose a convention and, for arrays, a component order explicitly. These names
refer directly to the same native functions; there is no per-call convention or
layout selection, conversion wrapper, or compulsory quaternion object.

| Representation | Backend-first imports | Native quaternion layout |
| --- | --- | --- |
| Python tuples | `quatops.python.hamilton`, `quatops.python.jpl` | Hamilton `wxyz`; JPL `xyzw` |
| Named tuples | `quatops.types.hamilton_namedtuple`, `quatops.types.jpl_namedtuple` | Hamilton `wxyz`; JPL `xyzw` |
| Frozen slotted dataclasses | `quatops.types.hamilton_dataclass`, `quatops.types.jpl_dataclass` | Hamilton `wxyz`; JPL `xyzw` |
| NumPy arrays | The four modules in the table below | Explicitly selected |

`quatops.python` itself directly exposes the Hamilton `wxyz` tuple backend,
and `quatops.numpy` directly exposes the Hamilton `wxyz` array backend.
`quatops.types.HamiltonQuaternion` and `quatops.types.JPLQuaternion` name the
corresponding frozen dataclass types; there is no bare default `Quaternion` in
`quatops.types`.

| Convention | Scalar-first `wxyz` | Scalar-last `xyzw` |
| --- | --- | --- |
| Hamilton (+ cross) | `quatops.numpy.hamilton_wxyz` | `quatops.numpy.hamilton_xyzw` |
| JPL (− cross) | `quatops.numpy.jpl_wxyz` | `quatops.numpy.jpl_xyzw` |

The existing convention-first paths `quatops.hamilton` and `quatops.jpl` remain
available, including all four `numpy_wxyz`/`numpy_xyzw` kernels. The top-level
`quatops.scalar`, `quatops.namedtuple`, and `quatops.dataclass` modules remain
Hamilton; `quatops.hamilton.numpy` is Hamilton `wxyz` and `quatops.jpl.numpy`
is JPL `xyzw`.

The tuple, NamedTuple, and dataclass operations each have one native layout per
convention: Hamilton `wxyz` and JPL `xyzw`. Their layout helpers explicitly
reorder values at a boundary, but do not add alternate native hot paths. Axis
angle and rotation-vector values have the same physical layouts in both
conventions. A component layout alone does not identify the multiplication
algebra or meaning of a rotation.

From the project directory, `python -m pip install .` installs the scalar
representations. Use `python -m pip install '.[numpy]'` for array modules. NumPy
is imported only when a NumPy backend is imported. Python 3.9 or newer is
required.

## Conventions

| Property | Hamilton modules | JPL modules |
| --- | --- | --- |
| Native scalar, NamedTuple, dataclass storage | `(w, x, y, z)` | `(x, y, z, w)` |
| NumPy storage | Explicit `wxyz` or `xyzw` module | Explicit `wxyz` or `xyzw` module |
| Product vector term | `aw*bv + bw*av + av × bv` | `aw*bv + bw*av - av × bv` |
| Product scalar term | `aw*bw - av · bv` | `aw*bw - av · bv` |
| Rotation action | Active, right-handed `q * (0, v) * inverse(q)` | Active, right-handed `q ⊙ (v, 0) ⊙ inverse(q)` |
| Composition | `multiply(q, r)` applies `r`, then `q` | `multiply(q, r)` applies `r`, then `q` |

Here `av` and `bv` are the vector components and `aw` and `bw` their scalar
components. For a passive coordinate transformation, apply `inverse(q)` to
the vector. Negating every component produces the *same* rotation; it is not
the same operation as taking the inverse.

For the same physical rotation, Hamilton `(w, x, y, z)` corresponds to JPL
`(w, -x, -y, -z)` in `wxyz` storage or `(-x, -y, -z, w)` in `xyzw` storage.
Use a JPL module's `from_hamilton` and `to_hamilton` for this convention
conversion: regardless of the JPL module's storage order, the Hamilton input
to `from_hamilton` and output from `to_hamilton` are `wxyz`. The layout
helpers `from_wxyz`, `to_wxyz`,
`from_xyzw`, and `to_xyzw` **only reorder components** within the chosen
convention; they do not negate the vector or change algebra. This distinction
also matters when comparing to SciPy, whose rotation quaternions are
Hamilton: after expressing a JPL value as `xyzw`, use SciPy scalar-last
values `(-jx, -jy, -jz, jw)`.

Convention names are not universal: JPL's own SPICE toolkit uses Hamilton
quaternions. See the [SPICE quaternion convention documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html).
The table above defines what these modules mean.

```python
from math import pi
from quatops.python import hamilton as h
from quatops.python import jpl as j

qh = h.from_axis_angle((0, 0, 1, pi / 2))  # wxyz, +z vector part
qj = j.from_hamilton(qh)                    # xyzw, -z vector part
assert j.to_hamilton(qj) == qh
vh = h.rotate(qh, (1, 0, 0))               # approximately (0, 1, 0)
vj = j.rotate(qj, (1, 0, 0))               # approximately (0, 1, 0)
axis_angle = j.to_axis_angle(qj)           # approximately (0, 0, 1, pi/2)
rotvec = j.to_rotvec(qj)                   # approximately (0, 0, pi/2)

# Named values and arrays are chosen explicitly within a convention.
from quatops.types import jpl_namedtuple as named
from quatops.numpy import hamilton_xyzw as hxy
from quatops.numpy import jpl_wxyz as jw
q = named.Quaternion(0, 0, 0, 1)
h_xyzw = hxy.from_wxyz(qh)                 # layout change only
j_wxyz = jw.from_hamilton(qh)              # layout and convention change
rotated = jw.rotate([j_wxyz], [1, 0, 0])   # shape (1, 3)
```

## API and numerical behavior

See [API.md](API.md) for operation signatures, validation, conversion and
numerical details. Both conventions offer conjugation, multiplication, norm,
normalization, inverse, rotation, explicit layout conversion, and axis-angle
and rotation-vector conversions. The JPL modules also offer explicit
Hamilton/JPL conversion. Axis-angle is `(ax, ay, az, angle)` in radians;
rotation vectors are axis times angle. These describe the same physical
rotations in both conventions. Quaternion algebra does not normalize inputs;
rotation-representation conversions explicitly normalize as documented.

Inputs must be finite and correctly shaped. Zero quaternions cannot be
normalized, inverted, or used to rotate; zero axes cannot be used in
axis-angle conversion. NumPy validates batches together, broadcasts leading
dimensions, uses float64, preserves input arrays and the caller's
`numpy.seterr` settings, and supports empty batches. Object and complex arrays
are rejected. Arithmetic yielding a non-finite result raises `OverflowError`;
finite underflow and rounding can lose information. Existing extreme
componentwise cancellation and subnormal conversion limits remain documented
in [API.md](API.md) and the [precision investigation](experiments/ROTATION_NUMERICS.md).

The test suite uses SciPy only as a development oracle. Install
`python -m pip install '.[dev]'` for optional test dependencies and run
`PYTHONPATH=src python -m unittest discover -s tests -v`. Operator overloads,
matrix or Euler-angle conversion, interpolation such as SLERP, angular
velocities, compiled acceleration, and performance guarantees are outside
this initial scope.
