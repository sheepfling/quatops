# Shared API contract

This is the **implemented, provisional** contract for explicit Hamilton and JPL
modules. Choose a convention and representation at import time; there is no
common runtime quaternion wrapper, convention flag, or automatic dispatch. The
distribution/import root is `quatops`; this unpublished candidate was previously
named `plainquat`, and no `plainquat` import shim is provided. For an
introduction, see the [README](../README.md).

## Convention, storage, and rotation

| | Hamilton | JPL |
| --- | --- | --- |
| Module family | `quatops.hamilton` | `quatops.jpl` |
| Scalar / namedtuple / dataclass storage | `(w, x, y, z)` | `(x, y, z, w)` |
| NumPy storage | `numpy_wxyz` or `numpy_xyzw` | `numpy_wxyz` or `numpy_xyzw` |
| Identity in `wxyz` order | `(1, 0, 0, 0)` | `(1, 0, 0, 0)` |
| Identity in `xyzw` order | `(0, 0, 0, 1)` | `(0, 0, 0, 1)` |
| Product vector part | `aw*bv + bw*av + av × bv` | `aw*bv + bw*av - av × bv` |
| Product scalar part | `aw*bw - av · bv` | `aw*bw - av · bv` |
| Active action | Vector part of `q * (0, v) * inverse(q)` | Vector part of `q ⊙ (v, 0) ⊙ inverse(q)` |
| Composition order | `multiply(q, r)` means `q * r`; its active rotation applies `r` first, then `q` | `multiply(q, r)` means JPL `q ⊙ r`; its active rotation applies `r` first, then `q` |

Here `av` and `bv` are the three vector components and `aw` and `bw` their
scalar components. In the action table, `(0, v)` and `(v, 0)` denote a pure
vector quaternion in scalar-first and scalar-last notation respectively. For
a given NumPy module its stored form is `(0, vx, vy, vz)` in `wxyz` or
`(vx, vy, vz, 0)` in `xyzw`. Both actions are active and right-handed. A passive
coordinate transform is the inverse action, obtained using `inverse(q)`;
`-q` and `q` represent the same rotation. The top-level `quatops.scalar`,
`quatops.namedtuple`, and `quatops.dataclass` modules remain Hamilton-compatible
imports. `quatops.numpy` is only a namespace; choose convention and order
submodules before using NumPy operations.

The same physical rotation is represented by Hamilton `h=(w, x, y, z)` in
`wxyz` order and JPL `j=(w, -x, -y, -z)` in `wxyz` order or
`j=(-x, -y, -z, w)` in `xyzw` order. Each JPL module supplies
`from_hamilton(h)` and `to_hamilton(j)` for this explicit conversion. The
Hamilton input to `from_hamilton` and output from `to_hamilton` always use
`wxyz`, regardless of the JPL module's storage order. These functions preserve
magnitude and do not normalize. The JPL `to_hamilton` result is a plain wxyz
tuple for scalar representations or a float64 array for NumPy; it is not a
JPL `Quaternion` with misleading component labels.

Hamilton/JPL labels alone do not determine storage or action. For example,
JPL's SPICE toolkit itself uses a Hamilton product; its documentation explains
the related engineering convention and sign mapping. [SPICE quaternion
conventions](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html).
For SciPy's Hamilton scalar-last quaternion input, first express a JPL value
in `xyzw` order and then convert it as `(-jx, -jy, -jz, jw)`. Merely
reordering JPL components does not make them a physically equivalent SciPy
quaternion.

## Representations and construction

Backend-first names are direct aliases to the native convention-specific
kernels, so importing a different name does not add per-call dispatch or
conversion. Convention-first `quatops.hamilton` and `quatops.jpl` modules
remain available.

| Representation | Hamilton | JPL |
| --- | --- | --- |
| Python tuples | `quatops.python.hamilton` | `quatops.python.jpl` |
| NamedTuple | `quatops.types.hamilton_namedtuple` | `quatops.types.jpl_namedtuple` |
| Frozen slotted dataclass | `quatops.types.hamilton_dataclass` | `quatops.types.jpl_dataclass` |
| NumPy `wxyz` | `quatops.numpy.hamilton.wxyz` | `quatops.numpy.jpl.wxyz` |
| NumPy `xyzw` | `quatops.numpy.hamilton.xyzw` | `quatops.numpy.jpl.xyzw` |

`quatops.python` and `quatops.numpy` are namespaces without default operations.
Import `quatops.python.hamilton` or `quatops.python.jpl` to choose a scalar
convention. `quatops.types.HamiltonQuaternion` and
`quatops.types.JPLQuaternion` alias the respective frozen dataclass classes;
`quatops.types` has no bare default `Quaternion`.

The convention-first NumPy paths `quatops.hamilton.numpy_wxyz`,
`quatops.hamilton.numpy_xyzw`, `quatops.jpl.numpy_wxyz`, and
`quatops.jpl.numpy_xyzw` re-export the corresponding fully specified native
functions. NumPy `multiply` and `rotate` do not select a convention or layout at
runtime and do not convert inputs on each call.

Tuple, NamedTuple, and dataclass modules have one native layout per convention:
Hamilton `wxyz` and JPL `xyzw`. Their layout helpers perform one-time reorders
but do not provide alternate native hot paths. Scalar operations accept a
four-component iterable and return plain tuples; NamedTuple and dataclass
modules return their own immutable `Quaternion` type for quaternion-valued
operations, with fields in that convention's order. The frozen slotted
dataclass is iterable but not a tuple or indexable. Constructors store values
without validating, coercing, or normalizing; functions validate inputs. NumPy
accepts real numeric arrays with final dimension 4 and returns float64 arrays;
vector inputs have final dimension 3.

The layout functions perform no convention conversion or implicit
normalization:

| Function | Input order | Result |
| --- | --- | --- |
| `from_wxyz(q)` | `(w, x, y, z)` | Quaternion in the selected module's native storage order |
| `to_wxyz(q)` | Selected module's native order | Components in `(w, x, y, z)` order |
| `from_xyzw(q)` | `(x, y, z, w)` | Quaternion in the selected module's native storage order |
| `to_xyzw(q)` | Selected module's native order | Components in `(x, y, z, w)` order |

In particular, `from_wxyz`/`to_wxyz` are validating layout identities in
`wxyz` modules, and `from_xyzw`/`to_xyzw` are validating layout identities in
`xyzw` modules. `to_wxyz` and `to_xyzw` return plain tuples in the scalar families and arrays in NumPy, so
labels cannot imply the wrong field order. Reordering does not convert Hamilton
to JPL; use the explicit conversion functions above for that.

## Shared operations

`q` and `r` are values in the selected convention and `v` is a three-component
vector. The two multiplication symbols have the different algebra specified
above; their names and composition order are shared.

| Signature | Meaning | Result |
| --- | --- | --- |
| `conjugate(q)` | Negate the vector components | Quaternion |
| `multiply(q, r)` | Product in the selected convention | Quaternion |
| `norm(q)` | Euclidean norm | Real scalar or array of scalars |
| `normalize(q)` | Divide by the norm; `q` must be nonzero | Unit quaternion |
| `inverse(q)` | Multiplicative inverse; `q` must be nonzero | Quaternion |
| `rotate(q, v)` | Active quaternion action; `q` must be nonzero | Three-component vector |
| `from_axis_angle(a)` | Convert `(axis_x, axis_y, axis_z, radians)` to a canonical unit quaternion | Quaternion |
| `to_axis_angle(q)` | Convert a nonzero quaternion to canonical unit axis and principal angle | AxisAngle |
| `from_rotvec(r)` | Convert axis-times-angle radians to a canonical unit quaternion | Quaternion |
| `to_rotvec(q)` | Convert a nonzero quaternion to a principal rotation vector | RotVec |

For NumPy, binary operations broadcast leading dimensions according to NumPy
rules. Thus `(2, 1, 4)` multiplied by `(1, 3, 4)` returns `(2, 3, 4)`; rotating
`(2, 1, 4)` quaternions by `(1, 3, 3)` vectors returns `(2, 3, 3)`. `norm`
removes the final axis and yields a NumPy scalar for shape `(4,)`. No output
memory layout is guaranteed. Array-like inputs must infer a boolean, signed or
unsigned integer, or real floating dtype. Complex, object (including
`Fraction` or `Decimal`), string, and other dtypes are rejected even when their
values look real. Values are cast to float64, which can round or underflow;
conversion to a non-finite float64 raises `ValueError`.

## Axis-angle and rotation-vector conversions

These are provisional candidate policies. AxisAngle is flat `(ax, ay, az,
angle)` in radians; RotVec is `(x, y, z)` with direction equal to the axis and
length equal to the angle in radians. The scalar backend returns plain tuples;
`namedtuple` and `dataclass` expose their own `AxisAngle` and `RotVec` classes;
NumPy returns float64 arrays of shape `(..., 4)` and `(..., 3)`. Constructors
store their arguments as supplied; operations validate them. Each conversion is
unary and preserves NumPy leading batch dimensions, including empty batches;
there is no independent axis/angle broadcasting in the flat layout.

`from_axis_angle` explicitly normalizes the supplied finite nonzero axis, even
at zero angle, and rejects a zero axis. A zero RotVec is valid and means identity.
Quaternion-to-representation conversion explicitly extracts a rotation from a
finite nonzero quaternion without requiring unit length; it does not preserve
the input's magnitude or the sign distinction between `q` and `-q`. Conversion
to quaternion returns a canonical unit quaternion. These are conversion
semantics, not implicit normalization of quaternion arithmetic.

Output axis-angle and rotation-vector angles use a principal range `[0, pi]`.
Identity uses axis +X and zero angle or the zero RotVec. Canonical quaternion
sign has `w > 0`; at `w == 0`, the first nonzero x/y/z component is positive
for Hamilton and negative for JPL, giving the same canonical physical axis.
Signed zeros in canonical outputs become positive zero. This half-turn tie is exact, without a tolerance. `from_axis_angle`
treats input `+math.pi` or `-math.pi` as an exact half-turn; `from_rotvec` does
so when its computed float64 norm equals `math.pi`. Adjacent angle values are
not snapped. Scalar and NumPy vector norm calculations can round a near-`pi`
norm differently by an ulp, so their raw quaternion components can differ at
this boundary even when the represented rotations agree within rounding.
`to_axis_angle` can return an angle rounded to `pi` even when the original
nonzero scalar part means its axis need not satisfy the exact-half-turn sign tie. At extreme axis component
ratios, the first nonzero component used for the exact-half-turn tie can also
underflow in the returned axis; sign selection still uses the original input.
A round trip preserves the rotation only to the extent that the intermediate
representation can express it, not the raw quaternion components.

All finite input angles are accepted. Outside the exact `pi` rule, sine and
cosine are evaluated on the half-angle without explicit modulo reduction; for
very large arguments the precision of platform/NumPy trigonometric functions
and their argument reduction is not guaranteed across platforms. If a RotVec
norm cannot be represented as a float, `from_rotvec` raises `OverflowError`.
No arbitrary small-angle threshold snaps a representable rotation to identity.
Nevertheless, finite precision has limits: the half-angle of a smallest
subnormal RotVec cannot be represented in a quaternion and may round to zero;
a quaternion whose rotation angle itself rounds to zero converts to identity.
Canonical quaternion input sign is chosen before scaling. Tiny or extreme
ratios can still lose information, so componentwise precision is not guaranteed.
For example, if `u` is the smallest positive binary64 subnormal, the current
`from_rotvec((5*u, 5*u, 0))` produces quaternion vector component magnitudes
`3*u`; the exact magnitudes are just below `2.5*u` and round to `2*u`. This is
a documented double-rounding limit, not a correctly-rounded conversion guarantee.

## Conventions and errors

- Hamilton and JPL quaternion storage and products differ as defined above.
  Both rotation actions are active and right-handed, and `multiply(q, r)`
  composes rotations so that `r` acts first and `q` second.
- Quaternion algebra has no implicit normalization. In particular, `rotate`
  uses the inverse and accepts any finite nonzero quaternion: nonzero scalar multiples represent
  the same rotation. Internal scaling limits avoidable overflow without
  modifying the input or its represented rotation.
- Operations reject malformed component shapes or counts, non-real or
  non-finite inputs, incompatible broadcast dimensions, and zero quaternions
  in `normalize`, `inverse`, and `rotate` with `ValueError`. One invalid entry
  rejects a whole NumPy call. `norm` of zero is zero.
- Arithmetic that produces a non-finite result raises `OverflowError`. Norm,
  normalization, and inverse use scaled calculations to reduce avoidable
  overflow or underflow; finite underflow and rounding follow floating-point
  arithmetic. Rotation scales the quaternion and forms each output component
  separately, combining opposite-signed terms to reduce avoidable overflow.
  Coefficients and individual products are rounded to float64; extreme
  quaternion ratios and tiny products can underflow, and cancellation can lose
  relative accuracy. In particular, Hamilton rotation of vector
  `(1e308, 1, 1)` by `(5, 12, 13, 0)` currently returns a first component
  near `1.11e292` instead of `17/13`; the corresponding JPL quaternion is
  `(-12, -13, 0, 5)` in `xyzw` or `(5, -12, -13, 0)` in `wxyz`. See [the precision investigation](ROTATION_NUMERICS.md)
  for a second example and measured alternatives. Other operations can overflow
  in intermediate calculations even when their exact result is representable.
  NumPy suppresses floating-point warnings locally, checks results, and preserves
  the caller's `numpy.seterr` configuration. Inputs are not mutated.

## Possible additions, not implemented

Operator overloads are not implemented. These are candidates for a later
decision, not part of the current API:

- **Rotation matrix conversion** could ease interoperability; it needs explicit
  matrix shape, active/passive, handedness, and invalid-matrix policies.
- **Interpolation** such as spherical interpolation could support animation; it
  needs decisions about shortest-path sign choice and endpoint behavior.

The current matrix is deliberately limited to the operations implemented and
covered by conformance tests. It does not claim to be a complete quaternion API.
