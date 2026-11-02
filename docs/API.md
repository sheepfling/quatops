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
| Composition order | `multiply(lhs, rhs)` means `lhs * rhs`; its active rotation applies `rhs` first, then `lhs` | `multiply(lhs, rhs)` means JPL `lhs ⊙ rhs`; its active rotation applies `rhs` first, then `lhs` |

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

The explicit Hamilton scalar module exports the same function objects as
`quatops.scalar`. JPL scalar operations use native xyzw formulas or direct
aliases for convention-neutral kernels; operation calls do not convert
quaternions through `from_hamilton`/`to_hamilton`. Dot, squared norm, skew,
SO(3) Jacobians, coefficient sign alignment, and sign continuity are direct
function aliases. Input validation and construction of the selected
NamedTuple or dataclass result remain part of each public call.

Run `.venv/bin/python experiments/run.py convention-benchmark` to compare
matched Hamilton/JPL calls across tuple, NamedTuple, dataclass, and NumPy
representations on the current machine.

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

### Python scalar input types

Scalar operation functions accept finite values implementing Python's
`SupportsFloat` protocol. This includes built-in numbers and NumPy floating
scalars such as `numpy.float32`; the scalar modules do not import or require
NumPy. Inputs are converted to built-in `float` values before calculation, and
scalar results remain built-in floats. The component-input annotations use a
`TypeVar` bounded by `SupportsFloat`, while output types stay explicit because
these APIs intentionally do not preserve input precision or type. Complex
values, strings, and values that cannot convert to a finite float are rejected.

The convention-first NumPy paths `quatops.hamilton.numpy_wxyz`,
`quatops.hamilton.numpy_xyzw`, `quatops.jpl.numpy_wxyz`, and
`quatops.jpl.numpy_xyzw` re-export the corresponding fully specified native
functions. NumPy operations do not select a convention or layout at runtime and
do not convert inputs on each call.

The standard NumPy modules store components on the last axis: quaternions have
shape `(..., 4)` and vectors have shape `(..., 3)`. Component-first adapters
are provided at `quatops.numpy.hamilton.components_first.wxyz`,
`quatops.numpy.hamilton.components_first.xyzw`,
`quatops.numpy.jpl.components_first.wxyz`, and
`quatops.numpy.jpl.components_first.xyzw`. They accept quaternions shaped
`(4, ...)` and vectors shaped `(3, ...)`; the `wxyz` or `xyzw` suffix still
selects coefficient order. Matching convention-first aliases end in
`numpy_wxyz_components_first` or `numpy_xyzw_components_first` under
`quatops.hamilton` and `quatops.jpl`.

Component-first operations follow the same validation and batch-broadcasting
rules as their trailing-axis counterparts. Quaternion, axis-angle, and
three-vector outputs put the component axis first. Matrix outputs put both
matrix axes first, producing
`(3, 3, ...)` for rotation/Jacobian matrices and `(4, 4, ...)` for quaternion
multiplication matrices. Scalar reductions such as `norm` return only the
broadcast batch shape. Thus a `(4, 4)` array is interpreted according to the
selected module: four component-first quaternions in a batch, or four
component-last quaternions in a batch. The adapter moves axes around the
existing NumPy kernels and does not duplicate the mathematical operations.
`from_rotation_matrix` accepts `(..., 3, 3)` input on trailing-axis modules and
`(3, 3, ...)` input on component-first modules.

Tuple, NamedTuple, and dataclass modules have one native layout per convention:
Hamilton `wxyz` and JPL `xyzw`. Their layout helpers perform one-time reorders
but do not provide alternate native hot paths. Scalar operations accept a
four-component iterable and return plain tuples; NamedTuple and dataclass
modules return their own immutable `Quaternion` type for quaternion-valued
operations, with fields in that convention's order. The frozen slotted
dataclass is iterable but not a tuple or indexable. Constructors store values
without validating, coercing, or normalizing; functions validate inputs. NumPy
accepts real numeric arrays and returns float64 arrays. The standard NumPy
modules require final dimensions 4 for quaternions and 3 for vectors; the
explicit `components_first` modules require first dimensions 4 and 3.

Typed quaternion equality is nominal: a Hamilton value compares equal only to
an instance of that same concrete Hamilton class, and likewise for JPL. This
also applies to the NamedTuple representation; it does not compare equal to a
plain tuple. Use `tuple(q)` when sequence comparison is intended. Plain tuple
scalar APIs carry no runtime convention tag, so their convention remains the
responsibility of the selected module.

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

`lhs` and `rhs` are values in the selected convention and `v` is a
three-component vector. The two multiplication symbols have the different
algebra specified above; their names and composition order are shared.

| Signature | Meaning | Result |
| --- | --- | --- |
| `add(lhs, rhs)` | Add corresponding components | Quaternion |
| `subtract(lhs, rhs)` | Subtract corresponding components | Quaternion |
| `negate(q)` | Negate every component | Quaternion |
| `scale(q, factor)` | Multiply each component by one finite real factor | Quaternion |
| `conjugate(q)` | Negate the vector components | Quaternion |
| `multiply(lhs, rhs)` | Product in the selected convention | Quaternion |
| `divide(lhs, rhs)` | Right quotient, `lhs * inverse(rhs)`; `rhs` must be nonzero | Quaternion |
| `right_divide(lhs, rhs)` | Explicit alias for `lhs * inverse(rhs)` | Quaternion |
| `left_divide(lhs, rhs)` | Left quotient, `inverse(lhs) * rhs`; `lhs` must be nonzero | Quaternion |
| `dot(lhs, rhs)` | Euclidean coefficient inner product | Real scalar or array |
| `norm(q)` | Euclidean norm | Real scalar or array of scalars |
| `norm_squared(q)` | Squared Euclidean coefficient norm | Real scalar or array of scalars |
| `normalize(q)` | Divide by the norm; `q` must be nonzero | Unit quaternion |
| `inverse(q)` | Multiplicative inverse; `q` must be nonzero | Quaternion |
| `rotate(q, v)` | Active quaternion action; `q` must be nonzero | Three-component vector |
| `to_rotation_matrix(q)` | Norm-aware active 3×3 rotation matrix; `q` must be nonzero | Matrix |
| `to_left_matrix(q)` | Matrix `L(q)` such that `L(q) r = multiply(q, r)` | Matrix |
| `to_right_matrix(q)` | Matrix `P(q)` such that `P(q) r = multiply(r, q)` | Matrix |
| `exp(q)` | Algebraic quaternion exponential | Quaternion |
| `log(q)` | Principal algebraic quaternion logarithm; zero is undefined | Quaternion |
| `sin(q)` | Algebraic quaternion sine | Quaternion |
| `cos(q)` | Algebraic quaternion cosine | Quaternion |
| `tan(q)` | Right quotient `sin(q) / cos(q)` | Quaternion |
| `sinh(q)` | Algebraic quaternion hyperbolic sine | Quaternion |
| `cosh(q)` | Algebraic quaternion hyperbolic cosine | Quaternion |
| `tanh(q)` | Right quotient `sinh(q) / cosh(q)` | Quaternion |
| `power(q, exponent)` | Integer or real power using the principal logarithm for non-integers | Quaternion |
| `pow_real(q, exponent)` | Alias for the principal real power | Quaternion |
| `sqrt(q)` | Principal square root | Quaternion |
| `slerp(q0, q1, t, shortest=True)` | Spherical interpolation between unit rotations; supports extrapolation | Quaternion |
| `rotation_distance(q0, q1)` | Shortest SO(3) geodesic distance in radians | Real scalar or array |
| `rotations_equivalent(q0, q1, atol=1e-12)` | Compare rotations using an angular tolerance in radians | Boolean or boolean array |
| `canonicalize(q)` | Choose a deterministic sign; preserve magnitude | Quaternion |
| `from_rotation_matrix(matrix, atol=1e-8)` | Validate and convert an active proper 3×3 rotation matrix | Unit quaternion |
| `from_two_vectors(source, target)` | Shortest active rotation mapping one nonzero vector to another | Unit quaternion |
| `skew_matrix(v)` | 3×3 cross-product matrix `[v]×` | Matrix |
| `so3_left_jacobian(r)` / `so3_right_jacobian(r)` | Left/right SO(3) exponential-map Jacobians | Matrix |
| `so3_left_jacobian_inverse(r)` / `so3_right_jacobian_inverse(r)` | Inverse left/right SO(3) Jacobians | Matrix |
| `integrate_angular_velocity(q, omega, dt, frame='body')` | Exact constant-rate rotation increment in the selected frame | Unit quaternion |
| `from_axis_angle(a)` | Convert `(axis_x, axis_y, axis_z, radians)` to a canonical unit quaternion | Quaternion |
| `to_axis_angle(q)` | Convert a nonzero quaternion to canonical unit axis and principal angle | AxisAngle |
| `from_rotvec(r)` | Convert axis-times-angle radians to a canonical unit quaternion | Quaternion |
| `to_rotvec(q)` | Convert a nonzero quaternion to a principal rotation vector | RotVec |
| `from_rotation_vector(r)` | Descriptive alias for `from_rotvec(r)` | Quaternion |
| `to_rotation_vector(q)` | Descriptive alias for `to_rotvec(q)` | RotVec |

`divide(lhs, rhs)` remains the right quotient for compatibility and is the
same operation as `right_divide`. `left_divide` uses the inverse of its first
argument. `dot` is the ordinary coefficient-space dot product; it is not a
rotation distance. `norm_squared(q)` is exactly `dot(q, q)` subject to floating
point overflow limits.

`rotation_distance` normalizes both nonzero inputs and returns an angle in
`[0, pi]`; it treats `q` and `-q` as the same attitude. `rotations_equivalent`
uses an absolute angular tolerance in radians. `canonicalize` preserves the
input magnitude and changes only its sign: it makes the scalar coefficient
positive, and for an exact zero scalar it makes the first nonzero vector
coefficient positive in Hamilton or negative in JPL. Zero is allowed and
remains zero.

`from_rotation_matrix` accepts active right-handed matrices only. It requires
`0 <= atol < 1`, checks orthogonality and determinant `+1` against that absolute
tolerance, then returns a
canonical unit quaternion. `from_two_vectors(source, target)` normalizes both
nonzero vectors and returns the shortest active alignment. For exactly
antiparallel vectors, it chooses a deterministic half-turn axis by crossing
the source with its least-aligned coordinate axis.

### Algebraic functions

For `q = (w, x, y, z)`, let `v = (x, y, z)` and `r = ||v||`. The
algebraic functions use component formulas that do not depend on the
Hamilton/JPL cross-product sign:

| Function | Formula |
| --- | --- |
| `exp(q)` | `e^w (cos(r), v sin(r)/r)` |
| `log(q)` | `(ln(||q||), (v/r) atan2(r, w))` |
| `sin(q)` | `(sin(w) cosh(r), v cos(w) sinh(r)/r)` |
| `cos(q)` | `(cos(w) cosh(r), -v sin(w) sinh(r)/r)` |
| `sinh(q)` | `(sinh(w) cos(r), v cosh(w) sin(r)/r)` |
| `cosh(q)` | `(cosh(w) cos(r), v sinh(w) sin(r)/r)` |

At `r = 0`, the ratios `sin(r)/r` and `sinh(r)/r` use their continuous
value `1`. The logarithm rejects the zero quaternion. A negative real
quaternion has no unique vector direction for its logarithm; this API chooses
the positive X axis, so `log((-a, 0, 0, 0)) = (ln(a), pi, 0, 0)` for `a > 0`.
`power(q, n)` uses exponentiation by squaring when `n` is an integer. Other
real exponents are computed as `exp(n * log(q))`; `sqrt(q)` is `power(q, 0.5)`.
For zero, exponent zero returns identity, a positive exponent returns zero, and
a negative exponent raises `ValueError`. A non-integer power of a negative
real follows the positive-X logarithm branch. NumPy operations preserve
leading batch dimensions. Algebraic operations use the input coefficient
branch in both Hamilton and JPL modules; rotation-convention conversion is
reserved for physical rotation operations.

### Hyperbolic functions

The quaternion hyperbolic sine and cosine use the formulas in the table above.
`tanh(q)` is computed as `sinh(q) / cosh(q)` with the library's right-quotient
definition; it raises `ValueError` when `cosh(q)` is zero.

### Inverse functions and stable logarithms

`asin`, `acos`, `atan`, `asinh`, `acosh`, and `atanh` apply the principal
complex function to the slice `w + r*i`, then map its imaginary part back onto
the input vector direction. This is independent of the Hamilton/JPL product
sign. For a real input outside the real function's domain, the selected slice
uses +X as its imaginary direction. The usual complex branch cuts apply;
singularities whose result is not finite raise `OverflowError`.

`expm1(q)` computes `exp(q) - 1` with a power series near zero, preserving
small results that direct subtraction would round away. `log1p(q)` similarly
computes `log(1 + q)` near zero. Outside their series regions they use the
corresponding algebraic functions directly. `log2(q)` and `log10(q)` scale the
principal natural logarithm by `1/log(2)` or `1/log(10)`.

`power_quaternion(q, exponent)` is explicitly defined as
`exp(exponent * log(q))`, with the exponent on the left in the selected
quaternion algebra. A zero exponent returns identity; a zero base with a
nonzero exponent and a zero base logarithm are undefined and raise
`ValueError`. Its branch for negative real bases follows the positive-X axis
choice documented for `log`.

`commutator(lhs, rhs)` returns `lhs*rhs - rhs*lhs`. `scalar_part(q)` and
`vector_part(q)` return the real and three imaginary coefficients in the
selected module's storage order; for JPL, `vector_part` is the stored xyz
coefficient triple. `dot` and `norm_squared` operate on all four coefficients.

### Local attitude, interpolation, and sampled rates

`quaternion_derivative(q, omega, frame='body')` returns the coefficient
derivative for a physical, right-handed rate in radians per second. Hamilton
uses `0.5*q*(0, omega)` for body rates and `0.5*(0, omega)*q` for world
rates. Under the JPL negative-cross product, the equivalent pure-rate
quaternion has scalar coefficient zero and vector coefficient `-omega`.
The input magnitude is preserved in the derivative; it is not normalized.

`integrate_angular_velocity_samples(q, omegas, times, frame='body',
method='midpoint')` starts with the attitude at `times[0]` and composes one
constant-rate increment over each adjacent timestamp interval. `method='left'`
uses the interval's first sample; `method='midpoint'` averages its endpoint
samples. Times must be finite, nonempty, and strictly increasing, with one
rate sample per timestamp. A single sample returns the normalized initial
attitude. This is a piecewise constant/midpoint method, not an adaptive ODE
solver.

`boxplus(q, delta, frame='body')` applies a radians rotation-vector increment
on the right for body coordinates or on the left for world coordinates.
`boxminus(q, reference, frame='body')` returns the principal local rotation
vector that takes `reference` to `q` in the corresponding coordinates. Both
normalize their attitude inputs and reject zero quaternions.

`nlerp(q0, q1, t, shortest=True)` normalizes a linear coefficient blend.
With the default shortest path it flips the second sign when needed. `t` may
lie outside `[0, 1]` for extrapolation; if the blend is zero, the result is
undefined and raises `ValueError`. `squad(q0, q1, a, b, t)` evaluates the
standard nested-SLERP SQUAD expression with caller-supplied endpoint and
control rotations. It does not calculate control points for a keyframe
sequence; callers should choose sign-consistent controls. Both return unit
rotations.

`align_sign(q, reference)` chooses the sign of `q` with nonnegative
coefficient dot product against `reference`. `make_sign_continuous` applies
that choice successively along a sequence. These functions preserve each
input's magnitude and the first input sign; they do not normalize.

For NumPy, quaternion components remain on the final axis in the standard
modules, and on axis zero in `components_first` modules. Sampled quaternion
sequences have shape `(..., samples, 4)` or `(4, ..., samples)` respectively;
angular-rate samples use `(..., samples, 3)` or `(3, ..., samples)`. The
timestamp sample axis is the final axis in either layout. Batch axes broadcast
as in the other NumPy APIs.

### Attitude conversions, alignment, statistics, and random sampling

`from_euler(angles, sequence='xyz')` and `to_euler(q, sequence='xyz')` use
radians and the six Tait-Bryan axis permutations. A lowercase sequence such as
`'xyz'` means fixed/extrinsic rotations; the uppercase form `'XYZ'` means
body/intrinsic rotations. At gimbal lock, `to_euler` sets the third angle to
zero. The conversions match SciPy's sequence convention.

`from_mrp(sigma)` converts modified Rodrigues parameters to a canonical unit
quaternion. `to_mrp(q)` first chooses the canonical sign and returns the
principal representative, whose norm is at most one. As with other rotation
conversions, `q` and `-q` yield the same MRP result.

`swing_twist(q, axis)` returns `(swing, twist)` such that the normalized input
is their coefficient product in that order. The twist axis must be nonzero.
When a half-turn has no projected twist component, twist is identity and
swing is the input attitude. Factor signs are retained so recomposition also
matches the normalized input coefficients, including a negative input sign.

`align_vectors(source, target, weights=None)` solves the weighted Wahba
problem for the active rotation mapping each source vector to its target.
Corresponding vectors must be nonzero, weights must be nonnegative with at
least one positive value, and ambiguous correspondences raise `ValueError`.
Vector lengths are retained and act as observation strengths. The solver
rescales each whole vector set internally to avoid overflow without changing
the optimum. Weights are normalized internally, so multiplying all weights
or either complete vector set by a common positive factor does not change the
answer.

`rotation_mean(quaternions, weights=None, method='markley', tolerance=1e-12,
max_iterations=100)` returns either the dominant-eigenvector Markley mean or an
iterative intrinsic Karcher mean. Each input is normalized; weights are
nonnegative and scale invariant. Intrinsic iteration starts at the Markley
mean and raises `RuntimeError` if it does not converge within the requested
limit. `rotation_covariance(quaternions, mean=None, weights=None,
frame='body')` returns the weighted population covariance of principal local
rotation errors as a 3×3 matrix. By default it centers on the Markley mean;
`frame` selects body or world tangent coordinates.

NumPy mean and covariance inputs use the sample axis immediately before the
four quaternion components: `(..., samples, 4)` in trailing layout and
`(4, ..., samples)` in component-first layout. Weights have shape
`(..., samples)` and broadcast over the batch axes. Covariance matrices place
their two matrix axes last in trailing layout and first in component-first
layout.

`random_quaternion(size=None, rng=None)` draws uniform unit quaternions on
the 3-sphere using Shoemake's construction; `random_rotation` is an alias for
uniform SO(3) sampling with the same quaternion result. The returned sign is
not canonicalized. Scalar APIs accept an object whose `random()` method draws
three values in `[0, 1)`; NumPy APIs accept an object whose `random(size)`
returns the requested values, such as `numpy.random.Generator`. NumPy `size`
may be an integer or a tuple of nonnegative dimensions.

`to_homogeneous_matrix(q, translation=(0, 0, 0))` returns an active rigid
transform with the rotation in its upper-left 3×3 block, translation in its
last column, and bottom row `(0, 0, 0, 1)`. `from_homogeneous_matrix(matrix,
atol=1e-8)` validates that bottom row and the proper rotation block, then
returns only the rotation quaternion; translation is ignored. `so3_adjoint(q)`
returns the SO(3) adjoint, equal to the active rotation matrix for this
three-dimensional rotation group.

### SLERP, SO(3) Jacobians, and angular velocity

`slerp(q0, q1, t, shortest=True)` first normalizes nonzero endpoints. With the
default `shortest=True`, it flips the second endpoint when their coefficient
dot product is negative, so `q` and `-q` represent the same endpoint. Set
`shortest=False` to preserve the supplied signs; exactly antipodal endpoints
then have no unique interpolation path and raise `ValueError`. `t` is any
finite real, including values outside `[0, 1]` for extrapolation. Nearly equal
endpoints use normalized linear interpolation to avoid division by a small
sine. NumPy broadcasts endpoint batch dimensions and returns float64 arrays.

For a physical rotation vector `r`, let `K = [r/||r||]×` and `theta=||r||`.
The Jacobians use the standard left/right exponential-map convention:

```text
J_left(r)  = I + (1-cos(theta))/theta * K + (1-sin(theta)/theta) * K^2
J_right(r) = J_left(-r)
```

Their inverses use `J_left^-1 = I - theta/2*K +
(1-theta/2*cot(theta/2))*K^2` and `J_right^-1(r)=J_left^-1(-r)`.
Series expansions are used near zero. Inverse Jacobians are undefined at
nonzero multiples of `2*pi` and raise `ValueError` there. `skew_matrix(v)`
returns the unnormalized cross-product matrix satisfying `[v]× u = v × u`.
The inverse implementations also reject inputs whose computed
`|sin(theta/2)|` is at most `1e-15` to avoid returning an unstable matrix.
NumPy accepts batched `(..., 3)` vectors and returns `(..., 3, 3)` matrices.

`integrate_angular_velocity(q, omega, dt, frame='body')` assumes `q` is an
active, right-handed attitude and `omega` is a physical angular-velocity
vector in radians per second, constant over `dt`. It forms the rotation vector
`omega * dt`, then right-composes its increment for `frame='body'` and
left-composes it for `frame='world'`. Inputs are normalized and the result is
unit length. This is exact for constant angular velocity; it is not a
time-varying numerical integrator. `dt` may be negative. NumPy broadcasts
quaternion and angular-velocity leading dimensions.

The exponential and logarithm follow the algebraic formulas in Joan Solà's
[*Quaternion kinematics for the error-state Kalman filter*](https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf).
The general quaternion sine and cosine are included as power-series-derived
extensions; the paper uses scalar sine and cosine in the exponential but does
not separately develop those general functions.

These are algebraic quaternion functions. The `from_rotation_vector` and
`to_rotation_vector` names make the physical SO(3) conversion distinct; they
alias `from_rotvec` and `to_rotvec`. Hamilton and JPL return different vector
coefficient signs for the same physical rotation, as described in
[Axis-angle and rotation-vector conversions](#axis-angle-and-rotation-vector-conversions).

For trailing-axis NumPy modules, binary quaternion operations broadcast leading
dimensions according to NumPy rules. Thus `(2, 1, 4)` added to `(1, 3, 4)`
returns `(2, 3, 4)`; rotating `(2, 1, 4)` quaternions by `(1, 3, 3)` vectors
returns `(2, 3, 3)`. `norm` removes the final axis and yields a NumPy scalar
for shape `(4,)`. Component-first modules apply the same rules to the batch
axes after the leading component axis, and `norm` removes the first axis. No
output memory layout is guaranteed. Array-like inputs must infer a boolean, signed or
unsigned integer, or real floating dtype. Complex, object (including
`Fraction` or `Decimal`), string, and other dtypes are rejected even when their
values look real. Values are cast to float64, which can round or underflow;
conversion to a non-finite float64 raises `ValueError`.

### Matrix representations

The plain scalar modules, NamedTuple modules, dataclass modules, and all
fully specified NumPy modules provide `to_rotation_matrix`, `to_left_matrix`,
and `to_right_matrix`. The Hamilton scalar-compatible path is also available
as `quatops.scalar` and through its Hamilton namespace; JPL uses
`quatops.jpl.scalar`, `quatops.jpl.namedtuple`, and `quatops.jpl.dataclass`.
These are functions on the selected module, not methods on the quaternion
value. Scalar and typed results are nested tuples; NumPy results are float64
arrays. For a trailing-axis NumPy input shaped `(..., 4)`, rotation matrices
have shape `(..., 3, 3)` and multiplication matrices have shape `(..., 4, 4)`.
For component-first inputs `(4, ...)`, these outputs have shapes `(3, 3, ...)`
and `(4, 4, ...)`. Batch dimensions are preserved, including empty batches.
Matrix input uses the module's native coefficient ordering.

`to_rotation_matrix(q)` implements the active action already used by
`rotate(q, v)`. It is norm-aware: any finite nonzero quaternion is accepted,
and its magnitude does not change the returned rotation. Hamilton uses the
positive cross-product term and JPL the negative term, independently of
`wxyz` or `xyzw` storage. The resulting matrix maps column vectors, so
`R @ v` agrees with `rotate(q, v)`; a passive coordinate transform is its
transpose.

The multiplication 4×4 matrices act on quaternion coefficient vectors, not
on 3D points. `to_left_matrix(q)` satisfies `L(q) @ r == multiply(q, r)`, while
`to_right_matrix(q)` satisfies `P(q) @ r == multiply(r, q)`. These algebra
identities also hold for non-unit quaternions. Their scalar and vector blocks
follow the selected algebra and storage order, so the matrices should be used
with coefficient vectors in that same module's order.

The separate `to_homogeneous_matrix` conversion returns a rigid transform for
points; its matrix convention and extraction behavior are documented in
[attitude conversions](#attitude-conversions-alignment-statistics-and-random-sampling).

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
  Both rotation actions are active and right-handed, and `multiply(lhs, rhs)`
  composes rotations so that `rhs` acts first and `lhs` second.
- Quaternion algebra has no implicit normalization. In particular, `rotate`
  uses the inverse and accepts any finite nonzero quaternion: nonzero scalar multiples represent
  the same rotation. Internal scaling limits avoidable overflow without
  modifying the input or its represented rotation.
- Operations reject malformed component shapes or counts, non-real or
  non-finite inputs, incompatible broadcast dimensions, and zero quaternions
  in `normalize`, `inverse`, `divide`, and `rotate` with `ValueError`. One invalid entry
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

## Current scope

The implemented API includes the algebraic, attitude, interpolation,
statistics, matrix, and conversion operations described above across the
tuple, NamedTuple, dataclass, and NumPy representations, including both NumPy
component-axis layouts. Operator overloads, dual quaternions, and SE(3)/pose
composition remain outside the core library.
