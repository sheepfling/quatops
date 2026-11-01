# Proposed axis-angle and rotation-vector contract

**Historical proposal:** a provisional conversion milestone has since been
implemented. See [API.md](API.md) for current behavior. This document preserves
the earlier options and open decisions; it does not override the implementation.

**Original proposal for review.** The original [quaternion API](API.md) and
its open numerical limitations motivated this design. The goal is a shared operation contract with
representation-specific, independently selected modules and kernels: no common
runtime object, automatic dispatch, or mandatory cross-backend wrapper.

## Representation and layout

A rotation vector (RotVec) is `axis * angle`, with angle in **radians**; it is
not an `(axis, angle)` pair. This proposal uses a flat axis-angle (AxisAngle)
`(ax, ay, az, angle)` and a RotVec `(x, y, z)`. Each quaternion backend would
accept and return its own representation:

| Existing module | Proposed AxisAngle | Proposed RotVec |
| --- | --- | --- |
| `plainquat.scalar` | four-float tuple `(ax, ay, az, angle)` | three-float tuple `(x, y, z)` |
| `plainquat.namedtuple` | `AxisAngle(ax, ay, az, angle)` named tuple | `RotVec(x, y, z)` named tuple |
| `plainquat.dataclass` | frozen slotted `AxisAngle(ax, ay, az, angle)` | frozen slotted `RotVec(x, y, z)` |
| `plainquat.numpy` | float64 array `(..., 4)` | float64 array `(..., 3)` |

Named constructors would store supplied values without validation, coercion or
normalization, like the current `Quaternion` constructors. Operations, not
constructors, validate input. A dataclass can be iterable in its documented
component order without being a tuple; no output-contiguity guarantee is
proposed. NumPy quaternion layout remains scalar-first `(..., 4)`.

### Flat versus nested

| Choice | Advantages | Costs |
| --- | --- | --- |
| **Flat** `(ax, ay, az, angle)` (proposed) | One consistent final component axis, matches existing array conventions, straightforward named fields and batch storage. | The axis and angle are less visibly separate, and must be assembled to vary/broadcast them independently. |
| **Nested** `(axis, angle)` | Mirrors the mathematical concept and allows separate axis and angle arguments to broadcast naturally. | Heterogeneous tuple structure; no single uniform NumPy `(..., 4)` layout without defining a separate array convention. |

An alternative is separate `axis` and `angle` arguments for construction
while keeping flat returned values. That would add a multi-argument broadcast
contract; it is not included in the minimal proposal below. **Rick's choice of
layout remains open.**

## Proposed shared functions

Each function is called from the explicitly selected quaternion module. `q` is
that module's scalar-first quaternion representation, `a` its AxisAngle, and
`r` its RotVec. Results use that same module's native representation.

| Signature | Proposed behavior | Return |
| --- | --- | --- |
| `from_axis_angle(a)` | Convert an axis and radians to a canonical unit quaternion. | Backend quaternion |
| `to_axis_angle(q)` | Convert a finite, nonzero quaternion to canonical axis and radians. | Backend AxisAngle |
| `from_rotvec(r)` | Convert axis-times-angle radians to a canonical unit quaternion. | Backend quaternion |
| `to_rotvec(q)` | Convert a finite, nonzero quaternion to a canonical rotation vector. | Backend RotVec |

These are proposed additions to the existing module-level API, not new classes
that dispatch across backends. The operation names and meaning would match;
the tuple, named tuple, dataclass, and NumPy result types remain distinct.
Direct AxisAngle-to-RotVec functions, matrix conversion, interpolation and
rotation-vector operations are separate scope decisions.

```python
# Illustrative call pattern from the original proposal.
from math import pi
from plainquat import namedtuple as quat

turn = quat.AxisAngle(0, 0, 1, pi / 2)
q = quat.from_axis_angle(turn)     # Quaternion, approximately (sqrt(.5), 0, 0, sqrt(.5))
r = quat.to_rotvec(q)             # RotVec, approximately (0, 0, pi/2)
assert r.z > 0

from plainquat import numpy as qnp
qs = qnp.from_rotvec([[0, 0, 0], [0, 0, pi / 2]])  # shape (2, 4)
a = qnp.to_axis_angle(qs)                              # shape (2, 4)
```

## Proposed numerical semantics

- All angles are radians. Rotations are active and right-handed, consistent
  with the current Hamilton quaternion convention.
- **Axis input policy to decide:** the recommended starting point accepts any
  finite **nonzero** axis and explicitly normalizes its direction inside
  `from_axis_angle`; this conversion would document that normalization as part
  of its definition. It would reject a zero axis even when the angle is zero.
  The alternative is requiring a unit axis, but that needs an explicit
  tolerance and a decision about whether near-unit inputs are divided by their
  norm. Neither constructor silently normalizes data. Existing quaternion
  algebra operations would keep their current no-implicit-normalization rule.
- Quaternion-to-rotation conversion explicitly obtains the orientation of a
  finite, nonzero quaternion by scale-safe normalization/direction extraction;
  zero and nonfinite quaternions are rejected. This is mathematical
  normalization of direction, not a requirement to materialize a rounded
  binary64 unit quaternion first: doing so can erase representable small
  rotations. The original input need not be unit length. A zero RotVec is
  valid identity.
- Output angles are proposed in `[0, pi]`. Identity becomes AxisAngle
  `(1, 0, 0, 0)` and RotVec `(0, 0, 0)`, with positive zero for canonical
  zero fields. The chosen +X axis is a convention, not an observable rotation.
- Canonicalize quaternion sign from the **original finite input**: choose
  `w > 0`; when `w == 0`, choose the sign whose first numerically nonzero
  component of `(x, y, z)` is positive, treating both signs of zero as zero.
  Thus `q` and `-q` yield the same canonical outputs, including an exact
  half-turn. `from_*` would also return a canonical unit quaternion. A
  quaternion → rotation representation → quaternion round trip preserves the
  **rotation**, not the original magnitude or raw `q` versus `-q` sign.
  Compare actions or unit-quaternion absolute dot products, not raw components.
- For a small nonzero vector part, the candidate formula is
  `2 * atan2(norm(x, y, z), w)` after choosing the canonical sign, with
  scale-safe norms and a carefully computed axis/RotVec. Avoid `acos(w)` and
  avoid an arbitrary near-zero snap that discards representable rotations.
  Intermediate underflow, angle rounding and conditioning need explicit tests;
  no correctness or performance guarantee is implied by this proposal.
- **Exactly-pi boundary to decide:** recommended tie detection uses `w == 0`
  on the original quaternion, with no tolerance snap. An angle from nonzero
  `w` can nevertheless round to floating-point `pi`; its axis could then have
  either first-nonzero sign. Applying the tie instead to every rounded `pi`
  output would give more uniform-looking values but could alter a near-pi
  rotation. For an input AxisAngle explicitly containing the floating value
  `pi`, whether conversion treats it as an exact half-turn and emits `w=+0`
  needs a matching explicit policy.
- Negative and greater-than-pi input angles/RotVec magnitudes may represent
  valid rotations; proposed canonicalization reduces them to the output range.
  **Open:** whether to accept all finite magnitudes, restrict inputs to a
  practical range, or specify accuracy limits for large-angle argument
  reduction. If a finite RotVec's norm cannot be represented, a proposed
  `OverflowError` is preferable to an unannounced nonfinite output.

## Array and error behavior under the proposal

Every conversion above is unary: NumPy keeps all leading dimensions and changes
only the final component width, `(..., 4)` ↔ `(..., 3)` as appropriate. A
single quaternion `(4,)` yields a single AxisAngle `(4,)` or RotVec `(3,)`;
empty leading batches are valid. There is **no** independent axis/angle
broadcasting in this flat API. If a separate two-argument constructor is later
added, its leading broadcast shapes and zero-axis policy must be specified.

For consistency with the current backend, NumPy inputs would use accepted
boolean, integer or real-floating dtypes and convert to float64; complex,
object and string dtypes would be rejected. A malformed shape, nonfinite
component, zero quaternion or zero axis would raise `ValueError`; any invalid
batch entry would reject the entire call. A nonfinite computed result would
raise `OverflowError`. Inputs are not mutated, and NumPy operations would
preserve the caller's `numpy.seterr` settings. Scalar modules use the existing
real-iterable validation policy. These rules are proposed, including the
zero-axis decision, and need testing before being promised.

## Test plan before implementation is accepted

- One shared conformance table exercises all four modules and their native
  return types. Identity `q=(±1,0,0,0)`, a +Z quarter-turn, and exact half-turn
  `q=(0,-1,0,0)` establish sign, radians and the proposed +X/tie conventions.
- Check `q` and `-q`, finite nonzero scaling, negative and over-pi angles,
  zero RotVec, signed zero, exactly pi, and values immediately on each side of
  pi. Compare reconstructed rotation actions or absolute unit-quaternion dot
  products; do not require raw quaternion sign equality.
- Include `(1, 1e-300, 0, 0)` and `(2, 5e-324, 0, 0)`: the latter has a
  representable angle of `5e-324`, although materializing a unit quaternion
  first loses its vector part. Test `atan2` behavior without an arbitrary
  identity threshold. Compare exact
  known cases and high-precision references where needed, and specify tolerances
  and unresolved underflow limits rather than declaring universal precision.
- Validate zero axes (including zero angle), nonunit axes according to the
  chosen policy, invalid/nonfinite inputs, overflow, bad scalar lengths,
  NumPy trailing shapes, integer promotion and unsupported dtypes. Exercise
  empty and mixed-invalid batches, shape preservation, input non-mutation and
  unchanged `numpy.seterr`. If multi-argument functions are added, add
  incompatible and leading-dimension broadcast tests.
- Use SciPy **only as a development oracle**: explicitly adapt scalar-first
  quaternions to SciPy's default scalar-last order, account for SciPy's
  quaternion normalization and sign choices, and compare rotations. Adapt
  manually rather than depending on newer keyword options, since the dev
  dependency allows SciPy 1.7. Test the
  proposed exact-pi tie independently instead of assuming an oracle's branch
  choice. SciPy documents [rotation vectors and radians](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_rotvec.html)
  and [quaternion layout and canonical signs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html).

## Decisions for Rick

1. Flat AxisAngle versus nested axis/angle, and whether independent array
   broadcasting is worth a separate two-argument function.
2. Accept and explicitly normalize any nonzero axis during conversion, or
   require a unit axis with a defined tolerance; decide zero-axis identity.
3. Exactly-pi and near-pi tie behavior, including floating `pi` input, and
   whether canonical quaternion outputs are desired.
4. Accepted magnitude range and accuracy policy for large angles and extreme
   quaternion/axis ratios. The separate quaternion rotation precision
   investigation remains open and should not be mistaken for a resolved bound.
